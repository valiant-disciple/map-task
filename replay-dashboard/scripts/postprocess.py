#!/usr/bin/env python3
"""
Session ZIP → CSV dataset pipeline with GT scoring, HR features, prosody, and ASR (Smallest Pulse).

Usage:
  python scripts/postprocess.py \
    --zip /path/to/session.zip \
    --gt-dir "Ground Truth Maps" \
    --out out_dir \
    --smallest-key $SMALLEST_AI_KEY

Outputs in --out:
  metrics.csv, strokes.csv, hr_matcher.csv, hr_director.csv, hr_stats.csv (HRV+RQA),
  audio_manifest.csv, manifest.csv, prosody.csv, speech.csv (if ASR key).
"""

import argparse
import datetime
import io
import json
import math
import os
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import librosa
import soundfile as sf
import requests

# Optional RQA
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.computation import RQAComputation
    from pyrqa.metric import EuclideanMetric
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.analysis_type import Cross
    HAS_RQA = True
except Exception:
    HAS_RQA = False

INF = 1e12


def epoch_to_iso(t) -> str:
    """Convert Unix epoch milliseconds to ISO 8601 string, or empty if invalid."""
    if t is None or t == "":
        return ""
    try:
        return datetime.datetime.fromtimestamp(int(t) / 1000, tz=datetime.timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return ""


def load_gt(gt_dir: str, map_number: int):
    path = os.path.join(gt_dir, f"gt_{map_number}.json")
    with open(path, "r") as f:
        return json.load(f)


def stroke_draw(draw: ImageDraw.ImageDraw, stroke: dict, erase: bool):
    pts = stroke.get("polyline") or stroke.get("points") or []
    if len(pts) < 2:
        return
    width = stroke.get("width", 3 if not erase else 20)
    xy = [(p["x"], p["y"]) for p in pts if isinstance(p, dict) and "x" in p and "y" in p]
    if len(xy) < 2:
        return
    color = 255 if erase else 0
    draw.line(xy, fill=color, width=int(width))


def strokes_to_mask(strokes: List[dict], width: int, height: int) -> np.ndarray:
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for s in strokes:
        mode = s.get("mode", "draw")
        erase = mode == "erase"
        stroke_draw(draw, s, erase)
    mask = np.array(img)
    return (mask < 250).astype(np.uint8)


def binary_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt_f = gt > 0
    pr_f = pred > 0
    tp = np.logical_and(gt_f, pr_f).sum()
    fp = np.logical_and(~gt_f, pr_f).sum()
    fn = np.logical_and(gt_f, ~pr_f).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "dice": f1}


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_f = gt.astype(np.float32)
    pr_f = pred.astype(np.float32)
    mu_x = gt_f.mean()
    mu_y = pr_f.mean()
    var_x = ((gt_f - mu_x) ** 2).mean()
    var_y = ((pr_f - mu_y) ** 2).mean()
    cov = ((gt_f - mu_x) * (pr_f - mu_y)).mean()
    C1 = 6.5025
    C2 = 58.5225
    return float(((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)))


def hausdorff(gt: np.ndarray, pred: np.ndarray) -> float:
    def coords(mask):
        yx = np.argwhere(mask > 0)
        return yx
    A = coords(gt)
    B = coords(pred)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    from scipy.spatial.distance import cdist
    D = cdist(A, B)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def chamfer(gt: np.ndarray, pred: np.ndarray) -> float:
    dt_gt = distance_transform_edt(gt == 0)
    dt_pr = distance_transform_edt(pred == 0)
    a = dt_gt[pred > 0]
    b = dt_pr[gt > 0]
    mean_a = np.sqrt(a.mean()) if a.size else 0.0
    mean_b = np.sqrt(b.mean()) if b.size else 0.0
    return float((mean_a + mean_b) / 2)


def boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue
            if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                out[y, x] = 1
                continue
            neigh = mask[y - 1:y + 2, x - 1:x + 2]
            if neigh.min() == 0:
                out[y, x] = 1
    return out


def boundary_f(gt: np.ndarray, pred: np.ndarray, tol: int = 2) -> Dict[str, float]:
    gt_b = boundary(gt)
    pr_b = boundary(pred)
    dt_gt = distance_transform_edt(gt_b == 0)
    dt_pr = distance_transform_edt(pr_b == 0)
    tot_p = pr_b.sum()
    tot_r = gt_b.sum()
    tp_p = (dt_gt[pr_b > 0] <= tol).sum() if tot_p else 0
    tp_r = (dt_pr[gt_b > 0] <= tol).sum() if tot_r else 0
    precision = tp_p / tot_p if tot_p else 0.0
    recall = tp_r / tot_r if tot_r else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_events(events: List[dict], map_number: int) -> List[dict]:
    out = []
    for idx, e in enumerate(events):
        if e.get("type") != "draw_stroke" or e.get("role") != "matcher":
            continue
        if e.get("payload", {}).get("mapNumber") not in (None, map_number):
            continue
        pts = e.get("payload", {}).get("polyline") or e.get("payload", {}).get("points") or []
        out.append({
            **e.get("payload", {}),
            "points": pts,
            "mode": e.get("payload", {}).get("mode", "draw"),
            "t": e.get("t"),
            "strokeIndex": idx
        })
    return out


def csv_write(rows: List[dict], path: str):
    """Write rows to CSV, unioning headers across all rows."""
    if not rows:
        return
    headers = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = []
            for h in headers:
                v = r.get(h, "")
                if v is None:
                    v = ""
                s = str(v)
                if any(c in s for c in [",", "\"", "\n"]):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(",".join(vals) + "\n")


def prosody_features(audio_bytes: bytes, sr_target: int = 16000) -> Dict[str, float]:
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr_target, mono=True)
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y).flatten()
        zcr = librosa.feature.zero_crossing_rate(y).flatten()
        f0 = librosa.yin(y, fmin=50, fmax=500)
        f0 = f0[np.isfinite(f0)]
        return {
            "duration_sec": duration,
            "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
            "rms_std": float(np.std(rms)) if rms.size else 0.0,
            "zcr_mean": float(np.mean(zcr)) if zcr.size else 0.0,
            "f0_mean": float(np.mean(f0)) if f0.size else 0.0,
            "f0_median": float(np.median(f0)) if f0.size else 0.0,
            "f0_coverage": float(f0.size / rms.size) if rms.size else 0.0,
        }
    except Exception:
        return {}


def asr_smallest(audio_bytes: bytes, api_key: str):
    url = "https://api.smallest.ai/waves/v1/pulse/get_text"
    params = {"language": "en", "word_timestamps": "true"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "audio/wav"}
    resp = requests.post(url, params=params, headers=headers, data=audio_bytes, timeout=120)
    resp.raise_for_status()
    return resp.json()


def hr_features(hr: List[dict], baseline_mean: float = None, baseline_n: int = 0) -> Dict[str, float]:
    """
    HRV + (optional) RQA. If baseline_mean is provided, bpm values are
    mean-centered using that baseline before computing features.
    """
    if not hr:
        return {}
    bpms = [r["bpm"] for r in hr if isinstance(r.get("bpm"), (int, float, np.floating))]
    if not bpms:
        return {}
    baseline_used = baseline_mean is not None
    if baseline_used:
        bpms = [b - baseline_mean for b in bpms]
    mean = float(np.mean(bpms))
    std = float(np.std(bpms))
    minv = float(np.min(bpms))
    maxv = float(np.max(bpms))
    diff = np.diff(bpms)
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    pnn50 = float(np.mean(np.abs(diff) > 50)) if diff.size else 0.0
    feats = {
        "bpm_mean": mean,
        "bpm_std": std,
        "bpm_min": minv,
        "bpm_max": maxv,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "baseline_mean": baseline_mean if baseline_used else "",
        "baseline_n": baseline_n if baseline_used else "",
    }
    if HAS_RQA and len(bpms) >= 3:
        try:
            ts = TimeSeries(bpms, embedding_dimension=2, time_delay=1)
            threshold = max(0.1 * std, 0.01) if std > 0 else 0.1
            settings = Settings(ts, threshold=threshold, metric=EuclideanMetric)
            comp = RQAComputation.create(settings)
            res = comp.run()
            feats.update({
                "rqa_rr": float(res.recurrence_rate),
                "rqa_det": float(res.determinism),
                "rqa_lam": float(res.laminarity),
                "rqa_entr": float(res.entropy_diagonal_lines),
            })
        except Exception:
            # If RQA fails (e.g., singular data), return HRV-only stats
            pass
    return feats


def crqa_features(bpms_m: List[float], bpms_d: List[float],
                  baseline_m: float = None, baseline_d: float = None) -> Dict[str, float]:
    """Cross-Recurrence Quantification Analysis between matcher and director HR."""
    if not HAS_RQA or len(bpms_m) < 3 or len(bpms_d) < 3:
        return {}
    if baseline_m is not None:
        bpms_m = [b - baseline_m for b in bpms_m]
    if baseline_d is not None:
        bpms_d = [b - baseline_d for b in bpms_d]
    std_both = float(np.std(bpms_m + bpms_d))
    threshold = max(0.1 * std_both, 0.01) if std_both > 0 else 0.1
    try:
        ts_m = TimeSeries(bpms_m, embedding_dimension=2, time_delay=1)
        ts_d = TimeSeries(bpms_d, embedding_dimension=2, time_delay=1)
        settings = Settings((ts_m, ts_d),
                            analysis_type=Cross,
                            neighbourhood=FixedRadius(threshold),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=0)
        comp = RQAComputation.create(settings)
        res = comp.run()
        return {
            "crqa_rr": float(res.recurrence_rate),
            "crqa_det": float(res.determinism),
            "crqa_lam": float(res.laminarity),
            "crqa_entr": float(res.entropy_diagonal_lines),
            "crqa_mean_diag": float(res.average_diagonal_line_length),
            "crqa_max_diag": float(res.longest_diagonal_line_length),
        }
    except Exception:
        return {}


def parse_hr_csv(csv_text: str, role: str, trial: int) -> List[dict]:
    if not csv_text:
        return []
    lines = csv_text.strip().splitlines()
    if len(lines) <= 1:
        return []
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 3:
            continue
        t = int(parts[0]) if parts[0].isdigit() else None
        try:
            bpm = float(parts[2]) if parts[2] else None
        except (ValueError, TypeError):
            bpm = None
        phase = parts[3] if len(parts) > 3 else ""
        rows.append({"t": t, "bpm": bpm, "phase": phase, "role": role, "trial": trial})
    return rows


def process_zip(zip_path: str, gt_dir: str, out_dir: str, asr_key: str = None):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        trial_dirs = sorted({p.split("/")[1] for p in zf.namelist() if p.startswith("trials/") and len(p.split("/")) > 2})
        metrics_rows = []
        strokes_rows = []
        hr_matcher_rows = []
        hr_director_rows = []
        hr_stats_rows = []
        audio_rows = []
        manifest_rows = []
        prosody_rows = []
        speech_rows = []
        gt_cache = {}
        ts_rows = []
        audio_out_dir = os.path.join(out_dir, "audio")
        os.makedirs(audio_out_dir, exist_ok=True)

        zip_baseline = {"director": None, "matcher": None}
        try:
            bl_bytes = zf.read("session/hr_baseline.json")
            bl = json.loads(bl_bytes.decode("utf-8"))
            if isinstance(bl.get("director"), (int, float)):
                zip_baseline["director"] = float(bl["director"])
            if isinstance(bl.get("matcher"), (int, float)):
                zip_baseline["matcher"] = float(bl["matcher"])
        except Exception:
            pass

        for tdir in trial_dirs:
            try:
                trial_idx = int(tdir.replace("T", ""))
            except (ValueError, TypeError):
                continue
            def read(path):
                try:
                    with zf.open(path) as f:
                        return f.read()
                except KeyError:
                    return None

            events_bytes = read(f"trials/{tdir}/events.json")
            strokes_bytes = read(f"trials/{tdir}/strokes.json")
            hr_m_bytes = read(f"trials/{tdir}/hr/hr_matcher.csv")
            hr_d_bytes = read(f"trials/{tdir}/hr/hr_director.csv")

            events = json.loads(events_bytes.decode("utf-8")) if events_bytes else []
            map_number = None
            for e in reversed(events):
                mn = e.get("payload", {}).get("mapNumber")
                if isinstance(mn, int):
                    map_number = mn
                    break
            if map_number is None:
                continue

            raw_strokes = parse_events(events, map_number)
            if not raw_strokes and strokes_bytes:
                try:
                    st = json.loads(strokes_bytes.decode("utf-8"))
                    if isinstance(st, list):
                        raw_strokes = st
                except Exception:
                    pass
            # filter drawable
            strokes = []
            for s in raw_strokes:
                pts = s.get("points") or s.get("polyline") or []
                pts = [p for p in pts if isinstance(p, dict) and "x" in p and "y" in p]
                if len(pts) < 2:
                    continue
                mode = s.get("mode", "draw")
                if mode not in ("draw", "erase"):
                    continue
                strokes.append({**s, "points": pts, "mode": mode})

            if map_number not in gt_cache:
                gt_cache[map_number] = load_gt(gt_dir, map_number)
            gt = gt_cache[map_number]
            width = gt.get("image", {}).get("width", 1024)
            height = gt.get("image", {}).get("height", 1024)
            gt_mask = strokes_to_mask(gt.get("strokes", []), width, height)
            pred_mask = strokes_to_mask(strokes, width, height)

            m = binary_metrics(gt_mask, pred_mask)
            s = ssim(gt_mask, pred_mask)
            h = hausdorff(gt_mask, pred_mask)
            cd = chamfer(gt_mask, pred_mask)
            bf = boundary_f(gt_mask, pred_mask, 2)
            coverage_gt = float(gt_mask.mean())
            coverage_pred = float(pred_mask.mean())

            metrics_rows.append({
                "sessionId": os.path.basename(zip_path),
                "trial": trial_idx,
                "mapNumber": map_number,
                **m,
                "ssim": s,
                "hausdorff": h,
                "chamfer": cd,
                "boundary_f1": bf["f1"],
                "boundary_p": bf["precision"],
                "boundary_r": bf["recall"],
                "coverage_gt": coverage_gt,
                "coverage_pred": coverage_pred,
            })

            for sidx, sraw in enumerate(strokes):
                stroke_t = sraw.get("t", "")
                for pidx, p in enumerate(sraw.get("points", [])):
                    pt_t = p.get("t", "") if isinstance(p, dict) else ""
                    t_val = pt_t if pt_t else stroke_t
                    strokes_rows.append({
                        "sessionId": os.path.basename(zip_path),
                        "trial": trial_idx,
                        "mapNumber": map_number,
                        "strokeIndex": sraw.get("strokeIndex", sidx),
                        "pointIndex": pidx,
                        "t_unix_ms": t_val,
                        "t_iso": epoch_to_iso(t_val),
                        "stroke_t_unix_ms": stroke_t,
                        "mode": sraw.get("mode", "draw"),
                        "x": p["x"],
                        "y": p["y"],
                    })

            # HR
            hr_m = parse_hr_csv(hr_m_bytes.decode("utf-8"), "matcher", trial_idx) if hr_m_bytes else []
            hr_d = parse_hr_csv(hr_d_bytes.decode("utf-8"), "director", trial_idx) if hr_d_bytes else []

            def add_hr_rows(role_rows, rows, role):
                baseline_mean = zip_baseline.get(role)
                baseline_n = 1 if baseline_mean is not None else 0
                for r in rows:
                    t_val = r.get("t", "")
                    role_rows.append({
                        "sessionId": os.path.basename(zip_path),
                        "trial": trial_idx,
                        "role": role,
                        "kind": "raw",
                        "t_unix_ms": t_val,
                        "t_iso": epoch_to_iso(t_val),
                        "bpm": r.get("bpm", ""),
                        "phase": r.get("phase", ""),
                        "n": "",
                        "bpm_mean": "",
                        "bpm_min": "",
                        "bpm_max": "",
                        "bpm_std": "",
                    })
                bpms = [r["bpm"] for r in rows if isinstance(r.get("bpm"), (int, float, np.floating))]
                if bpms:
                    role_rows.append({
                        "sessionId": os.path.basename(zip_path),
                        "trial": trial_idx,
                        "role": role,
                        "kind": "summary",
                        "t_unix_ms": "",
                        "t_iso": "",
                        "bpm": "",
                        "phase": "",
                        "n": len(bpms),
                        "bpm_mean": float(np.mean(bpms)),
                        "bpm_min": float(np.min(bpms)),
                        "bpm_max": float(np.max(bpms)),
                        "bpm_std": float(np.std(bpms)),
                        "baseline_mean": baseline_mean if baseline_mean is not None else "",
                        "baseline_n": baseline_n if baseline_mean is not None else "",
                    })
                    hf = hr_features(rows, baseline_mean=baseline_mean, baseline_n=baseline_n)
                    if hf:
                        stats_row = {"sessionId": os.path.basename(zip_path), "trial": trial_idx, "role": role}
                        stats_row.update(hf)
                        hr_stats_rows.append(stats_row)

            add_hr_rows(hr_matcher_rows, hr_m, "matcher")
            add_hr_rows(hr_director_rows, hr_d, "director")

            # Cross-recurrence (MDRQA) between matcher and director
            bpms_m = [r["bpm"] for r in hr_m if isinstance(r.get("bpm"), (int, float, np.floating))]
            bpms_d = [r["bpm"] for r in hr_d if isinstance(r.get("bpm"), (int, float, np.floating))]
            base_m = zip_baseline.get("matcher")
            base_d = zip_baseline.get("director")
            crqa = crqa_features(bpms_m, bpms_d, baseline_m=base_m, baseline_d=base_d)
            if crqa:
                crqa_row = {"sessionId": os.path.basename(zip_path), "trial": trial_idx, "role": "cross"}
                crqa_row.update(crqa)
                hr_stats_rows.append(crqa_row)

            # Audio
            for rel in zf.namelist():
                if not rel.startswith(f"trials/{tdir}/audio/") or rel.endswith("/"):
                    continue
                try:
                    with zf.open(rel) as f:
                        audio_bytes = f.read()
                except Exception:
                    continue
                fname = os.path.basename(rel)
                # Save original audio for downstream use
                try:
                    out_audio_path = os.path.join(audio_out_dir, f"T{trial_idx:02d}_{fname}")
                    with open(out_audio_path, "wb") as af:
                        af.write(audio_bytes)
                except Exception:
                    pass
                audio_rows.append({
                    "sessionId": os.path.basename(zip_path),
                    "trial": trial_idx,
                    "filename": fname,
                    "bytes": len(audio_bytes)
                })
                # Prosody
                pf = prosody_features(audio_bytes)
                if pf:
                    prosody_rows.append({"sessionId": os.path.basename(zip_path), "trial": trial_idx, "filename": fname, **pf})
                # ASR
                if asr_key:
                    try:
                        sr = 16000
                        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
                        buf = io.BytesIO()
                        sf.write(buf, y, sr, format="WAV")
                        asr_json = asr_smallest(buf.getvalue(), asr_key)
                        text = asr_json.get("text") or asr_json.get("transcript") or ""
                        conf = asr_json.get("confidence") or asr_json.get("score") or ""
                        speech_rows.append({
                            "sessionId": os.path.basename(zip_path),
                            "trial": trial_idx,
                            "filename": fname,
                            "text": text,
                            "confidence": conf
                        })
                    except Exception as e:
                        speech_rows.append({
                            "sessionId": os.path.basename(zip_path),
                            "trial": trial_idx,
                            "filename": fname,
                            "text": "",
                            "confidence": "",
                            "error": str(e)[:200]
                        })

            # Extract trial-level metadata from events
            trial_start_t = ""
            trial_end_t = ""
            target_reached = ""
            path_confidence = ""
            director_note = ""
            tlx_director = {}
            tlx_matcher = {}
            for e in events:
                etype = e.get("type", "")
                if etype == "trial_final_time":
                    t_val = e.get("t", "")
                    if not trial_end_t or (t_val and t_val > trial_end_t):
                        trial_end_t = t_val
                if etype == "draw_stroke" and not trial_start_t:
                    trial_start_t = e.get("t", "")
                if etype == "trial_success":
                    p = e.get("payload", {})
                    target_reached = p.get("targetReached", "")
                    path_confidence = p.get("pathConfidence", "")
                    director_note = p.get("note", "")
                if etype == "tlx_submit":
                    role = e.get("role", "")
                    p = e.get("payload", {})
                    if role == "director":
                        tlx_director = p
                    elif role == "matcher":
                        tlx_matcher = p

            manifest_rows.append({
                "sessionId": os.path.basename(zip_path),
                "trial": trial_idx,
                "mapNumber": map_number,
                "trial_start_ms": trial_start_t,
                "trial_start_iso": epoch_to_iso(trial_start_t),
                "trial_end_ms": trial_end_t,
                "trial_end_iso": epoch_to_iso(trial_end_t),
                "strokes": len(strokes),
                "strokePoints": sum(len(s.get("points", [])) for s in strokes),
                "hr_matcher": len(hr_m),
                "hr_director": len(hr_d),
                "audio_count": len([r for r in audio_rows if r["trial"] == trial_idx]),
                "coverage_gt": coverage_gt,
                "coverage_pred": coverage_pred,
                "target_reached": target_reached,
                "path_confidence": path_confidence,
                "director_note": director_note,
                "tlx_mental_d": tlx_director.get("mental", ""),
                "tlx_physical_d": tlx_director.get("physical", ""),
                "tlx_temporal_d": tlx_director.get("temporal", ""),
                "tlx_performance_d": tlx_director.get("performance", ""),
                "tlx_effort_d": tlx_director.get("effort", ""),
                "tlx_frustration_d": tlx_director.get("frustration", ""),
                "tlx_mental_m": tlx_matcher.get("mental", ""),
                "tlx_physical_m": tlx_matcher.get("physical", ""),
                "tlx_temporal_m": tlx_matcher.get("temporal", ""),
                "tlx_performance_m": tlx_matcher.get("performance", ""),
                "tlx_effort_m": tlx_matcher.get("effort", ""),
                "tlx_frustration_m": tlx_matcher.get("frustration", ""),
            })

            # Time-series correctness per stroke
            current_mask = np.zeros_like(pred_mask)
            step = 0
            for sraw in sorted(strokes, key=lambda s: s.get("t") or 0):
                # draw this stroke onto current_mask
                img = Image.fromarray((current_mask == 0).astype(np.uint8) * 255, mode="L")
                draw = ImageDraw.Draw(img)
                erase = sraw.get("mode", "draw") == "erase"
                stroke_draw(draw, sraw, erase)
                current_mask = (np.array(img) == 0).astype(np.uint8)

                step += 1
                if step % 5 != 0 and step != len(strokes):
                    continue  # downsample to every 5th stroke, keep final
                m_ts = binary_metrics(gt_mask, current_mask)
                cd_ts = chamfer(gt_mask, current_mask)
                bf_ts = boundary_f(gt_mask, current_mask, 2)
                t_val = sraw.get("t", "")
                ts_rows.append({
                    "sessionId": os.path.basename(zip_path),
                    "trial": trial_idx,
                    "mapNumber": map_number,
                    "step": step,
                    "t_unix_ms": t_val,
                    "t_iso": epoch_to_iso(t_val),
                    "iou": m_ts["iou"],
                    "f1": m_ts["f1"],
                    "dice": m_ts["dice"],
                    "precision": m_ts["precision"],
                    "recall": m_ts["recall"],
                    "chamfer": cd_ts,
                    "boundary_f1": bf_ts["f1"],
                    "boundary_p": bf_ts["precision"],
                    "boundary_r": bf_ts["recall"],
                    "coverage_pred": float(current_mask.mean()),
                })

        # Write CSVs
        out = os.path.join(out_dir, "metrics.csv"); csv_write(metrics_rows, out)
        out = os.path.join(out_dir, "strokes.csv"); csv_write(strokes_rows, out)
        out = os.path.join(out_dir, "hr_matcher.csv"); csv_write(hr_matcher_rows, out)
        out = os.path.join(out_dir, "hr_director.csv"); csv_write(hr_director_rows, out)
        out = os.path.join(out_dir, "audio_manifest.csv"); csv_write(audio_rows, out)
        out = os.path.join(out_dir, "manifest.csv"); csv_write(manifest_rows, out)
        if prosody_rows: csv_write(prosody_rows, os.path.join(out_dir, "prosody.csv"))
        if speech_rows: csv_write(speech_rows, os.path.join(out_dir, "speech.csv"))
        if hr_stats_rows: csv_write(hr_stats_rows, os.path.join(out_dir, "hr_stats.csv"))
        if ts_rows: csv_write(ts_rows, os.path.join(out_dir, "time_series_metrics.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Session ZIP path")
    ap.add_argument("--gt-dir", default="Ground Truth Maps", help="GT JSON directory (gt_#.json)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--smallest-key", default=None, help="Smallest Pulse API key (optional for ASR)")
    args = ap.parse_args()
    process_zip(args.zip, args.gt_dir, args.out, args.smallest_key or os.getenv("SMALLEST_AI_KEY"))


if __name__ == "__main__":
    main()
