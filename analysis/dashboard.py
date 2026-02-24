#!/usr/bin/env python3
"""
Map Task Analysis Dashboard (Self-Contained)
=============================================
Upload a session ZIP → automatic preprocessing → interactive visualisation.

Run:
    cd analysis && streamlit run dashboard.py
"""

import io
import os
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Optional heavy deps (graceful fallback) ─────────────────────────
try:
    from pydub import AudioSegment

    _HAS_PYDUB = True
except ImportError:
    _HAS_PYDUB = False

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    from scipy.interpolate import interp1d
    from scipy.signal import medfilt
    from scipy.stats import pearsonr

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ═════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Map Task Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent / "data"


# ═════════════════════════════════════════════════════════════════════
#  LOADING HELPERS
# ═════════════════════════════════════════════════════════════════════

def _jload(path: str | Path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _csv_load(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


# ═════════════════════════════════════════════════════════════════════
#  INLINE PREPROCESSING — Surveys
# ═════════════════════════════════════════════════════════════════════

TLX_DIMS = ["mental", "physical", "temporal", "performance", "effort", "frustration"]


def _score_tlx(raw_list: list) -> dict:
    """Score NASA-TLX from list of submissions. Uses last entry."""
    if not raw_list:
        return {"error": "empty"}
    entry = raw_list[-1] if isinstance(raw_list, list) else raw_list
    scores = {}
    vals = []
    for d in TLX_DIMS:
        v = entry.get(d)
        if v is not None:
            scores[d] = float(v)
            vals.append(float(v))
    scores["overall"] = round(np.mean(vals), 2) if vals else None
    return scores


def _score_psmm(raw_list: list) -> dict:
    """Score PSMM. Handles both flat list and nested-dict formats."""
    if not raw_list:
        return {"error": "empty"}
    entry = raw_list[-1] if isinstance(raw_list, list) else raw_list

    items = {}
    # The ZIP stores PSMM as {"0": {factor, itemNum, value, text}, "1": ...}
    for key, val in entry.items():
        if not isinstance(val, dict):
            continue
        item_num = val.get("itemNum")
        value = val.get("value")
        factor = val.get("factor", "")
        text = val.get("text", "")
        if item_num is not None and value is not None:
            items[int(item_num)] = {
                "value": float(value),
                "factor": factor,
                "text": text,
            }

    if not items:
        return {"error": "no items parsed"}

    task_vals = [v["value"] for k, v in items.items() if "task" in v["factor"]]
    team_vals = [v["value"] for k, v in items.items() if "team" in v["factor"]]
    all_vals = [v["value"] for v in items.values()]

    return {
        "task_mean": round(np.mean(task_vals), 3) if task_vals else None,
        "team_mean": round(np.mean(team_vals), 3) if team_vals else None,
        "overall": round(np.mean(all_vals), 3) if all_vals else None,
        "items": items,
    }


def _extract_demographics(events: list) -> dict:
    result = {}
    for e in events:
        if e.get("type") == "demographics":
            p = e.get("payload", {})
            role = e.get("role") or p.get("role", "unknown")
            result[role] = {
                "age": p.get("age"),
                "gender": p.get("gender"),
                "handedness": p.get("handedness"),
                "nativeLanguage": p.get("nativeLanguage"),
                "englishFluency": p.get("englishFluency"),
                "partnerFamiliarity": p.get("partnerFamiliarity"),
                "priorMapTask": p.get("priorMapTask"),
                "hearingDifficulties": p.get("hearingDifficulties"),
                "visionCorrected": p.get("visionCorrected"),
                "notes": p.get("notes", ""),
            }
    return result


def _extract_debrief(events: list) -> dict:
    for e in reversed(events):
        if e.get("type") == "debrief_submit":
            return e.get("payload", {})
    return {}


def _extract_trial_success(events: list) -> dict:
    for e in reversed(events):
        if e.get("type") == "trial_success":
            p = e.get("payload", {})
            return {
                "reported": True,
                "target_reached": p.get("targetReached"),
                "path_confidence": p.get("pathConfidence"),
                "note": p.get("note", ""),
            }
    return {"reported": False}


# ═════════════════════════════════════════════════════════════════════
#  INLINE PREPROCESSING — HR
# ═════════════════════════════════════════════════════════════════════

def _process_hr_csv(df: pd.DataFrame) -> dict:
    """Basic HR stats from a raw CSV DataFrame."""
    df = df.copy()
    df["timestamp_unix_ms"] = pd.to_numeric(df["timestamp_unix_ms"], errors="coerce")
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df = df.dropna(subset=["timestamp_unix_ms", "bpm"]).sort_values("timestamp_unix_ms").reset_index(drop=True)
    if len(df) == 0:
        return {"error": "empty"}
    t0 = df["timestamp_unix_ms"].iloc[0]
    df["t_sec"] = (df["timestamp_unix_ms"] - t0) / 1000.0
    # Clamp physio range
    df = df[(df["bpm"] >= 30) & (df["bpm"] <= 220)]
    stats = {
        "n": len(df),
        "duration_sec": round(float(df["t_sec"].iloc[-1] - df["t_sec"].iloc[0]), 2) if len(df) > 1 else 0,
        "mean": round(float(df["bpm"].mean()), 1),
        "std": round(float(df["bpm"].std()), 1),
        "min": round(float(df["bpm"].min()), 1),
        "max": round(float(df["bpm"].max()), 1),
    }
    ts = df[["t_sec", "bpm"]].to_dict(orient="records")
    return {"stats": stats, "timeseries": ts, "raw_df": df}


def _align_hr_pair(hr_dir: dict, hr_mat: dict) -> dict:
    """Align two HR time series and compute Pearson r."""
    df_a = hr_dir.get("raw_df")
    df_b = hr_mat.get("raw_df")
    if df_a is None or df_b is None or len(df_a) < 3 or len(df_b) < 3:
        return {"error": "insufficient data"}

    t_a = df_a["timestamp_unix_ms"].values / 1000.0
    t_b = df_b["timestamp_unix_ms"].values / 1000.0
    v_a = df_a["bpm"].values.astype(float)
    v_b = df_b["bpm"].values.astype(float)

    t_start = max(t_a[0], t_b[0])
    t_end = min(t_a[-1], t_b[-1])
    if t_end <= t_start + 2:
        return {"error": "no overlap"}

    common_t = np.arange(0, t_end - t_start, 1.0)
    if len(common_t) < 3:
        return {"error": "overlap too short"}

    abs_t = common_t + t_start

    if _HAS_SCIPY:
        ia = interp1d(t_a, v_a, kind="linear", bounds_error=False, fill_value="extrapolate")
        ib = interp1d(t_b, v_b, kind="linear", bounds_error=False, fill_value="extrapolate")
        aligned_a = ia(abs_t)
        aligned_b = ib(abs_t)
    else:
        aligned_a = np.interp(abs_t, t_a, v_a)
        aligned_b = np.interp(abs_t, t_b, v_b)

    r = float(np.corrcoef(aligned_a, aligned_b)[0, 1]) if len(aligned_a) > 2 else None
    if r is not None and np.isnan(r):
        r = None

    return {
        "t_sec": common_t.tolist(),
        "dir_bpm": aligned_a.tolist(),
        "mat_bpm": aligned_b.tolist(),
        "pearson_r": round(r, 4) if r is not None else None,
        "duration_sec": round(float(t_end - t_start), 1),
        "samples": len(common_t),
    }


# ═════════════════════════════════════════════════════════════════════
#  INLINE PREPROCESSING — Path (from events.json draw_stroke events)
# ═════════════════════════════════════════════════════════════════════

def _extract_strokes_from_events(events: list) -> list:
    """Extract draw strokes from events.json (draw_stroke events)."""
    strokes = []
    for e in events:
        if e.get("type") != "draw_stroke":
            continue
        p = e.get("payload", {})
        points = p.get("points", [])
        mode = p.get("mode", "draw")
        role = e.get("role", "matcher")
        if points and mode != "erase":
            strokes.append({"points": points, "mode": mode, "role": role})
    return strokes


def _path_length(points: list) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def _analyze_strokes(strokes: list) -> dict:
    all_points = []
    total_len = 0.0
    for s in strokes:
        pts = s["points"]
        all_points.extend(pts)
        total_len += _path_length(pts)
    return {
        "num_strokes": len(strokes),
        "total_points": len(all_points),
        "path_length": round(total_len, 1),
    }


# ═════════════════════════════════════════════════════════════════════
#  INLINE PREPROCESSING — Speech (optional, needs pydub + librosa)
# ═════════════════════════════════════════════════════════════════════

def _process_audio(webm_path: str) -> dict | None:
    """Process a single WebM file. Returns prosody features or None."""
    if not _HAS_PYDUB or not _HAS_LIBROSA:
        return None
    try:
        wav_path = webm_path.rsplit(".", 1)[0] + ".wav"
        if not os.path.exists(wav_path):
            audio = AudioSegment.from_file(webm_path, format="webm")
            audio.export(wav_path, format="wav")
        y, sr = librosa.load(wav_path, sr=None)
        dur = librosa.get_duration(y=y, sr=sr)

        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
        f0c = f0[~np.isnan(f0)]
        pitch = {
            "mean": round(float(np.mean(f0c)), 1) if len(f0c) > 0 else 0,
            "std": round(float(np.std(f0c)), 1) if len(f0c) > 0 else 0,
            "range": round(float(np.ptp(f0c)), 1) if len(f0c) > 0 else 0,
        }

        # Speaking rate
        non_silent = librosa.effects.split(y, top_db=25)
        speech_sec = sum(e - s for s, e in non_silent) / sr
        pauses = []
        for i in range(1, len(non_silent)):
            gap = (non_silent[i][0] - non_silent[i - 1][1]) / sr
            if gap > 0.3:
                pauses.append(round(gap, 2))

        # Energy time series for plot
        rms = librosa.feature.rms(y=y)[0]
        rms_t = librosa.times_like(rms, sr=sr)
        step = max(1, len(rms_t) // 300)
        energy_ts = [{"t": round(float(rms_t[i]), 2), "rms": round(float(rms[i]), 5)}
                     for i in range(0, len(rms_t), step)]

        # Pitch time series
        f0_t = librosa.times_like(f0, sr=sr)
        step2 = max(1, len(f0_t) // 300)
        pitch_ts = [{"t": round(float(f0_t[i]), 2), "hz": round(float(f0[i]), 1) if not np.isnan(f0[i]) else None}
                    for i in range(0, len(f0_t), step2)]

        return {
            "duration_sec": round(dur, 2),
            "speech_sec": round(speech_sec, 2),
            "speech_ratio": round(speech_sec / dur, 3) if dur > 0 else 0,
            "num_pauses": len(pauses),
            "mean_pause": round(float(np.mean(pauses)), 2) if pauses else 0,
            "pitch": pitch,
            "energy_ts": energy_ts,
            "pitch_ts": pitch_ts,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ═════════════════════════════════════════════════════════════════════
#  SESSION LOADER  (ZIP upload or data/ dir)
# ═════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Processing session…")
def load_session(session_dir: str) -> dict:
    """Load + preprocess an entire session from an extracted directory."""
    base = Path(session_dir)
    session_json = _jload(base / "session" / "session.json") or {}
    all_events = _jload(base / "session" / "events.json") or []

    meta = session_json.get("session", {})
    config = session_json.get("config", {})
    trial_meta_list = session_json.get("trials", [])

    # Demographics & debrief from global events
    demographics = _extract_demographics(all_events)
    debrief = _extract_debrief(all_events)

    # Build trial map from session.json
    trial_meta_map = {}
    for tm in trial_meta_list:
        ti = tm.get("trialIndex")
        trial_meta_map[ti] = tm

    # Discover trial folders
    trials_dir = base / "trials"
    trial_folders = sorted(trials_dir.iterdir()) if trials_dir.is_dir() else []

    trials = {}
    for tf in trial_folders:
        if not tf.is_dir() or not tf.name.startswith("T"):
            continue
        tname = tf.name
        ti = int(tname[1:])  # e.g. "T03" -> 3

        # Events
        trial_events = _jload(tf / "events.json") or []

        # Trial success
        trial_success = _extract_trial_success(trial_events)

        # Timing from session meta
        tmeta = trial_meta_map.get(ti, {})
        final_times = tmeta.get("finalTimes", [])
        map_num = tmeta.get("mapNumber")
        maps = tmeta.get("maps", {})

        # TLX
        tlx_dir = _score_tlx(_jload(tf / "tlx_director.json") or [])
        tlx_mat = _score_tlx(_jload(tf / "tlx_matcher.json") or [])

        # PSMM
        psmm_dir = _score_psmm(_jload(tf / "psmm_director.json") or [])
        psmm_mat = _score_psmm(_jload(tf / "psmm_matcher.json") or [])

        # HR
        hr_director = None
        hr_matcher = None
        hr_sync = None
        hr_dir_path = tf / "hr" / "hr_director.csv"
        hr_mat_path = tf / "hr" / "hr_matcher.csv"
        if hr_dir_path.exists():
            hr_director = _process_hr_csv(pd.read_csv(hr_dir_path))
        if hr_mat_path.exists():
            hr_matcher = _process_hr_csv(pd.read_csv(hr_mat_path))
        if hr_director and hr_matcher and "error" not in hr_director and "error" not in hr_matcher:
            hr_sync = _align_hr_pair(hr_director, hr_matcher)

        # Clean raw_df from HR results (not serializable)
        for hr in [hr_director, hr_matcher]:
            if hr and "raw_df" in hr:
                del hr["raw_df"]

        # Path (from events)
        strokes = _extract_strokes_from_events(trial_events)
        path_info = _analyze_strokes(strokes)

        # Speech (optional)
        speech = {}
        audio_dir = tf / "audio"
        if audio_dir.is_dir():
            for af in sorted(audio_dir.iterdir()):
                if af.suffix == ".webm":
                    role = "director" if "director" in af.name.lower() else "matcher"
                    sp = _process_audio(str(af))
                    if sp:
                        speech[role] = sp
                    else:
                        speech[role] = {"skipped": True, "file": af.name,
                                        "size_kb": round(af.stat().st_size / 1024, 1)}

        # Elapsed time
        elapsed_sec = None
        if final_times:
            # Find the actual elapsed from the first person who ended
            for ft in final_times:
                if ft.get("elapsedSec") and ft["elapsedSec"] > 0:
                    elapsed_sec = ft["elapsedSec"]
                    break

        trials[tname] = {
            "index": ti,
            "map_number": map_num,
            "maps": maps,
            "elapsed_sec": elapsed_sec,
            "final_times": final_times,
            "trial_success": trial_success,
            "tlx_director": tlx_dir,
            "tlx_matcher": tlx_mat,
            "psmm_director": psmm_dir,
            "psmm_matcher": psmm_mat,
            "hr_director": hr_director,
            "hr_matcher": hr_matcher,
            "hr_sync": hr_sync,
            "path": path_info,
            "strokes": strokes,
            "speech": speech,
            "event_count": len(trial_events),
        }

    return {
        "session_id": meta.get("id", "?"),
        "created_at": meta.get("createdAt"),
        "config": config,
        "demographics": demographics,
        "debrief": debrief,
        "trials": trials,
        "total_events": len(all_events),
    }


# ═════════════════════════════════════════════════════════════════════
#  FIND / EXTRACT SESSIONS
# ═════════════════════════════════════════════════════════════════════

def _find_existing_sessions() -> list[str]:
    """Find extracted sessions in data/ dir."""
    if not DATA_DIR.is_dir():
        return []
    results = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir() and (p / "session" / "session.json").exists():
            results.append(str(p))
    return results


def _extract_uploaded_zip(uploaded_file) -> str:
    """Extract uploaded ZIP to data/ dir, return extraction path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    name = uploaded_file.name.replace(".zip", "").replace("map_task_session_", "")
    out = DATA_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
        zf.extractall(out)
    return str(out)


# ═════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════

st.sidebar.title("🗺️ Map Task Dashboard")
st.sidebar.markdown("---")

# Upload
uploaded = st.sidebar.file_uploader("📤 Upload Session ZIP", type=["zip"])
if uploaded:
    with st.spinner("Extracting ZIP…"):
        extracted_path = _extract_uploaded_zip(uploaded)
    st.sidebar.success(f"Extracted → `{Path(extracted_path).name}`")

# Session list
existing = _find_existing_sessions()
if not existing:
    st.title("🗺️ Map Task Analysis Dashboard")
    st.info("Upload a session ZIP using the sidebar to get started.")
    st.stop()

session_labels = [Path(p).name for p in existing]
sel_idx = st.sidebar.selectbox("📂 Session", range(len(session_labels)), format_func=lambda i: session_labels[i])
session_path = existing[sel_idx]

# Load session
data = load_session(session_path)
trials = data["trials"]
trial_names = sorted(trials.keys())

if not trial_names:
    st.warning("No trial data found.")
    st.stop()

# Nav
view = st.sidebar.radio(
    "View",
    ["📊 Overview", "🔍 Trial Detail", "❤️ HR Analysis", "🎤 Speech",
     "📋 Surveys", "🗺️ Path", "📈 Trends"],
)

# Dep status
dep_status = []
if _HAS_PYDUB and _HAS_LIBROSA:
    dep_status.append("🎤 Speech ✅")
else:
    dep_status.append("🎤 Speech ❌ (install pydub+librosa)")
if _HAS_SCIPY:
    dep_status.append("📐 Scipy ✅")
else:
    dep_status.append("📐 Scipy ⚠️ (basic mode)")
st.sidebar.markdown("---")
st.sidebar.caption(" | ".join(dep_status))


# ═════════════════════════════════════════════════════════════════════
#  VIEW: OVERVIEW
# ═════════════════════════════════════════════════════════════════════

if view == "📊 Overview":
    st.title("📊 Session Overview")
    cfg = data["config"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Session", data["session_id"])
    c2.metric("Map Set", cfg.get("mapSet", "?"))
    c3.metric("Trials", len(trial_names))
    c4.metric("Warmup", cfg.get("warmupCount", "?"))
    c5.metric("Events", data["total_events"])

    # Demographics
    demo = data.get("demographics", {})
    if demo:
        st.markdown("---")
        st.subheader("👥 Participants")
        dcols = st.columns(len(demo))
        for i, (role, info) in enumerate(demo.items()):
            with dcols[i]:
                st.markdown(f"**{role.title()}**")
                if isinstance(info, dict):
                    for k, v in info.items():
                        if v not in (None, "", False):
                            st.text(f"  {k}: {v}")

    # Trial table
    st.markdown("---")
    st.subheader("📋 Trial Summary")
    rows = []
    for tn in trial_names:
        t = trials[tn]
        row = {
            "Trial": tn,
            "Map": t.get("map_number", "?"),
            "Elapsed (s)": t.get("elapsed_sec", "-"),
            "Target": "✅" if t["trial_success"].get("target_reached") else ("❌" if t["trial_success"].get("target_reached") is False else "-"),
            "Confidence": t["trial_success"].get("path_confidence", "-"),
            "TLX Dir": t["tlx_director"].get("overall", "-"),
            "TLX Mat": t["tlx_matcher"].get("overall", "-"),
            "PSMM Dir": t["psmm_director"].get("overall", "-"),
            "PSMM Mat": t["psmm_matcher"].get("overall", "-"),
            "Strokes": t["path"].get("num_strokes", 0),
            "Path Len": t["path"].get("path_length", 0),
        }
        hr_d = t.get("hr_director")
        hr_m = t.get("hr_matcher")
        row["HR Dir"] = hr_d["stats"]["mean"] if hr_d and "stats" in hr_d else "-"
        row["HR Mat"] = hr_m["stats"]["mean"] if hr_m and "stats" in hr_m else "-"
        sync = t.get("hr_sync")
        row["HR Sync r"] = sync.get("pearson_r", "-") if sync and "error" not in sync else "-"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Debrief
    debrief = data.get("debrief", {})
    if debrief:
        st.markdown("---")
        st.subheader("💬 Debrief")
        for k, v in debrief.items():
            if v:
                st.markdown(f"**{k}:** {v}")


# ═════════════════════════════════════════════════════════════════════
#  VIEW: TRIAL DETAIL
# ═════════════════════════════════════════════════════════════════════

elif view == "🔍 Trial Detail":
    st.title("🔍 Trial Detail")
    sel = st.sidebar.selectbox("Trial", trial_names)
    t = trials[sel]

    st.header(f"Trial {sel}  (Map {t.get('map_number', '?')})")
    mc = st.columns(5)
    mc[0].metric("Elapsed", f"{t.get('elapsed_sec', '?')}s")
    mc[1].metric("Target", "✅" if t["trial_success"].get("target_reached") else "❌")
    mc[2].metric("Confidence", f"{t['trial_success'].get('path_confidence', '?')}/7")
    mc[3].metric("Strokes", t["path"]["num_strokes"])
    mc[4].metric("Events", t["event_count"])

    st.markdown("---")

    # Surveys side by side
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### NASA-TLX")
        for role in ["director", "matcher"]:
            tlx = t[f"tlx_{role}"]
            if "error" not in tlx:
                vals = {d: tlx.get(d, 0) for d in TLX_DIMS}
                st.markdown(f"**{role.title()}** — Overall: **{tlx.get('overall', '?')}**/100")
                fig = go.Figure(go.Bar(x=list(vals.keys()), y=list(vals.values()),
                                       marker_color=["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6", "#e67e22"]))
                fig.update_layout(yaxis=dict(range=[0, 100]), height=250, margin=dict(t=10, b=30))
                st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("#### PSMM")
        for role in ["director", "matcher"]:
            psmm = t[f"psmm_{role}"]
            if "error" not in psmm:
                st.markdown(f"**{role.title()}** — Task: **{psmm.get('task_mean')}** | Team: **{psmm.get('team_mean')}** | Overall: **{psmm.get('overall')}**")
                items = psmm.get("items", {})
                if items:
                    idf = pd.DataFrame([
                        {"Item": f"Q{k}", "Value": v["value"],
                         "Factor": "Task" if "task" in v["factor"] else "Team"}
                        for k, v in sorted(items.items())
                    ])
                    fig = px.bar(idf, x="Item", y="Value", color="Factor",
                                 color_discrete_map={"Task": "#3498db", "Team": "#e74c3c"})
                    fig.update_layout(yaxis=dict(range=[0, 7]), height=250, margin=dict(t=10, b=30))
                    st.plotly_chart(fig, use_container_width=True)

    # HR
    st.markdown("---")
    st.markdown("#### ❤️ Heart Rate")
    hr_col1, hr_col2 = st.columns(2)
    for i, role in enumerate(["director", "matcher"]):
        hr = t.get(f"hr_{role}")
        with [hr_col1, hr_col2][i]:
            if hr and "stats" in hr:
                s = hr["stats"]
                st.markdown(f"**{role.title()}**: {s['mean']} ± {s['std']} bpm  ({s['n']} readings, {s['duration_sec']}s)")
                ts_data = hr.get("timeseries", [])
                if ts_data:
                    hdf = pd.DataFrame(ts_data)
                    fig = px.line(hdf, x="t_sec", y="bpm", title=f"{role.title()} HR")
                    fig.update_layout(height=250, margin=dict(t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No HR data for {role}")

    sync = t.get("hr_sync")
    if sync and "error" not in sync:
        st.markdown(f"**Synchronized** — Pearson r = **{sync.get('pearson_r')}** ({sync['duration_sec']}s overlap, {sync['samples']} samples)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["dir_bpm"], name="Director", line=dict(color="#e74c3c")))
        fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["mat_bpm"], name="Matcher", line=dict(color="#3498db")))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="BPM", height=350, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Path
    st.markdown("---")
    st.markdown("#### 🗺️ Drawn Path")
    if t["strokes"]:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, stroke in enumerate(t["strokes"]):
            pts = stroke["points"]
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Stroke {i + 1}",
                                     line=dict(width=2, color=colors[i % len(colors)]), showlegend=False))
        fig.update_layout(height=450, yaxis=dict(autorange="reversed", scaleanchor="x"),
                          xaxis=dict(constrain="domain"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No strokes recorded for this trial.")

    # Speech
    st.markdown("---")
    st.markdown("#### 🎤 Speech")
    for role in ["director", "matcher"]:
        sp = t["speech"].get(role)
        if not sp:
            st.info(f"No audio for {role}")
            continue
        if sp.get("skipped"):
            st.caption(f"**{role.title()}**: `{sp['file']}` ({sp['size_kb']} KB) — install pydub+librosa to process")
            continue
        if sp.get("error"):
            st.warning(f"{role.title()}: {sp['error']}")
            continue
        st.markdown(f"**{role.title()}**: {sp['duration_sec']}s total, {sp['speech_sec']}s speech ({sp['speech_ratio']:.0%}), "
                    f"{sp['num_pauses']} pauses, pitch {sp['pitch']['mean']} ± {sp['pitch']['std']} Hz")


# ═════════════════════════════════════════════════════════════════════
#  VIEW: HR ANALYSIS
# ═════════════════════════════════════════════════════════════════════

elif view == "❤️ HR Analysis":
    st.title("❤️ Heart Rate Analysis")
    sel = st.sidebar.selectbox("Trial", trial_names)
    t = trials[sel]

    # Individual
    for role in ["director", "matcher"]:
        hr = t.get(f"hr_{role}")
        if not hr or "stats" not in hr:
            st.info(f"No HR data for {role}")
            continue
        s = hr["stats"]
        st.subheader(f"{role.title()}")
        mc = st.columns(5)
        mc[0].metric("Mean", f"{s['mean']} bpm")
        mc[1].metric("Std", f"{s['std']}")
        mc[2].metric("Min", f"{s['min']}")
        mc[3].metric("Max", f"{s['max']}")
        mc[4].metric("Readings", s["n"])

        ts_data = hr.get("timeseries", [])
        if ts_data:
            hdf = pd.DataFrame(ts_data)
            fig = px.line(hdf, x="t_sec", y="bpm")
            fig.update_layout(height=300, margin=dict(t=10, b=30))
            st.plotly_chart(fig, use_container_width=True)

    # Sync
    sync = t.get("hr_sync")
    if sync and "error" not in sync:
        st.markdown("---")
        st.subheader("Synchrony")
        sc = st.columns(3)
        sc[0].metric("Pearson r", sync.get("pearson_r", "?"))
        sc[1].metric("Overlap", f"{sync['duration_sec']}s")
        sc[2].metric("Samples", sync["samples"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["dir_bpm"], name="Director", line=dict(color="#e74c3c", width=2)))
        fig.add_trace(go.Scatter(x=sync["t_sec"], y=sync["mat_bpm"], name="Matcher", line=dict(color="#3498db", width=2)))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="BPM", height=400,
                          legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
    elif sync:
        st.warning(f"Sync: {sync.get('error', 'unknown issue')}")

    # Cross-trial HR summary
    st.markdown("---")
    st.subheader("Cross-Trial HR")
    hr_rows = []
    for tn in trial_names:
        tt = trials[tn]
        row = {"Trial": tn}
        for role in ["director", "matcher"]:
            hr = tt.get(f"hr_{role}")
            row[f"{role.title()} Mean"] = hr["stats"]["mean"] if hr and "stats" in hr else None
            row[f"{role.title()} Std"] = hr["stats"]["std"] if hr and "stats" in hr else None
        s = tt.get("hr_sync")
        row["Sync r"] = s.get("pearson_r") if s and "error" not in s else None
        hr_rows.append(row)
    hdf = pd.DataFrame(hr_rows)
    st.dataframe(hdf, use_container_width=True, hide_index=True)

    if hdf["Sync r"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hdf["Trial"], y=hdf["Sync r"], mode="lines+markers",
                                 line=dict(color="#2ecc71", width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_title="Pearson r", yaxis=dict(range=[-1, 1]), height=350, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
#  VIEW: SPEECH
# ═════════════════════════════════════════════════════════════════════

elif view == "🎤 Speech":
    st.title("🎤 Speech Analysis")
    sel = st.sidebar.selectbox("Trial", trial_names)
    t = trials[sel]

    if not t["speech"]:
        st.info("No speech data. Upload audio files or install pydub + librosa.")
        st.stop()

    for role in ["director", "matcher"]:
        sp = t["speech"].get(role)
        st.subheader(role.title())
        if not sp:
            st.info("No audio file")
            continue
        if sp.get("skipped"):
            st.caption(f"`{sp['file']}` ({sp['size_kb']} KB)")
            st.warning("Install `pydub` and `librosa` for speech analysis")
            continue
        if sp.get("error"):
            st.error(sp["error"])
            continue

        mc = st.columns(5)
        mc[0].metric("Duration", f"{sp['duration_sec']}s")
        mc[1].metric("Speech", f"{sp['speech_sec']}s")
        mc[2].metric("Ratio", f"{sp['speech_ratio']:.0%}")
        mc[3].metric("Pauses", sp["num_pauses"])
        mc[4].metric("Pitch", f"{sp['pitch']['mean']} Hz")

        # Plots
        col_a, col_b = st.columns(2)
        with col_a:
            pts = sp.get("pitch_ts", [])
            if pts:
                pdf = pd.DataFrame(pts).dropna(subset=["hz"])
                if len(pdf) > 0:
                    fig = px.line(pdf, x="t", y="hz", title="Pitch (F0)")
                    fig.update_layout(height=250, margin=dict(t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)
        with col_b:
            ets = sp.get("energy_ts", [])
            if ets:
                edf = pd.DataFrame(ets)
                fig = px.line(edf, x="t", y="rms", title="Energy (RMS)")
                fig.update_layout(height=250, margin=dict(t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    # Cross-trial speech summary
    st.subheader("Cross-Trial Speech")
    sp_rows = []
    for tn in trial_names:
        tt = trials[tn]
        row = {"Trial": tn}
        for role in ["director", "matcher"]:
            sp = tt["speech"].get(role, {})
            if sp and not sp.get("skipped") and not sp.get("error"):
                row[f"{role[:3].title()} Speech%"] = sp.get("speech_ratio")
                row[f"{role[:3].title()} Pauses"] = sp.get("num_pauses")
                row[f"{role[:3].title()} Pitch"] = sp.get("pitch", {}).get("mean")
        sp_rows.append(row)
    if sp_rows:
        st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════
#  VIEW: SURVEYS
# ═════════════════════════════════════════════════════════════════════

elif view == "📋 Surveys":
    st.title("📋 Survey Results")
    sel = st.sidebar.selectbox("Trial", trial_names)
    t = trials[sel]

    # Trial Success
    st.subheader("🎯 Trial Success")
    ts = t["trial_success"]
    if ts.get("reported"):
        sc = st.columns(3)
        sc[0].metric("Target Reached", "✅ Yes" if ts.get("target_reached") else "❌ No")
        sc[1].metric("Path Confidence", f"{ts.get('path_confidence', '?')}/7")
        if ts.get("note"):
            sc[2].info(f"Note: {ts['note']}")
    else:
        st.info("Not reported")

    st.markdown("---")

    # TLX
    st.subheader("📊 NASA-TLX")
    tcol = st.columns(2)
    for i, role in enumerate(["director", "matcher"]):
        with tcol[i]:
            st.markdown(f"### {role.title()}")
            tlx = t[f"tlx_{role}"]
            if "error" in tlx:
                st.warning(tlx["error"])
                continue
            st.metric("Overall Workload", f"{tlx.get('overall', '?')}/100")
            vals = {d: tlx.get(d, 0) for d in TLX_DIMS}
            fig = go.Figure(go.Bar(
                x=list(vals.keys()), y=list(vals.values()),
                marker_color=["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6", "#e67e22"],
            ))
            fig.update_layout(yaxis=dict(range=[0, 100]), height=300, margin=dict(t=10, b=30))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # PSMM
    st.subheader("🧠 Perceived Shared Mental Models")
    pcol = st.columns(2)
    for i, role in enumerate(["director", "matcher"]):
        with pcol[i]:
            st.markdown(f"### {role.title()}")
            psmm = t[f"psmm_{role}"]
            if "error" in psmm:
                st.warning(psmm["error"])
                continue
            mc = st.columns(3)
            mc[0].metric("Task SMM", psmm.get("task_mean", "?"))
            mc[1].metric("Team SMM", psmm.get("team_mean", "?"))
            mc[2].metric("Overall", psmm.get("overall", "?"))
            items = psmm.get("items", {})
            if items:
                idf = pd.DataFrame([
                    {"Item": f"Q{k}", "Value": v["value"], "Factor": "Task" if "task" in v["factor"] else "Team",
                     "Text": v.get("text", "")}
                    for k, v in sorted(items.items())
                ])
                fig = px.bar(idf, x="Item", y="Value", color="Factor",
                             color_discrete_map={"Task": "#3498db", "Team": "#e74c3c"},
                             hover_data=["Text"])
                fig.update_layout(yaxis=dict(range=[0, 7]), height=300, margin=dict(t=10, b=30))
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
#  VIEW: PATH
# ═════════════════════════════════════════════════════════════════════

elif view == "🗺️ Path":
    st.title("🗺️ Path Analysis")
    sel = st.sidebar.selectbox("Trial", trial_names)
    t = trials[sel]

    pi = t["path"]
    mc = st.columns(3)
    mc[0].metric("Strokes", pi["num_strokes"])
    mc[1].metric("Points", pi["total_points"])
    mc[2].metric("Path Length", f"{pi['path_length']} px")

    if t["strokes"]:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, stroke in enumerate(t["strokes"]):
            pts = stroke["points"]
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     name=f"Stroke {i + 1}",
                                     line=dict(width=2, color=colors[i % len(colors)])))
        fig.update_layout(height=550, yaxis=dict(autorange="reversed", scaleanchor="x"),
                          xaxis=dict(constrain="domain"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        # Timing analysis
        st.subheader("⏱️ Stroke Timing")
        stroke_rows = []
        for i, stroke in enumerate(t["strokes"]):
            pts = stroke["points"]
            if len(pts) >= 2:
                dur = (pts[-1]["t"] - pts[0]["t"]) / 1000.0
                length = _path_length(pts)
                speed = length / dur if dur > 0 else 0
                stroke_rows.append({
                    "Stroke": i + 1,
                    "Points": len(pts),
                    "Duration (s)": round(dur, 2),
                    "Length (px)": round(length, 1),
                    "Speed (px/s)": round(speed, 1),
                })
        if stroke_rows:
            st.dataframe(pd.DataFrame(stroke_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No strokes recorded.")

    # Cross-trial path summary
    st.markdown("---")
    st.subheader("Cross-Trial Path")
    path_rows = []
    for tn in trial_names:
        tt = trials[tn]
        pi = tt["path"]
        path_rows.append({
            "Trial": tn,
            "Strokes": pi["num_strokes"],
            "Points": pi["total_points"],
            "Path Length": pi["path_length"],
        })
    pdf = pd.DataFrame(path_rows)
    st.dataframe(pdf, use_container_width=True, hide_index=True)
    if pdf["Path Length"].sum() > 0:
        fig = px.bar(pdf, x="Trial", y="Path Length", color="Strokes",
                     title="Path Length per Trial")
        fig.update_layout(height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
#  VIEW: CROSS-TRIAL TRENDS
# ═════════════════════════════════════════════════════════════════════

elif view == "📈 Trends":
    st.title("📈 Cross-Trial Trends")

    if len(trial_names) < 2:
        st.info("Need ≥ 2 trials.")
        st.stop()

    rows = []
    for tn in trial_names:
        t = trials[tn]
        row = {"Trial": tn}

        row["TLX Dir"] = t["tlx_director"].get("overall")
        row["TLX Mat"] = t["tlx_matcher"].get("overall")
        row["PSMM Task Dir"] = t["psmm_director"].get("task_mean")
        row["PSMM Task Mat"] = t["psmm_matcher"].get("task_mean")
        row["PSMM Team Dir"] = t["psmm_director"].get("team_mean")
        row["PSMM Team Mat"] = t["psmm_matcher"].get("team_mean")

        hr_d = t.get("hr_director")
        hr_m = t.get("hr_matcher")
        row["HR Dir"] = hr_d["stats"]["mean"] if hr_d and "stats" in hr_d else None
        row["HR Mat"] = hr_m["stats"]["mean"] if hr_m and "stats" in hr_m else None
        sync = t.get("hr_sync")
        row["HR Sync r"] = sync.get("pearson_r") if sync and "error" not in sync else None

        row["Path Length"] = t["path"]["path_length"]
        row["Strokes"] = t["path"]["num_strokes"]

        row["Target"] = 1 if t["trial_success"].get("target_reached") else 0
        row["Confidence"] = t["trial_success"].get("path_confidence")
        row["Elapsed (s)"] = t.get("elapsed_sec")

        for role in ["director", "matcher"]:
            sp = t["speech"].get(role, {})
            if sp and not sp.get("skipped") and not sp.get("error"):
                row[f"Speech% {role[:3]}"] = sp.get("speech_ratio")

        rows.append(row)

    df = pd.DataFrame(rows)

    # TLX trends
    st.subheader("📊 Workload (NASA-TLX)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Trial"], y=df["TLX Dir"], name="Director", mode="lines+markers", line=dict(color="#e74c3c")))
    fig.add_trace(go.Scatter(x=df["Trial"], y=df["TLX Mat"], name="Matcher", mode="lines+markers", line=dict(color="#3498db")))
    fig.update_layout(yaxis_title="Overall Workload (0-100)", height=350, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    # PSMM trends
    st.subheader("🧠 PSMM")
    fig = go.Figure()
    for role, color in [("Dir", "#e74c3c"), ("Mat", "#3498db")]:
        fig.add_trace(go.Scatter(x=df["Trial"], y=df[f"PSMM Task {role}"], name=f"Task ({role})",
                                 mode="lines+markers", line=dict(color=color)))
        fig.add_trace(go.Scatter(x=df["Trial"], y=df[f"PSMM Team {role}"], name=f"Team ({role})",
                                 mode="lines+markers", line=dict(color=color, dash="dash")))
    fig.update_layout(yaxis_title="PSMM (1-7)", yaxis=dict(range=[0.5, 7.5]), height=350, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    # HR sync trend
    if df["HR Sync r"].notna().any():
        st.subheader("❤️ HR Synchrony")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Trial"], y=df["HR Sync r"], mode="lines+markers",
                                 line=dict(color="#2ecc71", width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_title="Pearson r", yaxis=dict(range=[-1, 1]), height=350, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Path
    st.subheader("🗺️ Path")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Path Length", "Strokes"])
    fig.add_trace(go.Bar(x=df["Trial"], y=df["Path Length"], marker_color="#9b59b6"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["Trial"], y=df["Strokes"], marker_color="#f39c12"), row=1, col=2)
    fig.update_layout(height=350, showlegend=False, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    # Confidence & success
    st.subheader("🎯 Performance")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Path Confidence", "Target Reached"])
    fig.add_trace(go.Bar(x=df["Trial"], y=df["Confidence"], marker_color="#3498db"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["Trial"], y=df["Target"], marker_color="#2ecc71"), row=1, col=2)
    fig.update_yaxes(range=[0, 7], row=1, col=1)
    fig.update_yaxes(range=[0, 1.2], row=1, col=2)
    fig.update_layout(height=350, showlegend=False, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("🔗 Correlations")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=max(400, 50 * len(num_cols)), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Raw data table
    with st.expander("📊 Raw Data Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)
