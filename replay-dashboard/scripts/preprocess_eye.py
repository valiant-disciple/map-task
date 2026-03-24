#!/usr/bin/env python3
"""
Eye-tracker data preprocessor: Aurora (iMotions CSV) and SmartEye Pro 10 (.log TSV)
→ unified gaze CSV with AOI labels and trial alignment.

Usage:
  python scripts/preprocess_eye.py \
    --eye-file path/to/eye_data.csv \
    --format aurora \
    --role director \
    --zip path/to/session.zip \
    --out eye_preprocessed.csv
"""

import argparse
import datetime
import json
import math
import os
import re
import zipfile

AOI_CONFIG = {
    "director": {
        "map": {"x1": 252, "y1": 137, "x2": 889, "y2": 1017},
        "timer": {"x1": 613, "y1": 8, "x2": 735, "y2": 65},
        "toolbar": {"x1": 236, "y1": 0, "x2": 1018, "y2": 74},
    },
    "matcher": {
        "map": {"x1": 267, "y1": 137, "x2": 904, "y2": 1017},
        "timer": {"x1": 613, "y1": 8, "x2": 735, "y2": 65},
        "toolbar": {"x1": 236, "y1": 0, "x2": 1365, "y2": 74},
    },
}

OUTPUT_COLUMNS = [
    "t_unix_ms", "t_iso", "trial", "gaze_x", "gaze_y", "aoi",
    "pupil_left", "pupil_right", "head_pitch", "head_yaw", "head_roll",
    "fixation_idx", "fixation_x", "fixation_y", "fixation_duration",
    "saccade_idx", "saccade_amplitude", "saccade_peak_velocity",
    "saccade_direction", "gaze_velocity", "blink",
    "eyelid_left", "eyelid_right", "role", "source",
]

WINDOWS_EPOCH_OFFSET_MS = 11644473600000


def epoch_to_iso(t) -> str:
    if t is None or t == "":
        return ""
    try:
        return datetime.datetime.fromtimestamp(
            int(t) / 1000, tz=datetime.timezone.utc
        ).isoformat()
    except (ValueError, TypeError, OSError):
        return ""


def csv_write(rows: list, path: str):
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
                if any(c in s for c in [",", '"', "\n"]):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(",".join(vals) + "\n")


def _safe_float(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(v):
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def classify_aoi(gaze_x, gaze_y, role: str) -> str:
    if gaze_x is None or gaze_y is None:
        return "missing"
    aois = AOI_CONFIG.get(role, AOI_CONFIG["director"])
    t = aois["timer"]
    if t["x1"] <= gaze_x <= t["x2"] and t["y1"] <= gaze_y <= t["y2"]:
        return "timer"
    m = aois["map"]
    if m["x1"] <= gaze_x <= m["x2"] and m["y1"] <= gaze_y <= m["y2"]:
        return "map"
    tb = aois["toolbar"]
    if tb["x1"] <= gaze_x <= tb["x2"] and tb["y1"] <= gaze_y <= tb["y2"]:
        return "toolbar"
    return "other"


# ---------------------------------------------------------------------------
# Trial boundaries from session ZIP
# ---------------------------------------------------------------------------

def extract_trial_boundaries(zip_path: str) -> list:
    """Return sorted list of (trial_label, start_ms, end_ms)."""
    trials = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trial_dirs = sorted(
                {p.split("/")[1] for p in zf.namelist()
                 if p.startswith("trials/") and len(p.split("/")) > 2}
            )
            for tdir in trial_dirs:
                events_path = f"trials/{tdir}/events.json"
                try:
                    with zf.open(events_path) as ef:
                        events = json.loads(ef.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    continue
                if not events:
                    continue

                start_ms = None
                end_ms = None
                for e in events:
                    et = e.get("t")
                    if et is None:
                        continue
                    t_val = _safe_int(et)
                    if t_val is None:
                        continue
                    if e.get("type") == "draw_stroke" and start_ms is None:
                        start_ms = t_val
                    if e.get("type") == "trial_final_time":
                        end_ms = t_val

                if start_ms is None and events:
                    for e in events:
                        t_val = _safe_int(e.get("t"))
                        if t_val is not None:
                            start_ms = t_val
                            break
                if end_ms is None and events:
                    for e in reversed(events):
                        t_val = _safe_int(e.get("t"))
                        if t_val is not None:
                            end_ms = t_val
                            break
                if start_ms is not None and end_ms is not None:
                    trials.append((tdir, start_ms, end_ms))
    except (zipfile.BadZipFile, FileNotFoundError) as exc:
        print(f"Warning: could not read ZIP for trial boundaries: {exc}")
    return sorted(trials, key=lambda x: x[1])


def extract_flash_timestamps(zip_path: str) -> list:
    """Return list of {trial_label, flash_ts} from sync_flash events in events.json."""
    flashes = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trial_dirs = sorted(
                {p.split("/")[1] for p in zf.namelist()
                 if p.startswith("trials/") and len(p.split("/")) > 2}
            )
            for tdir in trial_dirs:
                events_path = f"trials/{tdir}/events.json"
                try:
                    with zf.open(events_path) as ef:
                        events = json.loads(ef.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    continue
                for e in events:
                    if e.get("type") == "sync_flash":
                        payload = e.get("payload", {})
                        flash_ts = _safe_int(payload.get("flashTs") or e.get("t"))
                        if flash_ts:
                            flashes.append({"trial": tdir, "flash_ts": flash_ts})
    except (zipfile.BadZipFile, FileNotFoundError):
        pass
    return flashes


def extract_clock_offset(zip_path: str) -> dict:
    """Return clock_offset event data from events.json if present."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trial_dirs = sorted(
                {p.split("/")[1] for p in zf.namelist()
                 if p.startswith("trials/") and len(p.split("/")) > 2}
            )
            # clock_offset is typically in the first trial's events
            for tdir in trial_dirs:
                events_path = f"trials/{tdir}/events.json"
                try:
                    with zf.open(events_path) as ef:
                        events = json.loads(ef.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    continue
                for e in events:
                    if e.get("type") == "clock_offset":
                        return e.get("payload", {})
            # Also check top-level events.json
            for name in zf.namelist():
                if name == "events.json" or name.endswith("/events.json"):
                    try:
                        with zf.open(name) as ef:
                            events = json.loads(ef.read().decode("utf-8"))
                        for e in events:
                            if e.get("type") == "clock_offset":
                                return e.get("payload", {})
                    except (KeyError, json.JSONDecodeError):
                        continue
    except (zipfile.BadZipFile, FileNotFoundError):
        pass
    return {}


def detect_flash_in_pupil(rows: list, flash_ts_unix_ms: int,
                          search_window_ms: int = 2000) -> int:
    """
    Detect the pupil constriction onset caused by a sync flash.

    Searches eye tracker data around flash_ts for a sharp pupil diameter drop.
    Returns the eye-tracker timestamp (t_unix_ms) of the detected flash onset,
    or 0 if not detected.
    """
    # Gather pupil samples in the search window
    window_start = flash_ts_unix_ms - search_window_ms
    window_end = flash_ts_unix_ms + search_window_ms
    samples = []
    for r in rows:
        t = _safe_int(r.get("t_unix_ms"))
        if t is None or t < window_start or t > window_end:
            continue
        pl = _safe_float(r.get("pupil_left"))
        pr = _safe_float(r.get("pupil_right"))
        vals = [v for v in [pl, pr] if v is not None and v > 0]
        if vals:
            samples.append((t, sum(vals) / len(vals)))

    if len(samples) < 10:
        return 0

    # Find the sharpest pupil diameter drop (constriction onset)
    # The flash causes a sudden decrease ~200-500ms after the flash
    # Look for the steepest negative derivative
    best_drop = 0
    best_t = 0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        if dt <= 0:
            continue
        dp = samples[i][1] - samples[i - 1][1]
        rate = dp / dt  # mm/ms — negative = constriction
        if rate < best_drop:
            best_drop = rate
            best_t = samples[i-1][0]

    return best_t


def assign_trial(t_unix_ms, trial_boundaries: list) -> str:
    if t_unix_ms is None:
        return "no_trial"
    for label, start, end in trial_boundaries:
        if start <= t_unix_ms <= end:
            return label
    return "no_trial"


# ---------------------------------------------------------------------------
# Aurora (iMotions) CSV parser
# ---------------------------------------------------------------------------

def _parse_csv_line(line: str) -> list:
    """Minimal CSV field splitter that handles quoted fields."""
    fields = []
    current = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == '"' and i + 1 < len(line) and line[i + 1] == '"':
                current.append('"')
                i += 2
                continue
            elif ch == '"':
                in_quotes = False
            else:
                current.append(ch)
        else:
            if ch == '"':
                in_quotes = True
            elif ch == ',':
                fields.append("".join(current))
                current = []
            else:
                current.append(ch)
        i += 1
    fields.append("".join(current))
    return fields


def _col(header_map: dict, row: list, name: str, default=""):
    idx = header_map.get(name)
    if idx is None or idx >= len(row):
        return default
    v = row[idx]
    return v if v != "" else default


def parse_aurora(eye_path: str, role: str, trial_boundaries: list) -> list:
    recording_unix_s = None
    data_header = None
    rows_out = []

    with open(eye_path, "r", encoding="utf-8", errors="replace") as f:
        found_data_marker = False
        for raw_line in f:
            line = raw_line.rstrip("\n\r")

            if line.startswith("#"):
                m = re.search(r"Unix time:\s*(\d+)", line)
                if m and recording_unix_s is None:
                    recording_unix_s = int(m.group(1))
                if line.startswith("#DATA"):
                    found_data_marker = True
                continue

            if found_data_marker and data_header is None:
                data_header = _parse_csv_line(line)
                continue

            if data_header is None:
                continue

            fields = _parse_csv_line(line)
            if not fields or len(fields) < 2:
                continue

            hmap = {name: i for i, name in enumerate(data_header)}

            ts_str = _col(hmap, fields, "Timestamp")
            ts_ms = _safe_float(ts_str)
            if ts_ms is None:
                continue

            if recording_unix_s is None:
                continue
            t_unix_ms = int(recording_unix_s * 1000 + ts_ms)

            gx = _safe_float(_col(hmap, fields, "Gaze X"))
            gy = _safe_float(_col(hmap, fields, "Gaze Y"))
            if gx is None or gy is None:
                gx = _safe_float(_col(hmap, fields, "Interpolated Gaze X"))
                gy = _safe_float(_col(hmap, fields, "Interpolated Gaze Y"))

            aoi = classify_aoi(gx, gy, role)
            trial = assign_trial(t_unix_ms, trial_boundaries)

            eyelid_l_str = _col(hmap, fields, "ET_EyelidOpeningLeft")
            eyelid_r_str = _col(hmap, fields, "ET_EyelidOpeningRight")

            # Infer blink from eyelid opening
            el = _safe_float(eyelid_l_str)
            er = _safe_float(eyelid_r_str)
            blink_detected = ""
            if el is not None and er is not None:
                if el < 0.2 and er < 0.2:
                    blink_detected = "1"
            elif el is not None and el < 0.2:
                blink_detected = "1"
            elif er is not None and er < 0.2:
                blink_detected = "1"

            rows_out.append({
                "t_unix_ms": t_unix_ms,
                "t_iso": epoch_to_iso(t_unix_ms),
                "trial": trial,
                "gaze_x": gx if gx is not None else "",
                "gaze_y": gy if gy is not None else "",
                "aoi": aoi,
                "pupil_left": _col(hmap, fields, "ET_PupilLeft"),
                "pupil_right": _col(hmap, fields, "ET_PupilRight"),
                "head_pitch": _col(hmap, fields, "ET_HeadRotationPitch"),
                "head_yaw": _col(hmap, fields, "ET_HeadRotationYaw"),
                "head_roll": _col(hmap, fields, "ET_HeadRotationRoll"),
                "fixation_idx": _col(hmap, fields, "Fixation Index"),
                "fixation_x": _col(hmap, fields, "Fixation X"),
                "fixation_y": _col(hmap, fields, "Fixation Y"),
                "fixation_duration": _col(hmap, fields, "Fixation Duration"),
                "saccade_idx": _col(hmap, fields, "Saccade Index"),
                "saccade_amplitude": _col(hmap, fields, "Saccade Amplitude"),
                "saccade_peak_velocity": _col(hmap, fields, "Saccade Peak Velocity"),
                "saccade_direction": _col(hmap, fields, "Saccade Direction"),
                "gaze_velocity": _col(hmap, fields, "Gaze Velocity"),
                "blink": blink_detected,
                "eyelid_left": eyelid_l_str,
                "eyelid_right": eyelid_r_str,
                "role": role,
                "source": "aurora",
            })

    return rows_out


# ---------------------------------------------------------------------------
# SmartEye Pro 10 .log (TSV) parser
# ---------------------------------------------------------------------------

def parse_smarteye(eye_path: str, role: str, trial_boundaries: list) -> list:
    rows_out = []

    with open(eye_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\n\r")
        headers = header_line.split("\t")
        hmap = {name: i for i, name in enumerate(headers)}

        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            fields = line.split("\t")

            obj_name = _col(hmap, fields,
                            "FilteredClosestWorldIntersection.objectName")
            if not obj_name or obj_name == "0":
                obj_name = _col(hmap, fields,
                                "ClosestWorldIntersection.objectName")
            if obj_name != "Screen2":
                continue

            rtc_str = _col(hmap, fields, "RealTimeClock")
            rtc = _safe_int(rtc_str)
            if rtc is None or rtc == 0:
                continue
            t_unix_ms = rtc // 10000 - WINDOWS_EPOCH_OFFSET_MS

            gx = _safe_float(
                _col(hmap, fields,
                     "FilteredClosestWorldIntersection.objectPoint.x"))
            gy = _safe_float(
                _col(hmap, fields,
                     "FilteredClosestWorldIntersection.objectPoint.y"))
            if gx is None or gy is None or (gx == 0 and gy == 0):
                gx_fb = _safe_float(
                    _col(hmap, fields,
                         "ClosestWorldIntersection.objectPoint.x"))
                gy_fb = _safe_float(
                    _col(hmap, fields,
                         "ClosestWorldIntersection.objectPoint.y"))
                if gx_fb is not None and gy_fb is not None:
                    gx, gy = gx_fb, gy_fb

            aoi = classify_aoi(gx, gy, role)
            trial = assign_trial(t_unix_ms, trial_boundaries)

            # Pupil: prefer filtered, fall back to raw, then combined
            # SmartEye reports in meters — convert to mm for consistency with Aurora
            pupil_left = (_col(hmap, fields, "FilteredLeftPupilDiameter")
                          or _col(hmap, fields, "LeftPupilDiameter"))
            pupil_right = (_col(hmap, fields, "FilteredRightPupilDiameter")
                           or _col(hmap, fields, "RightPupilDiameter"))
            if not pupil_left and not pupil_right:
                pd = (_col(hmap, fields, "FilteredPupilDiameter")
                      or _col(hmap, fields, "PupilDiameter"))
                pupil_left = pd
                pupil_right = pd
            # Convert meters to millimeters
            pl_f = _safe_float(pupil_left)
            pr_f = _safe_float(pupil_right)
            if pl_f is not None and 0 < pl_f < 0.1:  # clearly in meters
                pupil_left = str(pl_f * 1000)
            if pr_f is not None and 0 < pr_f < 0.1:
                pupil_right = str(pr_f * 1000)

            blink_val = _col(hmap, fields, "Blink")

            # Eyelid opening for blink detection fallback
            eyelid_left = _col(hmap, fields, "LeftEyelidOpening")
            eyelid_right = _col(hmap, fields, "RightEyelidOpening")

            rows_out.append({
                "t_unix_ms": t_unix_ms,
                "t_iso": epoch_to_iso(t_unix_ms),
                "trial": trial,
                "gaze_x": gx if gx is not None else "",
                "gaze_y": gy if gy is not None else "",
                "aoi": aoi,
                "pupil_left": pupil_left,
                "pupil_right": pupil_right,
                "head_pitch": _col(hmap, fields, "HeadPitch"),
                "head_yaw": _col(hmap, fields, "HeadHeading"),
                "head_roll": _col(hmap, fields, "HeadRoll"),
                "fixation_idx": _col(hmap, fields, "Fixation"),
                "fixation_x": "",
                "fixation_y": "",
                "fixation_duration": "",
                "saccade_idx": _col(hmap, fields, "Saccade"),
                "saccade_amplitude": "",
                "saccade_peak_velocity": "",
                "saccade_direction": "",
                "gaze_velocity": "",
                "blink": blink_val,
                "eyelid_left": eyelid_left,
                "eyelid_right": eyelid_right,
                "role": role,
                "source": "smarteye",
            })

    return rows_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Preprocess eye-tracker data to unified CSV")
    ap.add_argument("--eye-file", required=True, help="Eye tracker data file")
    ap.add_argument("--format", required=True, choices=["aurora", "smarteye"],
                    help="Input format: aurora or smarteye")
    ap.add_argument("--role", required=True, choices=["director", "matcher"],
                    help="Participant role")
    ap.add_argument("--zip", default=None,
                    help="Session ZIP for trial boundaries and sync flash detection")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--apply-offset", type=float, default=None,
                    help="Clock offset in ms (eye_tracker - frontend); subtracted from eye tracker timestamps")
    args = ap.parse_args()

    trial_boundaries = []
    flash_events = []
    clock_offset_info = {}
    if args.zip:
        trial_boundaries = extract_trial_boundaries(args.zip)
        flash_events = extract_flash_timestamps(args.zip)
        clock_offset_info = extract_clock_offset(args.zip)
        print(f"Found {len(trial_boundaries)} trial(s) in ZIP")
        for label, s, e in trial_boundaries:
            print(f"  {label}: {s} – {e}")
        if flash_events:
            print(f"Found {len(flash_events)} sync flash event(s)")
        if clock_offset_info:
            print(f"Clock offset (software): {clock_offset_info.get('offsetMs', '?')}ms "
                  f"(RTT: {clock_offset_info.get('rttMs', '?')}ms)")

    if args.format == "aurora":
        rows = parse_aurora(args.eye_file, args.role, trial_boundaries)
    else:
        rows = parse_smarteye(args.eye_file, args.role, trial_boundaries)

    if not rows:
        print("Warning: no gaze samples produced.")

    # ── Flash-based clock offset detection ──
    # For each sync_flash event, find the corresponding pupil constriction
    # in the eye tracker data and compute the offset.
    flash_offsets = []
    for fe in flash_events:
        flash_ts = fe["flash_ts"]
        detected_ts = detect_flash_in_pupil(rows, flash_ts)
        if detected_ts > 0:
            # offset = eye_tracker_time - frontend_time
            # Positive = eye tracker clock is ahead
            offset = detected_ts - flash_ts
            flash_offsets.append(offset)
            print(f"  Flash sync ({fe['trial']}): frontend={flash_ts}, "
                  f"eye_tracker={detected_ts}, offset={offset:+.0f}ms")
        else:
            print(f"  Flash sync ({fe['trial']}): pupil constriction not detected")

    # Compute median flash offset if we have detections
    applied_offset = 0
    if args.apply_offset is not None:
        applied_offset = args.apply_offset
        print(f"Applying manual offset: {applied_offset:+.0f}ms")
    elif flash_offsets:
        applied_offset = sorted(flash_offsets)[len(flash_offsets) // 2]
        print(f"Applying flash-detected median offset: {applied_offset:+.0f}ms")

    # Apply offset to all timestamps (corrects eye tracker timestamps to frontend clock)
    if applied_offset != 0:
        for r in rows:
            t = _safe_int(r.get("t_unix_ms"))
            if t is not None:
                corrected = t - int(applied_offset)
                r["t_unix_ms"] = corrected
                r["t_iso"] = epoch_to_iso(corrected)
        # Re-assign trials with corrected timestamps
        for r in rows:
            t = _safe_int(r.get("t_unix_ms"))
            r["trial"] = assign_trial(t, trial_boundaries)
        print(f"Corrected {len(rows)} timestamps by {-applied_offset:+.0f}ms")

    ordered_rows = []
    for r in rows:
        ordered_rows.append({col: r.get(col, "") for col in OUTPUT_COLUMNS})

    csv_write(ordered_rows, args.out)
    print(f"Wrote {len(ordered_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
