"""
Heart Rate Processing Pipeline
- Load HR CSV files (timestamp_unix_ms, timestamp_iso, bpm, phase)
- Baseline correction (subtract or ratio)
- Interpolation to uniform sampling rate
- Prepare paired time series for synchrony analysis (MdRQA-ready)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import medfilt


def load_hr_csv(csv_path: str) -> pd.DataFrame:
    """Load HR CSV and return DataFrame with parsed timestamps."""
    df = pd.read_csv(csv_path)
    df["timestamp_unix_ms"] = pd.to_numeric(df["timestamp_unix_ms"], errors="coerce")
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df = df.dropna(subset=["timestamp_unix_ms", "bpm"])
    df = df.sort_values("timestamp_unix_ms").reset_index(drop=True)
    # Relative time in seconds from first reading
    if len(df) > 0:
        df["t_sec"] = (df["timestamp_unix_ms"] - df["timestamp_unix_ms"].iloc[0]) / 1000.0
    else:
        df["t_sec"] = []
    return df


def baseline_correct(df: pd.DataFrame, method: str = "subtract") -> pd.DataFrame:
    """
    Baseline-correct HR data.
    method: 'subtract' (bpm - baseline_mean) or 'ratio' (bpm / baseline_mean)
    """
    baseline = df[df["phase"] == "baseline"]
    trial = df[df["phase"] == "trial"].copy()

    if len(baseline) == 0 or len(trial) == 0:
        trial["bpm_corrected"] = trial["bpm"]
        trial["baseline_mean"] = np.nan
        return trial

    baseline_mean = baseline["bpm"].mean()
    baseline_std = baseline["bpm"].std()

    if method == "subtract":
        trial["bpm_corrected"] = trial["bpm"] - baseline_mean
    elif method == "ratio":
        trial["bpm_corrected"] = trial["bpm"] / baseline_mean if baseline_mean > 0 else trial["bpm"]
    else:
        trial["bpm_corrected"] = trial["bpm"]

    trial["baseline_mean"] = baseline_mean
    trial["baseline_std"] = baseline_std
    return trial


def clean_hr(series: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Clean HR signal:
    1. Remove physiologically implausible values (< 30 or > 220 bpm)
    2. Median filter to remove transient spikes
    """
    cleaned = series.copy()
    # Clamp
    cleaned[(cleaned < 30) | (cleaned > 220)] = np.nan
    # Interpolate NaNs
    valid = ~np.isnan(cleaned)
    if valid.sum() < 2:
        return cleaned
    x = np.arange(len(cleaned))
    interp = interp1d(x[valid], cleaned[valid], kind="linear", bounds_error=False, fill_value="extrapolate")
    cleaned = interp(x)
    # Median filter
    if len(cleaned) >= kernel_size:
        cleaned = medfilt(cleaned, kernel_size=kernel_size)
    return cleaned


def resample_uniform(times: np.ndarray, values: np.ndarray, target_hz: float = 1.0) -> tuple:
    """
    Resample time series to uniform sampling rate.
    Returns (new_times, new_values).
    """
    if len(times) < 2:
        return times, values

    t_start = times[0]
    t_end = times[-1]
    step = 1.0 / target_hz
    new_times = np.arange(t_start, t_end, step)

    interp = interp1d(times, values, kind="linear", bounds_error=False, fill_value="extrapolate")
    new_values = interp(new_times)
    return new_times, new_values


def align_pair(
    df_a: pd.DataFrame, df_b: pd.DataFrame, target_hz: float = 1.0
) -> dict:
    """
    Align two HR time series to a common time axis.
    Both DataFrames must have 'timestamp_unix_ms' and 'bpm_corrected'.
    Returns dict with aligned arrays ready for synchrony.
    """
    if len(df_a) == 0 or len(df_b) == 0:
        return {"error": "Empty HR data for one or both participants"}

    # Use absolute timestamps for alignment
    t_a = df_a["timestamp_unix_ms"].values / 1000.0  # to seconds
    t_b = df_b["timestamp_unix_ms"].values / 1000.0
    v_a = df_a["bpm_corrected"].values
    v_b = df_b["bpm_corrected"].values

    # Clean both
    v_a = clean_hr(v_a)
    v_b = clean_hr(v_b)

    # Common time window
    t_start = max(t_a[0], t_b[0])
    t_end = min(t_a[-1], t_b[-1])
    if t_end <= t_start:
        return {"error": "No overlapping time window between participants"}

    # Resample to common grid
    step = 1.0 / target_hz
    common_t = np.arange(0, t_end - t_start, step)
    abs_common = common_t + t_start

    interp_a = interp1d(t_a, v_a, kind="linear", bounds_error=False, fill_value="extrapolate")
    interp_b = interp1d(t_b, v_b, kind="linear", bounds_error=False, fill_value="extrapolate")

    aligned_a = interp_a(abs_common)
    aligned_b = interp_b(abs_common)

    # Cross-correlation (basic synchrony indicator)
    if len(aligned_a) > 2:
        corr = float(np.corrcoef(aligned_a, aligned_b)[0, 1])
    else:
        corr = np.nan

    return {
        "t_sec": common_t.tolist(),
        "director_bpm": aligned_a.tolist(),
        "matcher_bpm": aligned_b.tolist(),
        "duration_sec": round(float(t_end - t_start), 2),
        "samples": len(common_t),
        "sampling_hz": target_hz,
        "pearson_r": round(corr, 4) if not np.isnan(corr) else None,
    }


def compute_windowed_correlation(
    series_a: np.ndarray, series_b: np.ndarray, window_sec: int = 30, hz: float = 1.0
) -> list:
    """
    Sliding window Pearson correlation.
    Returns list of {t_center, r, p} dicts.
    """
    from scipy.stats import pearsonr

    window_samples = int(window_sec * hz)
    step = max(1, window_samples // 2)  # 50% overlap
    results = []

    for start in range(0, len(series_a) - window_samples + 1, step):
        end = start + window_samples
        a_win = series_a[start:end]
        b_win = series_b[start:end]
        if np.std(a_win) < 1e-6 or np.std(b_win) < 1e-6:
            continue
        r, p = pearsonr(a_win, b_win)
        t_center = (start + end) / 2 / hz
        results.append({"t_sec": round(t_center, 2), "r": round(r, 4), "p": round(p, 6)})

    return results


def process_trial_hr(trial_dir: str, correction_method: str = "subtract", target_hz: float = 1.0) -> dict:
    """
    Process HR data for a single trial.
    Expects: trial_dir/hr/hr_director.csv, hr_matcher.csv
    """
    hr_dir = os.path.join(trial_dir, "hr")
    if not os.path.isdir(hr_dir):
        return {"error": "No HR directory found"}

    result = {}

    for role in ["director", "matcher"]:
        csv_path = os.path.join(hr_dir, f"hr_{role}.csv")
        if not os.path.exists(csv_path):
            result[role] = {"error": f"No HR file for {role}"}
            continue

        df = load_hr_csv(csv_path)
        df_corrected = baseline_correct(df, method=correction_method)

        stats = {}
        if len(df_corrected) > 0:
            stats = {
                "n_readings": len(df_corrected),
                "duration_sec": round(df_corrected["t_sec"].max() - df_corrected["t_sec"].min(), 2) if len(df_corrected) > 1 else 0,
                "bpm_mean": round(df_corrected["bpm"].mean(), 2),
                "bpm_std": round(df_corrected["bpm"].std(), 2),
                "bpm_min": round(df_corrected["bpm"].min(), 2),
                "bpm_max": round(df_corrected["bpm"].max(), 2),
                "bpm_corrected_mean": round(df_corrected["bpm_corrected"].mean(), 2),
                "baseline_mean": round(df_corrected["baseline_mean"].iloc[0], 2) if not pd.isna(df_corrected["baseline_mean"].iloc[0]) else None,
            }

        result[role] = {
            "stats": stats,
            "timeseries": df_corrected[["t_sec", "bpm", "bpm_corrected"]].to_dict(orient="records") if len(df_corrected) > 0 else [],
        }
        print(f"  [hr] {role}: {stats.get('n_readings', 0)} readings, mean={stats.get('bpm_mean', '?')} bpm")

    # Paired analysis
    if "error" not in result.get("director", {}) and "error" not in result.get("matcher", {}):
        dir_csv = os.path.join(hr_dir, "hr_director.csv")
        mat_csv = os.path.join(hr_dir, "hr_matcher.csv")
        if os.path.exists(dir_csv) and os.path.exists(mat_csv):
            df_dir = baseline_correct(load_hr_csv(dir_csv), correction_method)
            df_mat = baseline_correct(load_hr_csv(mat_csv), correction_method)
            aligned = align_pair(df_dir, df_mat, target_hz=target_hz)
            result["synchrony"] = aligned

            if "error" not in aligned and len(aligned.get("director_bpm", [])) > 30:
                windowed = compute_windowed_correlation(
                    np.array(aligned["director_bpm"]),
                    np.array(aligned["matcher_bpm"]),
                    window_sec=30,
                    hz=target_hz,
                )
                result["windowed_correlation"] = windowed

    # Save processed HR JSON
    json_path = os.path.join(hr_dir, "hr_processed.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [hr] Saved processed HR to {json_path}")

    return result
