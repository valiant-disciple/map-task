"""
Path Similarity Analysis
- Compare Matcher's drawn path to reference (Director's map route)
- Metrics: Fréchet distance, DTW distance, area between curves, completion %
"""

import os
import json
import numpy as np


def load_strokes(strokes_json_path: str) -> list:
    """Load strokes.json and return list of polylines."""
    if not os.path.exists(strokes_json_path):
        return []
    with open(strokes_json_path) as f:
        data = json.load(f)
    # Each stroke has 'polyline' which is a list of {x, y} or [x, y]
    polylines = []
    for stroke in data:
        if stroke.get("mode") == "erase":
            continue  # Skip eraser strokes
        pl = stroke.get("polyline", [])
        if isinstance(pl, list) and len(pl) > 0:
            points = []
            for p in pl:
                if isinstance(p, dict):
                    points.append([p.get("x", 0), p.get("y", 0)])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    points.append([p[0], p[1]])
            if points:
                polylines.append(np.array(points))
    return polylines


def merge_polylines(polylines: list) -> np.ndarray:
    """Concatenate all polylines into a single path (order matters)."""
    if not polylines:
        return np.array([]).reshape(0, 2)
    return np.vstack(polylines)


def frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute discrete Fréchet distance between two curves P and Q.
    Each is an Nx2 array of (x, y) points.
    """
    if len(P) == 0 or len(Q) == 0:
        return float("inf")

    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)

    def _dist(i, j):
        return np.sqrt(np.sum((P[i] - Q[j]) ** 2))

    def _compute(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = _dist(i, j)
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_compute(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_compute(0, j - 1), d)
        else:
            ca[i, j] = max(min(_compute(i - 1, j), _compute(i - 1, j - 1), _compute(i, j - 1)), d)
        return ca[i, j]

    # Iterative version (avoids recursion limit for large paths)
    for i in range(n):
        for j in range(m):
            d = _dist(i, j)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)

    return float(ca[n - 1, m - 1])


def dtw_distance(P: np.ndarray, Q: np.ndarray) -> tuple:
    """
    Dynamic Time Warping distance between two paths.
    Returns (dtw_distance, normalized_dtw_distance).
    """
    if len(P) == 0 or len(Q) == 0:
        return float("inf"), float("inf")

    n, m = len(P), len(Q)
    dtw_matrix = np.full((n + 1, m + 1), float("inf"))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(np.sum((P[i - 1] - Q[j - 1]) ** 2))
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    total = float(dtw_matrix[n, m])
    normalized = total / (n + m) if (n + m) > 0 else 0
    return total, normalized


def path_length(path: np.ndarray) -> float:
    """Total Euclidean length of a path."""
    if len(path) < 2:
        return 0.0
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def simplify_path(path: np.ndarray, tolerance: float = 2.0) -> np.ndarray:
    """
    Douglas-Peucker simplification to reduce point count.
    Keeps shape while reducing computation.
    """
    if len(path) <= 2:
        return path

    # Find point farthest from the line between first and last
    start, end = path[0], path[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return np.array([start, end])

    line_unit = line_vec / line_len
    offsets = path - start
    proj = np.dot(offsets, line_unit)
    proj_points = start + np.outer(proj, line_unit)
    dists = np.sqrt(np.sum((path - proj_points) ** 2, axis=1))

    max_idx = np.argmax(dists)
    max_dist = dists[max_idx]

    if max_dist > tolerance:
        left = simplify_path(path[: max_idx + 1], tolerance)
        right = simplify_path(path[max_idx:], tolerance)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def analyze_path_similarity(
    matcher_strokes_path: str,
    reference_path: np.ndarray | None = None,
    simplify_tolerance: float = 3.0,
) -> dict:
    """
    Analyze matcher's drawn path.
    If reference_path is provided, computes similarity metrics.
    """
    polylines = load_strokes(matcher_strokes_path)
    if not polylines:
        return {"error": "No matcher strokes found", "num_strokes": 0}

    matcher_path = merge_polylines(polylines)

    result = {
        "num_strokes": len(polylines),
        "total_points": len(matcher_path),
        "path_length_px": round(path_length(matcher_path), 2),
    }

    # Simplify for faster comparison
    matcher_simplified = simplify_path(matcher_path, tolerance=simplify_tolerance)
    result["simplified_points"] = len(matcher_simplified)

    if reference_path is not None and len(reference_path) > 0:
        ref_simplified = simplify_path(reference_path, tolerance=simplify_tolerance)

        # Fréchet distance
        fd = frechet_distance(matcher_simplified, ref_simplified)
        result["frechet_distance_px"] = round(fd, 2)

        # DTW distance
        dtw_total, dtw_norm = dtw_distance(matcher_simplified, ref_simplified)
        result["dtw_distance_px"] = round(dtw_total, 2)
        result["dtw_normalized_px"] = round(dtw_norm, 2)

        # Path length comparison
        ref_len = path_length(reference_path)
        mat_len = path_length(matcher_path)
        result["reference_path_length_px"] = round(ref_len, 2)
        result["length_ratio"] = round(mat_len / ref_len, 4) if ref_len > 0 else None

    return result


def process_trial_path(trial_dir: str, reference_path: np.ndarray | None = None) -> dict:
    """
    Process path data for a single trial.
    """
    strokes_path = os.path.join(trial_dir, "strokes.json")
    result = analyze_path_similarity(strokes_path, reference_path)

    # Save
    json_path = os.path.join(trial_dir, "path_analysis.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [path] {result.get('num_strokes', 0)} strokes, "
          f"length={result.get('path_length_px', 0)} px"
          f"{', Fréchet=' + str(result.get('frechet_distance_px', '?')) if 'frechet_distance_px' in result else ''}")

    return result
