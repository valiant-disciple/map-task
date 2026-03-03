#!/usr/bin/env python3
"""
Compute RQA metrics for a 1D time series using pyrqa.
Input: JSON on stdin: { "series": [..], "embedding": 1, "delay": 1, "radius": 0.1 }
Output: JSON on stdout with RQA metrics.
"""
import sys, json
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.computation import RQAComputation
    from pyrqa.metric import EuclideanMetric
except ImportError:
    sys.stderr.write("pyrqa is not installed. pip install pyrqa\n")
    sys.exit(1)

def main():
    raw = sys.stdin.read()
    data = json.loads(raw)
    series = data.get("series", [])
    embedding = int(data.get("embedding", 1))
    delay = int(data.get("delay", 1))
    radius = float(data.get("radius", 0.1))
    if len(series) < 4:
        print(json.dumps({"error": "series_too_short"}))
        return
    ts = TimeSeries(series, embedding_dimension=embedding, time_delay=delay, metric=EuclideanMetric)
    settings = Settings(
        ts,
        analysis_type=RQAComputation,
        neighbourhood=radius,
        similarity_measure=EuclideanMetric,
        theiler_corrector=1
    )
    comp = RQAComputation.create(settings)
    res = comp.run()
    out = {
        "rr": res.recurrence_rate,
        "det": res.determinism,
        "l": res.average_diagonal_line,
        "lmax": res.longest_diagonal_line,
        "entr": res.entropy_diagonal_lines,
        "lam": res.laminarity,
        "tt": res.trapping_time,
        "vmax": res.longest_vertical_line,
        "ratio": res.ratio_determinism_recurrence_rate,
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
