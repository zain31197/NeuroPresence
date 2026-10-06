"""Small summaries used in result files."""

import numpy as np


def summarise(values, digits=2):
    """Mean and tail percentiles of a list of timings."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), digits),
        "p95": round(float(np.percentile(arr, 95)), digits),
        "p99": round(float(np.percentile(arr, 99)), digits),
    }


def summarise_scores(values, threshold, digits=3):
    """Summary of per-frame scores where higher is better (for example CSIM)."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), digits),
        "min": round(float(arr.min()), digits),
        "p05": round(float(np.percentile(arr, 5)), digits),
        "share_at_or_above_threshold": round(float((arr >= threshold).mean()), digits),
        "threshold": threshold,
        "frames": int(arr.size),
    }
