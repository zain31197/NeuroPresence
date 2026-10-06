"""Read saved benchmark results (the JSON files scripts/benchmark.py writes)."""

import json
from pathlib import Path


def latest_benchmark(results_dir):
    """Summary of the most recent benchmark file in results_dir, or None if there is none."""
    newest = None
    for path in Path(results_dir).glob("benchmark*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "summary" not in data or "date" not in data:
            continue
        if newest is None or data["date"] > newest[1]["date"]:
            newest = (path, data)
    if newest is None:
        return None
    path, data = newest
    return {
        "file": path.name,
        "date": data["date"],
        "machine": data.get("machine", {}),
        "code": data.get("code", {}),
        "clips": data["summary"].get("pairs"),
        "summary": data["summary"],
        "targets": data.get("targets", {}),
    }
