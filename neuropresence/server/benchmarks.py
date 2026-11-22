"""Read saved benchmark results (the JSON files scripts/benchmark.py writes) and call measurements
(scripts/measure_call.py)."""

import json
from pathlib import Path

# What each saved run measured, in words. The files are named after the change they were made to judge.
LABELS = {
    "baseline": "Baseline",
    "steady": "Steadier crop and keypoints",
    "fast": "Faster motion reading",
    "tensorrt": "TensorRT",
    "identity": "Identity monitor",
    "consent": "Consent check and mark",
    "smooth": "Even motion",
    "rest": "Neutral pose follows the person",
    "lips": "Closed lips anchored",
    "half": "16-bit networks, for comparison",
    "int8": "Heaviest layers in 8 bit",
    "own_camera": "Own camera recording",
}


def _label(stem):
    name = stem.removeprefix("benchmark_").removesuffix("_windows_rtx5050")
    for key, label in LABELS.items():
        if name == key or name.startswith(key + "_"):
            return label
    return name.replace("_", " ").capitalize()


def _benchmarks(results_dir):
    found = []
    for path in Path(results_dir).glob("benchmark*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "summary" not in data or "date" not in data:
            continue
        found.append((path, data))
    return sorted(found, key=lambda item: item[1]["date"])


def _entry(path, data):
    return {
        "file": path.name,
        "label": _label(path.stem),
        "date": data["date"],
        "machine": data.get("machine", {}),
        "code": data.get("code", {}),
        "clips": data["summary"].get("pairs"),
        "summary": data["summary"],
        "targets": data.get("targets", {}),
    }


def latest_benchmark(results_dir):
    """Summary of the most recent benchmark file in results_dir, or None if there is none."""
    found = _benchmarks(results_dir)
    return _entry(*found[-1]) if found else None


def all_benchmarks(results_dir):
    """Every saved benchmark, oldest first: how the figures moved as the work went on."""
    return [_entry(path, data) for path, data in _benchmarks(results_dir)]


# The figures of a call measurement the app shows, from each phase's summary.
CALL_FIGURES = ("fps", "delay_ms", "render_ms", "identity", "gpu_util_pct", "gpu_memory_mb", "cpu_engine_pct", "voice_delay_ms")


def call_measurements(results_dir):
    """The newest measurement of the engine beside meeting applications, or None."""
    newest = None
    for path in Path(results_dir).glob("call_measurements*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "phases" not in data or "date" not in data:
            continue
        if newest is None or data["date"] > newest[1]["date"]:
            newest = (path, data)
    if newest is None:
        return None
    path, data = newest
    phases = []
    for key, phase in data["phases"].items():
        summary = phase.get("summary") or {}
        phases.append({
            "key": key,
            "title": phase.get("title", key),
            "measured_at": phase.get("measured_at"),
            "seconds": summary.get("running_seconds", summary.get("seconds")),
            "figures": {name: summary[name] for name in CALL_FIGURES if isinstance(summary.get(name), dict)},
            "answers": phase.get("answers"),
        })
    return {"file": path.name, "date": data["date"], "machine": data.get("machine", {}),
            "seconds_per_phase": data.get("seconds_per_phase"), "targets": data.get("targets", {}), "phases": phases}
