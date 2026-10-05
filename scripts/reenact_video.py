"""Offline check of stages 1-3: a source image driven by a recorded video.

Usage:
    python scripts/reenact_video.py --source me.jpg --driving clip.mp4 --out out.mp4

Writes a side-by-side video (driving frame | reenacted source) and prints the
per-stage latency, frame rate, and peak GPU memory.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuropresence.capture import FaceTracker, FrameSource
from neuropresence.capture.crop import letterbox, square_face_crop
from neuropresence.reenactment import ReenactmentEngine
from neuropresence.reenactment.engine import OUTPUT_SIZE

WARMUP_FRAMES = 10  # the first GPU calls are slow and would distort the averages


def summarise(values):
    arr = np.asarray(values)
    return {
        "mean": round(float(arr.mean()), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="image of the person to animate")
    parser.add_argument("--driving", required=True, help="video whose motion drives the source")
    parser.add_argument("--out", default="reenacted.mp4")
    parser.add_argument("--report", help="optional path for a JSON timing report")
    args = parser.parse_args()

    source = cv2.imread(args.source)
    if source is None:
        sys.exit(f"Could not read source image: {args.source}")

    engine = ReenactmentEngine()
    engine.set_source(source)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 25,
                             (OUTPUT_SIZE * 2, OUTPUT_SIZE))
    timings = {"tracker": [], "motion": [], "render": [], "compose": [], "total": []}
    frames = skipped = 0
    with FrameSource(args.driving) as video, FaceTracker() as tracker:
        while (frame := video.read()) is not None:
            start = time.perf_counter()
            result = tracker.process(frame)
            if not result.ok:
                skipped += 1
                continue
            face = square_face_crop(frame, result.bbox)
            output = engine.drive(face)
            total_ms = (time.perf_counter() - start) * 1000

            frames += 1
            if frames > WARMUP_FRAMES:
                timings["tracker"].append(result.latency_ms)
                timings["motion"].append(engine.last_timing_ms["motion"])
                timings["render"].append(engine.last_timing_ms["render"])
                timings["compose"].append(engine.last_timing_ms["compose"])
                timings["total"].append(total_ms)
            writer.write(np.hstack([letterbox(frame, OUTPUT_SIZE), letterbox(output, OUTPUT_SIZE)]))
    writer.release()

    if not timings["total"]:
        sys.exit(f"Too few usable frames ({frames} driven, {skipped} without one face).")
    report = {
        "frames_driven": frames,
        "frames_skipped": skipped,
        "latency_ms": {stage: summarise(v) for stage, v in timings.items()},
        "fps": round(1000 / float(np.mean(timings["total"])), 2),
        "peak_vram_gb": round(engine.peak_vram_gb(), 2),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
