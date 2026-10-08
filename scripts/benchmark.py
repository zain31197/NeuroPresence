"""Benchmark the pipeline on a list of (source, driving video) pairs.

Usage:
    python scripts/benchmark.py                                  # the sample clips
    python scripts/benchmark.py --manifest my_clips.json --out results/mine.json
    python scripts/benchmark.py --save-videos results/videos     # also keep side-by-side videos

For every pair it measures
  speed     - per-stage latency, frame rate and peak GPU memory of the pipeline;
  identity  - CSIM between the source face and the face in every output frame;
  motion    - how closely the output's head pose and expression follow the driver,
              and whether the output lags behind;
  stability - warping error (flicker) and landmark jitter (wobble) of the output,
              each next to the same measurement on the real driving video.
The measurements run outside the timed pipeline, so they do not slow the numbers.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker, FrameSource
from neuropresence.capture.crop import letterbox
from neuropresence.capture.enrol import first_frontal_frame
from neuropresence.evaluation import WarpingError, landmark_jitter, summarise, summarise_scores
from neuropresence.evaluation.motion import correlation, mean_abs_error, motion_lag
from neuropresence.identity import IdentityScorer
from neuropresence.pipeline import Pipeline
from neuropresence.reenactment import ReenactmentEngine
from neuropresence.reenactment.engine import LIVEPORTRAIT_DIR
from neuropresence.targets import TARGETS

WARMUP_FRAMES = 10  # the first GPU calls are slow and would distort the averages
METRIC_SIZE = 256  # frames are compared at this size for the warping error
PANEL = 512
# Eye corners and nose bridge: points that stay fixed to the skull while a
# person talks, so their movement is head movement, not expression.
STABLE_POINTS = [33, 133, 362, 263, 6, 168]
EYE_SPAN = (33, 263)  # outer eye corners; their distance is the unit of length
# The lag is read from yaw, pitch (degrees) and mouth opening (0..1). A signal
# that varies less than this in the driving video is too still to time.
LAG_MIN_STD = (2.0, 2.0, 0.03)


def load_source(pair):
    if pair["source"] != "first_frame":
        image = cv2.imread(str(ROOT / pair["source"]))
        if image is None:
            raise ValueError(f"Could not read source image: {pair['source']}")
        return image
    with FrameSource(str(ROOT / pair["driving"])) as frames, FaceTracker() as tracker:
        frame = first_frontal_frame(frames, tracker)
    if frame is None:
        raise ValueError(f"No front-facing frame to use as the source in {pair['driving']}")
    return frame


def read_face(track):
    """What the stability and motion measurements need from one tracked face."""
    if not track.ok:
        return None
    landmarks = track.landmarks.astype(np.float64)
    return {
        "points": landmarks[STABLE_POINTS],
        "eye_span": float(np.linalg.norm(landmarks[EYE_SPAN[0]] - landmarks[EYE_SPAN[1]])),
        "pose": np.array(track.pose_deg),  # yaw, pitch, roll in degrees
        "expression": np.array([score for name, score in sorted(track.blendshapes.items())
                                if name != "_neutral"]),
        "mouth": track.blendshapes["jawOpen"],
    }


def timeline(index, faces, key):
    """One measurement laid out by video frame number, NaN where the face was not found."""
    found = [face for face in faces if face is not None]
    if not found:
        raise ValueError("The face was not found in any frame.")
    values = np.full((index[-1] + 1, *np.shape(found[0][key])), np.nan)
    for i, face in zip(index, faces):
        if face is not None:
            values[i] = face[key]
    return values


def run_pair(pair, engine, scorer, measure, max_frames, video_dir):
    """Drive one pair through the pipeline and collect the raw measurements."""
    is_self = pair["source"] == "first_frame"
    engine.set_source(load_source(pair))
    source_embedding = scorer.embed(engine.source_frame)
    if source_embedding is None:
        raise ValueError(f"{pair['name']}: no single face found in the source image")

    writer = None
    if video_dir:
        writer = cv2.VideoWriter(str(video_dir / f"{pair['name']}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 25, (PANEL * 2, PANEL))
    timings = {"tracker": [], "motion": [], "render": [], "compose": [], "total": []}
    raw = {"index": [], "csim": [], "csim_real": [], "out_frames": [], "drv_frames": [],
           "out_faces": [], "drv_faces": []}
    fallback = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with FrameSource(str(ROOT / pair["driving"])) as video, FaceTracker() as tracker:
        pipeline = Pipeline(tracker, engine)
        index = -1
        while len(raw["index"]) < max_frames and (frame := video.read()) is not None:
            index += 1
            result = pipeline.step(frame, at=index / (video.fps or 25.0))  # timed by the clip, so a run repeats exactly
            if not result.live:
                fallback += 1
                continue
            raw["index"].append(index)
            if len(raw["index"]) > WARMUP_FRAMES:
                for stage in timings:
                    timings[stage].append(result.timing_ms[stage])

            # Measurements from here on are outside the timed pipeline.
            out_track, drv_track = measure.process(result.output), measure.process(frame)
            if out_track.ok:
                embedding = scorer.embed(result.output, out_track.landmarks)
                raw["csim"].append(scorer.similarity(source_embedding, embedding))
            if is_self and drv_track.ok:
                embedding = scorer.embed(frame, drv_track.landmarks)
                raw["csim_real"].append(scorer.similarity(source_embedding, embedding))
            raw["out_faces"].append(read_face(out_track))
            raw["drv_faces"].append(read_face(drv_track))
            raw["out_frames"].append(cv2.resize(engine.last_crop, (METRIC_SIZE, METRIC_SIZE),
                                                interpolation=cv2.INTER_AREA))
            raw["drv_frames"].append(result.driving_face)
            if writer is not None:
                writer.write(np.hstack([letterbox(frame, PANEL), letterbox(result.output, PANEL)]))
    if writer is not None:
        writer.release()
    if len(timings["total"]) < 3:
        raise ValueError(f"{pair['name']}: too few usable frames ({len(raw['index'])} driven)")
    if not raw["csim"]:
        raise ValueError(f"{pair['name']}: no output frame contained a single detectable face")

    report = {
        "name": pair["name"],
        "kind": "self-reenactment" if is_self else "cross-identity",
        "frames_driven": len(raw["index"]),
        "frames_fallback": fallback,
        "latency_ms": {stage: summarise(values) for stage, values in timings.items()},
        "fps": round(1000 / float(np.mean(timings["total"])), 2),
        "peak_vram_gb": round(engine.peak_vram_gb(), 2),
        "csim": summarise_scores(raw["csim"], TARGETS["csim"]),
        "output_frames_without_one_face": len(raw["index"]) - len(raw["csim"]),
    }
    if is_self:
        report["csim_real_video"] = summarise_scores(raw["csim_real"], TARGETS["csim"])
    return report, raw


def add_motion(report, raw):
    """How closely the output's pose and expression follow the driving video."""
    index, drv, out = raw["index"], raw["drv_faces"], raw["out_faces"]
    pose_error, pose_motion = mean_abs_error(timeline(index, drv, "pose"), timeline(index, out, "pose"))
    expression_error, expression_motion = mean_abs_error(timeline(index, drv, "expression"),
                                                         timeline(index, out, "expression"))

    def lag_signals(faces):  # yaw, pitch, mouth opening
        return np.column_stack([timeline(index, faces, "pose")[:, :2], timeline(index, faces, "mouth")])

    lag, _ = motion_lag(lag_signals(drv), lag_signals(out), LAG_MIN_STD)
    mouth = correlation(timeline(index, drv, "mouth"), timeline(index, out, "mouth"))
    report["motion"] = {
        "pose_error_deg": round(pose_error, 2),
        "pose_motion_deg": round(pose_motion, 2),
        "expression_error": round(expression_error, 4),
        "expression_motion": round(expression_motion, 4),
        "mouth_opening_correlation": None if mouth is None else round(mouth, 3),
        "lag_frames": None if lag is None else round(lag, 2),
    }


def add_stability(report, raw, warping):
    """Stability of the output next to the same measurement on the real driving video."""
    index = raw["index"]
    adjacent = [k for k in range(1, len(index)) if index[k] == index[k - 1] + 1]
    result = {}
    for label, frames in (("output", raw["out_frames"]), ("driving", raw["drv_frames"])):
        errors = warping.pair_errors([frames[k - 1] for k in adjacent], [frames[k] for k in adjacent])
        result[label] = round(float(errors.mean()) * 1000, 4)
    result["ratio"] = round(result["output"] / result["driving"], 3)
    report["warping_error_x1000"] = result

    result = {}
    for label, faces in (("output", raw["out_faces"]), ("driving", raw["drv_faces"])):
        # One length for the whole clip: a per-frame span would add its own noise.
        eye_span = float(np.nanmedian(timeline(index, faces, "eye_span")))
        result[label] = round(landmark_jitter(timeline(index, faces, "points"), eye_span) * 100, 4)
    result["ratio"] = round(result["output"] / result["driving"], 3)
    report["jitter_percent_of_eye_span"] = result


def git_revision(path):
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def mean_of(reports, *keys):
    values = []
    for report in reports:
        value = report
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(value)
    return round(float(np.mean(values)), 3) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=str(ROOT / "benchmarks" / "samples.json"))
    parser.add_argument("--out", default=str(ROOT / "results" / "benchmark.json"))
    parser.add_argument("--max-frames", type=int, help="frames per pair (default: from the manifest)")
    parser.add_argument("--save-videos", help="folder for side-by-side videos (driving | output)")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    max_frames = args.max_frames or manifest.get("max_frames", 150)
    video_dir = Path(args.save_videos) if args.save_videos else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)

    engine = ReenactmentEngine(tensorrt=True, compile_networks=True)  # as the app runs it
    print(f"Networks that draw the picture: {engine.accelerated or 'as they are'}")
    reports, raws = [], []
    with FaceTracker(video=False) as measure:
        scorer = IdentityScorer(tracker=measure)
        for pair in manifest["pairs"]:
            print(f"Running {pair['name']} ...", flush=True)
            report, raw = run_pair(pair, engine, scorer, measure, max_frames, video_dir)
            add_motion(report, raw)
            reports.append(report)
            raws.append(raw)
    # The optical-flow model is loaded only now, so it is not counted in the GPU memory above.
    warping = WarpingError()
    for report, raw in zip(reports, raws):
        add_stability(report, raw, warping)

    self_reports = [r for r in reports if r["kind"] == "self-reenactment"]
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": {
            "os": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "code": {"neuropresence": git_revision(ROOT), "liveportrait": git_revision(LIVEPORTRAIT_DIR)},
        "manifest": Path(args.manifest).name,
        "max_frames_per_pair": max_frames,
        "targets": TARGETS,
        "summary": {
            "pairs": len(reports),
            "fps": mean_of(reports, "fps"),
            "render_ms": mean_of(reports, "latency_ms", "render", "mean"),
            "pipeline_ms": mean_of(reports, "latency_ms", "total", "mean"),
            "peak_vram_gb": max(r["peak_vram_gb"] for r in reports),
            "csim_self_reenactment": mean_of(self_reports, "csim", "mean"),
            "csim_real_video": mean_of(self_reports, "csim_real_video", "mean"),
            "pose_error_deg": mean_of(reports, "motion", "pose_error_deg"),
            "pose_motion_deg": mean_of(reports, "motion", "pose_motion_deg"),
            "expression_error": mean_of(reports, "motion", "expression_error"),
            "expression_motion": mean_of(reports, "motion", "expression_motion"),
            "mouth_opening_correlation": mean_of(reports, "motion", "mouth_opening_correlation"),
            "lag_frames": mean_of(reports, "motion", "lag_frames"),
            "warping_error_ratio": mean_of(reports, "warping_error_x1000", "ratio"),
            "jitter_ratio": mean_of(reports, "jitter_percent_of_eye_span", "ratio"),
        },
        "notes": [
            "pipeline_ms covers tracker, crop, motion, render and compose; it excludes camera capture and display.",
            "peak_vram_gb is memory allocated by PyTorch for the pipeline, not memory reserved.",
            "CSIM compares each output frame with the source image. A face that never moved would score 1.0, "
            "so read it together with the motion errors.",
            "csim_real_video is the same CSIM measured on the real driving frames of the same person.",
            "Pose and expression are read with MediaPipe from the driving and the output frame and compared "
            "as change since the first driven frame; *_motion is how much the driver itself moved.",
            "lag_frames is positive when the output is late.",
            "Ratios are output divided by real driving video; 1.0 means as steady as real video.",
        ],
        "pairs": reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\n{'pair':13s} {'fps':>5s} {'render':>7s} {'CSIM':>6s} {'real':>6s} {'pose err':>9s} "
          f"{'expr err':>9s} {'lag':>6s} {'flicker x':>10s} {'jitter x':>9s}")
    for r in reports:
        real = r.get("csim_real_video", {}).get("mean", "-")
        motion = r["motion"]
        lag = "-" if motion["lag_frames"] is None else f"{motion['lag_frames']:+.2f}"
        print(f"{r['name']:13s} {r['fps']:5.1f} {r['latency_ms']['render']['mean']:7.1f} "
              f"{r['csim']['mean']:6.3f} {real:>6} "
              f"{motion['pose_error_deg']:4.1f}/{motion['pose_motion_deg']:<4.1f} "
              f"{motion['expression_error']:.3f}/{motion['expression_motion']:.3f} {lag:>6s} "
              f"{r['warping_error_x1000']['ratio']:10.2f} {r['jitter_percent_of_eye_span']['ratio']:9.2f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
