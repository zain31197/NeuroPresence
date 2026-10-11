"""How evenly is the movement shown when the camera and the output run at different rates?

A camera delivers 30 frames a second and the output is drawn at about 22. Each output frame is
made from the newest camera frame there is, so the camera frames that get shown are one and two
frames apart by turns, and a head that turns steadily is drawn moving in short and long steps.
"Even motion" reads the movement between two camera frames, for moments that are evenly spaced
(neuropresence/server/pacing.py, Pipeline.step: motion_at).

This script runs the pipeline over sample clips with the timing of a live session, once each way,
and measures what is shown: how far apart the moments are that consecutive output frames show,
how much the head in the output jumps from one frame to the next, and whether the output still
follows the driver as closely. The timing is worked out, not waited for, so every run is the same.

Usage:
    python scripts/study_pacing.py        # about 6 minutes, needs the GPU
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker
from neuropresence.consent.liveness import at_rest, read
from neuropresence.evaluation import landmark_jitter
from neuropresence.evaluation.motion import correlation
from neuropresence.pipeline import Pipeline
from neuropresence.reenactment import ReenactmentEngine
from neuropresence.server.pacing import MotionClock

DRIVING = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving"
OUT = ROOT / "results" / "pacing_study.json"
CLIPS = ("d3", "d6", "d9", "d10", "d11", "d12")  # the sample clips in which the head moves
SECONDS = 12.0
DRAW_SECONDS = 0.0455  # one output frame every 45.5 ms: 22 a second, as measured in a live session
READY_SECONDS = 0.014  # tracking and cropping a camera frame, before its turn on the GPU
STABLE_POINTS = [33, 133, 362, 263, 6, 168]  # eye corners and nose bridge: they move with the head, not the face
EYE_SPAN = (33, 263)


def faces_of(frames):
    """What the tracker reads in each picture, one at a time: head pose, mouth opening, steady points."""
    found = []
    with FaceTracker(video=False) as tracker:
        for frame in frames:
            track = tracker.process(frame)
            if not track.ok:
                found.append(None)
                continue
            points = track.landmarks
            found.append({"yaw": track.pose_deg[0], "pitch": track.pose_deg[1], "mouth": track.blendshapes.get("jawOpen", 0.0),
                          "points": points[STABLE_POINTS], "eye_span": float(np.linalg.norm(points[EYE_SPAN[0]] - points[EYE_SPAN[1]]))})
    return found


def at_moment(faces, fps, key, moment):
    """What the driver did at a moment between two of its frames."""
    position = min(max(moment * fps, 0.0), len(faces) - 1.0)
    low = int(np.floor(position))
    high = min(low + 1, len(faces) - 1)
    if faces[low] is None or faces[high] is None:
        return np.nan
    return faces[low][key] + (position - low) * (faces[high][key] - faces[low][key])


def session(engine, frames, fps, even):
    """The pipeline run as a live session runs it. Returns the output frames and the moment each one shows."""
    clock = MotionClock()
    outputs, moments = [], []
    with FaceTracker() as tracker:
        pipeline = Pipeline(tracker, engine)
        drawing = 0.5  # the first half second is left for the clock to have frames behind it
        while True:
            index = int((drawing - READY_SECONDS) * fps)  # the newest camera frame that is ready by now
            if index >= len(frames):
                break
            captured = index / fps
            motion_at = clock.moment(drawing, captured) if even else None
            result = pipeline.step(frames[index], at=captured, motion_at=motion_at)
            if result.live:
                outputs.append(result.output)
                moments.append(captured if result.motion_at is None else result.motion_at)
            drawing += DRAW_SECONDS
    return outputs, np.array(moments)


def measure(outputs, moments, driver, fps):
    shown = faces_of(outputs)
    steps = np.diff(moments) * 1000
    points = np.array([face["points"] if face else np.full((len(STABLE_POINTS), 2), np.nan) for face in shown])
    span = float(np.nanmedian([face["eye_span"] for face in shown if face]))
    settled = slice(20, None)  # after the clock has settled and the first fade is over

    def error(key):
        """How far the output is from the driver at the moment it shows, both as change since the first frame."""
        ours = np.array([face[key] if face else np.nan for face in shown])
        theirs = np.array([at_moment(driver, fps, key, moment) for moment in moments])
        ours, theirs = ours - ours[0], theirs - theirs[0]
        return float(np.nanmean(np.abs(ours - theirs)[settled]))

    mouth = correlation(np.array([at_moment(driver, fps, "mouth", moment) for moment in moments])[settled],
                        np.array([face["mouth"] if face else np.nan for face in shown])[settled])
    return {
        "frames": len(outputs),
        "moment_step_ms": {"mean": round(float(steps[settled].mean()), 1), "sd": round(float(steps[settled].std()), 1),
                           "shortest": round(float(steps[settled].min()), 1), "longest": round(float(steps[settled].max()), 1)},
        # How much the head changes its speed from one output frame to the next, as shown at even intervals.
        "head_acceleration_percent_of_eye_span": round(landmark_jitter(points[settled], span) * 100, 3),
        "pose_error_deg": {"yaw": round(error("yaw"), 2), "pitch": round(error("pitch"), 2)},
        "mouth_opening_correlation": None if mouth is None else round(float(mouth), 3),
    }


def main():
    engine = ReenactmentEngine(tensorrt=True, compile_networks=True)
    report = {"draw_seconds": DRAW_SECONDS, "ready_seconds": READY_SECONDS, "clips": {}}
    print("clip  camera | moments shown, apart (ms): newest frame   even motion | head acceleration: newest  even | "
          "yaw error: newest  even | mouth: newest  even")
    for name in CLIPS:
        capture = cv2.VideoCapture(str(DRIVING / f"{name}.mp4"))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while len(frames) < SECONDS * fps:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        driver = faces_of(frames)
        with FaceTracker(video=False) as still:  # the clip animates one of its own frames, as the benchmark's pairs do
            source = next((f for f in frames if at_rest(read(still.process(f)))), frames[0])
        entry = {"camera_fps": round(fps, 2)}
        for label, even in (("newest_frame", False), ("even_motion", True)):
            engine.set_source(source)
            outputs, moments = session(engine, frames, fps, even)
            entry[label] = measure(outputs, moments, driver, fps)
        report["clips"][name] = entry
        a, b = entry["newest_frame"], entry["even_motion"]
        print(f"{name:4s} {fps:5.1f}  | {a['moment_step_ms']['shortest']:5.1f} to {a['moment_step_ms']['longest']:5.1f}, sd {a['moment_step_ms']['sd']:4.1f}   "
              f"{b['moment_step_ms']['shortest']:5.1f} to {b['moment_step_ms']['longest']:5.1f}, sd {b['moment_step_ms']['sd']:4.1f} | "
              f"{a['head_acceleration_percent_of_eye_span']:6.3f} {b['head_acceleration_percent_of_eye_span']:6.3f} | "
              f"{a['pose_error_deg']['yaw']:5.2f} {b['pose_error_deg']['yaw']:5.2f} | "
              f"{a['mouth_opening_correlation']} {b['mouth_opening_correlation']}", flush=True)

    def mean(mode, *keys):
        values = []
        for entry in report["clips"].values():
            value = entry[mode]
            for key in keys:
                value = value[key]
            values.append(value)
        return round(float(np.mean(values)), 3)

    report["summary"] = {mode: {
        "moment_step_sd_ms": mean(mode, "moment_step_ms", "sd"),
        "head_acceleration_percent_of_eye_span": mean(mode, "head_acceleration_percent_of_eye_span"),
        "yaw_error_deg": mean(mode, "pose_error_deg", "yaw"), "pitch_error_deg": mean(mode, "pose_error_deg", "pitch"),
        "mouth_opening_correlation": mean(mode, "mouth_opening_correlation"),
    } for mode in ("newest_frame", "even_motion")}
    print("all clips:", json.dumps(report["summary"]))
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
