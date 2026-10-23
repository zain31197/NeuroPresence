"""Can the voice drive the picture with the camera off? The feasibility study of the audio driven live mode.

Three parts, each written to results/speech_study.json:

1. Directions. How LivePortrait's expression keypoints move when a mouth opens and when eyes blink,
   fitted on its ten bundled driving clips with the face tracker's readings. Written to
   neuropresence/speech/directions.json, which the speech stage uses.
2. Speed. The engine driven from camera frames (the motion network reads each one) against driven
   from speech (the movement is made from the sound; no motion network, no face tracker).
3. Agreement. For every recording made with `record_camera.py --speech` (a video and its sound in
   data/recordings/), the mouth the voice makes against the real lip gap in the video: correlation at
   the best lag, how often the mouth is open while the real lips are shut, and how often it is open.

    python scripts/study_speech.py                 # all three
    python scripts/study_speech.py --skip-fit      # keep the directions, measure speed and agreement
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio_ffmpeg  # noqa: E402

from neuropresence.capture import FaceTracker  # noqa: E402
from neuropresence.capture.crop import square_face_crop  # noqa: E402
from neuropresence.pipeline import lip_gap  # noqa: E402
from neuropresence.reenactment import ReenactmentEngine  # noqa: E402
from neuropresence.speech import SpeechMotion, SpeechMouth  # noqa: E402
from neuropresence.speech.motion import DIRECTIONS  # noqa: E402

CLIPS = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving"
SOURCE = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "source" / "s0.jpg"
RECORDINGS = ROOT / "data" / "recordings"
OUT = ROOT / "results" / "speech_study.json"
LIPS, EYES = [6, 12, 14, 17, 19, 20], [11, 13, 15, 16, 18]  # LivePortrait's own (live_portrait_pipeline.py)
TIMED_FRAMES = 300
SHUT_GAP = 0.01  # a real lip gap below this is lips shut
OPEN = 0.3  # an openness above this counts as an open mouth


def git_revision():
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return rev + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ------------------------------------------------------------- 1. directions


def read_clip(engine, path):
    """Per frame: the lip gap, the blink score and the engine's expression keypoints."""
    gaps, blinks, expressions = [], [], []
    capture = cv2.VideoCapture(str(path))
    with FaceTracker(video=True) as tracker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            track = tracker.process(frame)
            if not track.ok:
                continue
            gap = lip_gap(track)
            if gap is None:
                continue
            info, _ = engine.read(square_face_crop(frame, track.bbox, scale=2.0))
            gaps.append(gap)
            blinks.append((track.blendshapes.get("eyeBlinkLeft", 0.0) + track.blendshapes.get("eyeBlinkRight", 0.0)) / 2)
            expressions.append(info["exp"].numpy().reshape(21, 3))
    return np.array(gaps), np.array(blinks), np.array(expressions)


def direction(signals, changes, keep):
    """The one direction in keypoint space that the change follows, per unit of the signal, on `keep` only."""
    s = np.concatenate(signals)
    e = np.concatenate(changes).reshape(len(s), -1)
    fitted = ((s[:, None] * e).sum(axis=0) / (s ** 2).sum()).reshape(21, 3)
    out = np.zeros_like(fitted)
    out[keep] = fitted[keep]
    return out


def fit_directions(engine):
    opening, open_changes, blinking, blink_changes, frames = [], [], [], [], 0
    for path in sorted(CLIPS.glob("d*.mp4")):
        gaps, blinks, expressions = read_clip(engine, path)
        if len(gaps) < 30:
            continue
        frames += len(gaps)
        shut = gaps <= np.percentile(gaps, 10)  # each speaker's own closed mouth
        opening.append(gaps)
        open_changes.append(expressions - expressions[shut].mean(axis=0))
        eyes_open = blinks <= np.percentile(blinks, 30)
        blinking.append(blinks - blinks[eyes_open].mean())
        blink_changes.append(expressions - expressions[eyes_open].mean(axis=0))
        print(f"  {path.stem}: {len(gaps)} frames", flush=True)
    opened, blinked = direction(opening, open_changes, LIPS), direction(blinking, blink_changes, EYES)
    DIRECTIONS.write_text(json.dumps({
        "about": "Directions in LivePortrait's expression keypoints (21 x 3), fitted on its bundled driving clips "
                 f"({frames} frames) with the face tracker's readings: 'open' per unit of lip gap over eye span, measured "
                 "from each clip's own closed mouth; 'blink' per unit of the tracker's blink score, from open eyes. Each is "
                 "kept to its own keypoints, the lips [6, 12, 14, 17, 19, 20] and the eyes [11, 13, 15, 16, 18], as "
                 "LivePortrait does in its own retargeting. Written by scripts/study_speech.py.",
        "open": np.round(opened, 5).tolist(),
        "blink": np.round(blinked, 5).tolist(),
    }, indent=1), encoding="utf-8")
    return {"clips": len(opening), "frames": frames,
            "open_largest": {"keypoint": int(np.abs(opened).max(axis=1).argmax()), "per_unit_gap": round(float(np.abs(opened).max()), 4)},
            "blink_largest": {"keypoint": int(np.abs(blinked).max(axis=1).argmax()), "per_unit_score": round(float(np.abs(blinked).max()), 4)}}


# ------------------------------------------------------------------ 2. speed


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def measure_speed(engine):
    engine.set_source(cv2.imread(str(SOURCE)))
    faces = []
    capture = cv2.VideoCapture(str(CLIPS / "d6.mp4"))
    with FaceTracker(video=True) as tracker:
        while len(faces) < TIMED_FRAMES + 20:
            ok, frame = capture.read()
            if not ok:
                break
            track = tracker.process(frame)
            if track.ok:
                faces.append(square_face_crop(frame, track.bbox, scale=2.0))

    def camera():
        engine.reset_reference()
        for face in faces[:20]:
            engine.drive(None, motion=engine.read(face))
        motion, render, start = [], [], sync()
        for face in faces[20:20 + TIMED_FRAMES]:
            t0 = sync()
            reading = engine.read(face)
            motion.append(sync() - t0)
            engine.drive(None, motion=reading)
            render.append(engine.last_timing_ms["render"])
        return TIMED_FRAMES / (sync() - start), motion, render

    def speech():
        engine.reset_reference()
        neutral, rotation = engine.source_reading()
        moving, mouth = SpeechMotion(neutral, engine.rotation, seed=1), SpeechMouth(16000)
        sound = np.sin(2 * np.pi * 150 * np.arange(16000 * 12) / 16000).astype(np.float32) * 0.1
        engine.drive(None, motion=(neutral, rotation))
        motion, render, start = [], [], sync()
        for index in range(TIMED_FRAMES):
            t0 = sync()
            reading = moving.reading(mouth.feed(sound[index * 533:(index + 1) * 533]), index / 30)
            motion.append(sync() - t0)
            engine.drive(None, motion=reading)
            render.append(engine.last_timing_ms["render"])
        return TIMED_FRAMES / (sync() - start), motion, render

    out = {}
    for name, run in (("camera", camera), ("speech", speech)):
        fps, motion, render = run()
        out[name] = {"fps_back_to_back": round(fps, 1), "motion_ms": round(float(np.median(motion)) * 1000, 2),
                     "render_ms": round(float(np.median(render)), 2)}
        print(f"  {name}: {out[name]}", flush=True)
    out["note"] = ("Frames drawn one after another with nothing else running: the ceiling of each way. The camera way "
                   "also needs the face tracker on the CPU for every frame, which is not counted here.")
    return out


# -------------------------------------------------------------- 3. agreement


def read_sound(path):
    with wave.open(str(path)) as source:
        return np.frombuffer(source.readframes(source.getnframes()), np.int16).astype(np.float32) / 32768, source.getframerate()


def best_lag(computed, real, most=6):
    best = (0, -2.0)
    for lag in range(-most, most + 1):  # positive: the real mouth moves after the computed one
        x = computed[max(0, -lag): len(computed) - max(0, lag)]
        y = real[max(0, lag): len(real) - max(0, -lag)]
        if x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best[1]:
            best = (lag, r)
    return best


def agreement(recording):
    """The voice's mouth against the real lip gap of one recording made with record_camera.py --speech."""
    meta = json.loads(recording.with_suffix(".json").read_text(encoding="utf-8"))
    sound, rate = read_sound(RECORDINGS / meta["sound"]["file"])
    times = np.array(meta["times_s"]) - meta["times_s"][0]
    gaps = []
    capture = cv2.VideoCapture(str(recording))
    with FaceTracker(video=True) as tracker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            track = tracker.process(frame)
            gap = lip_gap(track) if track.ok else None
            gaps.append(np.nan if gap is None else gap)
    gaps = np.array(gaps[:len(times)])
    times = times[:len(gaps)]
    mouth, openness, fed = SpeechMouth(rate), [], 0
    for moment in times:
        upto = int(round(moment * rate))
        if upto > fed:
            mouth.feed(sound[fed:upto])
            fed = upto
        openness.append(mouth.openness)
    openness = np.array(openness)
    seen = ~np.isnan(gaps)
    real = np.where(seen, gaps, 0.0)
    lag, r = best_lag(openness[seen], real[seen])
    shut = seen & (real < SHUT_GAP)
    return {"frames": int(seen.sum()), "seconds": round(float(times[-1]), 1), "microphone": meta["sound"]["microphone"],
            "correlation": round(r, 3), "best_lag_frames": lag,
            "open_while_real_lips_shut": round(float(np.mean(openness[shut])), 3) if shut.any() else None,
            "open_share": round(float(np.mean(openness[seen] > OPEN)), 3),
            "real_open_share": round(float(np.mean(real[seen] > 0.03)), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-fit", action="store_true", help="keep neuropresence/speech/directions.json as it is")
    args = parser.parse_args()
    result = {"date": datetime.now().isoformat(timespec="seconds"),
              "machine": {"os": platform.platform(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"},
              "code": git_revision()}
    engine = ReenactmentEngine(tensorrt=True, compile_networks=True)  # as the app runs it
    if not args.skip_fit:
        print("Fitting the directions on the sample clips:")
        result["directions"] = fit_directions(engine)
    print("Speed, drawn from camera frames and from speech:")
    result["speed"] = measure_speed(engine)
    recordings = [path for path in sorted(RECORDINGS.glob("*.avi"))
                  if path.with_suffix(".json").exists()
                  and "sound" in json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))]
    result["agreement"] = {}
    for recording in recordings:
        print(f"Agreement on {recording.name}:")
        result["agreement"][recording.stem] = agreement(recording)
        print(f"  {result['agreement'][recording.stem]}")
    if not recordings:
        print("No recording with sound yet: make one with  python scripts/record_camera.py --speech --name <you>_speech")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
