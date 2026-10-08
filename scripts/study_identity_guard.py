"""How the live identity score behaves: in normal use, and when something has gone wrong.

The identity monitor scores the output against the picture it is made from (CSIM). Before it
is allowed to act on that score, this measures what it will see, so that its limits come from
data and not from a guess:

  normal   every pair of the benchmark, as the pipeline runs it;
  faults   the same clips with one thing wrong at a time: a neutral pose taken at a bad
           moment, the head pushed past its range with the limit off (and, for comparison,
           with it on), an exaggerated expression, a head of the wrong size, a covered mouth,
           and another person driving the picture.

For every case it reports the scores and, at each candidate limit, the share of frames below
it and the longest time spent below it. It then runs the guard itself (identity/guard.py) over
the recorded scores, at the two readings a second the live monitor takes, to see what it would do.

Usage:
    python scripts/study_identity_guard.py                      # about twenty minutes
    python scripts/study_identity_guard.py --pictures some/dir  # also save the worst output of each case
    python scripts/study_identity_guard.py --replay             # run the guard again over the scores already
                                                                # recorded, after a change to it: a few seconds
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker, FrameSource
from neuropresence.capture.enrol import first_frontal_frame
from neuropresence.capture.steady import SteadyCrop
from neuropresence.identity import SAME_PERSON_CSIM, IdentityGuard, IdentityScorer
from neuropresence.pipeline import Pipeline, at_rest
from neuropresence.reenactment import ReenactmentEngine
from neuropresence.targets import TARGETS

LIMITS = (0.80, 0.75, 0.70, 0.60, 0.50, 0.40, SAME_PERSON_CSIM, 0.30)
NORMAL_FRAMES = 150  # per pair, as in the benchmark
FAULT_CLIPS = ("d0", "d6", "d9", "d13")  # four different people
FAULT_FRAMES = 75  # per clip and fault: about three seconds
MONITOR_INTERVAL = 0.5  # the live monitor takes a reading this often
REPEATS = 4  # the guard is shown each clip this many times over, as if what it shows went on
NEUTRAL_FRAMES = 100  # a bad neutral pose is watched for longer: it lasts until it is retaken
CROP_SCALE = 2.0


def clip_path(name):
    return ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving" / f"{name}.mp4"


def read_clip(path, max_frames):
    with FrameSource(str(path)) as video:
        fps = video.fps or 25.0
        frames = []
        while len(frames) < max_frames and (frame := video.read()) is not None:
            frames.append(frame)
    return frames, fps


def score(scorer, measure, source_embedding, output):
    """The live score of one output frame, or None if no single face can be found in it."""
    track = measure.process(output)
    if not track.ok:
        return None
    return scorer.similarity(source_embedding, scorer.embed(output, track.landmarks))


def summary(scores, fps):
    """What a series of scores looks like to a monitor. A frame with no face counts as below every limit."""
    found = np.array([s for s in scores if s is not None], dtype=np.float64)
    low = np.array([-1.0 if s is None else s for s in scores], dtype=np.float64)
    report = {
        "frames": len(scores),
        "frames_without_a_face": int(len(scores) - len(found)),
        "mean": round(float(found.mean()), 3) if len(found) else None,
        "p05": round(float(np.percentile(found, 5)), 3) if len(found) else None,
        "p01": round(float(np.percentile(found, 1)), 3) if len(found) else None,
        "min": round(float(found.min()), 3) if len(found) else None,
        "below": {},
    }
    for limit in LIMITS:
        under = low < limit
        longest = run = 0
        for flag in under:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        report["below"][f"{limit:.2f}"] = {"share": round(float(under.mean()), 3) if len(low) else None,
                                           "longest_seconds": round(longest / fps, 2)}
    return report


def pooled(clips):
    """All the frames of several clips, each given as (scores, fps), as one series. The longest
    time under a limit is the longest in any one clip."""
    report = summary([s for scores, _ in clips for s in scores], 25.0)
    for limit in LIMITS:
        key = f"{limit:.2f}"
        report["below"][key]["longest_seconds"] = max(summary(scores, fps)["below"][key]["longest_seconds"]
                                                      for scores, fps in clips)
    report["guard"] = guard_response(clips)
    report["clips"] = [{"fps": round(fps, 2), "scores": [None if s is None else round(s, 3) for s in scores]}
                       for scores, fps in clips]
    return report


def guard_response(clips):
    """What the guard does with these scores, read twice a second, each clip repeated as if it went on.

    A fresh neutral pose cannot change a recording, so "fallback" here means: if the fresh neutral
    pose did not help. Every possible timing of the readings against the frames is tried.
    """
    runs, anchors, fallbacks = 0, [], []
    for scores, fps in clips:
        if not scores:
            continue
        step = max(1, round(fps * MONITOR_INTERVAL))
        series = list(scores) * REPEATS
        for phase in range(step):
            guard, anchored, fell = IdentityGuard(), None, None
            for index in range(phase, len(series), step):
                action = guard.sample(series[index], index / fps)
                if action == "anchor" and anchored is None:
                    anchored = index / fps
                if action == "fallback":
                    fell = index / fps
                    break
            runs += 1
            if anchored is not None:
                anchors.append(anchored)
            if fell is not None:
                fallbacks.append(fell)
    return {
        "runs": runs,
        "share_with_a_fresh_neutral_pose": round(len(anchors) / runs, 3) if runs else None,
        "seconds_to_it": round(float(np.median(anchors)), 1) if anchors else None,
        "share_ending_on_the_still_picture": round(len(fallbacks) / runs, 3) if runs else None,
        "seconds_to_that": round(float(np.median(fallbacks)), 1) if fallbacks else None,
    }


# ------------------------------------------------------------------ normal use


def normal_pair(pair, engine, scorer, measure):
    """One benchmark pair, run by the pipeline exactly as the benchmark runs it."""
    frames, fps = read_clip(ROOT / pair["driving"], NORMAL_FRAMES)
    if pair["source"] == "first_frame":
        with FaceTracker() as finder:
            source = first_frontal_frame(iter_frames(frames), finder)
    else:
        source = cv2.imread(str(ROOT / pair["source"]))
    engine.set_source(source)
    source_embedding = scorer.embed(engine.source_frame)
    scores = []
    with FaceTracker() as tracker:
        pipeline = Pipeline(tracker, engine)
        for index, frame in enumerate(frames):
            result = pipeline.step(frame, at=index / fps)
            if result.live:
                scores.append(score(scorer, measure, source_embedding, result.output))
    return scores, fps


class iter_frames:
    """A list of frames with the read() that first_frontal_frame expects."""

    def __init__(self, frames):
        self._frames = iter(frames)

    def read(self):
        return next(self._frames, None)


# ---------------------------------------------------------------------- faults


def turned(pitch=0.0, yaw=0.0):
    def change(engine, info):
        info = dict(info)
        info["pitch"], info["yaw"] = info["pitch"] + pitch, info["yaw"] + yaw
        return info
    return change


def resized(factor):
    def change(engine, info):
        info = dict(info)
        info["scale"] = info["scale"] * factor
        return info
    return change


def exaggerated(gain):
    def change(engine, info):
        info = dict(info)
        rest = engine._reference["info"]["exp"]
        info["exp"] = rest + gain * (info["exp"] - rest)
        return info
    return change


def covered_mouth(frame, track):
    """The camera frame with the lower part of the face hidden, as by a hand or a cup."""
    x, y, w, h = track.bbox
    hidden = frame.copy()
    cv2.rectangle(hidden, (int(x), int(y + 0.55 * h)), (int(x + w), int(y + 1.05 * h)), (70, 80, 95), -1)
    return hidden


def unrest(track):
    """How far a tracked face is from a resting one: a turned head and an open mouth both count."""
    yaw, pitch, _ = track.pose_deg
    return abs(yaw) / 15.0 + abs(pitch) / 15.0 + 3.0 * track.blendshapes.get("jawOpen", 0.0)


def driven(engine, tracker, frames, fps, neutral_face, limit=True, change=None, before=None):
    """The output for each frame that has one face, driven straight through the engine so that one
    thing can be altered: the movement that was read (`change`) or the camera frame (`before`)."""
    crop = SteadyCrop()
    engine.set_reference(neutral_face)
    outputs = []
    for index, frame in enumerate(frames):
        track = tracker.process(frame)
        if track.ok and before is not None:
            frame = before(frame, track)
            track = tracker.process(frame)
        if not track.ok:
            crop.reset()
            continue
        face, _ = crop(frame, track.landmarks, index / fps, CROP_SCALE)
        info, rotation = engine.read(face)
        if change is not None:
            info = change(engine, info)
            rotation = engine._rotation(info["pitch"], info["yaw"], info["roll"])
        outputs.append(engine.drive(face, at=index / fps, steady=True, limit=limit, motion=(info, rotation)))
    return outputs


def resting_face(frames, fps):
    """The face crop of the first frame at rest, which is what a session takes as its neutral pose."""
    crop = SteadyCrop()
    with FaceTracker() as tracker:
        first = None
        for index, frame in enumerate(frames):
            track = tracker.process(frame)
            if not track.ok:
                continue
            face, _ = crop(frame, track.landmarks, index / fps, CROP_SCALE)
            first = face if first is None else first
            if at_rest(track):
                return face
    return first


def least_resting_face(frames, fps):
    crop = SteadyCrop()
    worst, worst_face = -1.0, None
    with FaceTracker() as tracker:
        for index, frame in enumerate(frames):
            track = tracker.process(frame)
            if not track.ok:
                continue
            face, _ = crop(frame, track.landmarks, index / fps, CROP_SCALE)
            if unrest(track) > worst:
                worst, worst_face = unrest(track), face
    return worst_face, round(worst, 2)


FAULTS = (
    # name, what is wrong, keyword arguments for driven()
    ("none", "Nothing: the control, driven the same way as the faults", {}),
    ("pitch_20_no_limit", "Head pushed 20 degrees past where it is, head range off", {"limit": False, "change": turned(pitch=20)}),
    ("pitch_30_no_limit", "Head pushed 30 degrees, head range off", {"limit": False, "change": turned(pitch=30)}),
    ("pitch_45_no_limit", "Head pushed 45 degrees, head range off", {"limit": False, "change": turned(pitch=45)}),
    ("yaw_30_no_limit", "Head turned 30 degrees further, head range off", {"limit": False, "change": turned(yaw=30)}),
    ("yaw_45_no_limit", "Head turned 45 degrees further, head range off", {"limit": False, "change": turned(yaw=45)}),
    ("pitch_45_limit", "Head pushed 45 degrees, head range on", {"limit": True, "change": turned(pitch=45)}),
    ("yaw_45_limit", "Head turned 45 degrees further, head range on", {"limit": True, "change": turned(yaw=45)}),
    ("expression_x2", "Expression twice as strong as it was read", {"change": exaggerated(2.0)}),
    ("expression_x3", "Expression three times as strong", {"change": exaggerated(3.0)}),
    ("size_x1.3", "Head 1.3 times its size, head range off", {"limit": False, "change": resized(1.3)}),
    ("size_x0.75", "Head 0.75 times its size, head range off", {"limit": False, "change": resized(0.75)}),
    ("mouth_covered", "Lower part of the face hidden in the camera picture", {"before": covered_mouth}),
)


def study_faults(engine, scorer, measure, picture_dir):
    series = {name: [] for name, _, _ in FAULTS}
    series["bad_neutral"], series["other_person"] = [], []
    unrest_found = []
    clips = {name: read_clip(clip_path(name), NEUTRAL_FRAMES) for name in FAULT_CLIPS}
    for position, name in enumerate(FAULT_CLIPS):
        frames, fps = clips[name]
        with FaceTracker() as finder:
            source = first_frontal_frame(iter_frames(frames), finder)
        engine.set_source(source)
        source_embedding = scorer.embed(engine.source_frame)
        neutral = resting_face(frames, fps)

        def keep(fault, outputs, clip_fps):
            scores = [score(scorer, measure, source_embedding, output) for output in outputs]
            series[fault].append((scores, clip_fps))
            if picture_dir is not None and outputs:
                worst = int(np.argmin([-1.0 if s is None else s for s in scores]))
                label = "no face" if scores[worst] is None else f"{scores[worst]:.2f}"
                sheet = np.hstack([cv2.resize(engine.source_frame, (384, 384)), cv2.resize(outputs[worst], (384, 384))])
                cv2.putText(sheet, f"{fault} {name}: {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imwrite(str(picture_dir / f"{fault}_{name}.jpg"), sheet)

        def run(fault, clip_frames, clip_fps, *args, **how):  # a fresh tracker each time: it follows a face from frame to frame
            with FaceTracker() as tracker:
                keep(fault, driven(engine, tracker, clip_frames, clip_fps, *args, **how), clip_fps)

        for fault, _, how in FAULTS:
            run(fault, frames[:FAULT_FRAMES], fps, neutral, **how)
        # A neutral pose taken at the worst moment of the clip, then the whole clip driven from it.
        bad, how_bad = least_resting_face(frames, fps)
        unrest_found.append(how_bad)
        run("bad_neutral", frames, fps, bad)
        # Someone else sits down: the neutral pose is this person's, the movement another's.
        other_frames, other_fps = clips[FAULT_CLIPS[(position + 1) % len(FAULT_CLIPS)]]
        run("other_person", other_frames[:FAULT_FRAMES], other_fps, neutral)
        print(f"  faults on {name}: done", flush=True)

    described = {name: text for name, text, _ in FAULTS}
    described["bad_neutral"] = "Neutral pose taken at the least restful frame of the clip (head turned or mouth open)"
    described["other_person"] = "Another person drives the picture from this person's neutral pose"
    report = [{"fault": name, "what": described[name], **pooled(series[name])} for name in series]
    return report, unrest_found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "results" / "identity_guard_study.json"))
    parser.add_argument("--pictures", default=None, help="folder for the worst output of each fault, to look at")
    parser.add_argument("--replay", action="store_true",
                        help="do not record anything: run the guard again over the scores already in the result file")
    args = parser.parse_args()
    if args.replay:
        report = json.loads(Path(args.out).read_text(encoding="utf-8"))

        def again(case):
            return guard_response([(clip["scores"], clip["fps"]) for clip in case["clips"]])

        for pair, clip in zip(report["normal"], report["normal_all"]["clips"]):
            pair["guard"] = guard_response([(clip["scores"], clip["fps"])])
        report["normal_all"]["guard"] = again(report["normal_all"])
        for fault in report["faults"]:
            fault["guard"] = again(fault)
        report["guard_replayed"] = datetime.now().isoformat(timespec="seconds")
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        show(report)
        print(f"\nThe guard was run again over the recorded scores. Written to {args.out}")
        return
    picture_dir = Path(args.pictures) if args.pictures else None
    if picture_dir is not None:
        picture_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.loads((ROOT / "benchmarks" / "samples.json").read_text(encoding="utf-8"))["pairs"]
    engine = ReenactmentEngine(tensorrt=True)
    with FaceTracker(video=False) as measure:
        scorer = IdentityScorer(tracker=measure)

        normal, normal_scores = [], []
        for pair in pairs:
            scores, fps = normal_pair(pair, engine, scorer, measure)
            normal_scores.append((scores, fps))
            normal.append({"name": pair["name"], **summary(scores, fps), "guard": guard_response([(scores, fps)])})
            print(f"  normal {pair['name']}: mean {normal[-1]['mean']}, lowest {normal[-1]['min']}", flush=True)

        # What one score costs the monitor, which finds the face in the output by itself.
        frames, fps = read_clip(clip_path("d0"), 30)
        with FaceTracker() as tracker:
            outputs = driven(engine, tracker, frames, fps, resting_face(frames, fps))
        started = time.perf_counter()
        for output in outputs:
            scorer.embed(output)
        score_ms = (time.perf_counter() - started) * 1000 / len(outputs)

        faults, unrest_found = study_faults(engine, scorer, measure, picture_dir)

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": {"os": platform.platform(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "target_csim": TARGETS["csim"],
        "same_person_csim": SAME_PERSON_CSIM,
        "limits": list(LIMITS),
        "one_score_ms": round(score_ms, 1),
        "normal": normal,
        "normal_all": pooled(normal_scores),
        "faults": faults,
        "bad_neutral_unrest": unrest_found,
        "notes": [
            "A score is the cosine similarity of ArcFace embeddings of the output frame and the source picture.",
            "below[limit].share is the share of frames under the limit; longest_seconds is the longest unbroken time "
            "under it in any one clip. A frame in which no single face is found counts as under every limit.",
            f"Normal use: the {len(pairs)} pairs of the benchmark, {NORMAL_FRAMES} frames each, run by the pipeline.",
            f"Faults: {len(FAULT_CLIPS)} clips, {FAULT_FRAMES} frames each ({NEUTRAL_FRAMES} for the bad neutral pose), "
            "driven straight through the engine with the one thing altered.",
            f"guard: the guard run over the recorded scores, one reading every {MONITOR_INTERVAL} s, each clip repeated "
            f"{REPEATS} times, for every timing of the readings. A recording does not change when a fresh neutral pose "
            "is taken, so ending on the still picture means: if the fresh neutral pose did not help.",
            "The sample clips are short studio recordings. What a webcam session of many minutes looks like is not in here.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    show(report)
    print(f"\nWritten to {args.out}")


def show(report):
    print(f"\nOne score costs the monitor {report['one_score_ms']:.0f} ms.")
    print(f"{'case':28s} {'mean':>6s} {'min':>6s}   share below 0.80 / 0.75 / 0.50     guard: fresh neutral pose in, after s / still picture in, after s")
    rows = [("normal, all pairs", report["normal_all"])] + [(f["fault"], f) for f in report["faults"]]

    def shown(value, digits=2):
        return "   -" if value is None else f"{value:.{digits}f}"

    for name, r in rows:
        b = r["below"]
        g = r["guard"]
        print(f"{name:28s} {shown(r['mean'], 3):>6s} {shown(r['min'], 3):>6s}   "
              f"{shown(b['0.80']['share'])} / {shown(b['0.75']['share'])} / {shown(b['0.50']['share'])}            "
              f"{shown(g['share_with_a_fresh_neutral_pose'])}, {shown(g['seconds_to_it'], 1)} s / "
              f"{shown(g['share_ending_on_the_still_picture'])}, {shown(g['seconds_to_that'], 1)} s")


if __name__ == "__main__":
    main()
