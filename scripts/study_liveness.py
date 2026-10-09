"""What can pass the liveness prompt and the consent check? Measured on the sample material.

Before a face is verified and before every camera session, the person is asked for two actions
chosen at random (neuropresence/consent/liveness.py), and the face at the camera is compared with
the registered signatures all the way through (neuropresence/consent/check.py). This script
measures what the limits of both rest on, without a camera:

  1. What the sample clips contain: how far heads turn by themselves, how often people blink
     and how long a mouth stays open while talking.
  2. A recording played to the camera: the real challenge is run over each clip's readings, from
     every half second of it and for every order of actions.
  3. A photograph held to the camera: one frame of each clip, tilted as a flat print or a phone
     can be, swung from side to side, with the eyes covered and with a dark shape on the mouth.
  4. Whether the face stays recognisable all the way through: every clip's frames scored against
     its own first frame at rest and against the first frame of every other clip.
  5. With --stand-in: a stand-in that answers the prompts. The sample material has no recording
     of a person turning the head on request, so one is built from a clip's own frames (blink,
     open mouth) and from frames in which the engine turns that face (head turn). Needs the GPU.

Usage:
    python scripts/study_liveness.py              # about 5 minutes
    python scripts/study_liveness.py --stand-in   # also part 5, a few minutes more
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker, TrackStatus
from neuropresence.capture.crop import square_face_crop
from neuropresence.consent import ConsentCheck, liveness
from neuropresence.consent.liveness import ACTIONS, TURNS, LivenessChallenge, Reading, Stage, at_rest, read
from neuropresence.identity import SAME_PERSON_CSIM, IdentityScorer

DRIVING = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving"
OUT = ROOT / "results" / "liveness_study.json"
SCORE_EVERY = 3  # every third frame of a clip is scored for identity
START_EVERY_SECONDS = 0.5  # a replayed recording is tried from every half second of it
TILTS_DEG = (10, 20, 30, 40, 50, 60, 65, 70)
# How far the camera is from a photograph that fills its picture, in widths of the photograph. The nearest is
# a camera with a view 80 degrees wide, which is wider than most webcams; the nearer, the larger the turn read.
DISTANCES = (0.6, 0.8, 1.4, 3.0)
SWING_DEG, SWING_PERIOD_SECONDS, SWING_SECONDS, SWING_FPS = 60.0, 2.0, 12.0, 25.0
LOST = Reading(TrackStatus.NO_FACE)

WITH_A_TURN = [pair for pair in itertools.permutations(ACTIONS, 2) if any(action in TURNS for action in pair)]
WITHOUT_A_TURN = [pair for pair in itertools.permutations(ACTIONS, 2) if not any(action in TURNS for action in pair)]


# ------------------------------------------------------------------ the clips


def record(path, scorer):
    """Every frame of a clip as a reading, with the face signature of every third frame."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    readings, signatures, rest_frame, first_frame = [], [], None, None
    with FaceTracker() as tracker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            track = tracker.process(frame)
            reading = read(track)
            readings.append(reading)
            if track.ok:
                first_frame = frame if first_frame is None else first_frame
                if rest_frame is None and at_rest(reading) and abs(track.pose_deg[0]) <= 6:
                    rest_frame = frame
                if index % SCORE_EVERY == 0:
                    signatures.append(scorer.embed(frame, track.landmarks))
            index += 1
    capture.release()
    return {"fps": fps, "readings": readings, "signatures": signatures,
            "photo": rest_frame if rest_frame is not None else first_frame}


def runs(flags, fps):
    """The length in seconds of every unbroken run of True."""
    found, run = [], 0
    for flag in list(flags) + [False]:
        if flag:
            run += 1
        elif run:
            found.append(run / fps)
            run = 0
    return found


def describe(clip):
    """What a clip contains, in the terms the challenge uses."""
    readings, fps = clip["readings"], clip["fps"]
    seen = [r for r in readings if r.ok]
    window = int(round(liveness.ACTION_SECONDS * fps))
    largest_turn = 0.0
    for start, reading in enumerate(readings):
        if at_rest(reading):
            after = [r.yaw for r in readings[start:start + window + 1] if r.ok]
            largest_turn = max(largest_turn, max(abs(yaw - reading.yaw) for yaw in after))
    closed = runs([r.ok and r.eyes >= liveness.EYES_CLOSED for r in readings], fps)
    opened = runs([r.ok and r.jaw >= liveness.OPEN_JAW for r in readings], fps)
    return {
        "seconds": round(len(readings) / fps, 1),
        "share_with_one_face": round(len(seen) / len(readings), 3),
        "share_at_rest": round(sum(at_rest(r) for r in readings) / len(readings), 3),
        "yaw_deg": [round(min(r.yaw for r in seen), 1), round(max(r.yaw for r in seen), 1)],
        "largest_turn_from_rest_deg": round(largest_turn, 1),
        "eyes_closed": {"times": len(closed), "longest_s": round(max(closed, default=0.0), 2)},
        "mouth_wide_open": {"times": len(opened), "longest_s": round(max(opened, default=0.0), 2),
                            "times_held": sum(run >= liveness.OPEN_HOLD_SECONDS for run in opened)},
    }


# ------------------------------------------------- a recording played back


def attempt(readings, fps, start, actions, waits, there_and_back=True):
    """The challenge run over a recording from frame `start`, played over and over. An ordinary
    clip is played forwards and then backwards, so that it has no seam where the face would jump."""
    challenge = LivenessChallenge(actions, waits)
    count = len(readings)
    period = max(1, 2 * count - 2) if there_and_back else count
    longest = int((challenge.settle_seconds + liveness.ACTION_SECONDS) * (len(actions) + 1) * fps)
    for step in range(longest):
        index = (start + step) % period
        challenge.sample(readings[index if index < count else period - index], step / fps)
        if challenge.done:
            break
    return challenge


def replay(readings, fps, orders, seed=0, there_and_back=True):
    """How the challenge ends for a recording, over every start and every order of actions."""
    rng = random.Random(seed)
    ends = {"attempts": 0, "passed": 0}
    for start in range(0, len(readings), max(1, int(round(START_EVERY_SECONDS * fps)))):
        for actions in orders:
            waits = [rng.uniform(*liveness.WAIT_SECONDS) for _ in actions]
            challenge = attempt(readings, fps, start, actions, waits, there_and_back)
            ends["attempts"] += 1
            if challenge.stage is Stage.PASSED:
                ends["passed"] += 1
            else:
                key = f"failed_{challenge.failure}_at_step_{challenge.step + 1}"
                ends[key] = ends.get(key, 0) + 1
    ends["passed_share"] = round(ends["passed"] / ends["attempts"], 4) if ends["attempts"] else None
    return ends


def total(reports):
    summed = {}
    for report in reports:
        for key, value in report.items():
            if key != "passed_share":
                summed[key] = summed.get(key, 0) + value
    summed["passed_share"] = round(summed["passed"] / summed["attempts"], 4) if summed.get("attempts") else None
    return summed


def made_on_purpose(rest_seconds, fps=30.0):
    """The readings of a recording made to beat the prompt: every action, one after another, with a
    rest between them. No such recording exists in the sample material, so this one is written
    out as readings; it is what the tracker would read from a person doing exactly that."""
    def held(seconds, **reading):
        return [Reading(TrackStatus.OK, **reading)] * int(round(seconds * fps))

    def turn(sign):
        ramp = [Reading(TrackStatus.OK, yaw=sign * 40.0 * (step + 1) / 9) for step in range(9)]
        return ramp + held(0.4, yaw=sign * 40.0) + ramp[::-1]

    rest = held(rest_seconds, yaw=1.0, jaw=0.05, eyes=0.1)
    return (rest + turn(1) + rest + held(0.15, eyes=0.8) + rest + turn(-1) + rest + held(0.6, jaw=0.8))


# ----------------------------------------------------------- a photograph


def tilted(image, about_vertical_deg, distance=1.4):
    """The picture as a flat print turned about its upright axis, seen by a camera `distance` picture widths away."""
    h, w = image.shape[:2]
    f = distance * w
    angle = np.radians(about_vertical_deg)
    turn = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    corners = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=np.float64)
    turned = corners @ turn.T
    turned[:, 2] += f
    seen = np.column_stack([f * turned[:, 0] / turned[:, 2] + w / 2, f * turned[:, 1] / turned[:, 2] + h / 2])
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), seen.astype(np.float32))
    return cv2.warpPerspective(image, matrix, (w, h), borderValue=(90, 90, 90))


def eyes_covered(image, track):
    """The photograph with a strip of card over the eyes."""
    points = track.landmarks
    covered = image.copy()
    y, half = int((points[33][1] + points[263][1]) / 2), int(0.09 * track.bbox[3])
    cv2.rectangle(covered, (int(points[234][0]), y - half), (int(points[454][0]), y + half), (150, 170, 200), -1)
    return covered


def mouth_darkened(image, track):
    """The photograph with a dark oval where an open mouth would be."""
    points = track.landmarks
    darkened = image.copy()
    centre = (int(points[13][0]), int((points[13][1] + points[14][1]) / 2 + 0.04 * track.bbox[3]))
    cv2.ellipse(darkened, centre, (int(0.13 * track.bbox[2]), int(0.09 * track.bbox[3])), 0, 0, 360, (30, 20, 40), -1)
    return darkened


def photograph(name, photo):
    """What the tracker reads from one photograph, however it is held."""
    report = {}
    with FaceTracker(video=False) as still:
        plain_track = still.process(photo)
        if not plain_track.ok:
            return None
        plain = read(plain_track)
        turns, nearness = {}, {str(distance): 0.0 for distance in DISTANCES}
        for angle in TILTS_DEG:  # to either side, and from near and far: the largest turn read
            seen = []
            for distance in DISTANCES:
                for sign in (1, -1):
                    reading = read(still.process(tilted(photo, sign * angle, distance)))
                    if reading.ok:
                        seen.append(abs(reading.yaw - plain.yaw))
                        nearness[str(distance)] = max(nearness[str(distance)], seen[-1])
            turns[str(angle)] = round(max(seen), 1) if seen else None  # None: the face is no longer found
        report["turn_read_when_tilted_deg"] = turns
        covered, darkened = read(still.process(eyes_covered(photo, plain_track))), read(still.process(mouth_darkened(photo, plain_track)))
        report["eyes"] = {"plain": round(plain.eyes, 2), "covered": round(covered.eyes, 2) if covered.ok else None}
        report["mouth"] = {"plain": round(plain.jaw, 2), "dark_shape": round(darkened.jaw, 2) if darkened.ok else None}
    # Swung from side to side in front of the camera, as someone trying to fake a turn would.
    swung, largest, found = [], 0.0, []
    for distance in DISTANCES:
        readings = []
        with FaceTracker() as tracker:
            for index in range(int(SWING_SECONDS * SWING_FPS)):
                angle = SWING_DEG * np.sin(2 * np.pi * index / SWING_FPS / SWING_PERIOD_SECONDS)
                readings.append(read(tracker.process(tilted(photo, float(angle), distance))))
        seen = [r.yaw for r in readings if r.ok]
        largest = max([largest] + [abs(yaw - plain.yaw) for yaw in seen])
        nearness[str(distance)] = max([nearness[str(distance)]] + [abs(yaw - plain.yaw) for yaw in seen])
        found.append(len(seen) / len(readings))
        swung.append(replay(readings, SWING_FPS, WITH_A_TURN, seed=1))
    report["swung"] = {"largest_turn_read_deg": round(largest, 1), "share_face_found": round(min(found), 3), **total(swung)}
    # The nearer the camera, the larger the turn a tilted photograph reads as: the largest, tilted or swung, at each distance.
    report["largest_turn_read_by_distance_deg"] = {distance: round(turn, 1) for distance, turn in nearness.items()}
    report["held_still"] = replay([plain] * int(4 * SWING_FPS), SWING_FPS, WITH_A_TURN + WITHOUT_A_TURN, seed=2)
    return report


# ------------------------------------------------------------ recognition


def below_runs(scores, limit):
    """The longest unbroken run of scores under the limit, counted in scores."""
    return max((len(list(group)) for under, group in itertools.groupby(scores, lambda s: s < limit) if under), default=0)


def recognition(clips):
    """Each clip's faces against its own first face at rest, and against the first face of every other clip."""
    scorer = IdentityScorer.similarity
    own, others = {}, {}
    for name, clip in clips.items():
        faces = [s for s in clip["signatures"] if s is not None]
        if clip["reference"] is None or not faces:
            continue
        same = [scorer(clip["reference"], face) for face in faces]
        own[name] = {
            "faces": len(same), "mean": round(float(np.mean(same)), 3), "lowest": round(min(same), 3),
            "share_below_limit": round(sum(s < SAME_PERSON_CSIM for s in same) / len(same), 4),
            "longest_run_below_limit": below_runs(same, SAME_PERSON_CSIM),
        }
        for other, their in clips.items():
            if other == name or their["reference"] is None:
                continue
            different = [scorer(their["reference"], face) for face in faces]
            others[f"{name} as {other}"] = {
                "highest": round(max(different), 3),
                "share_at_or_above_limit": round(sum(s >= SAME_PERSON_CSIM for s in different) / len(different), 4),
            }
    return {
        "limit": SAME_PERSON_CSIM,
        "same_person": own,
        "same_person_lowest": min((entry["lowest"] for entry in own.values()), default=None),
        "same_person_faces": sum(entry["faces"] for entry in own.values()),
        "another_person_pairs": len(others),
        "another_person_highest": max((entry["highest"] for entry in others.values()), default=None),
        "another_person_pairs_ever_at_or_above_limit": sorted(pair for pair, entry in others.items()
                                                              if entry["share_at_or_above_limit"] > 0),
        "another_person": others,
    }


# ---------------------------------------------------------------- a stand-in


class StandIn:
    """A sample face that does what the prompt asks for.

    Its blink and its open mouth are frames of the clip itself. Its head turns are drawn by the
    reenactment engine, which turns the clip's resting frame by a set angle: the sample material
    has no person turning that far. What the tracker reads from each drawn frame is measured
    here, so a turn is answered with the frame that reads as far as a person would turn.
    """

    REACTS_AFTER_SECONDS = 0.35  # a person does not start at once
    TURNS_IN_SECONDS = 0.45

    def __init__(self, engine, path):
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        with FaceTracker() as tracker:
            readings = [read(tracker.process(frame)) for frame in frames]
        calm = [i for i, r in enumerate(readings) if at_rest(r) and abs(r.yaw) <= 6 and r.eyes <= 0.3]
        if not calm:
            raise ValueError("no frame at rest")
        self.rest = frames[calm[0]]
        base = readings[calm[0]]
        near = [i for i, r in enumerate(readings) if r.ok and abs(r.yaw - base.yaw) <= 8]
        blink = max(near, key=lambda i: readings[i].eyes - readings[i].jaw)
        opened = max(near, key=lambda i: readings[i].jaw - readings[i].eyes)
        if readings[blink].eyes < 0.7 or readings[opened].jaw < 0.75:
            raise ValueError("no clear blink or no wide open mouth in this clip")
        self.blink, self.open_mouth = frames[blink], frames[opened]
        # The head turns: the resting frame turned by the engine, each read back by the tracker.
        engine.set_source(self.rest)
        with FaceTracker(video=False) as still:
            track = still.process(self.rest)
            crop = square_face_crop(self.rest, track.bbox, scale=2.0)
            engine.set_reference(crop)
            info, _ = engine.read(crop)
            self.turned = []  # (the turn the tracker reads, the frame), from the furthest right to the furthest left
            for offset in range(-60, 61, 4):
                changed = dict(info)
                changed["yaw"] = info["yaw"] + offset
                frame = engine.drive(crop, limit=False, motion=(changed, engine._rotation(changed["pitch"], changed["yaw"], changed["roll"])))
                seen = still.process(frame)
                if seen.ok:
                    self.turned.append((float(seen.pose_deg[0] - track.pose_deg[0]), frame))
        self.turned.sort(key=lambda pair: pair[0])
        self.reach = (self.turned[0][0], self.turned[-1][0])  # how far the tracker reads the drawn turns, right and left

    def frame(self, action, since):
        """The camera frame `since` seconds after `action` was asked for (None: nothing is asked)."""
        doing = since - self.REACTS_AFTER_SECONDS
        if action is None or doing < 0:
            return self.rest
        if action == "blink":
            return self.blink if doing < 0.15 else self.rest
        if action == "open_mouth":
            return self.open_mouth
        sign = 1.0 if action == "turn_left" else -1.0
        # As far as a person asked to turn would go: a little past what counts, reached in under half a second.
        goal = sign * (liveness.TURN_DEG + 8.0) * min(1.0, doing / self.TURNS_IN_SECONDS)
        return min(self.turned, key=lambda pair: abs(pair[0] - goal))[1]


def answering(stand_in, scorer, signatures, actions, waits, does=None):
    """One run of the whole check on a stand-in, with the real tracker and the real face match.
    `does` can map what is asked for to what the stand-in does instead."""
    check = ConsentCheck(LivenessChallenge(actions, waits), signatures, scorer.similarity)
    at, asked, asked_at, largest = 0.0, None, 0.0, 0.0
    with FaceTracker() as tracker:
        while not check.done and at < 60.0:
            action = check.snapshot(at)["action"]
            if action != asked:
                asked, asked_at = action, at
            frame = stand_in.frame((does or {}).get(action, action), at - asked_at)
            track = tracker.process(frame)
            check.frame(read(track), at)
            if track.ok and check.wants_face(at):
                check.face(scorer.embed(frame, track.landmarks), at)
            at += 1.0 / 30.0
    return check, at


def answer_the_prompts(clips, scorer):
    from neuropresence.reenactment import ReenactmentEngine

    engine = ReenactmentEngine(tensorrt=True, compile_networks=True)
    rng = random.Random(4)
    report, stand_ins = {}, {}
    for name in clips:
        try:
            stand_ins[name] = StandIn(engine, DRIVING / f"{name}.mp4")
        except ValueError as err:
            print(f"{name}: not used as a stand-in ({err})")
    other_way = {"turn_left": "turn_right", "turn_right": "turn_left"}
    for name, stand_in in stand_ins.items():
        own = [clips[name]["reference"]]
        stranger = [clips[other]["reference"] for other in clips if other != name][:1]
        ends = {"turns_read_deg": [round(stand_in.reach[0], 1), round(stand_in.reach[1], 1)],
                "answers": 0, "passed": 0, "seconds": [], "lowest_match": None,
                "turning_the_other_way": {"answers": 0, "passed": 0},
                "against_another_persons_signature": {"answers": 0, "passed": 0}}
        for actions in WITH_A_TURN:
            waits = [rng.uniform(*liveness.WAIT_SECONDS) for _ in actions]
            check, took = answering(stand_in, scorer, own, actions, waits)
            ends["answers"] += 1
            ends["passed"] += bool(check.passed)
            if check.passed:
                ends["seconds"].append(took)
            else:
                ends.setdefault("refused_for", []).append(f"{'+'.join(actions)}: {check.failure}")
            if check.lowest is not None:
                ends["lowest_match"] = round(min(check.lowest, ends["lowest_match"] or 1.0), 3)
            wrong, _ = answering(stand_in, scorer, own, actions, waits, does=other_way)
            ends["turning_the_other_way"]["answers"] += 1
            ends["turning_the_other_way"]["passed"] += bool(wrong.passed)
            other, _ = answering(stand_in, scorer, stranger, actions, waits)
            ends["against_another_persons_signature"]["answers"] += 1
            ends["against_another_persons_signature"]["passed"] += bool(other.passed)
        took = ends.pop("seconds")
        ends["seconds_to_pass"] = {"mean": round(float(np.mean(took)), 1), "longest": round(max(took), 1)} if took else None
        report[name] = ends
        print(f"{name}: drawn turns read {ends['turns_read_deg'][0]:+.0f} to {ends['turns_read_deg'][1]:+.0f} deg | "
              f"answering: {ends['passed']} of {ends['answers']} pass, in {ends['seconds_to_pass'] and ends['seconds_to_pass']['mean']} s on average, "
              f"lowest match {ends['lowest_match']} | turning the other way: {ends['turning_the_other_way']['passed']} pass | "
              f"against another person's signature: {ends['against_another_persons_signature']['passed']} pass"
              + (f" | refused: {ends['refused_for']}" if ends.get("refused_for") else ""), flush=True)
    return report


# ------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stand-in", action="store_true", help="also run a stand-in that answers the prompts (needs the GPU)")
    args = parser.parse_args()

    scorer = IdentityScorer()
    clips = {}
    for path in sorted(DRIVING.glob("*.mp4"), key=lambda p: int(p.stem[1:])):
        clip = record(path, scorer)
        with FaceTracker(video=False) as still:
            track = still.process(clip["photo"])
            clip["reference"] = scorer.embed(clip["photo"], track.landmarks) if track.ok else None
        clips[path.stem] = clip
        print(f"read {path.stem}: {len(clip['readings'])} frames", flush=True)

    report = {
        "limits": {name.lower(): getattr(liveness, name) for name in (
            "ACTION_SECONDS", "SETTLE_SECONDS", "REST_SECONDS", "WAIT_SECONDS", "LOST_SECONDS", "REST_YAW_DEG", "REST_JAW",
            "TURN_DEG", "OPEN_JAW", "OPEN_HOLD_SECONDS", "EYES_CLOSED", "CLOSED_RISE", "REOPEN_RISE",
            "BLINK_MAX_SECONDS")},
        "orders_with_a_turn": len(WITH_A_TURN),
        "clips": {}, "photographs": {},
    }

    print("\n1. What the clips contain")
    print("clip    s   at rest | yaw min   max | largest turn from rest in 3 s | eyes closed: times, longest | mouth wide open: times, held 0.3 s, longest")
    for name, clip in clips.items():
        d = describe(clip)
        report["clips"][name] = {"contains": d}
        print(f"{name:4s} {d['seconds']:5.1f}  {d['share_at_rest']:5.0%}  | {d['yaw_deg'][0]:6.1f} {d['yaw_deg'][1]:6.1f} | "
              f"{d['largest_turn_from_rest_deg']:6.1f} | {d['eyes_closed']['times']:3d}  {d['eyes_closed']['longest_s']:.2f} s | "
              f"{d['mouth_wide_open']['times']:3d} {d['mouth_wide_open']['times_held']:3d}  {d['mouth_wide_open']['longest_s']:.2f} s")
    described = [entry["contains"] for entry in report["clips"].values()]
    report["clips_summary"] = {
        "largest_turn_from_rest_deg": max(d["largest_turn_from_rest_deg"] for d in described),
        "longest_eyes_closed_s": max(d["eyes_closed"]["longest_s"] for d in described),
        "longest_mouth_wide_open_s": max(d["mouth_wide_open"]["longest_s"] for d in described),
    }

    print("\n2. A recording played to the camera: attempts that pass")
    print("clip   with a turn asked for (as built)      if no turn were asked for (blink and open mouth only)")
    for name, clip in clips.items():
        built = replay(clip["readings"], clip["fps"], WITH_A_TURN)
        loose = replay(clip["readings"], clip["fps"], WITHOUT_A_TURN)
        report["clips"][name]["replayed"] = {"as_built": built, "without_a_turn": loose}
        print(f"{name:4s}   {built['passed']:4d} of {built['attempts']:5d} ({built['passed_share']:.1%})"
              f"                 {loose['passed']:4d} of {loose['attempts']:5d} ({loose['passed_share']:.1%})")
    report["replayed"] = {"as_built": total(entry["replayed"]["as_built"] for entry in report["clips"].values()),
                          "without_a_turn": total(entry["replayed"]["without_a_turn"] for entry in report["clips"].values())}
    for key, label in (("as_built", "with a turn asked for"), ("without_a_turn", "if no turn were asked for")):
        r = report["replayed"][key]
        print(f"all clips, {label}: {r['passed']} of {r['attempts']} ({r['passed_share']:.2%})")
        print("   " + ", ".join(f"{k}: {v}" for k, v in sorted(r.items()) if k.startswith("failed")))

    print("\n   A recording made on purpose, with every action in it (written out as readings, not filmed)")
    report["made_on_purpose"] = {}
    for rest_seconds in (1.0, 2.0, 3.0):
        made = replay(made_on_purpose(rest_seconds), 30.0, WITH_A_TURN, seed=3, there_and_back=False)
        report["made_on_purpose"][f"rest_{rest_seconds:g}_s"] = made
        print(f"   with {rest_seconds:g} s of rest between the actions: {made['passed']} of {made['attempts']} attempts pass ({made['passed_share']:.1%})")

    print("\n3. A photograph held to the camera")
    print("photo  turn read at a tilt of " + " ".join(f"{a:>5d}" for a in TILTS_DEG) + " | swung: largest turn, passes | eyes plain, covered | mouth plain, dark shape")
    for name, clip in clips.items():
        p = photograph(name, clip["photo"])
        if p is None:
            continue
        report["photographs"][name] = p
        cells = " ".join("  ---" if p["turn_read_when_tilted_deg"][str(a)] is None else f"{p['turn_read_when_tilted_deg'][str(a)]:5.1f}"
                         for a in TILTS_DEG)
        print(f"{name:4s}                      {cells} | {p['swung']['largest_turn_read_deg']:5.1f}  {p['swung']['passed']} of {p['swung']['attempts']} | "
              f"{p['eyes']['plain']:.2f} {p['eyes']['covered'] if p['eyes']['covered'] is not None else '---'} | "
              f"{p['mouth']['plain']:.2f} {p['mouth']['dark_shape'] if p['mouth']['dark_shape'] is not None else '---'}", flush=True)
    photos = list(report["photographs"].values())
    report["photographs_summary"] = {
        "photographs": len(photos),
        "largest_turn_read_when_tilted_deg": max(v for p in photos for v in p["turn_read_when_tilted_deg"].values() if v is not None),
        "largest_turn_read_when_swung_deg": max(p["swung"]["largest_turn_read_deg"] for p in photos),
        "largest_turn_read_by_distance_deg": {str(distance): max(p["largest_turn_read_by_distance_deg"][str(distance)] for p in photos)
                                              for distance in DISTANCES},
        "swung": total({k: v for k, v in p["swung"].items() if k not in ("largest_turn_read_deg", "share_face_found")} for p in photos),
        "held_still": total(p["held_still"] for p in photos),
        "highest_eye_score_with_eyes_covered": max(p["eyes"]["covered"] for p in photos if p["eyes"]["covered"] is not None),
        "highest_mouth_score_with_dark_shape": max(p["mouth"]["dark_shape"] for p in photos if p["mouth"]["dark_shape"] is not None),
    }
    s = report["photographs_summary"]
    print("largest turn read, by how far the camera is from the photograph (in widths of the photograph): "
          + ", ".join(f"{distance}: {turn} deg" for distance, turn in s["largest_turn_read_by_distance_deg"].items()))
    print(f"all {s['photographs']} photographs: largest turn read {s['largest_turn_read_when_tilted_deg']} deg tilted, "
          f"{s['largest_turn_read_when_swung_deg']} deg swung; passes {s['swung']['passed']} of {s['swung']['attempts']} swung, "
          f"{s['held_still']['passed']} of {s['held_still']['attempts']} held still")

    print("\n4. Is the face recognised all the way through?")
    r = report["recognition"] = recognition(clips)
    for name, entry in r["same_person"].items():
        print(f"{name:4s} against its own first face: mean {entry['mean']:.2f}, lowest {entry['lowest']:.2f}, "
              f"below {r['limit']}: {entry['share_below_limit']:.1%} of {entry['faces']} faces, longest run {entry['longest_run_below_limit']}")
    print(f"the same person: lowest {r['same_person_lowest']} over {r['same_person_faces']} faces")
    print(f"another person: highest {r['another_person_highest']} over {r['another_person_pairs']} pairs of clips; "
          f"pairs ever at or above the limit: {r['another_person_pairs_ever_at_or_above_limit'] or 'none'}")

    if args.stand_in:
        print("\n5. A stand-in that answers the prompts")
        report["stand_in"] = answer_the_prompts(clips, scorer)

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
