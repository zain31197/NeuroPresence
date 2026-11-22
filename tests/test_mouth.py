"""Does the picture's mouth follow the person's, and what is done when it does not."""

import numpy as np
import pytest

from neuropresence.pipeline import gap_of, lip_gap
from neuropresence.server.mouth import CLOSED, OPEN, REST_SECONDS, MouthWatch


def run(watch, seconds, camera, output, start=0.0, fps=20.0):
    """A stretch of a session: `camera(t)` the person's lips each frame, `output(t)` the output's twice a second."""
    actions, next_output = [], start
    for step in range(int(seconds * fps)):
        t = start + step / fps
        watch.camera(t, camera(t))
        if t >= next_output:
            next_output += 0.5
            what = watch.output(t, output(t))
            if what:
                actions.append((round(t, 2), what))
    return actions


def talking(t):
    """Lips opening and closing a few times a second, as in speech."""
    return 0.055 + 0.05 * np.sin(2 * np.pi * 3.1 * t)


def test_a_mouth_that_stands_open_while_the_person_is_silent_is_found():
    watch = MouthWatch()
    assert run(watch, 4, lambda t: 0.005, lambda t: 0.006) == []  # both closed: they agree
    actions = run(watch, 6, lambda t: 0.005, lambda t: 0.09, start=4.0)
    assert actions and actions[0][1] == "open" and actions[0][0] < 4.0 + 2.5  # within about two seconds


def test_a_mouth_that_stays_shut_while_the_person_speaks_is_found():
    watch = MouthWatch()
    assert run(watch, 5, talking, lambda t: 0.9 * talking(t)) == []  # it follows: nothing
    actions = run(watch, 8, talking, lambda t: 0.004, start=5.0)
    assert actions and actions[0][1] == "shut"


def test_speech_and_silence_that_are_followed_never_trip_it():
    watch = MouthWatch()
    for block in range(8):
        speaking = block % 2 == 0
        lips = talking if speaking else (lambda t: 0.004)
        assert run(watch, 6, lips, lambda t, lips=lips: 0.95 * lips(t) + 0.002, start=block * 6.0) == []


def test_a_short_difference_is_not_a_stuck_mouth():
    watch = MouthWatch()
    # half a second open by mistake, then right again: the words of a late mouth are not a stuck one
    actions = run(watch, 8, lambda t: 0.005, lambda t: 0.09 if 3.0 <= t < 3.5 else 0.005)
    assert actions == []


def test_it_is_not_repaired_again_before_a_while_has_passed():
    watch = MouthWatch()
    actions = run(watch, 3 * REST_SECONDS, lambda t: 0.005, lambda t: 0.09)
    times = [t for t, _ in actions]
    assert len(times) >= 2 and all(b - a >= REST_SECONDS - 0.6 for a, b in zip(times, times[1:]))


def test_missing_readings_are_left_out():
    watch = MouthWatch()
    assert watch.output(0.0, None) is None and watch.camera(0.0, None) is None
    assert run(watch, 4, lambda t: None if t % 1 < 0.5 else 0.004, lambda t: None) == []


def test_the_thresholds_leave_room_between_closed_and_open():
    assert 2 * CLOSED < OPEN  # a mouth at rest is never near the open line


def test_the_gap_is_read_from_landmarks_alone():
    points = np.zeros((478, 2))
    points[33], points[263] = (0.0, 0.0), (100.0, 0.0)  # the outer corners of the eyes, 100 apart
    points[13], points[14] = (50.0, 60.0), (50.0, 66.0)  # the lips, 6 apart
    assert gap_of(points) == pytest.approx(0.06)
    assert gap_of(None) is None and gap_of(points[:50]) is None

    class Track:
        ok, landmarks = True, points

    assert lip_gap(Track()) == pytest.approx(0.06)
    Track.ok = False
    assert lip_gap(Track()) is None


# ------------------------------------------------------------------ the engine's repair


def make_engine(mouth):
    import torch

    from neuropresence.reenactment.engine import LIP_KEYPOINTS, ReenactmentEngine

    def face(value):
        expression = torch.full((1, 21, 3), 0.01)
        expression[:, LIP_KEYPOINTS] = value
        return {"exp": expression}

    engine = ReenactmentEngine.__new__(ReenactmentEngine)  # only the logic: no model is loaded
    engine._reference = {"info": face(mouth), "rotation": "kept"}
    engine.anchor_lips, engine.lips_closed, engine._lips_at = True, True, None
    return engine, face, LIP_KEYPOINTS


def mouth_of(engine, keypoints):
    return float(engine._reference["info"]["exp"][:, keypoints].mean())


def test_a_mouth_standing_open_is_eased_to_how_the_closed_one_reads_now():
    engine, face, keypoints = make_engine(mouth=-0.04)  # the neutral mouth reads the closed lips as far too tight
    engine.lips_closed = True
    engine._follow_lips(face(0.0), at=0.0)
    engine.repair_mouth("open")
    for step in range(1, 14):
        engine._follow_lips(face(0.0), at=step * 0.05)
    assert mouth_of(engine, keypoints) == pytest.approx(0.0, abs=0.004)  # put right in well under a second
    others = [k for k in range(21) if k not in keypoints]
    assert float(engine._reference["info"]["exp"][:, others].mean()) == pytest.approx(0.01)  # only the mouth
    assert engine._reference["rotation"] == "kept"


def test_the_repair_is_eased_not_snapped():
    engine, face, keypoints = make_engine(mouth=-0.04)
    engine._follow_lips(face(0.0), at=0.0)
    engine.repair_mouth("open")
    engine._follow_lips(face(0.0), at=0.05)
    assert -0.04 < mouth_of(engine, keypoints) < -0.005  # on its way after one frame, not there yet


def test_a_mouth_staying_shut_goes_to_how_it_last_read_closed():
    from neuropresence.reenactment.engine import LIP_HOLD_SECONDS

    engine, face, keypoints = make_engine(mouth=0.0)
    for step in range(int((LIP_HOLD_SECONDS + 1.0) / 0.05)):
        engine._follow_lips(face(0.002), at=step * 0.05)  # the lips held together: how his closed mouth reads is kept
    engine._reference["info"]["exp"][:, keypoints] = 0.08  # the neutral mouth then drifted to one that was open
    engine.lips_closed = False
    engine.repair_mouth("shut")
    for step in range(1, 16):
        engine._follow_lips(face(0.07), at=5.0 + step * 0.05)  # speaking now
    assert mouth_of(engine, keypoints) == pytest.approx(0.002, abs=0.004)


def test_nothing_is_repaired_without_a_picture_that_was_enrolled_mouth_closed():
    engine, face, keypoints = make_engine(mouth=-0.04)
    engine.anchor_lips = False
    engine.repair_mouth("open")
    engine._follow_lips(face(0.0), at=0.0)
    engine._follow_lips(face(0.0), at=0.05)
    assert mouth_of(engine, keypoints) == pytest.approx(-0.04)
