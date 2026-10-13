"""The moment each frame's movement is of, and reading the movement between two camera frames."""

import numpy as np
import pytest

from neuropresence.capture import TrackResult, TrackStatus
from neuropresence.pipeline import LEAD_SECONDS, MAX_BETWEEN_SECONDS, Pipeline
from neuropresence.server import pacing
from neuropresence.server.pacing import MotionClock

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def run(clock, ages, every=0.045, start=10.0):
    """Feed the clock frames drawn `every` seconds apart whose newest camera frame is `ages` old in turn."""
    moments, now = [], start
    for age in ages:
        moments.append((now, now - age, clock.moment(now, now - age)))
        now += every
    return moments


def test_the_moment_settles_a_little_behind_the_oldest_the_newest_frame_gets():
    clock = MotionClock()
    ages = [0.014 + 0.033 * ((7 * step) % 10) / 10 for step in range(200)]  # from just in to a camera frame old
    run(clock, ages)
    assert 0.044 - LEAD_SECONDS <= clock.lag <= 0.052 - LEAD_SECONDS  # near the oldest, 47 ms, less what is read ahead
    behind = [now - moment for now, _, moment in run(clock, ages)]
    assert max(behind) - min(behind) < 0.004  # the same time back for every frame: that is what makes it even
    assert np.mean(behind) == pytest.approx(clock.lag, abs=0.002)


def test_the_moment_is_never_far_past_the_newest_frame():
    clock = MotionClock()
    moments = run(clock, [0.060] * 3)  # a frame older than the clock has learnt to expect
    # Only a little can be read past the newest frame, by carrying its movement on.
    assert all(moment == pytest.approx(captured + LEAD_SECONDS) for _, captured, moment in moments)
    assert pacing.START_SECONDS < clock.lag < 0.060  # and the clock is on its way there


def test_one_late_frame_does_not_move_the_clock_and_a_lasting_change_does():
    clock = MotionClock()
    after = lambda moments: moments[-1][0] + 0.045  # noqa: E731  (when the next frame is drawn)
    end = after(run(clock, [0.030] * 120))
    settled = clock.lag
    assert settled == pytest.approx(0.033 - LEAD_SECONDS, abs=0.001)
    end = after(run(clock, [0.110] + [0.030] * 30, start=end))  # one hiccup
    assert clock.lag == pytest.approx(settled, abs=0.001)
    end = after(run(clock, [0.060] * 120, start=end))  # the camera now delivers half as often
    assert clock.lag == pytest.approx(0.063 - LEAD_SECONDS, abs=0.002)
    run(clock, [0.020] * 200, start=end)  # and then faster than ever: the old ages are forgotten
    assert clock.lag == pytest.approx(max(pacing.LEAST_SECONDS, 0.023 - LEAD_SECONDS), abs=0.002)


def test_the_lag_keeps_within_its_bounds_and_starts_again():
    clock = MotionClock()
    run(clock, [0.5] * 200)
    assert clock.lag == pytest.approx(pacing.MOST_SECONDS, abs=0.002)
    run(clock, [0.0] * 400, start=50.0)
    assert clock.lag == pytest.approx(pacing.LEAST_SECONDS, abs=0.002)
    clock.reset()
    assert clock.lag == pacing.START_SECONDS


# ------------------------------------------------- in the pipeline


class Tracker:
    status = TrackStatus.OK

    def process(self, frame):
        if self.status is not TrackStatus.OK:
            return TrackResult(self.status, 1.0)
        return TrackResult(self.status, 1.0, bbox=(100, 100, 80, 80), pose_deg=(0.0, 0.0, 0.0))  # a face at rest


class Engine:
    """Reads a number from each face, one more each time, and draws nothing: what was asked of it is kept."""

    def __init__(self):
        self.source_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
        self.last_timing_ms = {"motion": 2.0, "render": 3.0, "compose": 1.0}
        self.reads = 0
        self.joined = []  # (before, after, share) of every reading taken between two
        self.drawn = []  # (at, motion) of every frame drawn

    def reset_reference(self):
        pass

    def set_reference(self, face):
        pass

    def read(self, face):
        self.reads += 1
        return float(self.reads)

    def between(self, before, after, share):
        self.joined.append((before, after, share))
        return before + share * (after - before)

    def drive(self, face, at=None, steady=False, limit=False, motion=None):
        self.drawn.append((at, motion))
        return np.full((720, 1280, 3), 200, dtype=np.uint8)


def test_the_movement_is_read_between_this_frame_and_the_one_before():
    engine = Engine()
    pipeline = Pipeline(Tracker(), engine)
    first = pipeline.step(FRAME, at=1.000, motion_at=0.980)
    assert first.motion_at == 1.000 and engine.drawn[-1] == (1.000, 1.0)  # nothing before it: the frame as it is
    second = pipeline.step(FRAME, at=1.066, motion_at=1.033)  # halfway between the two frames
    assert engine.joined == [(1.0, 2.0, pytest.approx(0.5))]
    assert second.motion_at == pytest.approx(1.033) and engine.drawn[-1] == (pytest.approx(1.033), pytest.approx(1.5))
    third = pipeline.step(FRAME, at=1.100, motion_at=1.0745)  # a quarter of the way from the second to the third
    assert engine.joined[-1] == (2.0, 3.0, pytest.approx(0.25))
    assert third.motion_at == pytest.approx(1.0745) and third.live
    assert engine.reads == 3  # one reading a frame, as without it


def test_a_moment_outside_the_two_frames_is_brought_to_the_nearer_one():
    engine = Engine()
    pipeline = Pipeline(Tracker(), engine)
    pipeline.step(FRAME, at=1.000, motion_at=1.000)
    late = pipeline.step(FRAME, at=1.033, motion_at=1.090)  # far later than the newest frame: read a little ahead, no more
    assert late.motion_at == pytest.approx(1.033 + LEAD_SECONDS)
    assert engine.drawn[-1][1] == pytest.approx(2.0 + LEAD_SECONDS / 0.033)  # carried on the way it was going
    early = pipeline.step(FRAME, at=1.066, motion_at=1.000)  # earlier than the frame before
    assert early.motion_at == pytest.approx(1.033) and engine.drawn[-1][1] == pytest.approx(2.0)


def test_readings_are_not_joined_across_a_lost_face_or_a_long_gap():
    engine = Engine()
    tracker = Tracker()
    pipeline = Pipeline(tracker, engine)
    pipeline.step(FRAME, at=1.000, motion_at=0.990)
    tracker.status = TrackStatus.NO_FACE
    pipeline.step(FRAME, at=1.033, motion_at=1.020)
    tracker.status = TrackStatus.OK
    back = pipeline.step(FRAME, at=1.066, motion_at=1.050)
    assert engine.joined == [] and back.motion_at == 1.066  # the face that came back is drawn as it is
    pipeline.step(FRAME, at=1.066 + MAX_BETWEEN_SECONDS + 0.01, motion_at=1.2)
    assert engine.joined == []  # too long ago to read between
    pipeline.step(FRAME, at=1.40, motion_at=1.38, reenact=False)
    pipeline.step(FRAME, at=1.44, motion_at=1.42)
    assert engine.joined == []  # and not across a stretch in which nothing was drawn
    pipeline.wait_for_neutral(seconds=0.0)
    pipeline.step(FRAME, at=1.48, motion_at=1.46)
    assert engine.joined == []  # nor across a fresh neutral pose


def test_without_a_moment_asked_for_the_frame_is_drawn_as_it_is():
    engine = Engine()
    pipeline = Pipeline(Tracker(), engine)
    result = pipeline.step(FRAME, at=1.0)
    assert engine.reads == 0 and result.motion_at is None
    assert engine.drawn == [(1.0, None)]  # the engine reads the movement itself, as before
    pipeline.step(FRAME, at=1.033, motion_at=1.02)
    pipeline.step(FRAME, at=1.066)  # switched off again
    pipeline.step(FRAME, at=1.100, motion_at=1.09)
    assert engine.joined == []  # the reading from before the switch is not used


def test_reading_the_movement_counts_toward_the_motion_stage():
    engine = Engine()
    pipeline = Pipeline(Tracker(), engine)
    plain = pipeline.step(FRAME, at=1.0).timing_ms["motion"]
    resampled = pipeline.step(FRAME, at=1.033, motion_at=1.02).timing_ms["motion"]
    assert plain == pytest.approx(2.0, abs=0.5) and resampled >= plain
