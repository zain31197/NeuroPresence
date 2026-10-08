import numpy as np
import pytest

from neuropresence.capture import TrackResult, TrackStatus
from neuropresence import pipeline as pipeline_module
from neuropresence.pipeline import Pipeline, at_rest


class FakeTracker:
    def __init__(self, status):
        self.status = status

    def process(self, frame):
        if self.status is TrackStatus.OK:
            return TrackResult(self.status, 1.0, bbox=(100, 100, 80, 80))
        return TrackResult(self.status, 1.0)


class FakeEngine:
    def __init__(self):
        self.source_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
        self.last_timing_ms = {"motion": 2.0, "render": 3.0, "compose": 1.0}
        self.driven_shapes = []
        self.neutral_faces = []
        self.resets = 0

    def reset_reference(self):
        self.resets += 1

    def set_reference(self, face):
        self.neutral_faces.append(face.shape)

    def drive(self, face, at=None, steady=False, limit=False, motion=None):
        self.driven_shapes.append(face.shape)
        return np.full((720, 1280, 3), 200, dtype=np.uint8)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_one_face_is_reenacted():
    engine = FakeEngine()
    result = Pipeline(FakeTracker(TrackStatus.OK), engine).step(FRAME)
    assert result.live
    assert result.output.mean() == 200
    assert engine.driven_shapes == [(256, 256, 3)]
    assert result.driving_face.shape == (256, 256, 3)
    assert set(result.timing_ms) == {"tracker", "motion", "render", "compose", "total"}


@pytest.mark.parametrize("status", [TrackStatus.NO_FACE, TrackStatus.MULTIPLE_FACES])
def test_unusable_frame_falls_back_to_enrolled_frame(status):
    engine = FakeEngine()
    result = Pipeline(FakeTracker(status), engine).step(FRAME)
    assert not result.live
    assert result.status is status
    assert result.output.mean() == 50
    assert engine.driven_shapes == []  # the GPU is not used for unusable frames
    assert result.driving_face is None
    assert set(result.timing_ms) == {"tracker", "total"}


def test_fallback_frame_is_a_copy():
    engine = FakeEngine()
    result = Pipeline(FakeTracker(TrackStatus.NO_FACE), engine).step(FRAME)
    result.output[:] = 0
    assert engine.source_frame.mean() == 50


def test_pipeline_requires_a_source():
    engine = FakeEngine()
    engine.source_frame = None
    with pytest.raises(ValueError):
        Pipeline(FakeTracker(TrackStatus.OK), engine)


# ------------------------------------------------------------- neutral pose


def face(yaw=0.0, roll=0.0, jaw=0.02, blink=0.1):
    return TrackResult(TrackStatus.OK, 1.0, bbox=(100, 100, 80, 80), pose_deg=(yaw, -9.7, roll),
                       blendshapes={"jawOpen": jaw, "eyeBlinkLeft": blink, "eyeBlinkRight": blink})


class Scripted:
    """A tracker that reports the given faces, one per frame, then keeps reporting the last."""

    def __init__(self, *tracks):
        self.tracks = list(tracks)

    def process(self, frame):
        return self.tracks.pop(0) if len(self.tracks) > 1 else self.tracks[0]


def test_at_rest_means_facing_the_camera_with_mouth_closed_and_eyes_open():
    assert at_rest(face())
    assert at_rest(face(yaw=10.0))  # a camera to one side is still fine
    assert not at_rest(face(yaw=20.0)) and not at_rest(face(roll=15.0))
    assert not at_rest(face(jaw=0.4)) and not at_rest(face(blink=0.8))
    assert not at_rest(TrackResult(TrackStatus.NO_FACE, 1.0))
    # Pitch does not count: a camera below the face, as on a laptop, sees a raised chin all the time.
    assert at_rest(TrackResult(TrackStatus.OK, 1.0, bbox=(0, 0, 8, 8), pose_deg=(0.0, -25.0, 0.0), blendshapes={}))


def test_the_neutral_pose_is_taken_from_the_first_frame_at_rest():
    engine = FakeEngine()
    talking, turned, resting = face(jaw=0.5), face(yaw=25.0), face()
    pipeline = Pipeline(Scripted(talking, turned, resting, talking), engine)
    pipeline.wait_for_neutral()
    assert engine.resets == 1 and pipeline.waiting_for_neutral
    for _ in range(2):  # talking, then turned away: the still picture is shown and nothing is measured from them
        result = pipeline.step(FRAME)
        assert result.live is False and result.output.mean() == 50
    assert engine.neutral_faces == [] and engine.driven_shapes == []
    result = pipeline.step(FRAME)  # at rest: this frame is the neutral pose, and the output goes live with it
    assert result.live is True and engine.neutral_faces == [(256, 256, 3)]
    assert pipeline.neutral_taken == 1 and not pipeline.waiting_for_neutral
    assert pipeline.step(FRAME).live is True  # talking again is now movement, measured from that pose
    assert engine.neutral_faces == [(256, 256, 3)]  # taken once


def test_without_a_frame_at_rest_the_wait_ends_and_the_current_frame_is_taken(monkeypatch):
    engine = FakeEngine()
    pipeline = Pipeline(Scripted(face(jaw=0.5)), engine)  # this person never stops talking
    pipeline.wait_for_neutral(seconds=30.0)
    assert pipeline.step(FRAME).live is False
    monkeypatch.setattr(pipeline_module.time, "perf_counter", lambda: 1e12)  # long after the wait has run out
    assert pipeline.step(FRAME).live is True and engine.neutral_faces == [(256, 256, 3)]


def test_a_pipeline_that_was_not_asked_to_wait_drives_from_the_first_frame():
    engine = FakeEngine()
    assert Pipeline(Scripted(face(jaw=0.5)), engine).step(FRAME).live is True
    assert engine.neutral_faces == [] and engine.resets == 0  # the engine takes its own first frame, as before


# ------------------------------------------------------------ hold and fade


class Switchable:
    """A tracker whose face can be taken away and given back."""

    def __init__(self):
        self.track = face()

    def process(self, frame):
        return self.track


def test_a_lost_face_is_held_for_a_moment_then_fades_to_the_still_picture():
    engine, tracker = FakeEngine(), Switchable()
    pipeline = Pipeline(tracker, engine)
    assert pipeline.step(FRAME, at=0.0).output.mean() == 200  # live from the first frame, with no fade in
    tracker.track = TrackResult(TrackStatus.NO_FACE, 1.0)
    held = pipeline.step(FRAME, at=0.1)
    assert held.live is False and held.output.mean() == 200  # one missed frame does not show
    fading = pipeline.step(FRAME, at=0.3)  # past the hold: part of the way to the still picture
    assert 50 < fading.output.mean() < 200
    assert pipeline.step(FRAME, at=1.0).output.mean() == 50  # and then the still picture, exactly
    tracker.track = face()
    back = pipeline.step(FRAME, at=1.1)  # the face returns: the live output fades in, it does not snap
    assert back.live is True and 50 < back.output.mean() < 200
    assert pipeline.step(FRAME, at=2.0).output.mean() == 200


def test_a_frame_that_is_held_back_is_drawn_but_not_shown():
    """While the output is arriving too late the frames are still drawn, so the delay stays
    measured, but what is shown goes to the still picture as it does for a lost face."""
    engine = FakeEngine()
    pipeline = Pipeline(Switchable(), engine)
    assert pipeline.step(FRAME, at=0.0).output.mean() == 200
    held = pipeline.step(FRAME, at=0.1, show=False)
    assert held.live is False and held.output.mean() == 200  # the last shown frame, for a moment
    assert "render" in held.timing_ms and len(engine.driven_shapes) == 2  # it was drawn all the same
    assert pipeline.step(FRAME, at=1.0, show=False).output.mean() == 50  # then the still picture
    assert len(engine.driven_shapes) == 3
    back = pipeline.step(FRAME, at=1.1)
    assert back.live is True and 50 < back.output.mean() < 200  # and it fades back in


def test_the_crop_window_is_reported_and_holds_still_against_tracker_noise():
    engine, tracker = FakeEngine(), Switchable()
    pipeline = Pipeline(tracker, engine)
    rng = np.random.default_rng(0)
    still = np.column_stack([rng.uniform(200, 400, 478), rng.uniform(100, 340, 478)]).astype(np.float32)
    windows = {True: [], False: []}
    for steady in (True, False):
        pipeline.steady_crop = steady
        for index in range(40):  # a head that does not move, tracked with an error of about a pixel
            noisy = still + rng.normal(0, 1.0, still.shape).astype(np.float32)
            noisy[0], noisy[1] = (200 + rng.normal(0, 3), 100), (400 + rng.normal(0, 3), 340)  # the box edges wobble most
            tracker.track = TrackResult(TrackStatus.OK, 1.0, landmarks=noisy, pose_deg=(0.0, 0.0, 0.0), blendshapes={},
                                        bbox=(int(noisy[:, 0].min()), int(noisy[:, 1].min()),
                                              int(np.ptp(noisy[:, 0])), int(np.ptp(noisy[:, 1]))))
            result = pipeline.step(np.zeros((480, 640, 3), np.uint8), at=10.0 * steady + index / 30)
            if steady:
                windows[True].append(result.crop_window)
            else:
                x, y, w, h = tracker.track.bbox
                windows[False].append((x + w / 2 - max(w, h), y + h / 2 - max(w, h), 2.0 * max(w, h)))
                assert result.crop_window is None  # only the steady crop has a window of its own to report
    wobble = {key: float(np.std(np.array(value)[10:, 0])) for key, value in windows.items()}
    assert wobble[True] < 0.35 * wobble[False]  # the steady window moves far less than the tracked box does
    assert engine.driven_shapes and set(engine.driven_shapes) == {(256, 256, 3)}
