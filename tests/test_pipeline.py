import numpy as np
import pytest

from neuropresence.capture import TrackResult, TrackStatus
from neuropresence.pipeline import Pipeline


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

    def drive(self, face):
        self.driven_shapes.append(face.shape)
        return np.full((720, 1280, 3), 200, dtype=np.uint8)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_one_face_is_reenacted():
    engine = FakeEngine()
    result = Pipeline(FakeTracker(TrackStatus.OK), engine).step(FRAME)
    assert result.live
    assert result.output.mean() == 200
    assert engine.driven_shapes == [(256, 256, 3)]
    assert set(result.timing_ms) == {"tracker", "motion", "render", "compose", "total"}


@pytest.mark.parametrize("status", [TrackStatus.NO_FACE, TrackStatus.MULTIPLE_FACES])
def test_unusable_frame_falls_back_to_enrolled_frame(status):
    engine = FakeEngine()
    result = Pipeline(FakeTracker(status), engine).step(FRAME)
    assert not result.live
    assert result.status is status
    assert result.output.mean() == 50
    assert engine.driven_shapes == []  # the GPU is not used for unusable frames
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
