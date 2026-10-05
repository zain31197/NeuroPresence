import math

import numpy as np
import pytest

from neuropresence.capture import FaceTracker, FrameSource, FrameSourceError, TrackStatus
from neuropresence.capture.tracker import DEFAULT_MODEL, classify_face_count, rotation_to_euler_deg


def rotation_from_euler_deg(yaw, pitch, roll):
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


@pytest.mark.parametrize("angles", [(0, 0, 0), (30, 0, 0), (0, -20, 0), (0, 0, 15), (25, -10, 5)])
def test_euler_roundtrip(angles):
    recovered = rotation_to_euler_deg(rotation_from_euler_deg(*angles))
    assert recovered == pytest.approx(angles, abs=1e-6)


def test_euler_ignores_scale():
    scaled = 7.5 * rotation_from_euler_deg(25, -10, 5)
    assert rotation_to_euler_deg(scaled) == pytest.approx((25, -10, 5), abs=1e-6)


def test_face_count_policy():
    assert classify_face_count(0) is TrackStatus.NO_FACE
    assert classify_face_count(1) is TrackStatus.OK
    assert classify_face_count(2) is TrackStatus.MULTIPLE_FACES


def test_missing_video_file_raises():
    with pytest.raises(FrameSourceError):
        FrameSource("does_not_exist.mp4")


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaceTracker(tmp_path / "missing.task")


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="run scripts/download_models.py first")
def test_blank_frames_report_no_face():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with FaceTracker() as tracker:
        results = [tracker.process(blank) for _ in range(3)]
    for result in results:
        assert result.status is TrackStatus.NO_FACE
        assert not result.ok
        assert result.landmarks is None
        assert result.latency_ms > 0
