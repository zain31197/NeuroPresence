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


def test_crop_is_square_and_centred():
    from neuropresence.capture.crop import square_face_crop

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[200:280, 280:360] = 255  # an 80x80 white "face" at the frame centre
    crop = square_face_crop(frame, (280, 200, 80, 80), scale=2.0, out_size=256)
    assert crop.shape == (256, 256, 3)
    assert crop[128, 128].tolist() == [255, 255, 255]
    assert crop[5, 5].tolist() == [0, 0, 0]


def test_crop_repeats_edge_past_frame():
    from neuropresence.capture.crop import square_face_crop

    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    crop = square_face_crop(frame, (0, 0, 100, 100), scale=2.0, out_size=256)
    assert crop.shape == (256, 256, 3)
    assert crop[5, 5].tolist() == [200, 200, 200]  # outside the frame: edge repeated
    assert crop[250, 250].tolist() == [200, 200, 200]  # inside the frame


def test_crop_rejects_empty_box():
    from neuropresence.capture.crop import square_face_crop

    with pytest.raises(ValueError):
        square_face_crop(np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 0, 0))
