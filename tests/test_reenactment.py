import cv2
import numpy as np
import pytest
import torch

from neuropresence.reenactment.engine import LIVEPORTRAIT_DIR, OUTPUT_SIZE

SAMPLE_SOURCE = LIVEPORTRAIT_DIR / "assets" / "examples" / "source" / "s9.jpg"
WEIGHTS = LIVEPORTRAIT_DIR / "pretrained_weights" / "liveportrait" / "base_models" / "warping_module.pth"

needs_liveportrait = pytest.mark.skipif(
    not (WEIGHTS.exists() and torch.cuda.is_available()),
    reason="needs a GPU and scripts/setup_liveportrait.py",
)


def test_missing_liveportrait_gives_setup_hint(tmp_path):
    from neuropresence.reenactment import ReenactmentEngine

    with pytest.raises(FileNotFoundError, match="setup_liveportrait"):
        ReenactmentEngine(liveportrait_dir=tmp_path)


@pytest.fixture(scope="module")
def engine():
    from neuropresence.reenactment import ReenactmentEngine

    return ReenactmentEngine()


@needs_liveportrait
def test_drive_before_source_is_rejected(engine):
    with pytest.raises(RuntimeError):
        engine.drive(np.zeros((256, 256, 3), dtype=np.uint8))


@needs_liveportrait
def test_source_without_face_is_rejected(engine):
    from neuropresence.reenactment import SourceError

    with pytest.raises(SourceError):
        engine.set_source(np.zeros((480, 640, 3), dtype=np.uint8))


@needs_liveportrait
def test_drive_returns_frame_and_timings(engine):
    source = cv2.imread(str(SAMPLE_SOURCE))
    engine.set_source(source)
    assert engine.source_crop.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)

    driving = cv2.resize(engine.source_crop, (300, 300))  # any size is accepted
    out = engine.drive(driving)
    assert out.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)
    assert out.dtype == np.uint8
    assert out.std() > 10  # a real image, not a flat frame
    assert set(engine.last_timing_ms) == {"motion", "render"}
    assert all(v > 0 for v in engine.last_timing_ms.values())


@needs_liveportrait
def test_reference_frame_reproduces_source(engine):
    """Driving with the reference frame itself means zero relative motion."""
    engine.set_source(cv2.imread(str(SAMPLE_SOURCE)))
    face = engine.source_crop
    first = engine.drive(face).astype(np.float32)
    again = engine.drive(face).astype(np.float32)
    assert np.abs(first - again).mean() < 1.0
    assert np.abs(first - engine.source_crop.astype(np.float32)).mean() < 12.0
