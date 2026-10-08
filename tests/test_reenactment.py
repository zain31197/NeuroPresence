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
    assert engine.source_frame.shape == source.shape

    driving = cv2.resize(engine.source_crop, (300, 300))  # any size is accepted
    out = engine.drive(driving)
    assert out.shape == source.shape  # the full source frame, face blended in
    assert out.dtype == np.uint8
    assert out.std() > 10  # a real image, not a flat frame
    assert set(engine.last_timing_ms) == {"motion", "render", "compose"}
    assert all(v > 0 for v in engine.last_timing_ms.values())


@needs_liveportrait
def test_reference_frame_reproduces_source(engine):
    """Driving with the reference frame itself means zero relative motion."""
    engine.set_source(cv2.imread(str(SAMPLE_SOURCE)))
    face = engine.source_crop
    first = engine.drive(face).astype(np.float32)
    again = engine.drive(face).astype(np.float32)
    assert np.abs(first - again).mean() < 1.0
    assert np.abs(first - engine.source_frame.astype(np.float32)).mean() < 8.0


@needs_liveportrait
def test_face_at_frame_edge_leaves_no_black_band(engine):
    """A head touching the top of the frame used to drag a black band into view."""
    source = cv2.imread(str(SAMPLE_SOURCE))[200:]  # cut so the forehead is at the top edge
    engine.set_source(source)

    def black_fraction(image):
        return float((image.max(axis=2) < 12).mean())

    assert black_fraction(engine.source_crop) <= black_fraction(source) + 0.01
    out = engine.drive(engine.source_crop)
    assert out.shape == source.shape
    assert black_fraction(out) <= black_fraction(source) + 0.01


@needs_liveportrait
def test_set_reference_fixes_the_neutral_pose(engine):
    """With the source's own face as the neutral pose, driving with it gives back the source."""
    engine.set_source(cv2.imread(str(SAMPLE_SOURCE)))
    neutral = engine.source_crop
    other = cv2.flip(neutral, 1)  # a different pose: the mirror image
    engine.set_reference(neutral)
    first = engine.drive(other).astype(np.float32)  # does not become the reference
    back = engine.drive(neutral).astype(np.float32)
    assert np.abs(back - engine.source_frame.astype(np.float32)).mean() < 8.0
    assert np.abs(first - back).mean() > 0.2  # the other pose really moved the face


def test_the_head_range_is_followed_exactly_inside_and_eases_to_a_stop_outside():
    import torch

    from neuropresence.reenactment.engine import POSE_RANGE_DEG, soft_limit

    free, most = POSE_RANGE_DEG["pitch"]
    angles = torch.tensor([0.0, 3.0, -free, free + 4.0, -36.0, 90.0, -200.0])
    limited = soft_limit(angles, free, most)
    assert torch.equal(limited[:3], angles[:3])  # small movements are not touched at all
    assert free < float(limited[3]) < free + 4.0  # past the free range it follows, but less and less
    assert -most < float(limited[4]) < -free  # the head thrown back 36 degrees, as on a real camera on 8 October 2026
    assert float(limited[5]) <= most and float(limited[6]) >= -most  # and never past the most that looks right
    steps = soft_limit(torch.linspace(-60, 60, 241), free, most)
    assert torch.all(steps[1:] >= steps[:-1])  # no jump anywhere: more movement in never gives less out
    assert float((steps[1:] - steps[:-1]).max()) <= 0.5 + 1e-4


def test_a_held_posture_becomes_the_rest_position_and_a_small_movement_does_not():
    import torch

    from neuropresence.reenactment.engine import POSE_RANGE_DEG, POSTURE_SETTLED, ReenactmentEngine

    def pose(pitch=0.0, yaw=0.0):
        return {"pitch": torch.tensor([[pitch]]), "yaw": torch.tensor([[yaw]]), "roll": torch.tensor([[0.0]]),
                "t": torch.zeros(1, 3), "scale": torch.ones(1, 1)}

    engine = ReenactmentEngine.__new__(ReenactmentEngine)  # only the posture logic: no model is loaded
    engine._rotation = lambda pitch, yaw, roll: (float(pitch), float(yaw), float(roll))
    engine._reference = {"info": pose(), "rotation": (0.0, 0.0, 0.0)}
    engine._posture_at, engine._settling = None, False
    free = POSE_RANGE_DEG["pitch"][0]

    def away(driving):
        return float(driving["pitch"] - engine._reference["info"]["pitch"])

    # Small movement, inside the free range, for ten seconds: the rest position does not move at all.
    for step in range(100):
        engine._follow_posture(pose(pitch=free * 0.8), at=step * 0.1)
    assert abs(away(pose(pitch=free * 0.8)) - free * 0.8) < 1e-4 and engine._settling is False

    # Leaning back 36 degrees and staying there, as on a real camera on 8 October 2026.
    leaning = pose(pitch=-36.0)
    engine._follow_posture(leaning, at=10.1)
    assert engine._settling is True and away(leaning) < -30  # at first the head is simply far from rest
    for step in range(1, 31):
        engine._follow_posture(leaning, at=10.1 + step * 0.1)
    assert -36 * 0.45 < away(leaning) < -36 * 0.30  # after three seconds about two thirds of it is taken up
    for step in range(31, 400):
        engine._follow_posture(leaning, at=10.1 + step * 0.1)
    assert abs(away(leaning)) <= free * POSTURE_SETTLED + 0.01 and engine._settling is False  # and then it rests there
    assert engine._reference["rotation"][0] < -30  # the stored rotation went with the angles

    # Sitting up again is, at first, a movement like any other, and then the new rest position in its turn.
    assert away(pose()) > 30
