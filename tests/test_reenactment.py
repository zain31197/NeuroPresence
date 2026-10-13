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
def test_the_movement_can_be_read_between_two_frames(engine):
    engine.set_source(cv2.imread(str(SAMPLE_SOURCE)))
    rest = engine.source_crop
    turned = cv2.warpAffine(rest, cv2.getRotationMatrix2D((256, 256), 12, 1.0), (512, 512), borderMode=cv2.BORDER_REPLICATE)
    engine.set_reference(rest)
    first, second = engine.read(rest), engine.read(turned)
    assert not first[1].is_cuda and all(not value.is_cuda for value in first[0].values())  # the arithmetic is the CPU's
    roll = lambda reading: float(reading[0]["roll"])  # noqa: E731
    assert abs(roll(second) - roll(first)) > 5  # the two faces differ by a tilt of the head
    half = engine.between(first, second, 0.5)
    assert roll(half) == pytest.approx((roll(first) + roll(second)) / 2, abs=1e-3)
    assert torch.allclose(half[0]["exp"], (first[0]["exp"] + second[0]["exp"]) / 2, atol=1e-5)
    assert engine.between(first, second, 0.0) is first and engine.between(first, second, 1.0) is second
    start = engine.drive(rest, motion=first).astype(np.float32)
    middle = engine.drive(rest, motion=half).astype(np.float32)
    end = engine.drive(rest, motion=second).astype(np.float32)
    # The frame drawn from it lies between the two: nearer to each than they are to each other.
    assert np.abs(middle - start).mean() < np.abs(end - start).mean()
    assert np.abs(middle - end).mean() < np.abs(end - start).mean()
    assert np.abs(engine.drive(rest).astype(np.float32) - start).mean() < 0.5  # handing a reading in changes nothing


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


def test_calibrate_pose_range_fits_the_free_to_most_ratio_to_what_was_measured():
    from neuropresence.reenactment.engine import POSE_RANGE_DEG, ReenactmentEngine

    engine = ReenactmentEngine.__new__(ReenactmentEngine)  # only the calibration arithmetic
    engine.pose_range_deg = dict(POSE_RANGE_DEG)
    default_ratio_yaw = POSE_RANGE_DEG["yaw"][0] / POSE_RANGE_DEG["yaw"][1]
    default_ratio_pitch = POSE_RANGE_DEG["pitch"][0] / POSE_RANGE_DEG["pitch"][1]

    # A person who registered a wider turn than the default gets a wider range, same ratio.
    engine.calibrate_pose_range(yaw_extreme_deg=60.0)
    free, most = engine.pose_range_deg["yaw"]
    assert most == 60.0
    assert free == pytest.approx(60.0 * default_ratio_yaw)
    assert engine.pose_range_deg["pitch"] == POSE_RANGE_DEG["pitch"]  # untouched: not given this time

    # up and down each register their own sign; the more extreme of the two sets the limit.
    engine.calibrate_pose_range(pitch_up_extreme_deg=-18.0, pitch_down_extreme_deg=30.0)
    free, most = engine.pose_range_deg["pitch"]
    assert most == 30.0
    assert free == pytest.approx(30.0 * default_ratio_pitch)
    assert engine.pose_range_deg["yaw"] == (pytest.approx(60.0 * default_ratio_yaw), 60.0)  # the earlier call is kept


def test_calibrate_pose_range_never_sets_most_below_the_generic_free_range():
    from neuropresence.reenactment.engine import POSE_RANGE_DEG, ReenactmentEngine

    engine = ReenactmentEngine.__new__(ReenactmentEngine)
    engine.pose_range_deg = dict(POSE_RANGE_DEG)
    # A person who (for whatever reason) registered a smaller angle than the generic free range
    # does not end up clamped tighter than everyone else's default.
    engine.calibrate_pose_range(yaw_extreme_deg=3.0)
    assert engine.pose_range_deg["yaw"][1] == POSE_RANGE_DEG["yaw"][0]


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
    engine.pose_range_deg = dict(POSE_RANGE_DEG)
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


def test_at_rest_the_neutral_pose_follows_the_person_and_a_passing_look_does_not_move_it():
    import torch

    from neuropresence.reenactment.engine import REST_EXPRESSION_SECONDS, REST_HOLD_SECONDS, REST_SECONDS, ReenactmentEngine

    def face(brows=0.0, yaw=0.0):
        return {"pitch": torch.tensor([[0.0]]), "yaw": torch.tensor([[yaw]]), "roll": torch.tensor([[0.0]]),
                "t": torch.zeros(1, 3), "scale": torch.ones(1, 1), "exp": torch.full((1, 21, 3), brows)}

    engine = ReenactmentEngine.__new__(ReenactmentEngine)  # only the logic: no model is loaded
    engine._rotation = lambda pitch, yaw, roll: (float(pitch), float(yaw), float(roll))
    # The neutral pose was taken on a poor frame: eyes half shut, head a little to one side.
    engine._reference = {"info": face(brows=-0.02, yaw=6.0), "rotation": (0.0, 6.0, 0.0)}
    engine._rest_since = engine._rest_at = None
    relaxed = face()

    def off():
        return float((relaxed["exp"] - engine._reference["info"]["exp"]).abs().max())

    # Talking for ten seconds: not at rest, so nothing follows and the picture keeps what it was given.
    engine.resting = False
    for step in range(100):
        engine._follow_rest(relaxed, at=step * 0.1)
    assert off() == pytest.approx(0.02)

    # At rest: nothing for the first second, then the neutral pose eases toward how the person looks now.
    engine.resting = True
    hold = int(REST_HOLD_SECONDS * 10)
    for step in range(hold):
        engine._follow_rest(relaxed, at=10.0 + step * 0.1)
    assert off() == pytest.approx(0.02)
    # A look held for five seconds still shows: most of it is left. The head has come further back than the look.
    for step in range(hold, hold + 50):
        engine._follow_rest(relaxed, at=10.0 + step * 0.1)
    assert off() > 0.02 * 0.8
    assert REST_EXPRESSION_SECONDS > REST_SECONDS and abs(float(engine._reference["info"]["yaw"])) < 6.0 * 0.72
    # Left at rest for two minutes and a half, the look and the head are back.
    for step in range(hold + 50, hold + 1500):
        engine._follow_rest(relaxed, at=10.0 + step * 0.1)
    assert off() < 0.0005 and abs(float(engine._reference["info"]["yaw"])) < 0.05
    assert abs(engine._reference["rotation"][1]) < 0.05

    # A blink or a word breaks the rest, and the wait starts again.
    engine.resting = False
    engine._follow_rest(face(brows=0.03), at=170.1)
    engine.resting = True
    for step in range(5):
        engine._follow_rest(face(brows=0.03), at=170.2 + step * 0.1)
    assert off() < 0.0005
