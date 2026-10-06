import cv2
import numpy as np
import pytest
import torch

from neuropresence.evaluation import landmark_jitter, summarise, summarise_scores
from neuropresence.evaluation.temporal import backward_warp, masked_mse, visibility_mask

# ---------------------------------------------------------------- warping maths


def ramp_image(h=16, w=24):
    """One-channel image whose value is its own x position, so shifts are easy to check."""
    return torch.arange(w, dtype=torch.float32).repeat(h, 1)[None, None]


def constant_flow(dx, dy, h=16, w=24):
    flow = torch.zeros(1, 2, h, w)
    flow[:, 0], flow[:, 1] = dx, dy
    return flow


def test_backward_warp_with_zero_flow_changes_nothing():
    image = torch.rand(1, 3, 16, 24)
    assert torch.allclose(backward_warp(image, constant_flow(0, 0)), image, atol=1e-5)


def test_backward_warp_samples_at_position_plus_flow():
    warped = backward_warp(ramp_image(), constant_flow(3, 0))
    # Each pixel now shows the value from 3 pixels to its right.
    assert torch.allclose(warped[0, 0, :, :21], ramp_image()[0, 0, :, :21] + 3, atol=1e-4)
    assert warped[0, 0, 0, 23] == 0  # sampled outside the image


def test_visibility_mask_accepts_flows_that_cancel():
    mask = visibility_mask(constant_flow(2, 1), constant_flow(-2, -1))
    assert mask[0, 2:12, 2:20].all()


def test_visibility_mask_rejects_flows_that_disagree():
    mask = visibility_mask(constant_flow(2, 0), constant_flow(4, 0))
    assert not mask.any()


def test_visibility_mask_rejects_pixels_that_came_from_outside():
    mask = visibility_mask(constant_flow(5, 0), constant_flow(-5, 0))
    assert not mask[0, :, -4:].any()  # their source lies beyond the right edge
    assert mask[0, 2:12, 2:15].all()


def test_masked_mse_counts_only_masked_pixels():
    a, b = torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)
    b[0, :, 0, 0] = 1.0  # one differing pixel
    everything = torch.ones(1, 4, 4, dtype=torch.bool)
    without_it = everything.clone()
    without_it[0, 0, 0] = False
    assert masked_mse(a, b, everything).item() == pytest.approx(1 / 16)
    assert masked_mse(a, b, without_it).item() == 0.0


# ------------------------------------------------------------- landmark jitter


def moving_track(frames=60, points=6, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform(100, 400, (1, points, 2))
    return start + np.arange(frames)[:, None, None] * np.array([1.2, -0.4])


def test_jitter_is_zero_for_still_and_steadily_moving_points():
    track = moving_track()
    assert landmark_jitter(np.tile(track[:1], (60, 1, 1)), 100.0) == pytest.approx(0.0, abs=1e-12)
    assert landmark_jitter(track, 100.0) == pytest.approx(0.0, abs=1e-9)


def test_jitter_grows_in_proportion_to_the_noise():
    noise = np.random.default_rng(1).normal(0, 1.0, (60, 6, 2))
    small = landmark_jitter(moving_track() + 0.5 * noise, 100.0)
    large = landmark_jitter(moving_track() + 1.0 * noise, 100.0)
    assert small > 0.005
    assert large == pytest.approx(2 * small, rel=1e-6)


def test_jitter_is_relative_to_the_scale():
    track = moving_track() + np.random.default_rng(2).normal(0, 1.0, (60, 6, 2))
    assert landmark_jitter(track, 50.0) == pytest.approx(2 * landmark_jitter(track, 100.0))
    # A face twice as large with twice the jitter in pixels is equally jittery.
    assert landmark_jitter(2 * track, 200.0) == pytest.approx(landmark_jitter(track, 100.0))


def test_jitter_skips_frames_next_to_a_gap():
    track = moving_track()
    track[30] += 50.0  # one wild frame
    with_gap = track.copy()
    with_gap[30] = np.nan  # the same frame marked as "face not found"
    assert landmark_jitter(track, 100.0) > 0.01
    assert landmark_jitter(with_gap, 100.0) == pytest.approx(0.0, abs=1e-9)


def test_jitter_rejects_unusable_input():
    with pytest.raises(ValueError):
        landmark_jitter(np.zeros((2, 6, 2)), 100.0)  # too short
    with pytest.raises(ValueError):
        landmark_jitter(np.zeros((10, 6)), 100.0)  # wrong shape
    with pytest.raises(ValueError):
        landmark_jitter(np.full((10, 6, 2), np.nan), 100.0)  # face never found


# ------------------------------------------------------------------- summaries


def test_summaries():
    assert summarise([10, 20, 30]) == {"mean": 20.0, "p95": 29.0, "p99": 29.8}
    scores = summarise_scores([0.9, 0.7, 0.85, 0.95], threshold=0.8)
    assert scores["mean"] == 0.85
    assert scores["min"] == 0.7
    assert scores["share_at_or_above_threshold"] == 0.75
    assert scores["frames"] == 4


# ------------------------------------------- warping error with the flow network


@pytest.fixture(scope="module")
def warping():
    from neuropresence.evaluation import WarpingError

    try:
        return WarpingError()
    except Exception as err:  # no internet for the first download of the flow weights
        pytest.skip(f"optical-flow model not available: {err}")


@pytest.fixture(scope="module")
def texture():
    noise = np.random.default_rng(0).integers(0, 256, (320, 320, 3), dtype=np.uint8)
    return cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 3), None, 0, 255, cv2.NORM_MINMAX)


def view(texture, dx=0.0, dy=0.0, gain=1.0):
    """A 256x256 window onto the texture, shifted by (dx, dy) and dimmed by gain."""
    matrix = np.float32([[1, 0, -32 - dx], [0, 1, -32 - dy]])
    image = cv2.warpAffine(texture, matrix, (256, 256), flags=cv2.INTER_LINEAR)
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def test_warping_error_is_zero_for_a_still_clip(warping, texture):
    assert warping([view(texture)] * 6) < 1e-7


def test_warping_error_sees_flicker(warping, texture):
    flicker = [view(texture, gain=1.0 if i % 2 == 0 else 0.9) for i in range(8)]
    assert warping(flicker) > 1e-3


def test_warping_error_matches_known_sensor_noise(warping, texture):
    rng = np.random.default_rng(3)
    sd = 3.0
    noisy = [np.clip(view(texture).astype(np.float32) + rng.normal(0, sd, (256, 256, 3)), 0, 255).astype(np.uint8)
             for _ in range(8)]
    # Two frames with independent noise differ by noise of variance 2 * sd^2.
    assert warping(noisy) == pytest.approx(2 * (sd / 255) ** 2, rel=0.2)


def test_warping_error_is_blind_to_rigid_jitter(warping, texture):
    """Shaking the whole picture is real movement, so the flow explains it away.

    This is why landmark jitter is measured as well.
    """
    rng = np.random.default_rng(4)
    shaken = [view(texture, *rng.uniform(-2, 2, 2)) for _ in range(8)]
    flicker = [view(texture, gain=1.0 if i % 2 == 0 else 0.95) for i in range(8)]
    assert warping(shaken) < 1e-4
    assert warping(flicker) > 10 * warping(shaken)


def test_warping_error_rejects_bad_clips(warping, texture):
    with pytest.raises(ValueError):
        warping([view(texture)])  # a single frame
    with pytest.raises(ValueError):
        warping([view(texture)[:250, :250]] * 2)  # size not a multiple of 8
