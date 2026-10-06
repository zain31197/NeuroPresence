import numpy as np
import pytest

from neuropresence.evaluation.motion import change_from_first, correlation, mean_abs_error, motion_lag


def head_motion(frames=150, signals=3, seed=0):
    """Smooth made-up motion: a few slow sine waves per signal, like a moving head."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None]
    periods = rng.uniform(20, 60, (3, signals))
    phases = rng.uniform(0, 2 * np.pi, (3, signals))
    return sum(5.0 * np.sin(2 * np.pi * t / periods[k] + phases[k]) for k in range(3))


def delayed(signal, frames):
    """The same motion arriving `frames` later (fractions by straight-line interpolation)."""
    t = np.arange(signal.shape[0], dtype=np.float64)
    return np.column_stack([np.interp(t - frames, t, signal[:, s]) for s in range(signal.shape[1])])


def test_change_from_first_uses_the_first_known_frame():
    values = np.array([[np.nan, np.nan], [10.0, 20.0], [13.0, 18.0]])
    changed = change_from_first(values)
    assert np.isnan(changed[0]).all()
    assert changed[1:].tolist() == [[0.0, 0.0], [3.0, -2.0]]
    with pytest.raises(ValueError):
        change_from_first(np.full((4, 2), np.nan))


def test_error_is_zero_when_the_output_copies_the_motion():
    motion = head_motion()
    error, size = mean_abs_error(motion, motion)
    assert error == pytest.approx(0.0, abs=1e-12)
    assert size > 1.0


def test_error_ignores_a_different_resting_pose():
    # A source whose head rests 12 degrees to the side is not a motion error.
    motion = head_motion()
    error, _ = mean_abs_error(motion, motion + 12.0)
    assert error == pytest.approx(0.0, abs=1e-9)


def test_error_measures_damped_motion():
    motion = head_motion()
    error, size = mean_abs_error(motion, 0.5 * motion)
    assert error == pytest.approx(0.5 * size, rel=1e-9)
    frozen_error, _ = mean_abs_error(motion, np.zeros_like(motion))
    assert frozen_error == pytest.approx(size, rel=1e-9)  # a frozen face misses all of it


def test_error_skips_frames_without_a_face():
    motion = head_motion()
    output = motion.copy()
    output[40:50] = np.nan
    error, _ = mean_abs_error(motion, output)
    assert error == pytest.approx(0.0, abs=1e-12)


def test_correlation():
    signal = head_motion(signals=1)[:, 0]
    assert correlation(signal, 2 * signal + 3) == pytest.approx(1.0)
    assert correlation(signal, -signal) == pytest.approx(-1.0)
    assert correlation(signal[:5], signal[:5]) is None  # too few frames
    assert correlation(signal, np.ones_like(signal)) is None  # one signal never changes


@pytest.mark.parametrize("delay", [0.0, 1.0, 2.0, -1.0, 0.5, 1.5])
def test_lag_is_recovered(delay):
    motion = head_motion()
    lag, match = motion_lag(motion, delayed(motion, delay), min_std=(1.0, 1.0, 1.0))
    assert lag == pytest.approx(delay, abs=0.2)
    assert match > 0.95


def test_lag_survives_noise_and_missing_frames():
    motion = head_motion()
    output = delayed(motion, 2.0) + np.random.default_rng(5).normal(0, 0.5, motion.shape)
    output[60:70] = np.nan
    lag, _ = motion_lag(motion, output, min_std=(1.0, 1.0, 1.0))
    assert lag == pytest.approx(2.0, abs=0.3)


def test_lag_is_not_reported_for_a_still_driver():
    still = 0.01 * head_motion()
    assert motion_lag(still, still, min_std=(1.0, 1.0, 1.0)) == (None, None)


def test_lag_uses_only_the_signals_that_move():
    motion = head_motion()
    motion[:, 2] *= 0.001  # third signal barely moves in the driver
    output = delayed(motion, 1.0)
    output[:, 2] = np.random.default_rng(6).normal(0, 1.0, motion.shape[0])  # and is pure noise in the output
    lag, _ = motion_lag(motion, output, min_std=(1.0, 1.0, 1.0))
    assert lag == pytest.approx(1.0, abs=0.2)


def test_lag_rejects_mismatched_input():
    with pytest.raises(ValueError):
        motion_lag(np.zeros((50, 3)), np.zeros((40, 3)), min_std=(1.0, 1.0, 1.0))
