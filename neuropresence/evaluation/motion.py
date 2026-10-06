"""How faithfully the output follows the driving face.

Identity similarity alone can be gamed by not moving: a frozen copy of the
source scores a perfect CSIM. These measurements are the counterweight. They
compare the head pose and the expression of the output with those of the
driving video, frame by frame.

All inputs are arrays over time with one row per frame and NaN rows for
frames where the face was not found.
"""

import numpy as np


def change_from_first(values):
    """Subtract each signal's value at the first frame where it is known.

    The first driven frame is the neutral pose for the reenactment, so motion
    is compared as change since that frame. This also makes a source and a
    driver with different resting poses comparable.
    """
    values = np.asarray(values, dtype=np.float64)
    known = np.isfinite(values).all(axis=1)
    if not known.any():
        raise ValueError("The face was not found in any frame.")
    return values - values[np.argmax(known)]


def mean_abs_error(driving, output):
    """Mean absolute difference between driving and output motion, and the
    mean absolute size of the driving motion itself for comparison."""
    d, o = change_from_first(driving), change_from_first(output)
    both = np.isfinite(d).all(axis=1) & np.isfinite(o).all(axis=1)
    if not both.any():
        raise ValueError("No frame has both a driving and an output face.")
    return float(np.abs(d[both] - o[both]).mean()), float(np.abs(d[both]).mean())


def correlation(a, b, min_samples=10):
    """Pearson correlation of two signals over the frames where both are known.

    Returns None if there are too few frames or a signal does not vary.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < min_samples or a[both].std() < 1e-9 or b[both].std() < 1e-9:
        return None
    return float(np.corrcoef(a[both], b[both])[0, 1])


def motion_lag(driving, output, min_std, max_lag=4):
    """Time shift, in frames, that best lines the output up with the driving motion.

    driving, output: (T, S) arrays holding S motion signals (for example yaw,
    pitch and mouth opening). min_std: one value per signal; a signal that
    varies less than this in the driving video is left out, because its
    correlation would be mostly noise. Positive lag means the output is late.

    The match is the correlation averaged over the signals, tried at every
    whole-frame shift from -max_lag to +max_lag; a parabola through the best
    shift and its two neighbours gives the fraction of a frame.
    Returns (lag, correlation at the best whole-frame shift), or (None, None)
    if no signal moves enough.
    """
    driving, output = np.asarray(driving, dtype=np.float64), np.asarray(output, dtype=np.float64)
    if driving.shape != output.shape or driving.ndim != 2:
        raise ValueError(f"Need two matching (T, S) arrays, got {driving.shape} and {output.shape}")
    frames = driving.shape[0]

    def spread(signal):
        known = signal[np.isfinite(signal)]
        return known.std() if known.size else 0.0

    usable = [s for s in range(driving.shape[1]) if spread(driving[:, s]) >= min_std[s]]
    if not usable or frames <= 2 * max_lag + 10:
        return None, None

    scores = []
    for shift in range(-max_lag, max_lag + 1):
        # Pair driving[t] with output[t + shift].
        d = driving[max(0, -shift):frames - max(0, shift)]
        o = output[max(0, shift):frames - max(0, -shift)]
        per_signal = [correlation(d[:, s], o[:, s]) for s in usable]
        per_signal = [c for c in per_signal if c is not None]
        if not per_signal:
            return None, None
        scores.append(float(np.mean(per_signal)))

    best = int(np.argmax(scores))
    fraction = 0.0
    if 0 < best < len(scores) - 1:
        before, peak, after = scores[best - 1], scores[best], scores[best + 1]
        curvature = before - 2 * peak + after
        if curvature < 0:
            fraction = 0.5 * (before - after) / curvature
    return best - max_lag + fraction, scores[best]
