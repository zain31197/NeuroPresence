"""Holding the driving signal steady.

The output's head trembled about 1.5 times as much as real video. Measured on
four sample clips (7 and 8 October 2026), the tremble has two sources, and each
is dealt with where it arises:

- The face crop was cut fresh around the tracked box of every frame, so the
  tracker's small errors shifted and resized the face inside the crop, and the
  model read that as movement. SteadyCrop filters the crop's position and size.
  Tremble against real video: 1.50 before, 1.08 with this alone.
- The model's own reading still trembles a little when the head is still. The
  engine filters the keypoints it gives the generator (see reenactment/engine.py).
  With both: 0.96.

Both use the One Euro filter, which smooths hard when the signal is nearly
still and lets movement through: pose error and mouth sync were unchanged, and
the head pose trails the camera by a quarter of a frame.

What did not work, so nobody tries it again without a reason: placing the crop
in fractions of a pixel (1.47), one fixed crop for the whole clip (1.41),
a crop built from the eye corners (1.93), and filtering the pose angles the
model reads (worse tremble, half a frame of delay).
"""

import math

import cv2
import numpy as np

CROP_MIN_CUTOFF_HZ = 0.3  # how hard the crop is held when the head is still
CROP_BETA = 4.0  # how quickly it lets go when the head moves; speeds are in face sizes per second
MAX_GAP_SECONDS = 0.5  # after a gap this long a filter starts again from the new reading


class OneEuro:
    """The One Euro filter (Casiez, Roussel and Vogel, 2012).

    A low-pass filter whose cutoff rises with the speed of the signal: at rest
    it smooths at min_cutoff, and each unit of speed adds beta to the cutoff.
    Readings carry their own time, so it works at any frame rate.
    """

    def __init__(self, min_cutoff, beta, speed_cutoff=1.0, max_gap=MAX_GAP_SECONDS):
        self.min_cutoff, self.beta, self.speed_cutoff, self.max_gap = min_cutoff, beta, speed_cutoff, max_gap
        self.reset()

    def reset(self):
        self._value = self._speed = self._at = None

    @staticmethod
    def _weight(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def __call__(self, value, at):
        """Filter one reading taken at time `at` (seconds). Returns the filtered value."""
        value = np.asarray(value, dtype=np.float64)
        if self._value is None or not 0.0 < at - self._at <= self.max_gap or value.shape != self._value.shape:
            self._value, self._speed, self._at = value, np.zeros_like(value), at
            return value
        dt = at - self._at
        w = self._weight(self.speed_cutoff, dt)
        self._speed = w * (value - self._value) / dt + (1.0 - w) * self._speed
        w = self._weight(self.min_cutoff + self.beta * np.abs(self._speed), dt)
        self._value, self._at = w * value + (1.0 - w) * self._value, at
        return self._value


def square_crop_at(frame, cx, cy, side, out_size=256):
    """An out_size square crop centred on (cx, cy), placed in fractions of a pixel.

    Where it runs past the frame, the edge pixels are repeated outward.
    """
    scale = 2.0 * out_size / side
    to_crop = np.array([[scale, 0.0, out_size - cx * scale], [0.0, scale, out_size - cy * scale]])
    double = cv2.warpAffine(frame, to_crop, (2 * out_size, 2 * out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(double, (out_size, out_size), interpolation=cv2.INTER_AREA)


class SteadyCrop:
    """Cuts the driving face crop from a window that holds still while the head does."""

    def __init__(self, min_cutoff=CROP_MIN_CUTOFF_HZ, beta=CROP_BETA):
        self._centre = OneEuro(min_cutoff, beta)
        self._size = OneEuro(min_cutoff / 2.0, beta)  # the face changes size more slowly than it moves
        self._unit = None

    def reset(self):
        self._centre.reset()
        self._size.reset()
        self._unit = None

    def __call__(self, frame, landmarks, at, scale=2.0, out_size=256):
        """Returns the crop and its window in the frame as (x, y, side)."""
        low, high = landmarks.min(axis=0), landmarks.max(axis=0)
        centre, size = (low + high) / 2.0, float((high - low).max())
        # Positions are filtered in face sizes, so the filter acts the same at any camera resolution.
        self._unit = self._unit or size
        centre = self._centre(centre / self._unit, at) * self._unit
        side = scale * float(self._size(size / self._unit, at)) * self._unit
        return square_crop_at(frame, centre[0], centre[1], side, out_size), (float(centre[0] - side / 2), float(centre[1] - side / 2), float(side))
