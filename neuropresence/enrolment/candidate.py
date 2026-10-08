"""A picture that has been checked but is not enrolled yet."""

from dataclasses import dataclass

import numpy as np

from ..capture.crop import MAX_PICTURE_DIM, limit_size, square_face_crop
from .checks import Check, all_passed, as_dicts, evaluate, first_hint, first_tip, judge, measure


@dataclass
class Candidate:
    # As it would be stored: at most MAX_PICTURE_DIM on its longer side, and never smaller
    # than that requires. A larger face gives a sharper output (see checks.py).
    image: np.ndarray
    origin: str  # "camera" or "upload"
    checks: list[Check]
    landmarks: np.ndarray | None  # of the one face, if there is exactly one
    neutral_face: np.ndarray | None  # that face's driving crop: the neutral pose for this picture
    score: float = 0.0  # higher is better; used to pick the best of several good frames
    signature: np.ndarray | None = None  # of an uploaded picture, once it has been compared with the verified face

    @property
    def passed(self):
        return all_passed(self.checks)

    def summary(self):
        height, width = self.image.shape[:2]
        return {
            "origin": self.origin,
            "width": width,
            "height": height,
            "checks": as_dicts(self.checks),
            "passed": self.passed,
            "hint": first_hint(self.checks),
            "tip": first_tip(self.checks),
        }


def prepare(image_bgr):
    """Bring a picture to the size it would be stored at, before it is tracked and checked."""
    return limit_size(image_bgr, MAX_PICTURE_DIM)


def make_candidate(image_bgr, track, origin):
    """Check a prepared picture whose face has been tracked."""
    if not track.ok:
        return Candidate(image_bgr, origin, evaluate(image_bgr, track, origin), None, None)
    measurements = measure(image_bgr, track)
    # Sharper is better, and so are wider-open eyes: this picks the still, unblinking frame.
    score = measurements["sharpness"] * (1.0 - measurements["eye_closed"])
    return Candidate(image_bgr, origin, judge(measurements, origin), track.landmarks,
                     square_face_crop(image_bgr, track.bbox), score)
