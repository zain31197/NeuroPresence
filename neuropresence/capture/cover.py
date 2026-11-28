"""Is the mouth covered in the camera picture: a hand over it, a cup, a sleeve?

The picture's mouth is drawn from the camera's. A hand over the mouth is read as a mouth doing something strange,
and the picture's mouth was drawn smeared until the identity guard noticed that the output no longer looked like
the picture, which it did in only about half the runs, because those scores sit on its limit (PROJECT_HISTORY.md,
9 October 2026). Once the cover is known, the engine holds the picture's mouth closed and keeps the cover out of
the neutral pose (reenactment/engine.py), and the rest of the face goes on moving.

How it is told: lips are redder than the cheeks beside them. The difference is measured as the median of the a*
channel of CIE Lab (green to red) inside the outer lip line, less its median on both cheeks, and compared with what
it usually is for this person in this light, learned at the start of the session and followed slowly while the
mouth is uncovered. A hand is the same skin as the cheeks and takes the difference away; a cup or a sleeve changes
it far, either way. Measured on 27 November 2026 (scripts/study_mouth_cover.py, results/mouth_cover_study.json).

When this person's lips are hardly redder than their cheeks in this light, a cover cannot be told from a mouth,
and it says so (`usable` False) and never calls the mouth covered: then nothing changes from before.
"""

from collections import deque

import cv2
import numpy as np

# MediaPipe Face Mesh: the outer and inner lines of the lips, and points in the middle of each cheek. The lips are
# the band between the two lines: inside the inner one an open mouth shows teeth and tongue, which are not lips.
OUTER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
INNER_LIPS = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
CHEEKS = (50, 280, 205, 425)
EYE_CORNERS = (33, 263)

LEAST_CONTRAST = 3.5  # lips at least this much redder than the cheeks (a* in 8-bit Lab), or it cannot tell ...
LEAST_OVER_NOISE = 3.0  # ... and at least this many times how much the readings wander from frame to frame:
# Zain's own camera reads 5.5 to 6.9 with a wander of 0.7 to 0.9 (7.7 times); a sample clip whose lips read 2.6
# with a wander of 1.3 (1.9 times) gave false alarms, since its lips cannot be told from its own noise.
LEARN_FRAMES = 20  # readings of the uncovered mouth that make the first baseline: about a second
FOLLOW_SECONDS = 20.0  # the baseline follows the light this slowly, and only while the mouth is uncovered
NOISE_SECONDS = 10.0  # the wander is measured all along, over about this long, since one calm second says little
COVERED_BELOW, COVERED_ABOVE = 0.35, 3.0  # a reading this far from the baseline, as a share of it, is a cover ...
NOISE_MARGIN = 3.5  # ... and at least this many times the wander from it, so that a noisy face is not called covered
UNCOVERED_BETWEEN = (0.55, 2.0)  # and one back within this is the mouth again
FRAMES_TO_COVER = 4  # in a row: about a fifth of a second at 22 frames a second
FRAMES_TO_UNCOVER = 6


def lip_redness(frame_bgr, landmarks):
    """How much redder the lips are than the cheeks in this frame, or None if the face is too small to say."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or len(points) <= max(OUTER_LIPS + INNER_LIPS + CHEEKS + EYE_CORNERS):
        return None  # not the face mesh's landmarks
    points = points[:, :2]
    eyes = float(np.linalg.norm(points[EYE_CORNERS[0]] - points[EYE_CORNERS[1]]))
    if eyes < 20.0:
        return None
    lips = points[list(OUTER_LIPS)]
    radius = max(3, int(0.06 * eyes))
    region = np.vstack([lips, points[list(CHEEKS)]])
    height, width = frame_bgr.shape[:2]
    x0, y0 = (np.floor(region.min(axis=0)) - radius).astype(int)
    x1, y1 = (np.ceil(region.max(axis=0)) + radius).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    # Only the lower face is turned into Lab: about 1 ms, against 4 for the whole picture. In floating point, so
    # that a* is not rounded to whole steps, which for faint lips are a third of the difference.
    red = cv2.cvtColor(frame_bgr[y0:y1, x0:x1].astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[:, :, 1]
    mask = np.zeros(red.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(lips - (x0, y0)).astype(np.int32)], 1)
    cv2.fillPoly(mask, [np.round(points[list(INNER_LIPS)] - (x0, y0)).astype(np.int32)], 0)
    if mask.sum() < 20:
        return None
    cheeks = []
    for index in CHEEKS:
        cx, cy = np.round(points[index] - (x0, y0)).astype(int)
        patch = red[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius]
        if patch.size:
            cheeks.append(patch.ravel())
    if not cheeks:
        return None
    return float(np.median(red[mask.astype(bool)])) - float(np.median(np.concatenate(cheeks)))


class MouthCover:
    """Follows one session's frames and says whether the mouth is covered. Not safe for two threads."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.base = None  # how much redder this person's lips usually are than their cheeks, in this light
        self.noise = None  # how much the readings wander from frame to frame around it
        self._learning = deque(maxlen=LEARN_FRAMES)
        self._last_at = None
        self._run = 0
        self.covered = False
        self.share = None  # the last reading as a share of the baseline

    @property
    def usable(self):
        """Can a cover be told from a mouth for this person in this light?"""
        # A plain bool: the session puts it in the status, and JSON refuses numpy's own bool.
        return bool(self.base is not None and self.base >= LEAST_CONTRAST
                    and self.base >= LEAST_OVER_NOISE * (self.noise or 0.0))

    def update(self, frame_bgr, landmarks, at):
        """One frame with its tracked face. Returns whether the mouth is covered now."""
        reading = lip_redness(frame_bgr, landmarks)
        elapsed = 0.0 if self._last_at is None else min(max(at - self._last_at, 0.0), 0.5)
        self._last_at = at
        if reading is None:
            return self.covered
        if not self.usable:
            # Until a baseline is usable, it is the median of the latest readings: a mouth covered at the very
            # start, or a light that hides the lips, gives none, and one is learned once the lips show.
            self._learning.append(reading)
            if len(self._learning) == LEARN_FRAMES:
                values = np.asarray(self._learning)
                self.base = float(np.median(values))
                self.noise = 1.4826 * float(np.median(np.abs(values - self.base)))  # a spread that one odd frame does not move
            else:
                self.base = self.noise = None
            self.covered, self._run, self.share = False, 0, None
            return False
        self.share = float(reading / self.base)
        margin = NOISE_MARGIN * self.noise
        far = (reading < self.base - max((1.0 - COVERED_BELOW) * self.base, margin)
               or reading > self.base + max((COVERED_ABOVE - 1.0) * self.base, margin))
        near = UNCOVERED_BETWEEN[0] <= self.share <= UNCOVERED_BETWEEN[1]
        against = near if self.covered else far
        self._run = self._run + 1 if against else 0
        if self._run >= (FRAMES_TO_UNCOVER if self.covered else FRAMES_TO_COVER):
            self.covered, self._run = not self.covered, 0
        if not self.covered and not far and elapsed > 0.0:
            # The light changes slowly, and the baseline follows it; the wander is kept up to date with it, each
            # reading counted at most three times the wander so far, so that one odd frame does not swell it.
            self.base = float(self.base + (reading - self.base) * (1.0 - np.exp(-elapsed / FOLLOW_SECONDS)))
            wander = 1.25 * min(abs(reading - self.base), 3.0 * self.noise)  # a mean distance, as a spread
            self.noise = float(self.noise + (wander - self.noise) * (1.0 - np.exp(-elapsed / NOISE_SECONDS)))
            if not self.usable:  # this person's lips, in this light, can no longer be told from the noise
                self._learning.clear()
                self.base = self.noise = None
        return self.covered
