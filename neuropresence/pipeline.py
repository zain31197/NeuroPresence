"""Stages 1-3 joined: one webcam frame in, one output frame out."""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .capture import TrackResult, TrackStatus
from .capture.crop import square_face_crop
from .capture.steady import SteadyCrop
from .consent.disclosure import mark

# The neutral pose is the person's own resting face at their camera: all movement is measured
# from it. It is not the enrolled picture's pose. That picture may come from another camera, and
# the difference between the two would then be applied as if it were movement: measured on a
# sample photograph driven from a differently framed camera, a head 0.87 times the size and
# turned 14 degrees, with the person sitting still.
NEUTRAL_MAX_YAW_DEG = 12.0  # at rest means facing the camera ...
NEUTRAL_MAX_ROLL_DEG = 10.0  # ... with the head upright ...
NEUTRAL_MAX_JAW_OPEN = 0.15  # ... the mouth closed ...
# The tracker's jaw score does not see lips that are a little apart: at a real camera it read 0.02 with
# the lips 0.09 of the eye span apart. So the gap between the lips is measured as well.
NEUTRAL_MAX_LIP_GAP = 0.02  # ... the lips together: their gap over the distance between the eyes ...
LIPS_CLOSED_GAP = 0.012  # under this the lips are closed (0.002 to 0.005 at rest, and in an enrolled picture)
UPPER_LIP, LOWER_LIP, EYE_OUTER = 13, 14, (33, 263)  # landmarks of the tracker
NEUTRAL_MAX_EYE_CLOSED = 0.5  # ... and the eyes open. Pitch is left out: it depends on how high the camera sits.
NEUTRAL_WAIT_SECONDS = 3.0  # after this long the current frame is taken, at rest or not

# When the face is lost the output does not snap to the still picture. The last live frame is held
# for a moment, because the tracker often misses a single frame, and then fades to the still
# picture. When the face comes back the output fades in the same way.
HOLD_SECONDS = 0.25
FADE_SECONDS = 0.3

# The movement may be read a little past the newest camera frame, by carrying on the way it was
# going: each millisecond of that is a millisecond less delay. It overshoots where the movement
# turns. Measured on 14 October 2026 on six sample clips (see server/pacing.py): 10 ms ahead, the
# delay 96 to 86 ms, steps as even, the head changing speed 2.95 against 2.70 (3.77 without even
# motion); 30 ms ahead, 67 ms, and the mouth then agrees less (0.875 against 0.899).
LEAD_SECONDS = 0.010
# Two readings of the movement further apart than this are not joined: the face was lost between them.
MAX_BETWEEN_SECONDS = 0.25


def lip_gap(track):
    """How far apart the lips are, over the distance between the eyes. None when it cannot be read."""
    points = getattr(track, "landmarks", None)
    if not track.ok or points is None or len(points) <= max(EYE_OUTER):
        return None
    eyes = float(np.linalg.norm(points[EYE_OUTER[0]] - points[EYE_OUTER[1]]))
    return float(np.linalg.norm(points[UPPER_LIP] - points[LOWER_LIP])) / eyes if eyes > 0 else None


def at_rest(track):
    """Is this a face to measure movement from: front-facing, upright, mouth closed, eyes open?"""
    if not track.ok or track.pose_deg is None:
        return False
    gap = lip_gap(track)
    if gap is not None and gap > NEUTRAL_MAX_LIP_GAP:
        return False
    yaw, _, roll = track.pose_deg
    scores = track.blendshapes or {}
    return (abs(yaw) <= NEUTRAL_MAX_YAW_DEG and abs(roll) <= NEUTRAL_MAX_ROLL_DEG
            and scores.get("jawOpen", 0.0) <= NEUTRAL_MAX_JAW_OPEN
            and max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0)) <= NEUTRAL_MAX_EYE_CLOSED)


@dataclass
class FrameResult:
    output: np.ndarray  # BGR frame to send onward, same size as the source image
    status: TrackStatus
    live: bool  # True if reenacted from this frame, False if the static fallback
    timing_ms: dict
    driving_face: np.ndarray | None = None  # the 256x256 face crop that drove this frame
    crop_window: tuple | None = None  # where that crop was cut from the camera frame, as (x, y, side)
    track: TrackResult | None = None  # what the tracker found in this frame
    motion_at: float | None = None  # the moment the movement shown is of, if it is not the frame's own


class Pipeline:
    """Reenacts when exactly one face is tracked; otherwise shows the enrolled frame.

    The static enrolled frame is the fallback so the output is never frozen
    mid-expression or blank. Every frame that leaves the pipeline, live or still,
    carries the disclosure mark (consent/disclosure.py). There is no way to turn it off.
    """

    def __init__(self, tracker, engine, crop_scale=2.0):
        if engine.source_frame is None:
            raise ValueError("Call engine.set_source() before building the pipeline.")
        self.tracker = tracker
        self.engine = engine
        self.crop_scale = crop_scale
        self.steady_crop = True  # hold the crop window still while the head is still
        self.steady_keypoints = True  # filter the keypoints given to the generator
        self.natural_range = True  # keep the head to the range that looks right on a body that stays still
        self._crop = SteadyCrop()
        self._level = None  # how much of the output is live: 1 live, 0 the still picture, between while fading
        self._last_live = None  # the newest live output and when it was made
        self._last_at = None
        self._still_of = self._still = None  # the enrolled frame with the mark on it, and which frame it was made from
        self._before = None  # the reading of the movement in the frame before, as (when it was taken, reading)
        self._neutral_deadline = None  # set while waiting to see the person at rest
        self.neutral_taken = 0  # how many times a neutral pose has been taken from the camera

    def wait_for_neutral(self, seconds=None):
        """Show the still picture until the person is seen at rest, then measure all movement from that frame."""
        self.engine.reset_reference()
        self._before = None
        self._neutral_deadline = time.perf_counter() + (NEUTRAL_WAIT_SECONDS if seconds is None else seconds)

    @property
    def waiting_for_neutral(self):
        return self._neutral_deadline is not None

    def prepare(self, frame_bgr, track, at):
        """The part of a frame that can be done ahead of its turn on the GPU: the face crop.

        Returns what step() takes as `prepared`, or None if there is no single face. The live
        session calls this on the tracking thread; whoever calls it owns the steady crop.
        """
        if not track.ok:
            self._crop.reset()
            return None
        window = None
        if self.steady_crop and track.landmarks is not None:
            face, window = self._crop(frame_bgr, track.landmarks, at, self.crop_scale)
        else:
            self._crop.reset()
            face = square_face_crop(frame_bgr, track.bbox, scale=self.crop_scale)
        # The reading of the movement is not done here although the engine could (engine.read): it needs
        # the GPU, so on another thread it only queues behind the frame being drawn. Tried on 8 October
        # 2026: 1.6 frames a second more, but 35 ms more delay from camera to output.
        return face, window, None

    def _movement(self, face, at, motion_at):
        """The movement to draw, and the moment it is of.

        Without `motion_at` that is simply what this frame shows. With it, the movement is read
        between this frame and the one before, for that moment: the camera and the output run at
        different rates, so the newest frame is sometimes just in and sometimes a frame old, and a
        steady turn drawn from it moves in short and long steps (see server/pacing.py).
        """
        if motion_at is None:
            self._before = None
            return None, None
        reading = self.engine.read(face)
        before, self._before = self._before, (at, reading)
        if before is None or not 0.0 < at - before[0] <= MAX_BETWEEN_SECONDS:
            return reading, at
        step = at - before[0]
        share = min(1.0 + LEAD_SECONDS / step, max(0.0, (motion_at - before[0]) / step))
        return self.engine.between(before[1], reading, share), before[0] + share * (at - before[0])

    def step(self, frame_bgr, reenact=True, at=None, track=None, prepared=None, show=True, motion_at=None):
        """Process one frame. With reenact=False the face is still tracked but
        the output stays on the enrolled frame. `at` is when the frame was taken,
        in seconds on any clock; it is what the steadying and the fades are timed by.
        Pass `track` if the face has already been tracked in this frame, as the live
        session does on another thread while the GPU draws the frame before.

        With show=False the frame is drawn all the same, so that its timings stay real,
        but it is held back and the output goes to the enrolled frame as it does for a
        lost face. The live session does this while its output is arriving too late.

        `motion_at`, on the same clock as `at`, asks for the movement as it was at that
        moment, read between this frame and the one before, and at most LEAD_SECONDS past
        this frame (see _movement)."""
        start = time.perf_counter()
        at = start if at is None else at
        if track is None:
            track = self.tracker.process(frame_bgr)
        timing = {"tracker": track.latency_ms}
        face = None
        live = track.ok and reenact
        window = motion = None
        ahead = prepared is not None  # the crop and the reading were done before this call
        if live:
            if ahead:
                face, window, motion = prepared
            elif self.steady_crop and track.landmarks is not None:
                face, window = self._crop(frame_bgr, track.landmarks, at, self.crop_scale)
            else:
                self._crop.reset()
                face = square_face_crop(frame_bgr, track.bbox, scale=self.crop_scale)
            if self._neutral_deadline is not None:
                if at_rest(track) or time.perf_counter() >= self._neutral_deadline:
                    self.engine.set_reference(face)
                    self._neutral_deadline = None
                    self.neutral_taken += 1
                else:
                    live = False  # not at rest yet: keep showing the still picture
        shown_at = None
        if live:
            reading = time.perf_counter()
            if motion is None:
                motion, shown_at = self._movement(face, at, motion_at)
            reading = (time.perf_counter() - reading) * 1000
            given = {} if motion is None else {"motion": motion}
            self.engine.resting = at_rest(track)  # at rest, the neutral pose follows the person (engine.py)
            gap = lip_gap(track)
            self.engine.lips_closed = gap is not None and gap <= LIPS_CLOSED_GAP
            output = self.engine.drive(face, at=at if shown_at is None else shown_at, steady=self.steady_keypoints,
                                       limit=self.natural_range, **given)
            timing.update(self.engine.last_timing_ms)
            timing["motion"] = timing.get("motion", 0.0) + reading  # reading the movement is part of that stage, wherever it is done
            if show:
                output = mark(output)  # once, here: holds and fades then only reuse frames that carry it
            else:
                output, live = None, False
        else:
            if track is None or not track.ok or not ahead:
                self._crop.reset()
            self._before = None  # nothing was drawn from this frame: the next reading stands alone
            output = None
        output = self._settle(output, at)
        timing["total"] = (time.perf_counter() - start) * 1000
        return FrameResult(output, track.status, live, timing, face, crop_window=window, track=track, motion_at=shown_at)

    def _settle(self, live_output, at):
        """Hold and fade between the live output and the still picture, so the two never snap."""
        still = self.still_picture()
        elapsed = 0.0 if self._last_at is None else max(0.0, at - self._last_at)
        self._last_at = at
        if self._level is None:  # the very first frame: nothing to fade from
            self._level = 1.0 if live_output is not None else 0.0
        if live_output is not None:
            self._last_live = (live_output, at)
            self._level = min(1.0, self._level + elapsed / FADE_SECONDS)
            shown = live_output
        elif self._last_live is None:
            return still.copy()
        else:
            shown, since = self._last_live
            if at - since <= HOLD_SECONDS:
                return shown  # one missed frame is not a lost face: keep the last live frame for a moment
            self._level = max(0.0, self._level - elapsed / FADE_SECONDS)
            if self._level == 0.0:
                self._last_live = None
                return still.copy()
        if self._level >= 1.0 or shown.shape != still.shape:
            return shown
        return cv2.addWeighted(shown, self._level, still, 1.0 - self._level, 0.0)

    def still_picture(self):
        """The enrolled frame as it is shown: with the disclosure mark, like every other output frame."""
        source = self.engine.source_frame
        if self._still_of is not source:
            self._still_of, self._still = source, mark(source.copy())
        return self._still
