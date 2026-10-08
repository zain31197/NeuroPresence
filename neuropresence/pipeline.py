"""Stages 1-3 joined: one webcam frame in, one output frame out."""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .capture import TrackResult, TrackStatus
from .capture.crop import square_face_crop
from .capture.steady import SteadyCrop

# The neutral pose is the person's own resting face at their camera: all movement is measured
# from it. It is not the enrolled picture's pose. That picture may come from another camera, and
# the difference between the two would then be applied as if it were movement: measured on a
# sample photograph driven from a differently framed camera, a head 0.87 times the size and
# turned 14 degrees, with the person sitting still.
NEUTRAL_MAX_YAW_DEG = 12.0  # at rest means facing the camera ...
NEUTRAL_MAX_ROLL_DEG = 10.0  # ... with the head upright ...
NEUTRAL_MAX_JAW_OPEN = 0.15  # ... the mouth closed ...
NEUTRAL_MAX_EYE_CLOSED = 0.5  # ... and the eyes open. Pitch is left out: it depends on how high the camera sits.
NEUTRAL_WAIT_SECONDS = 3.0  # after this long the current frame is taken, at rest or not

# When the face is lost the output does not snap to the still picture. The last live frame is held
# for a moment, because the tracker often misses a single frame, and then fades to the still
# picture. When the face comes back the output fades in the same way.
HOLD_SECONDS = 0.25
FADE_SECONDS = 0.3


def at_rest(track):
    """Is this a face to measure movement from: front-facing, upright, mouth closed, eyes open?"""
    if not track.ok:
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


class Pipeline:
    """Reenacts when exactly one face is tracked; otherwise shows the enrolled frame.

    The static enrolled frame is the fallback so the output is never frozen
    mid-expression or blank.
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
        self._neutral_deadline = None  # set while waiting to see the person at rest
        self.neutral_taken = 0  # how many times a neutral pose has been taken from the camera

    def wait_for_neutral(self, seconds=None):
        """Show the still picture until the person is seen at rest, then measure all movement from that frame."""
        self.engine.reset_reference()
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

    def step(self, frame_bgr, reenact=True, at=None, track=None, prepared=None):
        """Process one frame. With reenact=False the face is still tracked but
        the output stays on the enrolled frame. `at` is when the frame was taken,
        in seconds on any clock; it is what the steadying and the fades are timed by.
        Pass `track` if the face has already been tracked in this frame, as the live
        session does on another thread while the GPU draws the frame before."""
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
        if live:
            given = {} if motion is None else {"motion": motion}
            output = self.engine.drive(face, at=at, steady=self.steady_keypoints, limit=self.natural_range, **given)
            timing.update(self.engine.last_timing_ms)
        else:
            if track is None or not track.ok or not ahead:
                self._crop.reset()
            output = None
        output = self._settle(output, at)
        timing["total"] = (time.perf_counter() - start) * 1000
        return FrameResult(output, track.status, live, timing, face, crop_window=window, track=track)

    def _settle(self, live_output, at):
        """Hold and fade between the live output and the still picture, so the two never snap."""
        still = self.engine.source_frame
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
