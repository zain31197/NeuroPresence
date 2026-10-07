"""Stages 1-3 joined: one webcam frame in, one output frame out."""

import time
from dataclasses import dataclass

import numpy as np

from .capture import TrackResult, TrackStatus
from .capture.crop import square_face_crop

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
        self._neutral_deadline = None  # set while waiting to see the person at rest
        self.neutral_taken = 0  # how many times a neutral pose has been taken from the camera

    def wait_for_neutral(self, seconds=None):
        """Show the still picture until the person is seen at rest, then measure all movement from that frame."""
        self.engine.reset_reference()
        self._neutral_deadline = time.perf_counter() + (NEUTRAL_WAIT_SECONDS if seconds is None else seconds)

    @property
    def waiting_for_neutral(self):
        return self._neutral_deadline is not None

    def step(self, frame_bgr, reenact=True):
        """Process one frame. With reenact=False the face is still tracked but
        the output stays on the enrolled frame."""
        start = time.perf_counter()
        track = self.tracker.process(frame_bgr)
        timing = {"tracker": track.latency_ms}
        face = None
        live = track.ok and reenact
        if live:
            face = square_face_crop(frame_bgr, track.bbox, scale=self.crop_scale)
            if self._neutral_deadline is not None:
                if at_rest(track) or time.perf_counter() >= self._neutral_deadline:
                    self.engine.set_reference(face)
                    self._neutral_deadline = None
                    self.neutral_taken += 1
                else:
                    live = False  # not at rest yet: keep showing the still picture
        if live:
            output = self.engine.drive(face)
            timing.update(self.engine.last_timing_ms)
        else:
            output = self.engine.source_frame.copy()
        timing["total"] = (time.perf_counter() - start) * 1000
        return FrameResult(output, track.status, live, timing, face, track)
