"""Stages 1-3 joined: one webcam frame in, one output frame out."""

import time
from dataclasses import dataclass

import numpy as np

from .capture import TrackStatus
from .capture.crop import square_face_crop


@dataclass
class FrameResult:
    output: np.ndarray  # BGR frame to send onward, same size as the source image
    status: TrackStatus
    live: bool  # True if reenacted from this frame, False if the static fallback
    timing_ms: dict


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

    def step(self, frame_bgr):
        start = time.perf_counter()
        track = self.tracker.process(frame_bgr)
        timing = {"tracker": track.latency_ms}
        if track.ok:
            face = square_face_crop(frame_bgr, track.bbox, scale=self.crop_scale)
            output = self.engine.drive(face)
            timing.update(self.engine.last_timing_ms)
        else:
            output = self.engine.source_frame.copy()
        timing["total"] = (time.perf_counter() - start) * 1000
        return FrameResult(output, track.status, track.ok, timing)
