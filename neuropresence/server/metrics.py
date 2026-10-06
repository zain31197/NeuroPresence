"""Live figures for a running session, averaged over the last few seconds."""

import threading
import time
from collections import deque

import numpy as np

STAGES = ("tracker", "motion", "render", "compose")


class MetricsWindow:
    """Keeps the frames of the last `seconds` and summarises them on request."""

    def __init__(self, seconds=2.0):
        self._seconds = seconds
        self._frames = deque()  # (finished_at, timing_ms, end_to_end_ms, live, dropped)
        self._lock = threading.Lock()
        self.total_frames = 0

    def add(self, finished_at, timing_ms, end_to_end_ms, live, dropped):
        with self._lock:
            self._frames.append((finished_at, dict(timing_ms), end_to_end_ms, live, dropped))
            self.total_frames += 1
            self._trim(finished_at)

    def _trim(self, now):
        while self._frames and now - self._frames[0][0] > self._seconds:
            self._frames.popleft()

    def summary(self, now=None):
        """Averages over the window, or None values when no frame is recent enough."""
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._trim(now)
            frames = list(self._frames)
        if len(frames) < 2:
            return {"fps": None, "pipeline_ms": None, "end_to_end_ms": None, "render_ms": None,
                    "stages_ms": None, "dropped_share": None, "live_share": None}
        span = frames[-1][0] - frames[0][0]
        live = [f for f in frames if f[3]]
        produced = len(frames) + sum(f[4] for f in frames)
        return {
            # Frames finished per second of wall-clock time.
            "fps": round((len(frames) - 1) / span, 2) if span > 0 else None,
            "pipeline_ms": round(float(np.mean([f[1]["total"] for f in frames])), 1),
            # From the moment the frame arrived from the camera to the output being ready.
            "end_to_end_ms": round(float(np.mean([f[2] for f in frames])), 1),
            # GPU time of the reenactment stage, over the frames that were reenacted.
            "render_ms": round(float(np.mean([f[1]["render"] for f in live])), 1) if live else None,
            # Where an average frame spends its time. A frame that was not reenacted
            # spends nothing on motion, render and compose, so these add up to pipeline_ms.
            "stages_ms": {stage: round(float(np.mean([f[1].get(stage, 0.0) for f in frames])), 1)
                          for stage in STAGES},
            # Share of camera frames the pipeline was too busy to take.
            "dropped_share": round(sum(f[4] for f in frames) / produced, 3),
            "live_share": round(len(live) / len(frames), 3),
        }
