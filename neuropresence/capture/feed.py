"""A live feed of frames that always hands out the newest one.

The pipeline is slower than a camera. Reading frames one after another would
let unprocessed frames pile up, and the output would show the user's face from
further and further in the past. Here a capture thread keeps only the newest
frame: a frame the pipeline was too busy to take is dropped, never queued. This
is the frame-pacing rule of the design (stay current, never catch up later).

A recorded clip is played back at its own frame rate and treated the same way,
so it behaves like a camera pointed at a repeatable scene.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np

from .camera import FrameSource

DEFAULT_CLIP_FPS = 25.0


@dataclass
class FeedFrame:
    image: np.ndarray
    index: int  # counts every frame the source produced, including dropped ones
    captured_at: float  # time.perf_counter() when the frame arrived


class LiveFeed:
    def __init__(self, source=0, width=1280, height=720, loop=True):
        self._source, self._size, self._loop = source, (width, height), loop
        self._frames = FrameSource(source, width, height)  # raises FrameSourceError
        self.is_camera = self._frames.is_camera
        self._period = None if self.is_camera else 1.0 / (self._frames.fps or DEFAULT_CLIP_FPS)
        self._newest = None
        self._arrived = threading.Condition()
        self._stop = threading.Event()
        self.ended = False
        self._thread = threading.Thread(target=self._capture, name="live-feed", daemon=True)
        self._thread.start()

    def _capture(self):
        index, due, just_rewound = 0, time.perf_counter(), False
        try:
            while not self._stop.is_set():
                frame = self._frames.read()
                if frame is None:
                    # A camera that stops is an error; a clip starts again from the top.
                    if self.is_camera or not self._loop or just_rewound:
                        break
                    self._frames.release()
                    self._frames = FrameSource(self._source, *self._size)
                    just_rewound = True
                    continue
                just_rewound = False
                if self._period is not None:
                    due += self._period
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    else:
                        due = time.perf_counter()  # decoding fell behind; do not try to catch up
                with self._arrived:
                    self._newest = FeedFrame(frame, index, time.perf_counter())
                    self._arrived.notify_all()
                index += 1
        finally:
            self._frames.release()
            with self._arrived:
                self.ended = True
                self._arrived.notify_all()

    def read(self, after_index=-1, timeout=1.0):
        """Wait for a frame newer than after_index.

        Returns None if none arrives within timeout or the source has ended.
        Frames between after_index and the returned one were dropped.
        """
        deadline = time.perf_counter() + timeout
        with self._arrived:
            while True:
                if self._newest is not None and self._newest.index > after_index:
                    return self._newest
                remaining = deadline - time.perf_counter()
                if self.ended or remaining <= 0:
                    return None
                self._arrived.wait(remaining)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
