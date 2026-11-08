"""A live feed of frames that always hands out the newest one.

The pipeline is slower than a camera. Reading frames one after another would
let unprocessed frames pile up, and the output would show the user's face from
further and further in the past. Here a capture thread keeps only the newest
frame: a frame the pipeline was too busy to take is dropped, never queued. This
is the frame-pacing rule of the design (stay current, never catch up later).

A recorded clip is played back at its own frame rate and treated the same way,
so it behaves like a camera pointed at a repeatable scene.

A camera that fails to give a frame is not taken for one that has stopped. Webcams, and above
all a phone used as a webcam (Camo, over Wi-Fi or USB), miss a frame now and then; before 9
November 2026 one missed frame ended the session, which in a meeting looked like a crash after a
quarter of an hour or so. Now the feed keeps asking, opens the camera again after REOPEN_SECONDS
without a frame, and gives up only after GIVE_UP_SECONDS without one.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .camera import FrameSource, FrameSourceError

DEFAULT_CLIP_FPS = 25.0
RATE_FRAMES = 60  # the source's frame rate is read from when its last frames arrived: two seconds of a webcam
RETRY_SECONDS = 0.02  # a camera that gave no frame is asked again after this
REOPEN_SECONDS = 1.0  # without a frame for this long, the camera is closed and opened again ...
REOPEN_EVERY_SECONDS = 2.0  # ... and again this often while it stays silent ...
GIVE_UP_SECONDS = 15.0  # ... until it has given nothing for this long: then it has stopped


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
        self._arrivals = deque(maxlen=RATE_FRAMES)
        self._arrived = threading.Condition()
        self._stop = threading.Event()
        self.ended = False
        self.reopened = 0  # how many times the camera was opened again after it went silent
        self._last_frame_at = time.perf_counter()
        self._thread = threading.Thread(target=self._capture, name="live-feed", daemon=True)
        self._thread.start()

    @property
    def silent_seconds(self):
        """How long it has been since the source last gave a frame."""
        return time.perf_counter() - self._last_frame_at

    def _wait_for_camera(self, reopened_at):
        """A camera gave no frame: ask again, and open it again when it stays silent. Returns when it was
        last opened, or None once it has given nothing for GIVE_UP_SECONDS."""
        now = time.perf_counter()
        if now - self._last_frame_at >= GIVE_UP_SECONDS:
            return None
        if now - self._last_frame_at >= REOPEN_SECONDS and now - reopened_at >= REOPEN_EVERY_SECONDS:
            self._frames.release()
            try:
                self._frames = FrameSource(self._source, *self._size)
            except FrameSourceError:  # not back yet: the old one stays released, and reading it gives nothing
                pass
            self.reopened += 1
            return now
        self._stop.wait(RETRY_SECONDS)
        return reopened_at

    def _capture(self):
        index, due, just_rewound = 0, time.perf_counter(), False
        reopened_at = 0.0
        try:
            while not self._stop.is_set():
                frame = self._frames.read()
                if frame is None:
                    if self.is_camera:  # a frame missed is not a camera stopped (see the top of this file)
                        reopened_at = self._wait_for_camera(reopened_at)
                        if reopened_at is None:
                            break
                        continue
                    # A clip starts again from the top.
                    if not self._loop or just_rewound:
                        break
                    self._frames.release()
                    self._frames = FrameSource(self._source, *self._size)
                    just_rewound = True
                    continue
                just_rewound = False
                self._last_frame_at = time.perf_counter()
                if self._period is not None:
                    due += self._period
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    else:
                        due = time.perf_counter()  # decoding fell behind; do not try to catch up
                with self._arrived:
                    self._newest = FeedFrame(frame, index, time.perf_counter())
                    self._arrivals.append(self._newest.captured_at)
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

    @property
    def rate(self):
        """How many frames a second the source is delivering, or None before a few have arrived.

        The output cannot be faster than the camera, and a webcam in poor light often drops to
        15 frames a second or fewer by itself, so this is worth showing beside the output's rate.
        """
        with self._arrived:
            arrivals = list(self._arrivals)
        if len(arrivals) < 5 or arrivals[-1] <= arrivals[0]:
            return None
        return (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
