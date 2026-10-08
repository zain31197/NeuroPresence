"""The delay watchdog: when the output is too late to show.

A picture that arrives late shows lips that move after the voice. Above the limit the still
picture is the better thing to show. Measured on 9 October 2026 (scripts/study_delay.py,
results/delay_study.json), as the mean delay over two seconds in a live session:

  alone                                     79 to 88 ms
  GPU busy elsewhere half of the time       105 to 116 ms
  GPU busy elsewhere 75% of the time        126 to 142 ms, with single frames up to 173 ms
  GPU busy elsewhere all of the time        162 to 183 ms

Late frames are dropped and never queued, so under load the frame rate falls first (20, 13, 10,
7 frames a second in the four cases) and the delay follows slowly. Single frames pass the limit
long before the session as a whole does, so the watchdog reads the mean of two seconds, not frames.
"""

from collections import deque

WINDOW_SECONDS = 2.0
MIN_FRAMES = 5  # a mean over fewer frames than this says nothing yet
# Back under 90% of the limit, and for this long, before the live picture returns: a delay
# that hovers at the limit would otherwise switch the picture back and forth.
RELEASE_SHARE = 0.9
CLEAR_SECONDS = 1.0


class DelayWatchdog:
    """Says, from one frame's delay at a time, whether the live picture should be held back."""

    def __init__(self, limit_ms, window=WINDOW_SECONDS, release_share=RELEASE_SHARE, clear_seconds=CLEAR_SECONDS):
        self.limit_ms, self.window = limit_ms, window
        self.release_ms, self.clear_seconds = release_share * limit_ms, clear_seconds
        self.reset()

    def reset(self):
        self.held = False
        self.mean_ms = None
        self._frames = deque()  # (at, delay in ms) of the last `window` seconds
        self._clear_since = None

    def sample(self, delay_ms, at):
        """One frame's delay from camera to output, in ms, finished at `at` seconds. Returns whether
        the live picture is held back from now on."""
        self._frames.append((at, delay_ms))
        while self._frames and at - self._frames[0][0] > self.window:
            self._frames.popleft()
        if len(self._frames) < MIN_FRAMES or at - self._frames[0][0] < 0.5 * self.window:
            return self.held
        self.mean_ms = sum(delay for _, delay in self._frames) / len(self._frames)
        if not self.held:
            if self.mean_ms > self.limit_ms:
                self.held, self._clear_since = True, None
        elif self.mean_ms >= self.release_ms:
            self._clear_since = None
        else:
            if self._clear_since is None:
                self._clear_since = at
            if at - self._clear_since >= self.clear_seconds:
                self.held = False
        return self.held
