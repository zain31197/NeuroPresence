"""How far behind the present the movement is shown, so that it can be shown evenly.

A camera delivers 30 frames a second and the output is drawn at about 22, so each output frame is
made from the newest camera frame there is, and that frame is sometimes just in and sometimes 33 ms
old. Measured on 11 October 2026 (a live session with a 1280 x 720 camera at 30 frames a second):
the output frames came 45 ms apart, evenly, but the camera frames they showed were 33 and 67 ms
apart by turns. A head turning steadily was drawn moving in short and long steps.

So the movement is not taken from the newest frame as it is. It is read between the last two
camera frames, for a moment that lies a fixed time back (see Pipeline.step: motion_at). That time
has to be long enough for the newest frame to have been taken already, which is what this clock
works out: a little more than how old the newest frame has lately been at its worst. With a
30 frames a second camera it settles near 54 ms, where the newest frame alone was 34 ms old on
average. Measured in the same session: the moments shown by one frame and the next were then
46 ms apart to within 2 ms (16 ms before), and the movement was 100 ms old when it was shown,
steadily, where it had been 62 to 95 ms old by turns.

Since 14 October 2026 the moment may lie up to 10 ms past the newest frame (pipeline.py:
LEAD_SECONDS), which makes the lag, and so the delay, that much shorter. Measured on the six
sample clips of scripts/study_pacing.py: delay 96 to 86 ms, steps as even as before.
"""

from collections import deque

from ..pipeline import LEAD_SECONDS

START_SECONDS = 0.045  # until the first frames have been seen
WINDOW_SECONDS = 3.0  # how far back the ages of frames are looked at
MARGIN_SECONDS = 0.003
EASE = 0.05  # the share of the way the lag moves toward its goal with each frame: it must not jump
LEAST_SECONDS, MOST_SECONDS = 0.015, 0.120


class MotionClock:
    """Says, for each frame about to be drawn, which moment its movement should be of."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.lag = START_SECONDS
        self._ages = deque()  # (at, how old the newest camera frame was), of the last few seconds

    def moment(self, now, captured_at):
        """The moment whose movement the frame drawn at `now` should show, given that the newest
        camera frame was taken at `captured_at`. Never more than LEAD_SECONDS past that frame."""
        age = max(0.0, now - captured_at)
        self._ages.append((now, age))
        while self._ages and now - self._ages[0][0] > WINDOW_SECONDS:
            self._ages.popleft()
        # Nearly the oldest the newest frame has been: one late frame in twenty may fall outside.
        ages = sorted(seen for _, seen in self._ages)
        goal = ages[min(len(ages) - 1, int(0.95 * len(ages)))] + MARGIN_SECONDS - LEAD_SECONDS
        self.lag += EASE * (min(MOST_SECONDS, max(LEAST_SECONDS, goal)) - self.lag)
        return min(now - self.lag, captured_at + LEAD_SECONDS)
