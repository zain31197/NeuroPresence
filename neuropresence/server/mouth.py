"""Does the picture's mouth follow the person's? Watched while a session runs, so that a mouth that has stopped
following is put right and said so in the event list, whatever made it stop.

In a Google Meet call on 22 November 2026 (Zain's recording) the picture's mouth stood open for about ten seconds
while he listened, and stayed shut in two places where Meet heard him speak. The cause could not be read from a
recording of the output alone: the engine kept no record of the person's own lips. Replaying his recordings through
the live pipeline did not show it either (correlation 0.92, open by mistake 0.6% of the time). So the lips at the
camera and in the output are now measured while the session runs (the output twice a second, by the identity check,
which reads that face anyway) and compared here. When they disagree for long enough the neutral mouth is put right:
movement is measured from it, so a neutral mouth that has drifted is what a stuck mouth comes down to.

Both gaps are the distance between the lips over the distance between the eyes (pipeline.lip_gap).
"""

from collections import deque

CLOSED = 0.015  # at or under this the lips are together (at rest he reads 0.002 to 0.009)
OPEN = 0.035  # at or over this the output's mouth is clearly open
SPEAKING = 0.04  # the upper quarter of the camera's last seconds is at or over this while he speaks
SAMPLES_OPEN = 3  # output readings in a row (1.5 s) with the mouth open while the camera's is together
SAMPLES_SHUT = 8  # output readings in a row (4 s) with the mouth together while the camera's speaks
WINDOW_SECONDS = 3.0  # how far back the camera's own lips are looked at
REST_SECONDS = 8.0  # between two repairs, so one that did not help is not repeated at once
# The person's lips last read longer ago than this: covered by a hand (capture/cover.py), or the face lost. There is
# nothing to compare the output with, and the readings from before say nothing about now: on 27 November 2026 a
# hand held over a talking mouth for 10 s was taken for a mouth that stayed shut while its owner spoke.
STALE_SECONDS = 0.5


class MouthWatch:
    """Compares the lips at the camera with the lips in the output. Not safe for two threads: the caller locks."""

    def __init__(self, window=WINDOW_SECONDS):
        self._window = window
        self.reset()

    def reset(self):
        self._camera = deque()  # (at, gap), of every frame in the last seconds
        self._output = deque(maxlen=SAMPLES_SHUT)
        self._last_repair = None
        self.camera_gap = self.output_gap = None  # the last readings, for the session's figures
        self.repairs = 0

    def camera(self, at, gap):
        """One reading of the person's lips, every drawn frame."""
        if gap is None:
            return
        self.camera_gap = gap
        self._camera.append((at, gap))
        while self._camera and at - self._camera[0][0] > self._window:
            self._camera.popleft()

    def output(self, at, gap):
        """One reading of the output's lips (twice a second). Returns what to repair: "open" when the output's mouth
        stands open while the person's is closed, "shut" when it stays shut while they speak, or None."""
        if gap is None:
            return None
        self.output_gap = gap
        if not self._camera or at - self._camera[-1][0] > STALE_SECONDS:
            self._output.clear()  # the output's run counts only beside fresh readings of the person
            return None
        self._output.append(gap)
        if self._last_repair is not None and at - self._last_repair < REST_SECONDS:
            return None
        gaps = sorted(g for _, g in self._camera)
        if len(gaps) < 10 or self._camera[-1][0] - self._camera[0][0] < 0.6 * self._window:
            return None
        highest, upper = gaps[-1], gaps[int(0.75 * (len(gaps) - 1))]
        recent = list(self._output)
        if len(recent) >= SAMPLES_OPEN and highest <= CLOSED and min(recent[-SAMPLES_OPEN:]) >= OPEN:
            return self._repair(at, "open")
        if len(recent) >= SAMPLES_SHUT and upper >= SPEAKING and max(recent) <= CLOSED:
            return self._repair(at, "shut")
        return None

    def _repair(self, at, what):
        self._last_repair = at
        self.repairs += 1
        self._output.clear()
        return what
