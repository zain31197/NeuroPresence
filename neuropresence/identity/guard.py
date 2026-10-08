"""Acting on the identity score: what to do when the output stops looking like the enrolled picture.

The limits come from a measurement (scripts/study_identity_guard.py, results/identity_guard_study.json,
9 October 2026), on the sample clips:

  normal use, 1,128 frames of 8 pairs     mean 0.89; the lowest single frame 0.72, none below 0.70;
                                           the lowest mean of any three seconds 0.82
  head pushed 30 / 45 degrees past its     mean 0.73 / 0.55: the face is foreshortened, then the top of
  range, with the head range switched off  the head is all that shows
  head turned 45 degrees further           mean 0.65: a profile on shoulders that face the camera
  the same two with the head range on      mean 0.85 and 0.83: the range keeps the face recognisable
  mouth covered in the camera picture      mean 0.72: the mouth is drawn smeared
  expression three times too strong        mean 0.79, with single frames down to 0.43

So single frames cannot be judged: in normal use 9% of them are below the 0.80 the project aims
for on average, and a strong expression dips for a frame or two. What separates a broken output
from a normal one is the mean over a few seconds, and the level between the two groups is 0.75:
no three seconds of normal use average below 0.82, and the outputs that look wrong average 0.73
or less. Run over the recorded scores, two readings a second, this guard does nothing on any of
the eight normal pairs. With the head pushed down 45 degrees it ends on the still picture in
every run, and with the head turned 45 degrees in nine of ten, about four and a half seconds
after the fault starts. With the mouth covered it acts in about half the runs: those scores sit
on the level.

One thing a live session showed that the recordings could not (9 October 2026): a fresh neutral
pose taken while the mouth was covered made the cover part of the neutral pose. The output looked
right while the cover lasted (0.94), and was off once it was taken away (0.80, down from 0.86
before). A fresh neutral pose that helped is therefore watched: if the score then steps down
again, the neutral pose is taken once more.
"""

from collections import deque
from enum import Enum

LOW_CSIM = 0.75  # the mean of the last few seconds below this: the output no longer looks right
COLLAPSE_CSIM = 0.50  # readings below this in a row: do not wait for the mean
COLLAPSE_READINGS = 2
WINDOW_SECONDS = 3.0  # how much of the recent past the mean is taken over
MIN_READINGS = 4  # and how many readings it needs before it counts
# An anchor that worked and a second drop soon after it: the fresh neutral pose was not the cure.
RETRY_SECONDS = 30.0
# After an anchor that worked, a mean this far under the one it reached means that what was wrong
# when the neutral pose was taken has since gone, and has left the neutral pose wrong.
FOLLOW_STEP = 0.08


class GuardState(str, Enum):
    STEADY = "steady"  # the output looks like the picture
    DIPPING = "dipping"  # the newest readings are low, but not for long enough to act on
    ANCHORING = "anchoring"  # a fresh neutral pose was taken; waiting to see whether it helped
    FALLBACK = "fallback"  # the still picture is shown until the person resumes


class IdentityGuard:
    """Decides, from one identity reading at a time, whether anything has to be done.

    sample() returns what to do now, or None:
      "anchor"     take a fresh neutral pose and start the filters again. Either the score has been
                   low for a while (reason "low"), or it has stepped down after an earlier anchor
                   (reason "changed");
      "fallback"   that did not help, or the score collapsed: show the still picture and stay on it;
      "recovered"  the fresh neutral pose helped.
    A short dip returns nothing. Reacting to ordinary variation would disturb the picture more than
    the variation does.

    It holds no clock of its own: every reading carries its time, so it can be run on a recording.
    """

    def __init__(self, low=LOW_CSIM, collapse=COLLAPSE_CSIM, window=WINDOW_SECONDS, retry=RETRY_SECONDS):
        self.low, self.collapse, self.window, self.retry = low, collapse, window, retry
        self.reset()

    def reset(self):
        """A new session or a new picture: nothing is known yet."""
        self.state = GuardState.STEADY
        self.since = None  # when the state was entered
        self.mean = None  # the mean the last decision was taken on
        self.reason = None  # why the last anchor was asked for: "low" or "changed"
        self._readings = deque()  # (at, score) of the last `window` seconds
        self._anchored_at = None
        self._reached = None  # the mean a fresh neutral pose brought the score to, while that is watched

    def resume(self, at):
        """The person asked to go live again after a fallback. The next few seconds have to prove it."""
        self._enter(GuardState.ANCHORING, at)
        self._anchored_at, self.reason, self._reached = at, "low", None

    def _enter(self, state, at):
        self.state, self.since = state, at
        self._readings.clear()

    def _anchor(self, at, reason):
        self._enter(GuardState.ANCHORING, at)
        self._anchored_at, self.reason, self._reached = at, reason, None
        return "anchor"

    def sample(self, score, at):
        """One reading: the similarity of the live output to the picture, or None if no single
        face could be found in the output, which is as low as a reading gets."""
        if self.state is GuardState.FALLBACK:
            return None
        if self.since is None:
            self.since = at
        value = 0.0 if score is None else float(score)
        self._readings.append((at, value))
        while self._readings and at - self._readings[0][0] > self.window:
            self._readings.popleft()

        newest = [reading for _, reading in list(self._readings)[-COLLAPSE_READINGS:]]
        if len(newest) == COLLAPSE_READINGS and all(reading < self.collapse for reading in newest):
            self.mean = sum(newest) / len(newest)
            self._enter(GuardState.FALLBACK, at)
            return "fallback"

        # The mean counts once the readings cover most of the window.
        covered = len(self._readings) >= MIN_READINGS and at - self._readings[0][0] >= 0.6 * self.window
        mean = sum(reading for _, reading in self._readings) / len(self._readings)
        if not covered:
            if self.state is not GuardState.ANCHORING:
                self.state = GuardState.DIPPING if value < self.low else GuardState.STEADY
            return None
        self.mean = mean

        if self.state is GuardState.ANCHORING:
            if mean < self.low:
                self._enter(GuardState.FALLBACK, at)
                return "fallback"
            # Watch what a first anchor reached. One taken because the score stepped down is not
            # watched again, or two neutral poses could chase each other.
            self._reached = mean if self.reason == "low" else None
            self._enter(GuardState.STEADY, at)
            return "recovered"
        if mean >= self.low:
            if self._reached is not None and mean < self._reached - FOLLOW_STEP:
                return self._anchor(at, "changed")
            self.state = GuardState.DIPPING if value < self.low else GuardState.STEADY
            return None
        if self._anchored_at is not None and at - self._anchored_at < self.retry:
            self._enter(GuardState.FALLBACK, at)
            return "fallback"
        return self._anchor(at, "low")
