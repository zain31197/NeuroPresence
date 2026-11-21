"""A limit on repeated attempts at the consent check.

The liveness prompt is passed by a recording made on purpose about one attempt in seven (consent/liveness.py,
10 October 2026), and attempts were unlimited, so someone with such a recording could simply try until it got
through. The mid report promised to limit repeated attempts; this is that limit.

A few checks that do not pass cost nothing: a person can miss an action, look away, or be shown a photograph
during a demonstration. After that every further one that does not pass makes the next attempt wait, twice as
long each time. A check that passes forgets them all, and so does an hour without one.

What it buys, worked out for a recording that passes one attempt in seven: without the limit the expected wait
is seven attempts of about fifteen seconds each, under two minutes. With it, the first three attempts are free
and the attempts after them wait 1, 2, 4 and 8 minutes, about a quarter of an hour in all. It slows such an
attempt down; it does not stop it, and that limit is stated in the report. Until 21 November 2026 the counts
were kept in memory only and a restart of the engine cleared them; now they are read back from the database
(the consent_checks table) when the engine starts.
"""

import threading
import time

FREE_FAILURES = 3  # checks that do not pass before any waiting starts
FIRST_WAIT_SECONDS = 60.0  # the wait after the first failure beyond those, doubled after each further one
LONGEST_WAIT_SECONDS = 900.0
FORGET_SECONDS = 3600.0  # failures older than this no longer count


class Attempts:
    """Counts the consent checks that did not pass, and says how long the next one has to wait."""

    def __init__(self, clock=time.time, failures=()):
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = list(failures)  # when each check that did not pass ended, seconds since 1970

    def failed(self):
        with self._lock:
            self._failures.append(self._clock())

    def passed(self):
        with self._lock:
            self._failures.clear()

    def _recent(self, now):
        self._failures = [at for at in self._failures if now - at < FORGET_SECONDS]
        return self._failures

    def wait_seconds(self):
        """How long until the next check may start: 0 when it may start now."""
        with self._lock:
            now = self._clock()
            failures = self._recent(now)
            if len(failures) < FREE_FAILURES:
                return 0.0
            wait = min(FIRST_WAIT_SECONDS * 2 ** (len(failures) - FREE_FAILURES), LONGEST_WAIT_SECONDS)
            return max(0.0, failures[-1] + wait - now)

    def snapshot(self):
        with self._lock:
            failures = len(self._recent(self._clock()))
        return {"failures": failures, "free": FREE_FAILURES, "wait_seconds": round(self.wait_seconds(), 1)}

    def refusal(self):
        """The words for an attempt made too soon, or None when it may go ahead."""
        wait = self.wait_seconds()
        if wait <= 0.0:
            return None
        failures = self.snapshot()["failures"]
        return (f"{failures} checks in a row did not pass. So that a photograph or a recording cannot simply be "
                f"tried again and again, the next check can start in {_duration(wait)}.")


def _duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"
