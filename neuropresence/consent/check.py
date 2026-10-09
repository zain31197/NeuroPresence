"""The consent check: a picture is animated only for the person it belongs to.

Two things are asked of whoever sits at the camera. That they are a living person answering now,
which is the liveness prompt (liveness.py). And that they are the registered person, which is the
face match: the face in the camera picture is compared with every signature registered for the
user, and the best of them decides (0.35 or more is the same person, see identity/scorer.py).

The two are tied together. The face is matched about four times a second all the way through the
prompt, so the face that answers is the face that matches: a photograph of the user cannot be
held up for the match while someone else does the moving. Measured on 10 October 2026
(scripts/study_liveness.py, results/liveness_study.json): scored against its own first frame, no
face of the same person fell below 0.49 in 1,474 frames of the sample clips, through blinks,
talking and turns; scored against another person's first frame, none rose above 0.23.

After the session has started the camera face is still compared once a second (ConsentWatch): the
person who passed the check may get up and someone else sit down.
"""

from ..identity.scorer import SAME_PERSON_CSIM
from .liveness import Stage

SCORE_SECONDS = 0.25  # how often the face is matched during the prompt
MISSES_TO_REFUSE = 2  # a face that does not match is looked at again in the next frame; two in a row refuse
MIN_MATCHES = 3  # the face has to be matched at least this often before the check can pass
CONFIRM_SECONDS = 2.0  # after the last action the face is matched once more, within this long

NOT_THE_VERIFIED_FACE = ("The face at the camera is not the face that was verified, so the session was not started. "
                         "A picture is animated only for the person it belongs to.")
THE_FACE_CHANGED = "The face at the camera changed during the check. Only you should be in front of the camera."


class ConsentCheck:
    """The liveness prompt and the face match together. It is given every frame's reading, and the
    face signature of the frames it asks for, and ends as passed or refused.

    `signatures` are those registered for the user. Without any, which is the case while a face is
    being verified for the first time, the first face seen is the one all later ones have to match.
    """

    def __init__(self, challenge, signatures=None, similarity=None, limit=SAME_PERSON_CSIM):
        self.challenge = challenge
        self.registered = bool(signatures)
        self.signatures = list(signatures or [])
        self._similarity = similarity or (lambda a, b: float(a @ b))
        self.limit = limit
        self.passed = None  # None while it runs, then True or False
        self.failure = None  # in one word: one of the challenge's, or "identity"
        self.reason = ""
        self.match = None  # how alike the last face scored was
        self.lowest = None  # the least alike of all the faces scored
        self.matches = 0
        self._misses = 0
        self._scored_at = None
        self._again = False  # score the very next frame
        self._passed_at = None  # when the prompt was passed; one more match is needed after it

    @property
    def done(self):
        return self.passed is not None

    def frame(self, reading, at):
        """One frame's reading (liveness.read), taken at `at` seconds on any clock."""
        if self.done:
            return
        if not reading.ok:
            self._again = True  # whoever is there when the face is back is matched at once
        stage = self.challenge.sample(reading, at)
        if stage is Stage.FAILED:
            self._refuse(self.challenge.failure, self.challenge.reason)
        elif stage is Stage.PASSED:
            if self._passed_at is None:
                self._passed_at, self._again = at, True
            elif at - self._passed_at > CONFIRM_SECONDS:
                self._refuse("identity", "Your face could not be matched after the last action. Try again.")

    def wants_face(self, at):
        """Should the face in the frame taken at `at` be scored?"""
        return not self.done and (self._again or self._scored_at is None or at - self._scored_at >= SCORE_SECONDS)

    def face(self, signature, at):
        """The signature of the face in the frame taken at `at`, when wants_face() asked for it."""
        if self.done or signature is None:
            return
        self._scored_at, self._again = at, False
        if not self.signatures:
            self.signatures = [signature]  # the face being verified: all later faces are matched with this one
        self.match = max(self._similarity(known, signature) for known in self.signatures)
        self.lowest = self.match if self.lowest is None else min(self.lowest, self.match)
        if self.match < self.limit:
            self._misses += 1
            self._again = True
            if self._misses >= MISSES_TO_REFUSE:
                self._refuse("identity", NOT_THE_VERIFIED_FACE if self.registered else THE_FACE_CHANGED)
            return
        self._misses = 0
        self.matches += 1
        if self._passed_at is not None and at >= self._passed_at and self.matches >= MIN_MATCHES:
            self.passed = True

    def same_face(self, signature):
        """Is this the face the check was made on? For a picture taken after it has passed."""
        return bool(self.signatures) and signature is not None and max(
            self._similarity(known, signature) for known in self.signatures) >= self.limit

    def _refuse(self, failure, reason):
        self.passed, self.failure, self.reason = False, failure, reason

    def snapshot(self, at):
        """How the check stands, for the app to show."""
        state = self.challenge.snapshot(at)
        if self.passed is None and self.challenge.stage is Stage.PASSED:
            state["stage"], state["prompt"] = "confirming", "Hold still for a moment"
        elif self.passed is not None:
            state["stage"], state["prompt"] = ("passed" if self.passed else "refused"), ""
        state["reason"] = self.reason
        state["failure"] = self.failure
        state["match"] = None if self.match is None else round(self.match, 3)
        return state


# ---------------------------------------------------------- during a session

WATCH_SECONDS = 1.0  # how often the camera face is compared during a session
MISSES_TO_HOLD = 3  # this many in a row that do not match: the still picture is shown
MATCHES_TO_RELEASE = 2  # this many in a row that match again: the live picture returns


class ConsentWatch:
    """Says, from one face match at a time, whether the output has to be held back because the
    person at the camera is no longer the registered one."""

    def __init__(self, limit=SAME_PERSON_CSIM):
        self.limit = limit
        self.reset()

    def reset(self):
        self.held = False
        self.match = None  # how alike the last face compared was
        self._run = 0  # how many readings in a row have gone against the present state

    def sample(self, match):
        """One comparison of the camera face with the registered signatures. Returns "hold" when the
        output has to be held back from now on, "release" when it may return, or None."""
        self.match = match
        against = match >= self.limit if self.held else match < self.limit
        self._run = self._run + 1 if against else 0
        if self._run < (MATCHES_TO_RELEASE if self.held else MISSES_TO_HOLD):
            return None
        self.held, self._run = not self.held, 0
        return "hold" if self.held else "release"
