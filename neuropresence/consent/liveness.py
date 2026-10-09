"""The liveness prompt: is a living person at the camera, answering now?

A photograph of the enrolled person matches their face signature as well as they do, and so
does a recording. So before a face is verified, and before every camera session, the person is
asked to do two things chosen at random, one after the other, each within three seconds: blink,
turn the head to the left, turn it to the right, or open the mouth. What was asked for cannot be
known in advance, which is what a recording cannot answer; a photograph cannot do any of it.

Measured on 10 October 2026 (scripts/study_liveness.py, results/liveness_study.json):

  A photograph tilted or swung in front of the camera, by up to 65 degrees   reads as a head turn of 32.5 degrees at most
    the same, unless the camera sees wider than most webcams and the
    photograph fills its whole picture                                       28 degrees at most
  A photograph with the eyes covered                                         the eye score stays at 0.24 or under
  A dark shape drawn over the mouth of a photograph                          reads as a mouth open by 0.42 at most
  Mouths in the sample clips while talking                                   pass 0.6 in 57 places, and stay there 0.3 s in 11

So a turn counts from 35 degrees, and one of the two actions is always a turn: it is the one thing
a flat picture does not show, however it is held, and with holes cut for the eyes and the mouth
too. Blinking and opening the mouth happen by themselves in any recording of a person talking, so
they are the second action and never the only ones.

What gets through, with these limits. A photograph: none of 13,296 attempts, held still or swung.
The twelve sample clips played back as recordings, from every half second and for every order of
actions: none of 3,110 attempts (one of 622 if no turn were asked for). A recording made on
purpose, with every action in it one after another: about one attempt in nine, which is what
asking at random leaves to a recording that holds the right answer somewhere. A stand-in that does
what is asked passes every time, in under five seconds on average.

The challenge keeps no clock of its own: it is given each reading with its time, so it can be run
over a recording.
"""

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median

from ..capture import TrackStatus

ACTIONS = ("blink", "turn_left", "turn_right", "open_mouth")
TURNS = ("turn_left", "turn_right")
ACTIONS_ASKED = 2

# What the person is told, and how a missing action is named when the time runs out.
PROMPTS = {
    "blink": "Blink",
    "turn_left": "Turn your head to your left",
    "turn_right": "Turn your head to your right",
    "open_mouth": "Open your mouth wide",
}
MISSED = {
    "blink": "No blink was seen",
    "turn_left": "Your head did not turn to your left",
    "turn_right": "Your head did not turn to your right",
    "open_mouth": "Your mouth did not open wide",
}
FACE_THE_CAMERA = "Face the camera, with your mouth closed"
FACE_IT_AGAIN = "Good. Face the camera again"

ACTION_SECONDS = 3.0  # each action has to be seen within this long of being asked for
SETTLE_SECONDS = 8.0  # to be seen at rest: before the first action, and again after each
REST_SECONDS = 0.5  # at rest for this long before the next action is asked for ...
WAIT_SECONDS = (0.3, 1.2)  # ... and a wait of a random length on top, so the moment cannot be known either
LOST_SECONDS = 0.5  # no face, or two, for longer than this ends the challenge

# At rest: facing the camera, mouth closed, eyes open. Every action is measured from there.
REST_YAW_DEG = 12.0
REST_JAW = 0.30
TURN_DEG = 35.0  # a turn of the head, from where it was at rest
OPEN_JAW = 0.60  # the mouth counts as wide open from here (MediaPipe jawOpen), twice what it may be at rest ...
OPEN_HOLD_SECONDS = 0.3  # ... once it has stayed there this long: a word is shorter
EYES_CLOSED = 0.50  # both eyes at this score (MediaPipe eyeBlink) are closed; open eyes scored up to 0.43 ...
CLOSED_RISE = 0.25  # ... and this far above where they were at rest, because open eyes differ between people
REOPEN_RISE = 0.12  # open again: back within this of where they were at rest
BLINK_MAX_SECONDS = 1.0  # eyes kept shut for longer are not a blink


class Stage(str, Enum):
    SETTLE = "settle"  # waiting to see the face at rest
    ACT = "act"  # an action has been asked for
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class Reading:
    """What the challenge looks at in one tracked frame."""

    status: TrackStatus
    yaw: float = 0.0  # degrees; positive is the person's own left
    jaw: float = 0.0  # how far the mouth is open, 0 to 1
    eyes: float = 0.0  # how closed the more open eye is, 0 to 1: a blink closes both

    @property
    def ok(self):
        return self.status is TrackStatus.OK


def read(track):
    """The reading of a tracked frame (see capture/tracker.py)."""
    if not track.ok:
        return Reading(track.status)
    scores = track.blendshapes or {}
    return Reading(track.status, yaw=float(track.pose_deg[0]), jaw=float(scores.get("jawOpen", 0.0)),
                   eyes=float(min(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))))


def at_rest(reading):
    return (reading.ok and abs(reading.yaw) <= REST_YAW_DEG and reading.jaw <= REST_JAW
            and reading.eyes < EYES_CLOSED)


def pick_actions(rng=None, count=None):
    """The actions to ask for, in order: chosen at random, and one of them always a turn of the head."""
    rng = rng or random.SystemRandom()
    count = ACTIONS_ASKED if count is None else count
    if count <= 0:
        return ()
    while True:
        chosen = tuple(rng.sample(ACTIONS, count))
        if any(action in TURNS for action in chosen):
            return chosen


def pick_waits(count, rng=None):
    """How long to wait at rest before each action is asked for."""
    rng = rng or random.SystemRandom()
    return tuple(rng.uniform(*WAIT_SECONDS) for _ in range(count))


class LivenessChallenge:
    """Asks for the given actions one at a time and says, from one reading at a time, how it stands."""

    def __init__(self, actions=None, waits=None, action_seconds=None, settle_seconds=None, rest_seconds=None):
        self.actions = pick_actions() if actions is None else tuple(actions)
        self.waits = pick_waits(len(self.actions)) if waits is None else tuple(waits)
        self.action_seconds = ACTION_SECONDS if action_seconds is None else action_seconds
        self.settle_seconds = SETTLE_SECONDS if settle_seconds is None else settle_seconds
        self.rest_seconds = REST_SECONDS if rest_seconds is None else rest_seconds
        self.stage = Stage.SETTLE
        self.step = 0  # how many of the actions have been seen
        self.reason = ""  # why it failed, in words for the person ...
        self.failure = None  # ... and in one word: "no_face", "faces", "rest", "wrong" or "late"
        self._phase_at = None  # when the current wait for rest began
        self._deadline = None  # by when the action asked for has to be seen
        self._rest = deque()  # the readings of the unbroken rest so far, as (at, reading)
        self._base = None  # the reading at rest that the action asked for is measured from
        self._gap_since = self._open_since = self._closed_since = None
        self._turned = 0.0  # how far the head has turned the way asked, as a share of a full turn, for the app

    @property
    def done(self):
        return self.stage in (Stage.PASSED, Stage.FAILED)

    @property
    def action(self):
        """The action being asked for right now, or None."""
        return self.actions[self.step] if self.stage is Stage.ACT else None

    def sample(self, reading, at):
        """One reading, taken at `at` seconds on any clock. Returns the stage the challenge is in."""
        if self.done:
            return self.stage
        if self._phase_at is None:
            self._phase_at = at
        if not reading.ok:
            self._rest.clear()
            self._open_since = self._closed_since = None
            self._gap_since = at if self._gap_since is None else self._gap_since
            if at - self._gap_since > LOST_SECONDS:
                if reading.status is TrackStatus.MULTIPLE_FACES:
                    return self._fail("faces", "More than one face is in the picture. Only you should be in view.")
                return self._fail("no_face", "Your face left the picture. Stay in view of the camera.")
        else:
            self._gap_since = None
        if self.stage is Stage.SETTLE:
            self._settle(reading, at)
        else:
            self._act(reading, at)
        return self.stage

    def snapshot(self, at):
        """How the challenge stands, for the app to show."""
        acting = self.stage is Stage.ACT
        if acting:
            hint = PROMPTS[self.actions[self.step]]
        elif self.stage is Stage.SETTLE:
            hint = FACE_IT_AGAIN if self.step else FACE_THE_CAMERA
        else:
            hint = ""
        return {
            "stage": self.stage.value,
            "step": min(self.step + 1, len(self.actions)),
            "steps": len(self.actions),
            "action": self.actions[self.step] if acting else None,
            "prompt": hint,
            "seconds": self.action_seconds,
            "seconds_left": round(max(0.0, self._deadline - at), 1) if acting else None,
            # For a turn: how much of it has been seen, 0 to 1, so the app can show how far there is to go.
            "progress": round(self._turned, 2) if acting and self.actions[self.step] in TURNS else None,
            "reason": self.reason,
        }

    # ------------------------------------------------------------- the stages

    def _fail(self, failure, reason):
        self.stage, self.failure, self.reason = Stage.FAILED, failure, reason
        return self.stage

    def _settle(self, reading, at):
        if at_rest(reading):
            self._rest.append((at, reading))
        else:
            self._rest.clear()
        last = self.step == len(self.actions)
        needed = self.rest_seconds + (0.0 if last else self.waits[self.step])
        if self._rest and at - self._rest[0][0] >= needed:
            if last:
                self.stage = Stage.PASSED
                return
            # The action is measured from how the face was in the last moments of the rest.
            recent = [r for seen, r in self._rest if at - seen <= max(self.rest_seconds, 0.2)] or [self._rest[-1][1]]
            self._base = Reading(TrackStatus.OK, yaw=median(r.yaw for r in recent), jaw=median(r.jaw for r in recent),
                                 eyes=median(r.eyes for r in recent))
            self.stage, self._deadline = Stage.ACT, at + self.action_seconds
            self._open_since = self._closed_since = None
            self._turned = 0.0
        elif at - self._phase_at > self.settle_seconds:
            self._fail("rest", "You were not seen facing the camera with your mouth closed and your eyes open. "
                       "Sit facing the camera and try again.")

    def _act(self, reading, at):
        asked = self.actions[self.step]
        seen = self._seen(reading, at) if reading.ok else None
        if seen == asked:
            self.step += 1
            self.stage, self._phase_at = Stage.SETTLE, at
            self._rest.clear()
        elif seen is not None and seen != "blink":  # people blink all the time: a blink is never the wrong answer
            self._fail("wrong", f"{MISSED[asked]}: something else was done.")
        elif at > self._deadline:
            self._fail("late", f"{MISSED[asked]} within {self.action_seconds:g} seconds.")

    def _seen(self, reading, at):
        """The action this reading completes, or None."""
        base = self._base
        turn = reading.yaw - base.yaw
        asked = self.actions[self.step]
        if asked in TURNS:
            self._turned = min(1.0, max(0.0, turn if asked == "turn_left" else -turn) / TURN_DEG)
        if turn >= TURN_DEG:
            return "turn_left"
        if turn <= -TURN_DEG:
            return "turn_right"
        if reading.jaw >= OPEN_JAW:
            self._open_since = at if self._open_since is None else self._open_since
            if at - self._open_since >= OPEN_HOLD_SECONDS:
                return "open_mouth"
        else:
            self._open_since = None
        if reading.eyes >= max(EYES_CLOSED, base.eyes + CLOSED_RISE):
            self._closed_since = at if self._closed_since is None else self._closed_since
        elif self._closed_since is not None and reading.eyes <= base.eyes + REOPEN_RISE:
            closed_for, self._closed_since = at - self._closed_since, None
            if closed_for <= BLINK_MAX_SECONDS:
                return "blink"
        return None
