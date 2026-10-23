"""The movement to draw when the voice drives the picture: the mouth, blinks, and a little life in the head.

The engine draws relative movement: how far a driving face has moved from its neutral pose. With the
camera off there is no driving face, so the movement is made here and handed to the engine as a
reading, exactly as the motion network would have handed it over (engine.drive(..., motion=...)).
The neutral pose is the picture's own: drawn with no change, the picture comes out as it is.

Three things are added to it:
- the mouth, along the direction in which LivePortrait's lip keypoints move when a mouth opens,
  as far as the voice says (mouth.py);
- a blink every few seconds, along the direction the eye keypoints move in a blink;
- a slow sway of the head of a degree or so, and a slight nod with loud speech. A head that never
  moves while the lips talk looks like a photograph with a moving mouth.

The two directions were fitted on LivePortrait's bundled driving clips (directions.json).
"""

import json
import math
import random
from pathlib import Path

import torch

DIRECTIONS = Path(__file__).with_name("directions.json")
MOUTH_GAP = 0.22  # a fully open mouth, as lip gap over eye span: a wide vowel in the clips the directions come from
BLINK_DEPTH = 0.75  # a full blink, in the tracker's blink score
BLINK_SECONDS = 0.22  # how long a blink takes, closing and opening
BLINK_EVERY = (2.5, 6.0)  # seconds between blinks, chosen at random in this range
SWAY_DEG = {"yaw": 1.4, "pitch": 0.9, "roll": 0.6}  # the head's slow sway, at most
SWAY_PERIODS = (7.3, 11.9)  # two slow waves added together, in seconds, so the sway never repeats exactly
NOD_DEG = 1.2  # how far the head dips with the loudest speech


def load_directions(path=DIRECTIONS):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return (torch.tensor(data["open"], dtype=torch.float32).reshape(1, 21, 3),
            torch.tensor(data["blink"], dtype=torch.float32).reshape(1, 21, 3))


class SpeechMotion:
    """Readings for engine.drive from how open the mouth should be, frame by frame.

    `neutral` is the picture's own reading and `rotation` turns pitch, yaw and roll into the engine's
    rotation (the engine's own function). `seed` makes the blinks and the sway repeat, for tests.
    """

    def __init__(self, neutral, rotation, seed=None, directions=None):
        self.neutral = neutral
        self._rotation = rotation
        self.open_direction, self.blink_direction = directions or load_directions()
        self._random = random.Random(seed)
        self._phases = {axis: (self._random.uniform(0, 2 * math.pi), self._random.uniform(0, 2 * math.pi))
                        for axis in SWAY_DEG}
        self._next_blink = None
        self._blink_start = None

    def blink(self, at):
        """How closed the eyes are at `at`: 0 open, 1 a full blink."""
        if self._next_blink is None:
            self._next_blink = at + self._random.uniform(*BLINK_EVERY)
        if self._blink_start is None and at >= self._next_blink:
            self._blink_start = at
        if self._blink_start is None:
            return 0.0
        into = (at - self._blink_start) / BLINK_SECONDS
        if into >= 1.0:
            self._blink_start = None
            self._next_blink = at + self._random.uniform(*BLINK_EVERY)
            return 0.0
        return math.sin(math.pi * into)  # closes and opens again

    def sway(self, at):
        """The head's slow turn away from the picture's own pose, in degrees per axis."""
        out = {}
        for axis, most in SWAY_DEG.items():
            first, second = self._phases[axis]
            wave = 0.6 * math.sin(2 * math.pi * at / SWAY_PERIODS[0] + first) + 0.4 * math.sin(2 * math.pi * at / SWAY_PERIODS[1] + second)
            out[axis] = most * wave
        return out

    def reading(self, openness, at):
        """The reading to hand to engine.drive(motion=...) for a mouth this open at time `at` (seconds)."""
        info = dict(self.neutral)
        info["exp"] = (self.neutral["exp"] + openness * MOUTH_GAP * self.open_direction
                       + self.blink(at) * BLINK_DEPTH * self.blink_direction)
        turn = self.sway(at)
        turn["pitch"] += NOD_DEG * openness
        for axis, degrees in turn.items():
            info[axis] = self.neutral[axis] + degrees
        return info, self._rotation(info["pitch"], info["yaw"], info["roll"])
