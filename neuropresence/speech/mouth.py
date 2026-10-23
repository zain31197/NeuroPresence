"""How open the mouth is, from the sound of the voice, as it arrives.

With the camera off, the mouth of the picture follows the voice. This first version reads only how
loud the voice is, in steps of 10 ms: speech is loud when the mouth is open on a vowel and quiet when
the lips meet. No network is used, so it costs nothing on the graphics card.

Loudness in a room depends on the microphone, its distance and the noise around it, so it is judged
against two levels that follow the sound: the floor (the room when nobody speaks), the quietest moment
of the last second, and the peak (the voice at its loudest), which goes up at once and down only
slowly. Speech always pauses between words, so the floor stays on the room while someone talks; a
steady sound, a fan or the hiss of a microphone just switched on, becomes the floor within a second
and does not move the mouth. Only for that first second after silence, a microphone just switched on,
can its hiss open the mouth: telling hiss from a voice by how much it varies would also shut the
mouth on a long held vowel, which looks worse. Sound a few decibels above the floor starts to open the mouth, the peak opens it fully. Only
the top of the range counts: sound far below the peak is not speech, however far above the floor it is.

The opening follows the sound with a short time constant when it grows and a slightly longer one when
it shrinks, which keeps the lips from fluttering with every small dip in loudness.

The settings were chosen against Zain's own mouth in a recorded Meet call (23 October 2026): with them
the mouth was open in 40% of the frames against 23% for the real one, and stayed shut in pauses.
Agreement with the real lip gap was modest (r = 0.29): loudness cannot tell lips that meet on an "m"
from a vowel at the same volume.
"""

import math
from collections import deque

import numpy as np

HOP_SECONDS = 0.010  # the sound is judged in steps of this length
FLOOR_WINDOW_SECONDS = 1.0  # the floor is the quietest step of this last stretch of sound
FLOOR_UP_SECONDS = 0.1  # when the quiet moments get louder, the floor rises with this time constant
PEAK_DOWN_SECONDS = 8.0  # the peak falls this slowly after a loud moment
PEAK_UP_SECONDS = 0.03  # and rises this fast
MIN_RANGE_DB = 25.0  # the floor and the peak are kept at least this far apart, so that noise alone never opens the mouth wide
SPEECH_RANGE_DB = 25.0  # the mouth is judged within this much below the peak: in a very quiet room (or behind a
                        # meeting's noise suppression) faint sound far above the floor is still not speech
GATE_DB = 10.0  # this far above the floor the mouth starts to open
OPEN_SECONDS = 0.025  # the mouth opens with this time constant
CLOSE_SECONDS = 0.040  # and closes with this one
CURVE = 2.0  # above 1 keeps the mouth nearer closed except at the loudest moments, the vowels
SILENT_DB = -100.0  # what an empty step reads as


def _share(seconds, step=HOP_SECONDS):
    """How far a value following with time constant `seconds` moves in one step."""
    return 1.0 - math.exp(-step / seconds)


class SpeechMouth:
    """Feed it the microphone's samples as they come; read `openness` (0 closed, 1 fully open) at any time."""

    def __init__(self, rate=16000):
        self.rate = rate
        self.hop = max(1, int(round(rate * HOP_SECONDS)))
        self._pending = np.zeros(0, np.float32)
        self.floor = None
        self.peak = None
        self.level = SILENT_DB  # the loudness of the last step, in dB
        self.openness = 0.0
        self._opening, self._closing = _share(OPEN_SECONDS), _share(CLOSE_SECONDS)
        self._floor_up = _share(FLOOR_UP_SECONDS)
        self._recent = deque(maxlen=max(1, int(round(FLOOR_WINDOW_SECONDS / HOP_SECONDS))))
        self._peak_up, self._peak_down = _share(PEAK_UP_SECONDS), _share(PEAK_DOWN_SECONDS)

    def reset(self):
        self.__init__(self.rate)

    def feed(self, samples):
        """Take a block of mono samples (floats in -1..1) of any length. Returns the openness after it."""
        samples = np.asarray(samples, np.float32).reshape(-1)
        if len(self._pending):
            samples = np.concatenate([self._pending, samples])
        whole = len(samples) // self.hop
        for index in range(whole):
            step = samples[index * self.hop:(index + 1) * self.hop]
            rms = float(np.sqrt(np.mean(step * step)))
            self._step(20.0 * math.log10(rms) if rms > 0 else SILENT_DB)
        self._pending = samples[whole * self.hop:]
        return self.openness

    def _step(self, level):
        self.level = level
        if self.floor is None:
            self.floor, self.peak = level, level + MIN_RANGE_DB
        self._recent.append(level)
        quietest = min(self._recent)
        self.floor = quietest if quietest < self.floor else self.floor + (quietest - self.floor) * self._floor_up
        self.peak += (level - self.peak) * (self._peak_up if level > self.peak else self._peak_down)
        self.peak = max(self.peak, self.floor + MIN_RANGE_DB)
        floor = max(self.floor, self.peak - SPEECH_RANGE_DB)
        above = level - floor - GATE_DB
        span = self.peak - floor - GATE_DB
        target = min(max(above / span, 0.0), 1.0) ** CURVE if above > 0 else 0.0
        self.openness += (target - self.openness) * (self._opening if target > self.openness else self._closing)
