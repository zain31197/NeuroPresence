"""The voice driving the picture in a session: the microphone in, the frame to draw now out.

A session with the camera off starts like any camera session, with the liveness check and the face
matched at the camera; then the camera is closed and this takes over. It opens the microphone at
16 kHz, feeds Ditto's live stream (ditto.py), and tells the session which of Ditto's frames to draw
at each moment: the one whose speech was heard PLAY_DELAY ago. The voice that goes to the meeting is
held back by as much (server/virtual_microphone.py), so lips and words arrive together.

It also keeps the lips shut while nobody speaks: the loudness of every frame of speech is read as it
comes in (mouth.py), and the gate (ditto.py: VoiceGate) is stepped frame by frame in order.

Ditto takes about 10 s to load and, the first time on a machine, about a minute to compile its motion
model; it is kept loaded between sessions.
"""

import math
import sys
import threading
import time
from collections import namedtuple

import numpy as np

from .ditto import FPS, PLAY_DELAY, SAMPLES_PER_FRAME, VoiceGate
from .mouth import SpeechMouth

RATE = 16000
BLOCK = 320  # 20 ms handed over at a time
WARM_SECONDS = 600.0  # how long to wait for the motion model to be compiled before giving up
DITTO_LATE = 1  # Ditto's lips move this many frames after the real ones (ditto.py), so each is shown this much sooner
STEP = 0.25  # a new picture is drawn once the moment shown has moved on by a quarter of a frame or more

# What to draw now: Ditto's frame `number` and the next (`after`, None when not ready), `share` of the way
# from one to the other; the voice for that moment (0 to 1); when frame `number`'s speech was heard; and the
# frames passed over since the last drawn.
Due = namedtuple("Due", "number frame after share voice heard skipped")


def open_microphone(name, on_samples):
    """Start reading the microphone called `name` (the system's own when None) into on_samples(samples).
    Returns an object with close(). Raises RuntimeError with what to do when it cannot be opened."""
    try:
        import sounddevice as sd
    except ImportError as err:
        raise RuntimeError("The voice mode needs the sounddevice package: pip install sounddevice.") from err
    from ..server.virtual_microphone import _device_index, list_devices

    found = list_devices()
    name = name if name in found["microphones"] else found["default"]
    if name is None:
        raise RuntimeError("No microphone was found. Connect one and try again.")
    extra = sd.WasapiSettings(auto_convert=True) if sys.platform == "win32" else None
    try:
        stream = sd.InputStream(device=_device_index(sd, name, "input"), channels=1, samplerate=RATE, blocksize=BLOCK,
                                dtype="float32", latency="low", callback=lambda data, *_: on_samples(data[:, 0].copy()),
                                extra_settings=extra)
        stream.start()
    except Exception as err:  # taken by another program, or a format it refuses
        raise RuntimeError(f"The microphone could not be opened ({err}).") from err

    class Microphone:
        microphone = name

        def close(self):
            stream.stop()
            stream.close()

    return Microphone()


def _ditto_stream():
    from .ditto import DittoStream
    return DittoStream()


class VoiceDrive:
    """The microphone, Ditto's live stream and the silence gate, for one session at a time.

    `stream_factory` makes Ditto's stream (once; it is kept) and `opener` opens the microphone; the
    tests pass stand-ins for both.
    """

    def __init__(self, stream_factory=_ditto_stream, opener=open_microphone, delay=PLAY_DELAY):
        self._make_stream, self._opener, self.delay = stream_factory, opener, delay
        self.stream = None
        self._microphone = None
        self._lock = threading.Lock()
        self.microphone = None

    def start(self, picture, microphone=None, progress=None):
        """Load Ditto if needed, start it on `picture` (a file), and open the microphone. Blocks until ready."""
        if self.stream is None:
            if progress:
                progress("Loading the voice model. A few seconds, the first time")
            self.stream = self._make_stream()
        if progress:
            progress("Preparing the voice model for this card. About a minute the first time on a computer")
        self.stream.start(picture)
        if not self.stream.ready_to_listen.wait(WARM_SECONDS):
            self.stream.stop()
            raise RuntimeError("The voice model could not be made ready.")
        self._mouth, self._gate = SpeechMouth(RATE), VoiceGate()
        self._pending = 0  # samples since the last whole frame of speech
        self._openness = []  # how loud each whole frame of speech was, for the gate
        self._voiced = []  # the gate, stepped in order, one value a frame
        self._began = None  # when the first sample was heard
        self.last = -1  # the last frame handed out
        self._shown = -1.0  # the moment last handed out, in frames
        self._second = 0  # the last whole second at which what had been drawn was let go of
        self.missed = 0  # frames whose moment came before they were ready, each counted once
        self._missed_at = -1
        self._microphone = self._opener(microphone, self._heard)
        self.microphone = getattr(self._microphone, "microphone", microphone)

    def _heard(self, samples):
        """The microphone's callback: to Ditto, and the loudness of every whole frame for the gate."""
        if self._began is None:
            self._began = time.perf_counter() - len(samples) / RATE
        self.stream.feed(samples)
        with self._lock:
            self._mouth.feed(samples)
            self._pending += len(samples)
            while self._pending >= SAMPLES_PER_FRAME:
                self._pending -= SAMPLES_PER_FRAME
                self._openness.append(self._mouth.openness)

    def _voice(self, frame):
        """The gate for this frame, stepped through every frame before it in order."""
        with self._lock:
            while len(self._voiced) <= frame and len(self._voiced) < len(self._openness):
                self._voiced.append(self._gate.step(self._openness[len(self._voiced)]))
            return self._voiced[frame] if frame < len(self._voiced) else 1.0

    @property
    def blink(self):
        """Ditto's blink (15 frames of change to the eye keypoints), or None when its stream has none."""
        return (getattr(self.stream, "defaults", None) or {}).get("delta_eye_arr")

    def due(self, now):
        """What to draw now, as a Due, or None if there is nothing new to draw.

        The moment shown is the one `delay` behind now, in Ditto's frames, with the fraction: drawn at about
        16 frames a second, Ditto's 25 would otherwise be shown at uneven steps, and the movement would
        judder. Frames whose moment has passed are skipped, as the camera mode skips camera frames."""
        if self._began is None:
            return None
        moment = (now - self.delay - self._began) * FPS - 1 + DITTO_LATE  # frame k's speech ended `delay` before moment k
        frame = math.floor(moment)
        if frame < 0 or moment < self._shown + STEP:
            return None
        ready = self.stream.ready.get(frame)
        if ready is None:
            if frame != self._missed_at and frame > self.last:
                self.missed, self._missed_at = self.missed + 1, frame
            return None
        following = self.stream.ready.get(frame + 1)
        share = moment - frame if following is not None else 0.0
        skipped = max(frame - self.last - 1, 0)
        self.last, self._shown = frame, moment
        spoken = max(frame - DITTO_LATE, 0)  # the speech whose lips frame `frame` shows: the voice is held back to it
        heard = self.stream.heard(spoken) or (self._began + (spoken + 1) / FPS)
        second = frame // FPS
        if second > self._second:  # once a second, what has been drawn is let go of
            self._second = second
            if second > 1:
                self.stream.forget_before((second - 1) * FPS)
        voice = self._voice(frame)
        if following is not None:
            voice += (self._voice(frame + 1) - voice) * share
        return Due(frame, ready[0], following[0] if following is not None else None, share, voice, heard, skipped)

    def stop(self):
        if self._microphone is not None:
            self._microphone.close()
            self._microphone = None
        if self.stream is not None:
            self.stream.stop()
