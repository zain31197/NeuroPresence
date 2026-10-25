"""The voice drive: which frame to draw now, the gate stepped in order, and letting go. No Ditto, no microphone."""

import threading

import numpy as np
import pytest

from neuropresence.speech.ditto import FPS, SAMPLES_PER_FRAME
from neuropresence.speech.live import VoiceDrive

DELAY = 0.8


class Stream:
    """Stands in for Ditto's live stream: every frame of speech fed is ready at once."""

    def __init__(self):
        self.ready, self.fed, self.started, self.stopped, self.forgotten = {}, 0, None, False, []
        self.ready_to_listen = threading.Event()

    def start(self, picture):
        self.started = picture
        self.ready_to_listen.set()

    def feed(self, samples):
        self.fed += len(samples)
        for frame in range(len(self.ready), self.fed // SAMPLES_PER_FRAME):
            self.ready[frame] = ({"frame": frame}, 0.0)

    def heard(self, frame):
        return None  # the drive then works it out from when the first sample arrived

    def forget_before(self, frame):
        self.forgotten.append(frame)
        for old in [key for key in self.ready if key < frame]:
            del self.ready[old]

    def stop(self):
        self.stopped = True


class Microphone:
    microphone = "Test microphone"

    def __init__(self, on_samples):
        self.on_samples, self.closed = on_samples, False

    def close(self):
        self.closed = True


def drive(clock):
    made = []

    def opener(name, on_samples):
        made.append(Microphone(on_samples))
        return made[-1]

    d = VoiceDrive(stream_factory=Stream, opener=opener, delay=DELAY)
    d.start("picture.png")
    return d, made[0]


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr("neuropresence.speech.live.time.perf_counter", lambda: now["t"])
    return now


def speak(microphone, seconds, loud=True):
    samples = int(seconds * 16000)
    t = np.arange(samples) / 16000
    wave = (0.3 * np.sin(2 * np.pi * 150 * t) if loud else 0.0005 * np.random.default_rng(0).standard_normal(samples))
    for start in range(0, samples, 320):
        microphone.on_samples(wave[start:start + 320].astype(np.float32))


def test_nothing_is_drawn_before_the_microphone_has_heard_anything(clock):
    d, _ = drive(clock)
    assert d.due(clock["t"] + 5.0) is None


def test_the_frame_drawn_is_the_one_whose_speech_ended_the_delay_ago(clock):
    d, microphone = drive(clock)
    speak(microphone, 3.0)
    began = clock["t"] - 320 / 16000  # the first block's samples were heard before it arrived
    got = d.due(began + DELAY + 2.0)
    number, frame, voice, heard, skipped = got
    assert number == int(2.0 * FPS) - 1 and frame == {"frame": number}
    assert heard == pytest.approx(began + (number + 1) / FPS)
    assert skipped == number  # nothing was drawn before: every earlier frame was skipped
    assert d.due(began + DELAY + 2.0) is None  # the same moment again: nothing new
    later = d.due(began + DELAY + 2.0 + 3 / FPS)
    assert later[0] == number + 3 and later[4] == 2


def test_a_frame_not_ready_in_time_is_counted_once_and_skipped(clock):
    d, microphone = drive(clock)
    speak(microphone, 1.0)
    began = clock["t"] - 320 / 16000
    assert d.due(began + DELAY + 2.0) is None  # its speech has not even been fed
    assert d.due(began + DELAY + 2.0) is None
    assert d.missed == 1


def test_the_lips_are_shut_in_silence_and_move_with_the_voice(clock):
    d, microphone = drive(clock)
    speak(microphone, 1.5, loud=False)
    speak(microphone, 1.5, loud=True)
    began = clock["t"] - 320 / 16000
    quiet = d.due(began + DELAY + 1.0)
    talking = d.due(began + DELAY + 2.8)
    assert quiet[2] < 0.05 and talking[2] > 0.95


def test_what_has_been_drawn_is_let_go_of_once_a_second(clock):
    d, microphone = drive(clock)
    speak(microphone, 4.0)
    began = clock["t"] - 320 / 16000
    d.due(began + DELAY + 2.0 + 0.5 / FPS)  # frame 49: not a whole second
    assert not d.stream.forgotten
    d.due(began + DELAY + 51 / FPS)  # frame 50
    assert d.stream.forgotten == [50 - FPS] and min(d.stream.ready) == 50 - FPS


def test_stopping_closes_the_microphone_and_stops_the_model_but_keeps_it_loaded(clock):
    d, microphone = drive(clock)
    stream = d.stream
    d.stop()
    assert microphone.closed and stream.stopped
    d.start("picture.png")
    assert d.stream is stream  # loaded once, kept between sessions
