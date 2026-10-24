"""Ditto live: the bookkeeping of speech fed, features made and motion released. Stand-ins replace Ditto's
networks: each frame of speech carries its own number, and so does everything made from it."""

import threading

import numpy as np
import pytest

from neuropresence.speech.ditto import CHUNK, PAD_FRONT, SAMPLES_PER_FRAME, DittoStream

HOP = 5
SEQ = 80


class SpeechEncoder:
    """Stands in for HuBERT: the 5 frames of a window carry the numbers of the speech frames they come from."""

    def __call__(self, window, chunksize=CHUNK):
        assert len(window) == sum(chunksize) * SAMPLES_PER_FRAME + 80
        new = window[PAD_FRONT:PAD_FRONT + chunksize[1] * SAMPLES_PER_FRAME]
        return new.reshape(chunksize[1], SAMPLES_PER_FRAME).mean(axis=1, keepdims=True)

    def wav2feat(self, audio):
        return np.zeros((len(audio) // SAMPLES_PER_FRAME, 1), np.float32)


class MotionModel:
    """Stands in for Ditto's motion model: each run writes 80 frames that carry the numbers of their features."""

    seq_frames = SEQ
    fuse_length = HOP

    def __init__(self):
        self.kp_cond = np.zeros((1, 3), np.float32)

    def __call__(self, conditions, motion):
        written = conditions[:, :, :1]
        return written if motion is None else np.concatenate([motion, written[:, -HOP:]], 1)

    def cvt_fmt(self, motion):
        return [{"speech": float(value)} for value in motion[0, :, 0]]


def stream():
    s = DittoStream.__new__(DittoStream)
    s.hop, s.steps = HOP, 10
    s._lock, s._feature_lock = threading.Lock(), threading.Lock()
    s._wake, s._features_in, s._stop = threading.Event(), threading.Event(), threading.Event()
    s.wav2feat, s.audio2motion = SpeechEncoder(), MotionModel()
    s.conditions = lambda window, index: window
    s._reset(SEQ - HOP)
    return s


def speech(frames):
    """Each frame of speech carries its own number plus one (so that silence, 0, is told apart)."""
    return np.repeat(np.arange(1, frames + 1, dtype=np.float32), SAMPLES_PER_FRAME)


def work(s):
    while s._read_speech() or s._write_motion():
        pass


def test_frames_come_out_in_order_each_with_its_own_speech():
    s = stream()
    sound = speech(200)
    for start in range(0, len(sound), 320):  # 20 ms at a time, as a microphone hands it over
        s.feed(sound[start:start + 320])
        work(s)
    released = sorted(s.ready)
    assert released == list(range(len(released))) and len(released) > 150
    assert all(s.ready[k][0]["speech"] == pytest.approx(k + 1) for k in released)


def test_a_frame_is_released_only_once_the_speech_after_it_has_been_heard():
    s = stream()
    s.feed(speech(100))
    work(s)
    # Features exist for whole windows of 5 frames whose 2 frames after have arrived: frames 0 to 94 here.
    features_upto = (100 - CHUNK[2]) // CHUNK[1] * CHUNK[1] - 1
    assert max(s.ready) == features_upto - HOP  # the last hop waits for the next run to be blended into it
    s.feed(speech(10)[:SAMPLES_PER_FRAME * 10])
    work(s)
    assert max(s.ready) > features_upto - HOP  # more speech, more frames


def test_heard_gives_the_moment_the_speech_of_a_frame_had_all_arrived():
    s = stream()
    sound = speech(20)
    s.feed(sound[:5 * SAMPLES_PER_FRAME])
    first = s._arrivals[-1][1]
    s.feed(sound[5 * SAMPLES_PER_FRAME:])
    second = s._arrivals[-1][1]
    assert s.heard(4) == first and s.heard(5) == second and s.heard(19) == second
    assert s.heard(20) is None  # not yet spoken


def test_the_speech_already_read_is_let_go():
    s = stream()
    s.feed(speech(400))
    work(s)
    assert len(s._audio) < 2 * (sum(CHUNK) * SAMPLES_PER_FRAME + 80)  # only what the next window needs
    assert len(s._features) <= 2 * SEQ
