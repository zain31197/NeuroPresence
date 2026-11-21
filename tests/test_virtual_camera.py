"""The virtual camera: what a meeting application is sent. No camera driver is used: the device is a stand-in."""

import threading
import time

import cv2
import numpy as np
import pytest

from neuropresence.server.virtual_camera import (FPS, KEEPALIVE_SECONDS, NOT_RUNNING, SAFE_HEIGHT_SHARE, SIZE,
                                                  VirtualCamera, card, fit)


class Device:
    """Stands in for the camera device: keeps what it is sent."""

    device = "Test Camera"

    def __init__(self, fail_after=None):
        self.frames, self.times, self.closed, self.fail_after = [], [], False, fail_after

    def send(self, frame):
        if self.fail_after is not None and len(self.frames) >= self.fail_after:
            raise RuntimeError("the device was taken away")
        self.frames.append(frame.copy())
        self.times.append(time.perf_counter())

    def close(self):
        self.closed = True


def wait_for(test, seconds=2.0):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if test():
            return True
        time.sleep(0.01)
    return False


def camera(picture, device=None, fps=200):
    device = device or Device()
    opened = []

    def opener(width, height, rate):
        opened.append((width, height, rate))
        return device

    return VirtualCamera(picture, opener=opener, fps=fps), device, opened


def test_a_frame_is_set_in_the_middle_at_the_size_of_the_device():
    upright = np.full((1280, 960, 3), 200, np.uint8)  # an upright meeting picture
    sent = fit(upright)
    assert sent.shape == (SIZE[1], SIZE[0], 3)
    rows, cols = np.nonzero(sent.max(axis=2))
    assert rows.max() - rows.min() + 1 == round(SAFE_HEIGHT_SHARE * 720)  # 605 of the 720 rows
    assert abs(rows.min() - (719 - rows.max())) <= 1 and abs(cols.min() - (1279 - cols.max())) <= 1  # in the middle
    assert sent[rows.min():rows.max() + 1, cols.min():cols.max() + 1].min() == 200  # whole, nothing cut off
    small = fit(np.full((360, 640, 3), 50, np.uint8))
    assert small.shape == (720, 1280, 3) and small.max() == 50 and np.count_nonzero(small.max(axis=2)) > 0.8 * 720 * 1280 * 0.8


def test_a_wide_meeting_tile_leaves_the_picture_and_its_mark_whole():
    from neuropresence.consent import is_marked, mark
    output = mark(np.full((1280, 960, 3), 120, np.uint8))
    sent = fit(output)
    for tile in (16 / 9, 1.97, 2.1):  # 1.97 to 1: Google Meet's main tile in the call of 18 October 2026
        shown = round(1280 / tile)
        top = (720 - shown) // 2
        seen = sent[top:top + shown]
        rows = np.nonzero(seen.max(axis=2).max(axis=1))[0]
        picture = seen[rows.min():rows.max() + 1]
        cols = np.nonzero(picture.max(axis=2).max(axis=0))[0]
        assert rows.min() > 0 and rows.max() < shown - 1  # black above and below: none of the picture was cut
        assert is_marked(cv2.resize(picture[:, cols.min():cols.max() + 1], (960, 1280), interpolation=cv2.INTER_CUBIC))


def middle(frame):
    """The pixels in the middle of a frame sent to the device, where the picture always is."""
    return frame[300:420, 560:720]


def test_the_card_shows_nothing_of_the_user():
    shown = card()
    assert shown.shape == (SIZE[1], SIZE[0], 3)
    assert 0.9 < (shown.reshape(-1, 3) == shown[0, 0]).all(axis=1).mean() < 1.0  # a plain ground, and some writing on it
    assert NOT_RUNNING == "NeuroPresence is not running"


def test_it_is_off_until_switched_on_and_opens_the_device_at_one_size_and_rate():
    cam, device, opened = camera(lambda: None, fps=FPS)
    assert cam.snapshot() == {"on": False, "device": None, "reason": None, "showing": "nothing", "size": [1280, 720], "fps": 20}
    assert opened == [] and device.frames == []
    state = cam.switch(True)
    assert opened == [(1280, 720, 20)]  # the rate the drawing is held to while it is on, so each frame is picked up once
    assert state["on"] is True and state["device"] == "Test Camera" and state["reason"] is None
    cam.switch(True)  # already on: the device is not opened twice
    assert len(opened) == 1
    cam.close()


def test_without_a_session_the_card_is_sent_and_with_one_its_output():
    showing = {"frame": None}
    cam, device, _ = camera(lambda: showing["frame"])
    cam.switch(True)
    assert wait_for(lambda: len(device.frames) >= 3)
    assert cam.snapshot()["showing"] == "card" and np.array_equal(device.frames[-1], card())
    output = np.full((720, 1280, 3), 170, np.uint8)
    showing["frame"] = output
    assert wait_for(lambda: cam.snapshot()["showing"] == "session" and middle(device.frames[-1]).min() == 170)
    count = len(device.frames)
    assert wait_for(lambda: len(device.frames) >= count + 5)  # the same frame is sent again until the next is drawn
    assert all(middle(frame).min() == 170 for frame in device.frames[count:count + 5])
    showing["frame"] = None  # the session has ended
    assert wait_for(lambda: cam.snapshot()["showing"] == "card" and np.array_equal(device.frames[-1], card()))
    cam.close()


def test_switched_off_it_leaves_the_card_behind_and_lets_the_device_go():
    cam, device, _ = camera(lambda: np.full((720, 1280, 3), 170, np.uint8))
    cam.switch(True)
    assert wait_for(lambda: len(device.frames) >= 2 and middle(device.frames[-1]).min() == 170)
    state = cam.switch(False)
    assert state["on"] is False and state["showing"] == "nothing"
    assert device.closed and np.array_equal(device.frames[-1], card())  # never a frozen face
    sent = len(device.frames)
    time.sleep(0.05)
    assert len(device.frames) == sent


def test_frames_are_sent_at_the_rate_of_the_device():
    cam, device, _ = camera(lambda: None, fps=50)
    cam.switch(True)
    time.sleep(0.5)
    cam.close()
    assert 18 <= len(device.frames) <= 30  # about 25 in half a second, and not as fast as the loop could go


def test_a_device_that_cannot_be_opened_says_what_to_do():
    def opener(width, height, fps):
        raise RuntimeError("No virtual camera could be opened. Install OBS Studio.")

    cam = VirtualCamera(lambda: None, opener=opener)
    state = cam.switch(True)
    assert state["on"] is False and state["reason"] == "No virtual camera could be opened. Install OBS Studio."
    cam.close()  # nothing to close, and no error


def test_a_device_that_fails_while_running_switches_the_camera_off_with_the_reason():
    cam, device, _ = camera(lambda: None, device=Device(fail_after=3))
    cam.switch(True)
    assert wait_for(lambda: cam.snapshot()["on"] is False)
    assert cam.snapshot()["reason"] == "The virtual camera stopped: the device was taken away"
    assert device.closed


def test_fitting_is_done_once_for_a_frame_that_is_sent_again(monkeypatch):
    calls = []
    import neuropresence.server.virtual_camera as module

    real = module.fit
    monkeypatch.setattr(module, "fit", lambda frame, size: calls.append(1) or real(frame, size))
    frame = np.full((1280, 960, 3), 99, np.uint8)
    cam, device, _ = camera(lambda: frame)
    cam.switch(True)
    assert wait_for(lambda: len(device.frames) >= 10)
    cam.close()
    assert len(calls) == 1
    assert pytest.approx(99) == float(middle(device.frames[5]).mean())


def test_it_counts_the_new_pictures_apart_from_the_frames_sent_again():
    """On a clock of its own the camera sends at its rate however fast the session draws; how many of the frames
    were new pictures is what tells a smooth picture from a jerky one, so it is counted on its own."""
    frames = [np.full((360, 640, 3), 10 * i, np.uint8) for i in range(4)]
    shown = {"i": 0}
    cam, device, _ = camera(lambda: frames[shown["i"]])
    cam.switch(True)
    for i in range(1, 4):
        assert wait_for(lambda: len(device.frames) >= 5 * i)
        shown["i"] = i  # the session draws a new picture
    assert wait_for(lambda: cam.fresh == 4)
    assert wait_for(lambda: cam.sent >= 25)
    cam.close()
    assert cam.fresh == 4


class Session:
    """Stands in for the session: makes frames one at a time, and lets a reader wait for the next one."""

    def __init__(self):
        self.made, self.frame = 0, None
        self._ready = threading.Condition()

    def make(self, value):
        with self._ready:
            self.frame = np.full((360, 640, 3), value, np.uint8)
            self.made += 1
            self._ready.notify_all()
        return time.perf_counter()

    def wait(self, seen, timeout):
        with self._ready:
            if self.made == seen:
                self._ready.wait(timeout)
            return self.made


def test_with_a_session_each_frame_is_sent_once_as_soon_as_it_is_finished():
    """A second clock between the drawing and the meeting holds frames for one tick or two by turns, and drops a
    frame when two are finished within one tick (scripts/study_meeting_cadence.py). Waiting on the session instead,
    each frame goes out once, as it is finished."""
    session, device = Session(), Device()
    cam = VirtualCamera(lambda: session.frame, opener=lambda w, h, r: device, wait=session.wait)
    cam.switch(True)
    assert wait_for(lambda: len(device.frames) == 1)  # no session yet: the card, once
    made = []
    for value in (10, 20, 30, 40, 50, 60):
        made.append(session.make(value))
        time.sleep(0.05)
    cam.close()
    values = [int(middle(frame).min()) for frame in device.frames[1:-1]]  # between the first card and the last
    assert values == [10, 20, 30, 40, 50, 60]  # every frame, once, in order: none dropped and none sent twice
    late = [sent - finished for sent, finished in zip(device.times[1:-1], made)]
    assert max(late) < 0.03  # sent as soon as it was finished, not on the camera's next tick
    assert cam.fresh == 6


def test_with_a_session_the_last_frame_is_sent_again_only_after_a_pause():
    session, device = Session(), Device()
    cam = VirtualCamera(lambda: session.frame, opener=lambda w, h, r: device, wait=session.wait)
    cam.switch(True)
    session.make(90)
    assert wait_for(lambda: len(device.frames) >= 2)
    time.sleep(2.5 * KEEPALIVE_SECONDS)
    cam.close()
    again = sum(1 for frame in device.frames[1:-1] if middle(frame).min() == 90)
    assert 2 <= again <= 4  # sent, then again about every KEEPALIVE_SECONDS: a reader is not left waiting, nor flooded
