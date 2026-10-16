"""The virtual microphone: the voice held back to stay in step with the picture. No sound device is used."""

import json
import time

import numpy as np
import pytest

import neuropresence.server.virtual_microphone as vm
from neuropresence.server.virtual_microphone import (BLOCK, CATCH_UP_MS, MAX_DELAY_MS, MIN_DELAY_MS, RATE, DelayLine,
                                                     VirtualMicrophone)

MS = RATE // 1000  # samples in a millisecond


def run(line, seconds, signal=None, write_every=BLOCK):
    """Write `signal` (or silence) a block at a time, reading a block after each, as the two devices do."""
    blocks = int(seconds * RATE / BLOCK)
    signal = np.zeros(blocks * write_every, np.float32) if signal is None else signal
    out = []
    for n in range(blocks):
        line.write(signal[n * write_every:(n + 1) * write_every])
        out.append(line.read(BLOCK))
    return np.concatenate(out)


def test_before_anything_is_heard_it_sends_silence():
    line = DelayLine(delay_ms=100)
    assert not line.read(BLOCK).any()


def test_a_sound_comes_out_as_late_as_the_delay():
    line = DelayLine(delay_ms=100)
    signal = np.zeros(RATE, np.float32)
    signal[12000] = 1.0
    out = run(line, 1.0, signal)
    # The reader stays the delay behind the writer; it reads just after a block is written.
    assert int(np.argmax(out)) == 12000 + 100 * MS - BLOCK
    assert out.max() == pytest.approx(1.0)
    assert line.delay_ms == pytest.approx(100.0)


def test_the_delay_stays_between_its_limits():
    assert DelayLine(delay_ms=0).delay_ms == MIN_DELAY_MS
    assert DelayLine(delay_ms=5000).delay_ms == MAX_DELAY_MS


def test_a_small_change_is_reached_by_playing_slightly_faster_with_no_gap_and_no_jump():
    line = DelayLine(delay_ms=100)
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(4 * RATE) / RATE).astype(np.float32)
    run(line, 1.0, tone[:RATE])
    line.set_delay(85)  # the picture got 15 ms quicker
    out = run(line, 3.0, tone[RATE:])
    assert line.jumps == 0
    assert abs(line.delay_ms - 85) <= CATCH_UP_MS
    # Never silent, and the tone is never broken: at most a sample's worth of slope between neighbours.
    assert np.abs(out[BLOCK:]).max() > 0.45
    assert np.abs(np.diff(out)).max() < 0.03


def test_a_large_change_is_made_at_once_with_a_fade_and_no_click():
    line = DelayLine(delay_ms=MIN_DELAY_MS)
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(2 * RATE) / RATE).astype(np.float32)
    run(line, 1.0, tone[:RATE])
    line.set_delay(110)  # a session has started
    out = run(line, 1.0, tone[RATE:])
    assert line.jumps == 1
    assert line.delay_ms == pytest.approx(110.0)
    assert np.abs(np.diff(out)).max() < 0.05  # a jump without a fade would step by up to 1.0


def test_two_clocks_that_drift_apart_do_not_add_up():
    line = DelayLine(delay_ms=100)
    # The microphone runs 0.2 per cent fast: 481 samples for every 480 the cable takes. Over a minute
    # that would add 120 ms of delay.
    run(line, 60.0, np.zeros(int(60 * RATE / BLOCK) * 481, np.float32), write_every=481)
    assert abs(line.delay_ms - 100) <= CATCH_UP_MS
    assert line.jumps == 0


def test_a_device_that_asks_for_several_blocks_as_it_opens_still_gets_the_delay_right_within_half_a_second():
    line = DelayLine(delay_ms=100)
    line.write(np.zeros(BLOCK, np.float32))
    for _ in range(3):  # the cable fills its own buffer first: three blocks at once
        line.read(BLOCK)
    run(line, 0.5)
    assert line.jumps == 1  # set right once, with a fade
    assert line.delay_ms == pytest.approx(100, abs=1)
    run(line, 2.0)
    assert line.jumps == 1 and line.delay_ms == pytest.approx(100, abs=1)


def test_sound_handed_over_unevenly_does_not_set_off_catching_up():
    line = DelayLine(delay_ms=100)
    rng = np.random.default_rng(0)
    waiting, catching = 0, 0
    # As measured on the cable: blocks arrive late and then two at once, the lead swinging by up to 20 ms.
    for n in range(3000):
        waiting += BLOCK
        if rng.random() < 0.6:
            line.write(np.zeros(waiting, np.float32))
            waiting = 0
        line.read(BLOCK)
        catching += line._catching != 0
    assert line.jumps == 0
    assert catching < 0.02 * 3000  # played at its own speed nearly all the time
    assert abs(line.delay_ms - 100) < 6


# ------------------------------------------------------------- the switch


class Streams:
    def __init__(self, microphone, cable, on_input, on_output):
        self.microphone, self.cable, self.on_input, self.on_output = microphone, cable, on_input, on_output
        self.closed = False

    def close(self):
        self.closed = True


class Sound:
    """Stands in for the sound system: which devices there are, and the streams that were opened."""

    def __init__(self, cable="CABLE Input (VB-Audio Virtual Cable)", microphones=("Headset", "Webcam"), default="Webcam"):
        self.found = {"microphones": list(microphones), "default": default, "cable": cable}
        self.opened, self.fail = [], None

    def devices(self):
        return dict(self.found)

    def opener(self, microphone, cable, on_input, on_output):
        if self.fail:
            raise RuntimeError(self.fail)
        self.opened.append(Streams(microphone, cable, on_input, on_output))
        return self.opened[-1]


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(vm, "UPDATE_SECONDS", 0.01)


def microphone(sound=None, delay=None, settings=None):
    sound = sound or Sound()
    picture = {"ms": delay}
    mic = VirtualMicrophone(lambda: picture["ms"], devices=sound.devices, opener=sound.opener, settings=settings)
    return mic, sound, picture


def wait_for(test, seconds=2.0):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if test():
            return True
        time.sleep(0.01)
    return False


def test_it_is_off_until_switched_on_and_lists_the_microphones():
    mic, sound, _ = microphone()
    assert mic.snapshot() == {"on": False, "reason": None, "cable": "CABLE Input (VB-Audio Virtual Cable)",
                              "meeting_device": "CABLE Output (VB-Audio Virtual Cable)",
                              "microphone": "Webcam", "microphones": ["Headset", "Webcam"], "offset_ms": 0.0,
                              "picture_ms": None, "delay_ms": None, "level": 0.0}
    assert sound.opened == []


def test_it_sends_the_microphone_into_the_cable_once_switched_on():
    mic, sound, _ = microphone()
    state = mic.change(on=True)
    assert state["on"] is True and state["reason"] is None
    streams = sound.opened[0]
    assert (streams.microphone, streams.cable) == ("Webcam", "CABLE Input (VB-Audio Virtual Cable)")
    voice = np.full(BLOCK, 0.4, np.float32)
    for _ in range(10):  # 100 ms of voice
        streams.on_input(voice)
        out = streams.on_output(BLOCK)
    assert out.max() == pytest.approx(0.4)  # heard, after the least delay
    assert mic.snapshot()["level"] == pytest.approx(0.4)
    mic.change(on=False)
    assert streams.closed and mic.snapshot()["on"] is False


def test_without_a_cable_it_says_what_to_install():
    mic, sound, _ = microphone(Sound(cable=None))
    state = mic.change(on=True)
    assert state["on"] is False and state["reason"].startswith("No virtual cable was found.")
    assert sound.opened == []


def test_the_meeting_is_told_the_other_end_of_the_cable():
    from neuropresence.server.virtual_microphone import meeting_name
    assert meeting_name("CABLE Input (VB-Audio Virtual Cable)") == "CABLE Output (VB-Audio Virtual Cable)"
    assert meeting_name("NeuroPresence") == "Monitor of NeuroPresence"
    assert meeting_name(None) is None


def test_without_a_microphone_it_says_so():
    mic, _, _ = microphone(Sound(microphones=(), default=None))
    assert mic.change(on=True)["reason"] == "No microphone was found. Connect one and try again."


def test_a_device_that_cannot_be_opened_leaves_it_off_with_the_reason():
    sound = Sound()
    sound.fail = "The microphone or the cable could not be opened (busy)."
    mic, _, _ = microphone(sound)
    state = mic.change(on=True)
    assert state["on"] is False and state["reason"] == sound.fail


def test_the_delay_follows_the_picture_and_the_adjustment():
    mic, _, picture = microphone()
    mic.change(on=True)
    assert mic.line.target == MIN_DELAY_MS * MS  # no picture of the user is sent: no added delay
    picture["ms"] = 104.0  # a camera session is running
    assert wait_for(lambda: mic.line.target == 104 * MS)
    mic.change(offset_ms=20)
    assert wait_for(lambda: mic.line.target == 124 * MS)
    picture["ms"] = 108.0  # a small change is not chased
    time.sleep(0.1)
    assert mic.snapshot()["picture_ms"] == 104.0
    picture["ms"] = 95.0  # one beyond the margin is
    assert wait_for(lambda: mic.line.target == 115 * MS)
    picture["ms"] = None  # the session stopped
    assert wait_for(lambda: mic.line.target == MIN_DELAY_MS * MS)
    mic.close()


def test_the_adjustment_goes_two_hundred_ms_either_way():
    mic, _, _ = microphone()
    assert mic.change(offset_ms=500)["offset_ms"] == 200
    assert mic.change(offset_ms=-500)["offset_ms"] == -200


def test_choosing_another_microphone_while_on_opens_it_in_place_of_the_first():
    mic, sound, _ = microphone()
    mic.change(on=True)
    state = mic.change(microphone="Headset")
    assert state["on"] is True and state["microphone"] == "Headset"
    assert sound.opened[0].closed and sound.opened[1].microphone == "Headset"
    mic.close()
    assert sound.opened[1].closed


def test_the_microphone_and_the_adjustment_are_kept_but_it_always_starts_off(tmp_path):
    settings = tmp_path / "virtual_microphone.json"
    mic, sound, _ = microphone(settings=settings)
    mic.change(on=True, microphone="Headset", offset_ms=-30)
    assert json.loads(settings.read_text()) == {"microphone": "Headset", "offset_ms": -30}
    mic.close()
    again, _, _ = microphone(sound, settings=settings)
    assert again.snapshot()["on"] is False
    assert again.snapshot()["microphone"] == "Headset" and again.offset_ms == -30


def test_a_kept_microphone_that_is_gone_falls_back_to_the_default(tmp_path):
    settings = tmp_path / "virtual_microphone.json"
    settings.write_text(json.dumps({"microphone": "Old headset", "offset_ms": 0}))
    mic, sound, _ = microphone(settings=settings)
    mic.change(on=True)
    assert sound.opened[0].microphone == "Webcam"
    mic.close()


def test_a_damaged_settings_file_is_ignored(tmp_path):
    settings = tmp_path / "virtual_microphone.json"
    settings.write_text("{not json")
    mic, _, _ = microphone(settings=settings)
    assert mic.offset_ms == 0.0 and mic.snapshot()["microphone"] == "Webcam"
