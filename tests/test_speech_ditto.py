"""Ditto's motion turned into our engine's readings, and the silence gate. Ditto itself is not loaded."""

import numpy as np
import pytest
import torch

import neuropresence.speech.ditto as ditto
from neuropresence.speech.ditto import DittoReadings, VoiceGate, bin66_to_degree

LIPS, EYES = [6, 12, 14, 17, 19, 20], [11, 13, 15, 16, 18]


def bins_for(degrees):
    """66 bin scores whose expected angle is `degrees`: a narrow peak, as the network writes them."""
    centre = (degrees + 97.5) / 3
    return (-0.5 * ((np.arange(66) - centre) / 0.6) ** 2).reshape(1, 66).astype(np.float32)


def frame(pitch=0.0, yaw=0.0, roll=0.0, t=(0.0, 0.0, 0.0), exp=None):
    return {"pitch": bins_for(pitch), "yaw": bins_for(yaw), "roll": bins_for(roll),
            "t": np.array([t], np.float32), "exp": (np.zeros((1, 63), np.float32) if exp is None else exp.reshape(1, 63)),
            "scale": np.ones((1, 1), np.float32)}


def neutral():
    return {"pitch": torch.tensor([[8.0]]), "yaw": torch.tensor([[-4.0]]), "roll": torch.tensor([[1.0]]),
            "t": torch.tensor([[0.01, 0.02, 0.0]]), "scale": torch.tensor([[1.3]]), "exp": torch.full((1, 21, 3), 0.01)}


def rotation(pitch, yaw, roll):
    return torch.tensor([[float(pitch), float(yaw), float(roll)]])


def test_the_angle_is_read_from_the_bins_as_liveportrait_does():
    for degrees in (-30.0, -4.3, 0.0, 8.2, 45.0):
        assert bin66_to_degree(bins_for(degrees))[0] == pytest.approx(degrees, abs=0.05)
    flat = np.zeros((1, 66))
    assert bin66_to_degree(flat)[0] == pytest.approx(65 / 2 * 3 - 97.5)  # no preference: the middle


def test_the_first_frame_draws_the_picture_as_it_is_but_for_ditto_s_tilt():
    readings = DittoReadings(neutral(), rotation)
    info, turned = readings.reading(frame(pitch=3.0, yaw=7.0, t=(0.05, 0.0, 0.0)), voice=1.0)
    assert float(info["pitch"]) == pytest.approx(8.0 + ditto.TILT_DEG)
    assert float(info["yaw"]) == pytest.approx(-4.0) and float(info["roll"]) == pytest.approx(1.0)
    assert torch.allclose(info["t"], neutral()["t"])
    assert torch.equal(info["scale"], neutral()["scale"])
    assert torch.equal(turned, rotation(info["pitch"], info["yaw"], info["roll"]))


def test_later_frames_move_the_head_by_how_far_ditto_moved_it_from_its_first_frame():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame(pitch=3.0, yaw=7.0, t=(0.05, 0.0, 0.0)))
    info, _ = readings.reading(frame(pitch=5.0, yaw=1.0, roll=-2.0, t=(0.07, -0.01, 0.0)))
    assert float(info["pitch"]) == pytest.approx(8.0 + 2.0 + ditto.TILT_DEG, abs=0.05)
    assert float(info["yaw"]) == pytest.approx(-4.0 - 6.0, abs=0.05)
    assert float(info["roll"]) == pytest.approx(1.0 - 2.0, abs=0.05)
    assert torch.allclose(info["t"], torch.tensor([[0.03, 0.01, 0.0]]), atol=1e-6)


def test_only_the_lips_and_the_eyes_take_ditto_s_expression():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame())
    moved = np.full((21, 3), 0.1, np.float32)
    info, _ = readings.reading(frame(exp=moved))
    change = (info["exp"] - neutral()["exp"])[0]
    others = [k for k in range(21) if k not in LIPS + EYES]
    assert torch.allclose(change[others], torch.zeros(len(others), 3))
    assert torch.allclose(change[LIPS], torch.full((6, 3), 0.1 * ditto.LIP_SCALE))
    assert change[EYES].abs().min() > 0.09  # the eyes as Ditto writes them (with the gaze correction, which is tiny here)


def test_in_silence_the_lips_are_the_picture_s_own_but_the_eyes_still_move():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame())
    info, _ = readings.reading(frame(exp=np.full((21, 3), 0.1, np.float32)), voice=0.0)
    change = (info["exp"] - neutral()["exp"])[0]
    assert torch.allclose(change[LIPS], torch.zeros(6, 3))
    assert change[EYES].abs().min() > 0.09


def test_the_eyes_turn_back_toward_the_viewer_when_the_head_turns():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame())
    info, _ = readings.reading(frame(yaw=10.0))
    flat = (info["exp"] - neutral()["exp"]).reshape(63)
    dx = 10.0 * ditto.GAZE_X
    assert float(flat[33]) == pytest.approx(dx * 0.0007, abs=1e-5)
    assert float(flat[45]) == pytest.approx(dx * 0.001, abs=1e-5)


def test_the_gate_opens_with_a_voice_and_closes_a_moment_after_it():
    gate = VoiceGate(fps=25)
    assert gate.step(0.0) == 0.0
    opened = [gate.step(0.5) for _ in range(5)]
    assert opened[1] > 0.6 and opened[-1] > 0.95  # within two frames
    after = [gate.step(0.0) for _ in range(40)]
    held = int(ditto.VOICE_HOLD_SECONDS * 25)
    assert min(after[:held]) > 0.95  # held through the end of a word
    assert after[-1] < 0.01  # and shut again within a second and a half


def test_a_short_pause_between_words_does_not_close_the_gate():
    gate = VoiceGate(fps=25)
    for _ in range(10):
        gate.step(0.6)
    pause = [gate.step(0.0) for _ in range(5)]  # 200 ms between two words
    assert min(pause) > 0.95
