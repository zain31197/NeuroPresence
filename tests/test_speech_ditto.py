"""Ditto's motion turned into our engine's readings, and the silence gate. Ditto itself is not loaded."""

import numpy as np
import pytest
import torch

import neuropresence.speech.ditto as ditto
from neuropresence.speech.ditto import DittoReadings, VoiceGate, bin66_to_degree

LIPS, EYES = [6, 12, 14, 17, 19, 20], [11, 13, 15, 16, 18]
GAZE = [11, 15]  # the eye keypoints the gaze correction moves, with Ditto's tilt in every frame


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


def test_later_frames_move_the_head_by_a_share_of_how_far_ditto_moved_it_from_its_first_frame():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame(pitch=3.0, yaw=7.0, t=(0.05, 0.0, 0.0)))
    info, _ = readings.reading(frame(pitch=5.0, yaw=1.0, roll=-2.0, t=(0.07, -0.01, 0.0)))
    k = ditto.HEAD_SCALE
    assert float(info["pitch"]) == pytest.approx(8.0 + 2.0 * k + ditto.TILT_DEG, abs=0.05)
    assert float(info["yaw"]) == pytest.approx(-4.0 - 6.0 * k, abs=0.05)
    assert float(info["roll"]) == pytest.approx(1.0 - 2.0 * k, abs=0.05)
    assert torch.allclose(info["t"], torch.tensor([[0.01 + 0.02 * k, 0.02 - 0.01 * k, 0.0]]), atol=1e-6)


def test_only_the_lips_take_ditto_s_expression():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame())
    moved = np.full((21, 3), 0.1, np.float32)
    info, _ = readings.reading(frame(exp=moved))
    change = (info["exp"] - neutral()["exp"])[0]
    others = [k for k in range(21) if k not in LIPS + GAZE]
    assert torch.allclose(change[others], torch.zeros(len(others), 3))
    share = ditto.LIP_GAIN * (0.1 - ditto.LIP_REST) / 0.1
    assert torch.allclose(change[LIPS], torch.full((6, 3), 0.1 * share))


def lips_open(by):
    """Ditto's lips moved from its first frame, the lower lip opened `by`."""
    exp = np.full((21, 3), by / 2, np.float32)
    exp[ditto.LIP_POINT] = by
    return exp


def test_lips_open_by_less_than_ditto_s_hover_are_drawn_shut():
    readings = DittoReadings(neutral(), rotation)
    first, _ = readings.reading(frame())
    for by in (-0.01, 0.0, ditto.LIP_REST * 0.9):
        info, _ = readings.reading(frame(exp=lips_open(by)))
        assert torch.allclose(info["exp"], first["exp"])


def test_above_the_hover_the_opening_grows_by_lip_gain():
    readings = DittoReadings(neutral(), rotation)
    first, _ = readings.reading(frame())
    drawn = []
    for by in (0.03, 0.05):
        info, _ = readings.reading(frame(exp=lips_open(by)))
        drawn.append(float(info["exp"][0, ditto.LIP_POINT[0], ditto.LIP_POINT[1]] - first["exp"][0, ditto.LIP_POINT[0], ditto.LIP_POINT[1]]))
    assert drawn[0] == pytest.approx(ditto.LIP_GAIN * (0.03 - ditto.LIP_REST), abs=1e-6)
    assert drawn[1] - drawn[0] == pytest.approx(ditto.LIP_GAIN * 0.02, abs=1e-6)


def test_in_silence_the_lips_are_the_picture_s_own():
    readings = DittoReadings(neutral(), rotation)
    first, _ = readings.reading(frame())
    info, _ = readings.reading(frame(exp=np.full((21, 3), 0.1, np.float32)), voice=0.0)
    assert torch.allclose(info["exp"], first["exp"])


def blink_template():
    """15 frames closing and opening the eyes, as Ditto's template does: none at its first frame."""
    template = np.zeros((15, 21, 3), np.float32)
    template[:, EYES, 1] = np.sin(np.linspace(0, np.pi, 15))[:, None] * 0.02
    return template.reshape(15, 63)


def test_the_eyes_blink_with_ditto_s_template_every_60_to_100_frames():
    template = blink_template()
    readings = DittoReadings(neutral(), rotation, blink=template, seed=3)
    open_eyes = DittoReadings(neutral(), rotation).reading(frame())[0]["exp"]
    eyes = [float((readings.reading(frame(), number=n)[0]["exp"] - open_eyes)[0, EYES, 1].mean())
            for n in range(1000)]
    closing = [n for n in range(1, 1000) if eyes[n] > 0.0 and eyes[n - 1] == 0.0]
    gaps = np.diff(closing)
    assert 9 <= len(closing) <= 15 and gaps.min() >= 60 + 1 and gaps.max() <= 100 + 15
    start = closing[0] - 1  # the template's first frame is no change
    for i in range(15):
        assert eyes[start + i] == pytest.approx(float(template.reshape(15, 21, 3)[i, EYES, 1].mean()), abs=1e-7)


def test_the_blinks_fall_on_the_same_frames_whatever_order_they_are_asked_for_in():
    a = DittoReadings(neutral(), rotation, blink=blink_template(), seed=5)
    b = DittoReadings(neutral(), rotation, blink=blink_template(), seed=5)
    b.reading(frame(), number=900)  # far ahead first
    for n in (0, 70, 140, 500, 900):
        assert torch.equal(a.reading(frame(), number=n)[0]["exp"], b.reading(frame(), number=n)[0]["exp"])


def test_between_two_frames_the_movement_is_taken_the_share_of_the_way():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame(), number=0)
    lips = np.zeros((21, 3), np.float32)
    lips[LIPS] = 0.1
    one, two = frame(yaw=0.0), frame(yaw=10.0, exp=lips)
    info, turned = readings.between(one, two, 0.25, number=1)
    assert float(info["yaw"]) == pytest.approx(-4.0 + 0.25 * 10.0 * ditto.HEAD_SCALE, abs=0.05)
    share = ditto.LIP_GAIN * (0.1 - ditto.LIP_REST) / 0.1
    assert float(info["exp"][0, LIPS[0], 0] - 0.01) == pytest.approx(0.25 * 0.1 * share, abs=1e-5)
    assert torch.equal(turned, rotation(info["pitch"], info["yaw"], info["roll"]))
    alone, _ = readings.between(one, None, 0.6, number=1)  # the next frame not ready: this one as it is
    assert torch.equal(alone["exp"], readings.reading(one, number=1)[0]["exp"])


def test_the_eyes_turn_back_toward_the_viewer_when_the_head_turns():
    readings = DittoReadings(neutral(), rotation)
    readings.reading(frame())
    info, _ = readings.reading(frame(yaw=10.0))
    flat = (info["exp"] - neutral()["exp"]).reshape(63)
    dx = 10.0 * ditto.HEAD_SCALE * ditto.GAZE_X
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
