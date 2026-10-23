"""The voice driving the picture: the mouth from the sound, and the movement handed to the engine. No engine is used."""

import numpy as np
import pytest
import torch

import neuropresence.speech.motion as motion_module
from neuropresence.speech import SpeechMotion, SpeechMouth, load_directions

RATE = 16000
LIPS, EYES = [6, 12, 14, 17, 19, 20], [11, 13, 15, 16, 18]


def noise(seconds, db, seed=0):
    """Room noise at about this loudness."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * RATE)) * 10 ** (db / 20)).astype(np.float32)


def vowel(seconds, db):
    """A steady voiced sound at about this loudness."""
    t = np.arange(int(seconds * RATE)) / RATE
    wave = np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 280 * t) + 0.3 * np.sin(2 * np.pi * 700 * t)
    return (wave / np.sqrt(np.mean(wave ** 2)) * 10 ** (db / 20)).astype(np.float32)


def run(mouth, audio, block=533):
    """Feed the sound in blocks, as the microphone hands it over; the openness after each block."""
    return np.array([mouth.feed(audio[i:i + block]) for i in range(0, len(audio), block)])


def test_silence_and_room_noise_keep_the_mouth_shut():
    assert run(SpeechMouth(RATE), np.zeros(RATE, np.float32)).max() == 0.0
    assert run(SpeechMouth(RATE), noise(3.0, -55)).max() < 0.05


def test_noise_that_starts_after_silence_moves_the_mouth_for_about_a_second_only():
    """A microphone switched on after silence: its hiss is new, so for about a second it reads like a voice."""
    mouth = SpeechMouth(RATE)
    run(mouth, np.zeros(RATE, np.float32))
    after = run(mouth, noise(3.0, -55))
    per_block = 533 / RATE
    assert after[int(1.5 / per_block):].max() < 0.05


def test_a_voice_opens_the_mouth_and_it_closes_again_when_the_voice_stops():
    mouth = SpeechMouth(RATE)
    run(mouth, noise(2.0, -55))
    talking = run(mouth, np.concatenate([vowel(0.4, -20), noise(0.3, -55)] * 3))
    per_block = 533 / RATE
    assert talking.max() > 0.8  # a vowel opens it
    assert talking[int(0.15 / per_block)] > 0.5  # within 150 ms
    assert talking[-1] < 0.05  # and it is shut again 300 ms after the last one


def test_the_same_voice_opens_the_mouth_alike_at_a_near_and_a_far_microphone():
    """Only how loud the voice is against itself and the room counts, not the microphone's level."""
    near, far = SpeechMouth(RATE), SpeechMouth(RATE)
    pattern = lambda shift: np.concatenate([noise(2.0, -60 + shift)] + [vowel(0.3, -18 + shift), vowel(0.3, -28 + shift),
                                                                        noise(0.2, -60 + shift)] * 4)
    a, b = run(near, pattern(0)), run(far, pattern(-15))
    assert np.abs(a - b).max() < 0.05


def test_a_quieter_syllable_opens_the_mouth_less_than_a_loud_vowel():
    mouth = SpeechMouth(RATE)
    run(mouth, noise(1.0, -60))
    run(mouth, vowel(0.5, -18))  # sets how loud this voice gets
    loud = run(mouth, vowel(0.3, -18))[-1]
    soft = run(mouth, vowel(0.3, -26))[-1]
    assert loud > 0.8 and 0.05 < soft < loud * 0.7


def test_the_size_of_the_blocks_does_not_change_the_result():
    audio = np.concatenate([noise(1.0, -55), vowel(0.4, -20), noise(0.4, -55)])
    small, large = SpeechMouth(RATE), SpeechMouth(RATE)
    run(small, audio, block=160)
    run(large, audio, block=1600)
    assert small.openness == pytest.approx(large.openness, abs=1e-6)


# --------------------------------------------------------------- the motion


def neutral():
    return {"pitch": torch.tensor([[2.0]]), "yaw": torch.tensor([[-3.0]]), "roll": torch.tensor([[0.5]]),
            "t": torch.zeros(1, 3), "scale": torch.tensor([[1.6]]), "exp": torch.zeros(1, 21, 3)}


def rotation(pitch, yaw, roll):
    return torch.tensor([[float(pitch), float(yaw), float(roll)]])  # stands in for the engine's rotation


def test_the_directions_move_only_their_own_keypoints():
    open_direction, blink_direction = load_directions()
    others = [k for k in range(21) if k not in LIPS]
    assert open_direction[0, others].abs().max() == 0 and open_direction[0, LIPS].abs().max() > 0.1
    others = [k for k in range(21) if k not in EYES]
    assert blink_direction[0, others].abs().max() == 0 and blink_direction[0, EYES].abs().max() > 0.01


def test_with_the_mouth_shut_and_no_blink_only_the_head_sways_a_little(monkeypatch):
    monkeypatch.setattr(motion_module, "BLINK_EVERY", (100.0, 100.0))  # no blink in this test
    speech = SpeechMotion(neutral(), rotation, seed=3)
    for at in np.arange(0, 20, 0.5):
        info, turned = speech.reading(0.0, float(at))
        assert torch.equal(info["exp"], neutral()["exp"])
        assert abs(float(info["yaw"] - neutral()["yaw"])) <= motion_module.SWAY_DEG["yaw"] + 1e-6
        assert abs(float(info["roll"] - neutral()["roll"])) <= motion_module.SWAY_DEG["roll"] + 1e-6
        assert torch.equal(turned, rotation(info["pitch"], info["yaw"], info["roll"]))


def test_an_open_mouth_moves_the_lips_and_nods_the_head_slightly(monkeypatch):
    monkeypatch.setattr(motion_module, "BLINK_EVERY", (100.0, 100.0))
    shut = SpeechMotion(neutral(), rotation, seed=3).reading(0.0, 1.0)[0]
    wide = SpeechMotion(neutral(), rotation, seed=3).reading(1.0, 1.0)[0]
    moved = (wide["exp"] - shut["exp"]).abs()[0].sum(dim=1)
    assert set(torch.nonzero(moved).flatten().tolist()) <= set(LIPS) and moved[19] > 0.03
    assert float(wide["pitch"] - shut["pitch"]) == pytest.approx(motion_module.NOD_DEG)


def test_blinks_come_every_few_seconds_and_are_quick():
    speech = SpeechMotion(neutral(), rotation, seed=7)
    step = 1 / 30
    closed = [speech.blink(i * step) for i in range(int(60 / step))]
    starts = [i for i in range(1, len(closed)) if closed[i] > 0 and closed[i - 1] == 0]
    gaps = np.diff(starts) * step
    assert 9 <= len(starts) <= 25  # one every 2.5 to 6 seconds over a minute
    assert gaps.min() >= motion_module.BLINK_EVERY[0] and gaps.max() <= motion_module.BLINK_EVERY[1] + 0.3
    first = closed[starts[0]:]
    length = next(i for i, value in enumerate(first) if value == 0) * step
    assert length == pytest.approx(motion_module.BLINK_SECONDS, abs=step)
    assert max(first[:8]) > 0.9  # the eyes close fully once in a blink


def test_the_same_seed_gives_the_same_movement():
    a, b = SpeechMotion(neutral(), rotation, seed=11), SpeechMotion(neutral(), rotation, seed=11)
    for at in np.arange(0, 10, 0.1):
        assert torch.equal(a.reading(0.4, float(at))[0]["exp"], b.reading(0.4, float(at))[0]["exp"])
