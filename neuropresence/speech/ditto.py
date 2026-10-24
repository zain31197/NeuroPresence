"""Ditto's speech to motion, drawn by our own engine.

Ditto (Ant Group, ACM MM 2025, Apache-2.0; fetched by scripts/setup_ditto.py) turns speech into
LivePortrait's own motion: a HuBERT speech encoder reads the sound, and a diffusion model writes, for
every frame at 25 a second, the head's angles (as 66 bins each), its shift, and the 21 expression
keypoints. On Zain's speech recording its mouth followed his real one far more closely than loudness
does (r = 0.77 against 0.44). Its own drawing runs in plain PyTorch and is too slow for live use on
this card, so only its speech part is used here, and the motion is handed to our engine, which draws
with TensorRT and puts the mark on every frame.

The conversion follows what Ditto itself does with its motion (motion_stitch.py): every frame is
taken relative to the first one it writes and added to the picture's own pose; of the expression only
the lips and the eyes are driven; the head is tilted down by two degrees; and the eyes are corrected
for the turn of the head, so that the gaze stays on the viewer.

One setting differs. Drawn by our engine, the same frames open the mouth 26% wider than Ditto's own
drawing does (measured on 110 frames, correlation 0.996), and with 10 denoising steps the mouth moves a
little more than with 50. On Zain's speech recording the mouth then opened to a median lip gap of 0.18
where his real one opened to 0.07 and Ditto's own drawing to 0.09. The lips' movement is therefore
halved (LIP_SCALE), which leaves when they move, and so the agreement with the real lips, as it is.

Ditto moves the lips a little even in silence. A gate keeps them shut while no voice is heard: the
voice is taken from the loudness detector (mouth.py), held open for a moment after the last sound so
that the ends of words are not cut, and the lips are eased back to the picture's own when it closes.
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch

DITTO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "ditto"
CONFIG = DITTO_DIR / "checkpoints" / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"
MODELS = DITTO_DIR / "checkpoints" / "ditto_pytorch"
FPS = 25  # Ditto writes motion at this rate
SAMPLES_PER_FRAME = 640  # at 16 kHz
LIPS, EYES = [6, 12, 14, 17, 19, 20], [11, 13, 15, 16, 18]  # LivePortrait's own (live_portrait_pipeline.py)
TILT_DEG = 2.0  # Ditto's own default (overall_ctrl_info delta_pitch)
GAZE_X, GAZE_Y = 0.26, 0.28  # Ditto's correction of the eyes per degree of turn (motion_stitch.py: _fix_gaze)
LIP_SCALE = 0.5  # the lips' movement is scaled by this before our engine draws it: see below
STEPS = 10  # denoising steps: as good as Ditto's default of 50 on Zain's recording (0.975 on the mouth), five times faster
VOICE_HOLD_SECONDS = 0.30  # the gate stays open this long after the last sound
GATE_OPEN_SECONDS = 0.04  # and opens and closes with these time constants
GATE_CLOSE_SECONDS = 0.15
VOICED = 0.02  # an openness from the loudness detector above this counts as a voice


def bin66_to_degree(bins):
    """LivePortrait's head angle from its 66 bin scores (as Ditto writes them), in degrees."""
    bins = np.asarray(bins, dtype=np.float64).reshape(-1, 66)
    weights = np.exp(bins - bins.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights * np.arange(66)).sum(axis=1) * 3 - 97.5


# ------------------------------------------------------------ the conversion


def _degrees(frame):
    return {axis: float(bin66_to_degree(frame[axis])[0]) for axis in ("pitch", "yaw", "roll")}


class DittoReadings:
    """Turns Ditto's frames into readings for engine.drive(motion=...), from the picture's own reading.

    `neutral` is engine.source_reading()[0]; `rotation` is engine.rotation.
    """

    def __init__(self, neutral, rotation):
        self.neutral = neutral
        self._rotation = rotation
        self._first = None
        mask = np.zeros((21, 3), np.float32)
        mask[LIPS + EYES] = 1.0
        self._driven = torch.from_numpy(mask).reshape(1, 21, 3)
        lips = np.zeros((21, 3), np.float32)
        lips[LIPS] = 1.0
        self._lips = torch.from_numpy(lips).reshape(1, 21, 3)

    def reading(self, frame, voice=1.0):
        """`frame` is one of Ditto's frames ({pitch, yaw, roll: 66 bins, t: 3, exp: 63, scale}); `voice`
        from 0 (silent: the lips are the picture's own) to 1."""
        if self._first is None:
            self._first = {"angles": _degrees(frame), "t": np.asarray(frame["t"], np.float32).reshape(1, 3),
                           "exp": np.asarray(frame["exp"], np.float32).reshape(1, 21, 3)}
        first = self._first
        angles = _degrees(frame)
        turn = {axis: angles[axis] - first["angles"][axis] for axis in angles}
        turn["pitch"] += TILT_DEG
        change = torch.from_numpy(np.asarray(frame["exp"], np.float32).reshape(1, 21, 3) - first["exp"]) * self._driven
        change = change * (1 - self._lips) + change * self._lips * (LIP_SCALE * float(voice))
        exp = (self.neutral["exp"] + change).clone()
        # The eyes follow the turn of the head back toward the viewer (Ditto: _fix_gaze, _eye_delta).
        dx, dy = turn["yaw"] * GAZE_X, turn["pitch"] * GAZE_Y
        flat = exp.reshape(1, 63)
        flat[0, 33] += dx * (0.0007 if dx > 0 else 0.001)
        flat[0, 45] += dx * (0.001 if dx > 0 else 0.0007)
        flat[0, 34] += dy * -0.001
        flat[0, 46] += dy * -0.001
        info = dict(self.neutral)
        info["exp"] = flat.reshape(1, 21, 3)
        for axis, degrees in turn.items():
            info[axis] = self.neutral[axis] + degrees
        info["t"] = self.neutral["t"] + torch.from_numpy(np.asarray(frame["t"], np.float32).reshape(1, 3) - first["t"])
        return info, self._rotation(info["pitch"], info["yaw"], info["roll"])


class VoiceGate:
    """Opens with a voice and closes a moment after it: how much the lips may move, 0 to 1, frame by frame."""

    def __init__(self, fps=FPS):
        self.fps = fps
        self.level = 0.0
        self._since_voice = float("inf")
        self._open = 1.0 - np.exp(-1.0 / (fps * GATE_OPEN_SECONDS))
        self._close = 1.0 - np.exp(-1.0 / (fps * GATE_CLOSE_SECONDS))

    def step(self, openness):
        """`openness` from the loudness detector for this frame. Returns how open the gate is."""
        self._since_voice = 0.0 if openness > VOICED else self._since_voice + 1.0 / self.fps
        target = 1.0 if self._since_voice <= VOICE_HOLD_SECONDS else 0.0
        self.level += (target - self.level) * (self._open if target > self.level else self._close)
        return self.level


# ------------------------------------------------------------ Ditto itself


def _blend_images_cy(mask_warped, frame_warped, frame_rgb, result):
    """Ditto's blend of a drawn face into the picture, in numpy: its own is Cython and needs a C compiler.
    Not used here (our engine draws), but Ditto imports it."""
    mask = mask_warped[..., None]
    np.copyto(result, np.clip(mask * frame_warped + (1.0 - mask) * frame_rgb, 0, 255).astype(np.uint8))


def _import_ditto():
    if not (DITTO_DIR / "core").exists():
        raise RuntimeError("Ditto is not installed. Run: python scripts/setup_ditto.py")
    if str(DITTO_DIR) not in sys.path:
        sys.path.insert(0, str(DITTO_DIR))
    if "core.utils.blend" not in sys.modules:
        blend = types.ModuleType("core.utils.blend")
        blend.blend_images_cy = _blend_images_cy
        sys.modules["core.utils.blend"] = blend


class DittoMotion:
    """Ditto's speech to motion for one picture: speech at 16 kHz in, Ditto's frames at 25 a second out."""

    def __init__(self, steps=STEPS):
        _import_ditto()
        from core.atomic_components.audio2motion import Audio2Motion
        from core.atomic_components.avatar_registrar import AvatarRegistrar
        from core.atomic_components.cfg import parse_cfg
        from core.atomic_components.condition_handler import ConditionHandler
        from core.atomic_components.wav2feat import Wav2Feat

        (registrar_cfg, conditions_cfg, lmdm_cfg, _, _, _, wav2feat_cfg, defaults) = parse_cfg(str(CONFIG), str(MODELS), {})
        self.defaults = defaults
        self.steps = steps
        self.registrar = AvatarRegistrar(**registrar_cfg)
        self.conditions = ConditionHandler(**conditions_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.wav2feat = Wav2Feat(**wav2feat_cfg)

    def set_picture(self, path):
        """Read the picture as Ditto does (its own crop and motion reading), to start every clip from."""
        d = self.defaults
        info = self.registrar(str(path), max_dim=d.get("max_size", 1920), n_frames=-1,
                              crop_scale=d.get("crop_scale", 2.3), crop_vx_ratio=d.get("crop_vx_ratio", 0),
                              crop_vy_ratio=d.get("crop_vy_ratio", -0.125), crop_flag_do_rot=d.get("crop_flag_do_rot", True))
        self.conditions.setup(info, d.get("emo", 4), eye_f0_mode=False, ch_info=None)
        self.audio2motion.setup(self.conditions.x_s_info_0, overlap_v2=d.get("overlap_v2", 10), fix_kp_cond=0,
                                sampling_timesteps=self.steps, online_mode=False,
                                v_min_max_for_clip=d.get("v_min_max_for_clip"), smo_k_d=d.get("smo_k_d", 3))

    def frames(self, audio):
        """All of Ditto's frames for a stretch of speech (16 kHz mono floats), as Ditto's offline pipeline makes them."""
        conditions = self.conditions(self.wav2feat.wav2feat(np.asarray(audio, np.float32)), 0)
        a2m, done, index = self.audio2motion, None, 0
        while index < len(conditions):
            clip = conditions[index:index + a2m.seq_frames][None]
            if clip.shape[1] < a2m.seq_frames:
                clip = np.concatenate([clip, np.stack([clip[:, -1]] * (a2m.seq_frames - clip.shape[1]), 1)], 1)
            done = a2m(clip, done)
            index += a2m.valid_clip_len
        done = a2m._smo(done[:, :len(conditions)], 0, len(conditions))
        return a2m.cvt_fmt(done)
