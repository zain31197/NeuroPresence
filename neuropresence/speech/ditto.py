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

The lips are set differently. While a person speaks, Ditto's lips hover half open and rarely close
between syllables. Fitted on Zain's speech recording (27 October 2026) against his real lips, read as
the camera mode reads them: when his lips were shut mid sentence, Ditto's were still open by 0.015 at
the lower lip, and drawn so the mouth was parted 75% of the time where his was 53%, with a median lip
gap four times his. So Ditto's opening, at the lower lip's up and down, counts from LIP_REST, and
above it is scaled by LIP_GAIN, all the lips' movement with it (lip_share). Fitted by matching the two
spreads of opening, quantile by quantile: drawn this way, the opening's median, 90th percentile and
share of time open are within a few percent of his, and the lips shut when his did. Ditto's lips also
move one frame after his (r 0.83 at that lag, 0.74 without), which the drive allows for (live.py).

Ditto moves the lips a little even in silence. A gate keeps them shut while no voice is heard: the
voice is taken from the loudness detector (mouth.py), held open for a moment after the last sound so
that the ends of words are not cut, and the lips are eased back to the picture's own when it closes.
"""

import bisect
import sys
import threading
import time
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
LIP_REST = 0.015  # Ditto's lip opening (keypoint 19, up and down) that counts as shut while speaking: see above
LIP_GAIN = 0.57  # and its opening above that is scaled by this
LIP_POINT = (19, 1)  # the lower lip, up and down: the coordinate that moves most with the real lip gap
HEAD_SCALE = 0.45  # and the head's: Zain's head turned and nodded 0.8 degrees (spread) while he spoke, the voice
                  # driven one 1.8 and 2.1
BLINK_EVERY = (60, 100)  # frames between blinks, at random in this range: Ditto's own (motion_stitch.py)
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


def lip_share(change):
    """How much of Ditto's lip movement to draw, for its change from its first frame: none while its
    lower lip is open by less than LIP_REST, and above that the opening scaled by LIP_GAIN."""
    opening = float(change[0, LIP_POINT[0], LIP_POINT[1]])
    return LIP_GAIN * (opening - LIP_REST) / opening if opening > LIP_REST else 0.0


class DittoReadings:
    """Turns Ditto's frames into readings for engine.drive(motion=...), from the picture's own reading.

    `neutral` is engine.source_reading()[0]; `rotation` is engine.rotation; `blink` is Ditto's blink,
    15 frames of change to the eye keypoints (its configuration's delta_eye_arr), or None for no blinks.

    Three things are done as Ditto's own stitching does them, and two are measured settings. Seen beside
    Zain's camera (26 October 2026) the voice driven picture looked fake for three reasons:
    - it never blinked: Ditto's speech model leaves the eyes as they are, and its stitching adds blinks
      from a template every 2.4 to 4 seconds. That is done here now, the eyes driven by the template only;
    - the head moved 2.5 times as much as his did while he spoke (1.8 and 2.1 degrees of spread in turn and
      nod, against 0.8 and 0.8), so the head's movement is scaled by HEAD_SCALE;
    - drawn at about 16 frames a second from Ditto's 25, the frames shown were uneven steps apart, so the
      movement is taken between Ditto's two frames around the moment shown (between()).
    """

    def __init__(self, neutral, rotation, blink=None, seed=None):
        self.neutral = neutral
        self._rotation = rotation
        self._first = None
        lips = np.zeros((21, 3), np.float32)
        lips[LIPS] = 1.0
        self._lips = torch.from_numpy(lips).reshape(1, 21, 3)
        self._blink = None if blink is None else torch.as_tensor(np.asarray(blink, np.float32)).reshape(-1, 1, 21, 3)
        self._random = np.random.default_rng(seed)
        self._blinks = []  # the frames at which blinks start, made as far ahead as needed
        self._count = 0

    def _blink_at(self, number):
        """The change to the eye keypoints at this frame: Ditto's template while a blink runs, else none."""
        if self._blink is None:
            return None
        length = len(self._blink)
        while not self._blinks or self._blinks[-1] <= number:
            last = self._blinks[-1] + length if self._blinks else 0
            self._blinks.append(last + int(self._random.integers(BLINK_EVERY[0], BLINK_EVERY[1] + 1)))
        index = bisect.bisect_right(self._blinks, number) - 1
        if index < 0 or number - self._blinks[index] >= length:
            return None
        return self._blink[number - self._blinks[index]]

    def reading(self, frame, voice=1.0, number=None):
        """`frame` is one of Ditto's frames ({pitch, yaw, roll: 66 bins, t: 3, exp: 63, scale}); `voice`
        from 0 (silent: the lips are the picture's own) to 1; `number` the frame's number, for the blinks."""
        if number is None:
            number, self._count = self._count, self._count + 1
        if self._first is None:
            self._first = {"angles": _degrees(frame), "t": np.asarray(frame["t"], np.float32).reshape(1, 3),
                           "exp": np.asarray(frame["exp"], np.float32).reshape(1, 21, 3)}
        first = self._first
        angles = _degrees(frame)
        turn = {axis: (angles[axis] - first["angles"][axis]) * HEAD_SCALE for axis in angles}
        turn["pitch"] += TILT_DEG
        change = torch.from_numpy(np.asarray(frame["exp"], np.float32).reshape(1, 21, 3) - first["exp"])
        exp = (self.neutral["exp"] + change * self._lips * (lip_share(change) * float(voice))).clone()
        blink = self._blink_at(number)
        if blink is not None:
            exp = exp + blink
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
        shift = np.asarray(frame["t"], np.float32).reshape(1, 3) - first["t"]
        info["t"] = self.neutral["t"] + torch.from_numpy(shift) * HEAD_SCALE
        return info, self._rotation(info["pitch"], info["yaw"], info["roll"])

    def between(self, frame, after, share, voice=1.0, number=0):
        """The reading `share` of the way (0 to 1) from frame `number` to the next one, `after` (or None)."""
        info, rotation = self.reading(frame, voice, number)
        if after is None or share <= 0.0:
            return info, rotation
        later, _ = self.reading(after, voice, number + 1)
        blended = {key: torch.lerp(value, later[key], float(share)) for key, value in info.items()}
        return blended, self._rotation(blended["pitch"], blended["yaw"], blended["roll"])


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


# ------------------------------------------------------------- live, as speech arrives

STREAM_STEPS = 10  # denoising steps live: with the model compiled, about 64 ms a run on the RTX 5050
HOP = 5  # frames of new motion per run, of the 80 it writes. Ditto's own live setting is 10; measured on the
         # RTX 5050 with our engine drawing: hop 5, motion ready a median 0.54 s (at most 0.77 s) after the speech,
         # 16.5 frames a second drawn; hop 10, 0.84 s (at most 1.22 s), 20.4 a second; hop 7 between the two
PLAY_DELAY = 0.8  # with hop 5, the picture plays this far behind the voice: no frame was missing at it
CHUNK = (3, 5, 2)  # the speech encoder's window, in frames: before, new, after
WINDOW = sum(CHUNK) * SAMPLES_PER_FRAME + 80  # 6480 samples, as Ditto cuts them
PAD_FRONT = WINDOW - sum(CHUNK[1:]) * SAMPLES_PER_FRAME  # silence before the first sample, as Ditto's offline cut


class DittoStream(DittoMotion):
    """Ditto live: speech fed as it arrives, its frames ready one hop at a time, on a thread of its own.

    It works as Ditto's own live pipeline does. Every 5 frames (200 ms) of speech, once the 2 frames
    after them have arrived too, the speech encoder reads a window of 10 frames. Every 10 new frames of
    that, the motion model writes 80 frames, 70 of them over the previous run, and the 10 that the
    previous run left unfinished are blended with it and released. So a frame is ready only after the
    speech of the next hop has been heard: the price of motion that flows across the joins.

    The speech encoder runs on the CPU and the motion model on the graphics card, each on a thread of
    its own. One after the other they took 0.45 s for every 0.4 s of speech on the RTX 5050 and fell
    further behind with every sentence; side by side the slower of the two sets the pace. Call
    start(), wait for `ready_to_listen` (the motion model is compiled first), then feed speech.

    `ready` maps a frame number (25 a second, counted from the first sample fed) to Ditto's frame and
    the moment it was ready; `heard(n)` is the moment the speech of frame n had all arrived.
    """

    def __init__(self, steps=STREAM_STEPS, hop=HOP, compiled=True):
        super().__init__(steps)
        self.hop = hop
        self.compiled = compiled
        self.ready_to_listen = threading.Event()  # set once the motion model is compiled and warm
        self._lock = threading.Lock()  # the speech fed and read
        self._feature_lock = threading.Lock()  # the speech features made and used
        self._wake = threading.Event()
        self._features_in = threading.Event()
        self._stop = threading.Event()
        self._threads = []

    def start(self, picture):
        self.set_picture(picture)
        a2m, overlap = self.audio2motion, self.audio2motion.seq_frames - self.hop
        a2m.setup(self.conditions.x_s_info_0, overlap_v2=overlap, fix_kp_cond=0, sampling_timesteps=self.steps,
                  online_mode=True, v_min_max_for_clip=self.defaults.get("v_min_max_for_clip"),
                  smo_k_d=self.defaults.get("smo_k_d", 3))
        self._reset(overlap)
        self._stop.clear()
        self._threads = [threading.Thread(target=self._run_speech, name="ditto-speech", daemon=True),
                         threading.Thread(target=self._run_motion, name="ditto-motion", daemon=True)]
        for thread in self._threads:
            thread.start()

    def _reset(self, overlap):
        """Start the speech, its features and the motion afresh (a new session)."""
        self._audio = np.zeros(PAD_FRONT, np.float32)  # the speech not yet read, with Ditto's silence before it
        self._audio_from = 0  # the position of _audio[0] in the whole padded stream
        self._fed = 0  # samples of speech fed
        self._arrivals = []  # (samples fed so far, when)
        self._features = self.wav2feat.wav2feat(np.zeros(overlap * SAMPLES_PER_FRAME, np.float32))  # silence before
        self._next_feature = 0  # the frame whose features come next
        self._local, self._global, self._cond_start = 0, 0, -len(self._features)
        self._motion, self._valid_from, self._released = None, None, 0
        self.ready = {}
        self.runs = []  # how long each run of the motion model took

    def feed(self, samples):
        samples = np.asarray(samples, np.float32).reshape(-1)
        with self._lock:
            self._audio = np.concatenate([self._audio, samples])
            self._fed += len(samples)
            self._arrivals.append((self._fed, time.perf_counter()))
        self._wake.set()

    def heard(self, frame):
        """When the speech of this frame had all arrived, or None if it has not yet."""
        needed = (frame + 1) * SAMPLES_PER_FRAME
        with self._lock:
            index = bisect.bisect_left(self._arrivals, (needed, float("-inf")))
            return self._arrivals[index][1] if index < len(self._arrivals) else None

    def forget_before(self, frame):
        """Let go of the frames before this one, and of when their speech arrived: drawn, they are not needed."""
        for old in [key for key in self.ready if key < frame]:
            self.ready.pop(old, None)
        needed = (frame + 1) * SAMPLES_PER_FRAME
        with self._lock:
            index = bisect.bisect_left(self._arrivals, (needed, float("-inf")))
            del self._arrivals[:max(0, index - 1)]

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._features_in.set()
        for thread in self._threads:
            thread.join(timeout=5.0)

    def _run_speech(self):
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            while not self._stop.is_set() and self._read_speech():
                self._features_in.set()

    def _warm(self):
        """Compile the motion model's denoiser (CUDA graphs) and run it once, on this thread.

        Its many small steps cost more in starting work on the graphics card than in the work: one
        run of 5 steps took 242 ms as it is and 32 ms compiled, with the same output to four decimals.
        CUDA graphs belong to the thread that records them, so this is done on the motion thread.
        Compiling takes about 50 s the first time.
        """
        lmdm = self.audio2motion.lmdm
        network = lmdm.model.model
        # Silence made into conditions exactly as speech is, so that the compiled model meets the same shape
        # (1103 numbers a frame, not the 1059 the model's own setting gives) and is not compiled again live.
        with self._feature_lock:
            quiet = self._features[:lmdm.seq_frames]
        quiet = np.concatenate([quiet] + [quiet[-1:]] * (lmdm.seq_frames - len(quiet)), 0)
        silence = self.conditions(quiet, self._cond_start)[None].astype(np.float32)
        self.audio2motion.kp_cond = np.array(self.audio2motion.kp_cond, dtype=np.float32, copy=True)
        # Once as it is first: the rotary position embedding keeps what it computes on its first run, and
        # kept inside a CUDA graph it would be overwritten by the graph's next run.
        lmdm(self.audio2motion.kp_cond, silence, self.steps)
        if self.compiled and not getattr(network, "_compiled_by_us", False):
            network.guided_forward = torch.compile(network.guided_forward, mode="reduce-overhead", dynamic=False)
            network._compiled_by_us = True
        # Through the same path as live (a first run, then runs blended into it), so that every variant the
        # compiler makes is made now and not on the first sentence; then the motion is started afresh.
        a2m, motion = self.audio2motion, None
        for _ in range(3):
            a2m.kp_cond = np.array(a2m.kp_cond, dtype=np.float32, copy=True)
            motion = a2m(silence, motion)
        a2m.setup(self.conditions.x_s_info_0, overlap_v2=a2m.seq_frames - self.hop, fix_kp_cond=0,
                  sampling_timesteps=self.steps, online_mode=True,
                  v_min_max_for_clip=self.defaults.get("v_min_max_for_clip"), smo_k_d=self.defaults.get("smo_k_d", 3))
        self.ready_to_listen.set()

    def _run_motion(self):
        self._warm()
        while not self._stop.is_set():
            self._features_in.wait(0.05)
            self._features_in.clear()
            while not self._stop.is_set() and self._write_motion():
                pass

    def _read_speech(self):
        """Five more frames of speech features, if their window (and the two frames after it) has arrived."""
        start = self._next_feature * SAMPLES_PER_FRAME
        with self._lock:
            if self._audio_from + len(self._audio) < start + WINDOW:
                return False
            window = self._audio[start - self._audio_from:start - self._audio_from + WINDOW].copy()
        made = self.wav2feat(window, chunksize=CHUNK)
        with self._feature_lock:
            self._features = np.concatenate([self._features, made], 0)
        self._next_feature += CHUNK[1]
        with self._lock:  # what the next window will not need again
            keep_from = self._next_feature * SAMPLES_PER_FRAME
            self._audio = self._audio[keep_from - self._audio_from:]
            self._audio_from = keep_from
        return True

    def _write_motion(self):
        """One run of the motion model, if a hop of new speech features has come in."""
        a2m = self.audio2motion
        with self._feature_lock:
            if len(self._features) < self._local + a2m.seq_frames:
                return False
            window = self._features[self._local:self._local + a2m.seq_frames].copy()
        conditions = self.conditions(window, self._global + self._cond_start)
        # Ditto takes the frame to continue from as a slice of its growing history, whose memory layout
        # changes as it grows; the compiled model is tied to one layout and would be compiled again each run.
        a2m.kp_cond = np.array(a2m.kp_cond, dtype=np.float32, copy=True)  # a real copy: a slice of one row counts as compact to NumPy but keeps its odd stride
        began = time.perf_counter()
        self._motion = a2m(np.array(conditions[None], dtype=np.float32, copy=True), self._motion)
        self.runs.append(time.perf_counter() - began)
        if self._valid_from is None:  # the first run: its newest frames wait for the next one to be blended in
            self._valid_from = self._motion.shape[1] - a2m.fuse_length
        else:
            finished = a2m.cvt_fmt(self._motion[:, self._valid_from:self._valid_from + self.hop])
            now = time.perf_counter()
            for frame in finished:
                self.ready[self._released] = (frame, now)
                self._released += 1
            self._valid_from += self.hop
        self._local += self.hop
        self._global += self.hop
        if self._motion.shape[1] > a2m.seq_frames * 2:  # keep only what the blending can still need
            cut = self._motion.shape[1] - a2m.seq_frames * 2
            self._motion = self._motion[:, cut:]
            self._valid_from -= cut
        if self._local > a2m.seq_frames:  # and the features likewise
            cut = self._local - a2m.seq_frames
            with self._feature_lock:
                self._features = self._features[cut:]
            self._local -= cut
        return True

