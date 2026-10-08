"""Stage 3: animate the enrolled source face with the live driving face.

The neural networks are LivePortrait's (third_party/LivePortrait). This module
splits their use into "prepare the source once" and "drive one frame", which is
what a live loop needs, and times each part.
"""

import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..capture.crop import MAX_PICTURE_DIM, limit_size
from ..capture.steady import OneEuro

log = logging.getLogger("neuropresence.reenactment")
TENSORRT_DIR = Path(__file__).resolve().parents[2] / "models" / "tensorrt"  # converted engines, kept between runs
LIVEPORTRAIT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "LivePortrait"
INPUT_SIZE = 256  # LivePortrait network input
OUTPUT_SIZE = 512  # LivePortrait generator output
MAX_SOURCE_DIM = MAX_PICTURE_DIM  # larger source images are scaled down to this
# The keypoints given to the generator are filtered so a still head is drawn still (see capture/steady.py
# for the measurements). The filter lets go as soon as anything moves: mouth sync was unchanged.
KEYPOINT_MIN_CUTOFF_HZ = 2.0
KEYPOINT_BETA = 400.0  # speeds are in keypoint units per second, which are small numbers

# How far the head may turn from its neutral pose. The body in the picture stays still, so a head
# that turns as far as a real one can looks wrong on it: a real neck and shoulders would follow.
# Looked at on a picture turned in steps (8 October 2026): natural up to about 12 degrees of nod
# and 18 of turn; the face stretches from 20 degrees of nod and is distorted at 30 to 45. Movement
# is followed exactly up to the first number, then eases toward the second, which it never passes.
POSE_RANGE_DEG = {"pitch": (8.0, 15.0), "yaw": (12.0, 22.0), "roll": (8.0, 15.0)}
SCALE_RANGE = (0.03, 0.07)  # the same for the size of the head: it does not grow or shrink against the body

# A posture is not a gesture. A nod or a turn should show; leaning back in the chair and staying
# there should not leave the head tilted on a body that never moves, which looks glued on. So a
# pose held beyond the free range becomes, over a few seconds, the new rest position, and the head
# eases back to how it sits in the picture. Movement inside the free range is never touched.
POSTURE_SECONDS = 3.0  # how quickly a held posture becomes the rest position
POSTURE_SETTLED = 0.25  # the easing stops once the head is back within this share of the free range

# The model draws about as sharply as the face crop it is given, and that crop is softened on its way
# down to the model's 256 pixels. Strengthening its edges a little first gives the detail back. Measured
# on five pictures (8 October 2026), as the output's fine detail against the picture's own:
#   as it was 0.73 (worst picture 0.57);  at 0.2: 1.08 (worst 0.87);  at 0.5: 1.58, more than the picture has.
# Identity match to the picture stayed at 0.99. Averaging the picture down more carefully, the "correct"
# way, made the output softer (0.76 of what it was).
#
# It is off (0.0) because it is not free. In the full benchmark, at 0.1 and at 0.2 alike, the head
# tremble went from 0.99 of real video to 1.035, past its target of 1.0, and flicker from 0.75 to 0.81
# and 0.86: a sharper picture shows small movements more. At 0.1 the output keeps 0.93 of the picture's
# detail in place of 0.73. Whether that is worth 4% more tremble is a choice, not a measurement.
SOURCE_SHARPEN = 0.0


def soft_limit(value, free, most):
    """`value` unchanged while within `free` of zero, then easing toward `most`, which it never passes."""
    over = value.abs() - free
    eased = free + (most - free) * torch.tanh(over.clamp(min=0.0) / (most - free))
    return torch.where(over > 0, eased * value.sign(), value)


class SourceError(ValueError):
    """The source image cannot be used (no face found)."""




def _load_liveportrait(root):
    weights = root / "pretrained_weights" / "liveportrait" / "base_models" / "warping_module.pth"
    if not weights.exists():
        raise FileNotFoundError(
            f"LivePortrait is not set up at {root}. Run: python scripts/setup_liveportrait.py"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.config.crop_config import CropConfig
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.camera import get_rotation_matrix
    from src.utils.cropper import Cropper

    return CropConfig, InferenceConfig, LivePortraitWrapper, get_rotation_matrix, Cropper


class _HalfPrecision(torch.nn.Module):
    """Runs a network with half-precision weights and gives its results back in full precision.

    LivePortrait's own half-precision switch keeps the weights in full precision and converts them
    on every call. Converting once is faster (measured on the RTX 5050: render 93 to 86 ms) and
    changes the picture by 0.15 of 255 on average. Only the networks that draw the picture run in
    half precision; reading the movement, and the keypoint arithmetic, stay in full precision.
    """

    def __init__(self, network):
        super().__init__()
        self.network = network.half()

    def forward(self, *args, **kwargs):
        def half(value):
            return value.half() if torch.is_tensor(value) and value.is_floating_point() else value

        def full(value):
            if torch.is_tensor(value):
                return value.float()
            if isinstance(value, dict):
                return {key: full(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(full(item) for item in value)
            return value

        return full(self.network(*[half(arg) for arg in args], **{key: half(arg) for key, arg in kwargs.items()}))


class ReenactmentEngine:
    def __init__(self, liveportrait_dir=LIVEPORTRAIT_DIR, half_precision=True, compile_networks=False, tensorrt=False):
        # How much of the card was free before anything was loaded: the engine's own use is measured from it.
        self._gpu_free_before = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
        CropConfig, InferenceConfig, Wrapper, self._rotation, Cropper = _load_liveportrait(
            Path(liveportrait_dir)
        )
        self._crop_cfg = CropConfig()
        # The wrapper's own half-precision switch stays off: see _HalfPrecision.
        self._wrapper = Wrapper(InferenceConfig(flag_use_half_precision=False))
        if half_precision and torch.cuda.is_available():
            # Only the two networks that draw the picture. The one that reads the movement stays in full
            # precision: in half precision its readings were noisier, and the benchmark showed it at once
            # (tremble 0.98 to 1.46, flicker 0.73 to 1.12, mouth correlation 0.96 to 0.93).
            for name in ("warping_module", "spade_generator"):
                setattr(self._wrapper, name, _HalfPrecision(getattr(self._wrapper, name)))
        self._device = self._wrapper.device
        self.compiled = False  # True once the two heavy networks have been compiled for this GPU
        self.accelerated = None  # "tensorrt" or "compiled" once the two networks that draw have been made faster
        if tensorrt and half_precision and torch.cuda.is_available():
            self._use_tensorrt()
        if self.accelerated is None and compile_networks and torch.cuda.is_available():
            self._compile()
        # The cropper runs once per source, so the CPU is fast enough, and it
        # avoids ONNX Runtime's GPU build, which needs CUDA libraries PyTorch lacks.
        self._cropper = Cropper(crop_cfg=self._crop_cfg, flag_force_cpu=True)
        self._source = None
        self._reference = None
        self._paste = None
        self._gpu = None
        self.source_frame = None  # the full source image, shown as the static fallback frame
        self.source_crop = None  # 512x512 BGR face crop that the networks animate
        self._last_out = None  # the generator's latest output, still on the GPU
        self.last_timing_ms = {}
        self.last_motion = {}
        self._steady = OneEuro(KEYPOINT_MIN_CUTOFF_HZ, KEYPOINT_BETA)
        self._posture_at = None  # when the posture was last looked at
        self._settling = False  # True while a held posture is being taken as the new rest position
        # The generic defaults, until (and unless) calibrate_pose_range fits them to this person.
        self.pose_range_deg = dict(POSE_RANGE_DEG)
        self.scale_range = SCALE_RANGE

    def calibrate_pose_range(self, *, yaw_extreme_deg=None, pitch_up_extreme_deg=None, pitch_down_extreme_deg=None):
        """Replace the generic pose-range clamp with one fitted to this person's own registered
        poses, instead of the default (set by looking at one person's picture turned in steps).

        Each `most` is set to the angle this person actually showed, turned to the side or tilted
        up/down during enrolment (see enrolment/checks.py: left/right register yaw, up/down
        register pitch). `free`, the range followed exactly before easing starts, keeps the same
        ratio to `most` that the default pair had, so a person who turns further also gets to move
        further before the engine starts easing them to a stop, rather than everyone easing at the
        same fixed angle regardless of their own range.

        An axis whose extreme is not given (None) keeps whatever it already had: this can be
        called once per registered pose as it comes in, or once with everything at the end.
        """
        updated = dict(self.pose_range_deg)
        if yaw_extreme_deg is not None:
            most = max(abs(yaw_extreme_deg), POSE_RANGE_DEG["yaw"][0])
            updated["yaw"] = (most * (POSE_RANGE_DEG["yaw"][0] / POSE_RANGE_DEG["yaw"][1]), most)
        if pitch_up_extreme_deg is not None or pitch_down_extreme_deg is not None:
            most = max(abs(pitch_up_extreme_deg or 0.0), abs(pitch_down_extreme_deg or 0.0), POSE_RANGE_DEG["pitch"][0])
            updated["pitch"] = (most * (POSE_RANGE_DEG["pitch"][0] / POSE_RANGE_DEG["pitch"][1]), most)
        self.pose_range_deg = updated

    def _use_tensorrt(self):
        """Run the warping network and the generator through TensorRT (see accelerate.py for the measurements).

        The first time on a machine this converts them, which takes about a minute and a half;
        after that the converted engines are loaded from models/tensorrt. If TensorRT is not
        installed or the conversion fails, the networks are left as they were.
        """
        w = self._wrapper
        try:
            from .accelerate import to_tensorrt

            warping, generator = to_tensorrt(w.warping_module.network, w.spade_generator.network, TENSORRT_DIR)
            feature = torch.zeros(1, 32, 16, 64, 64, device=self._device)
            keypoints = torch.rand(1, 21, 3, device=self._device) * 0.2
            kept = w.warping_module, w.spade_generator
            w.warping_module, w.spade_generator = warping, generator
            try:
                w.warp_decode(feature, keypoints, keypoints * 1.01)  # prove it runs before relying on it
                torch.cuda.synchronize()
            except Exception:
                w.warping_module, w.spade_generator = kept
                raise
            self.accelerated = "tensorrt"
        except Exception as err:
            log.warning("TensorRT is not used: %s", (str(err).splitlines() or [type(err).__name__])[0][:200])
            return
        # The network that reads the movement, separately: if it does not convert, the other two still run fast.
        reader = w.motion_extractor
        try:
            from .accelerate import motion_to_tensorrt

            w.motion_extractor = motion_to_tensorrt(reader, TENSORRT_DIR)
            w.get_kp_info(torch.rand(1, 3, 256, 256, device=self._device))
            torch.cuda.synchronize()
        except Exception as err:
            w.motion_extractor = reader
            log.warning("The movement reader stays in PyTorch: %s", (str(err).splitlines() or [type(err).__name__])[0][:200])

    def _compile(self):
        """Compile the warping network and the generator for this GPU: about a minute, once per start.

        Measured on the RTX 5050: render 85 to 66 ms, the picture changed by 0.19 of 255. PyTorch's
        compiler needs Triton, which on Windows comes from the triton-windows package. If it is
        missing or the compile fails, the networks are left as they were and the engine still works.
        """
        w = self._wrapper
        heavy = [m.network if isinstance(m, _HalfPrecision) else m for m in (w.warping_module, w.spade_generator)]
        kept = [(holder, holder.network) for holder in (w.warping_module, w.spade_generator) if isinstance(holder, _HalfPrecision)]
        if len(kept) != 2:
            return  # only the half-precision arrangement has been measured
        try:
            for holder, network in kept:
                holder.network = torch.compile(network)
            feature = torch.zeros(1, 32, 16, 64, 64, device=self._device)
            keypoints = torch.rand(1, 21, 3, device=self._device) * 0.2
            for _ in range(3):  # the first call does the compiling; the next ones settle it
                w.warp_decode(feature, keypoints, keypoints * 1.01)
            torch.cuda.synchronize()
            self.compiled = True
            self.accelerated = "compiled"
        except Exception as err:
            for (holder, _), network in zip(kept, heavy):
                holder.network = network
            log.warning("The networks could not be compiled and run as they are: %s", str(err).splitlines()[0][:200])

    def set_source(self, image_bgr):
        """Crop the source face and cache everything that does not change per frame."""
        image_bgr = limit_size(image_bgr, MAX_SOURCE_DIM)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        found = self._cropper.crop_source_image(rgb, self._crop_cfg)
        if found is None:
            raise SourceError("No face found in the source image.")
        # Re-crop with the frame edge repeated outward. LivePortrait fills the
        # part of the crop that lies outside the image with black, and that
        # black band then gets warped into view when the head moves.
        to_crop, to_frame = found["M_o2c"][:2], found["M_c2o"][:2]
        crop_rgb = cv2.warpAffine(rgb, to_crop, (OUTPUT_SIZE, OUTPUT_SIZE),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        w = self._wrapper
        small = cv2.resize(crop_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        small += SOURCE_SHARPEN * (small - cv2.GaussianBlur(small, (0, 0), 1.0))  # see SOURCE_SHARPEN
        source = w.prepare_source(np.clip(small, 0, 255).astype(np.uint8))
        info = w.get_kp_info(source)
        self._source = {
            "info": info,
            "canonical_kp": info["kp"],
            "rotation": self._rotation(info["pitch"], info["yaw"], info["roll"]),
            "feature": w.extract_feature_3d(source),
            "kp": w.transform_keypoint(info),
        }
        self.source_frame = image_bgr
        self.source_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        self._paste = self._prepare_paste_back(image_bgr, to_frame)
        self._gpu = None  # the GPU's copy of it, made when first needed
        self.reset_reference()

    def _prepare_paste_back(self, frame_bgr, to_frame):
        """Precompute how the animated crop is blended back into the source frame.

        Only the rectangle the crop covers is touched each frame; the rest of
        the frame is the untouched source, so the background cannot drift.
        """
        h, w = frame_bgr.shape[:2]
        mask = self._wrapper.inference_cfg.mask_crop.astype(np.float32) / 255.0
        mask_frame = cv2.warpAffine(mask, to_frame, (w, h), flags=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(mask_frame.max(axis=2) > 0)
        if len(xs) == 0:
            raise SourceError("The face crop does not overlap the source image.")
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        mask_roi = mask_frame[y0:y1, x0:x1]
        to_roi = to_frame.copy()
        to_roi[:, 2] -= (x0, y0)
        return {
            "roi": (x0, y0, x1, y1),
            "to_roi": to_roi,
            "mask": mask_roi,
            "background": (1.0 - mask_roi) * frame_bgr[y0:y1, x0:x1].astype(np.float32),
        }

    @property
    def last_crop(self):
        """The generator's 512x512 output for the latest driven frame, as a BGR picture."""
        if self._last_out is None:
            return None
        return cv2.cvtColor(self._wrapper.parse_output(self._last_out)[0], cv2.COLOR_RGB2BGR)

    def _prepare_compose(self):
        """What the GPU needs to blend the generator's output into the frame: done once per picture.

        The same arithmetic as _paste_back, on the GPU: for every pixel of the region the face
        covers, where in the 512x512 output it comes from, the blending mask, and the part of the
        frame that shows through. Blending there and bringing back one finished region is faster
        than bringing the output back and blending on the CPU.
        """
        p = self._paste
        x0, y0, x1, y1 = p["roi"]
        to_crop = cv2.invertAffineTransform(p["to_roi"])
        xs, ys = np.meshgrid(np.arange(x1 - x0, dtype=np.float32), np.arange(y1 - y0, dtype=np.float32))
        u = to_crop[0, 0] * xs + to_crop[0, 1] * ys + to_crop[0, 2]
        v = to_crop[1, 0] * xs + to_crop[1, 1] * ys + to_crop[1, 2]
        # grid_sample wants positions from -1 to 1 across the picture, measured at pixel centres.
        grid = np.stack([(2.0 * u + 1.0) / OUTPUT_SIZE - 1.0, (2.0 * v + 1.0) / OUTPUT_SIZE - 1.0], axis=-1)
        chw = lambda image: torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(self._device)[None]  # noqa: E731
        return {"grid": torch.from_numpy(grid).to(self._device)[None], "mask": chw(p["mask"]) * 255.0,
                "background": chw(p["background"])}

    def _compose(self, out):
        """Blend the generator's output (1x3x512x512, RGB, 0 to 1, on the GPU) into the source frame."""
        if self._gpu is None:
            self._gpu = self._prepare_compose()
        g = self._gpu
        bgr = out[:, [2, 1, 0]].clamp(0.0, 1.0).float()
        warped = torch.nn.functional.grid_sample(bgr, g["grid"], mode="bilinear", padding_mode="zeros", align_corners=False)
        region = (g["mask"] * warped + g["background"]).clamp_(0.0, 255.0).to(torch.uint8)
        x0, y0, x1, y1 = self._paste["roi"]
        frame = self.source_frame.copy()
        frame[y0:y1, x0:x1] = region[0].permute(1, 2, 0).cpu().numpy()
        return frame

    def _paste_back(self, crop_bgr):
        p = self._paste
        x0, y0, x1, y1 = p["roi"]
        warped = cv2.warpAffine(crop_bgr, p["to_roi"], (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        frame = self.source_frame.copy()
        frame[y0:y1, x0:x1] = (p["mask"] * warped + p["background"]).astype(np.uint8)
        return frame

    def clear_source(self):
        """Forget the source picture. drive() cannot be used until set_source() is called again."""
        self._source = self._paste = self._reference = None
        self.source_frame = self.source_crop = self._last_out = None

    def reset_reference(self):
        """Forget the neutral driving pose; the next driven frame becomes the new neutral."""
        self._reference = None
        self._steady.reset()
        self._posture_at, self._settling = None, False

    def set_reference(self, driving_face_bgr):
        """Declare this cropped driving face to be the neutral pose.

        When the source picture was taken from the camera, passing the same
        frame here makes the output start exactly on the source: every later
        movement is then measured from the pose the picture itself shows.
        """
        info, rotation = self._read_motion(driving_face_bgr)
        self._reference = {"info": info, "rotation": rotation}
        self._steady.reset()
        self._posture_at, self._settling = None, False

    def _follow_posture(self, driving, at):
        """Let a posture held beyond the free range become the rest position, a little each frame."""
        now = time.perf_counter() if at is None else at
        elapsed = 0.0 if self._posture_at is None else min(max(now - self._posture_at, 0.0), 0.5)
        self._posture_at = now
        rest = self._reference["info"]
        # How far the head is from rest on its worst axis, in free ranges: above 1 it is outside.
        away = max(float((driving[axis] - rest[axis]).abs().max()) / self.pose_range_deg[axis][0] for axis in self.pose_range_deg)
        if away > 1.0:
            self._settling = True
        elif away < POSTURE_SETTLED:
            self._settling = False
        if not self._settling or elapsed == 0.0:
            return
        share = 1.0 - math.exp(-elapsed / POSTURE_SECONDS)
        moved = dict(rest)
        for key in ("pitch", "yaw", "roll", "t", "scale"):  # where the head is, not what the face is doing
            moved[key] = rest[key] + (driving[key] - rest[key]) * share
        self._reference = {"info": moved, "rotation": self._rotation(moved["pitch"], moved["yaw"], moved["roll"])}

    def _read_motion(self, driving_face_bgr):
        w = self._wrapper
        face = cv2.resize(driving_face_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        info = w.get_kp_info(w.prepare_source(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
        return info, self._rotation(info["pitch"], info["yaw"], info["roll"])

    def read(self, driving_face_bgr):
        """Read the movement from a driving face crop, to be handed to drive() as `motion`.

        It does not depend on the picture being animated, so the live session does it on the
        tracking thread while the GPU is still drawing the frame before.
        """
        return self._read_motion(driving_face_bgr)

    def drive(self, driving_face_bgr, at=None, steady=False, limit=False, motion=None):
        """Animate the source with one cropped driving face.

        Returns the source frame (same size as the source image, BGR) with the
        animated face blended in. Motion is relative: the source moves by how
        much the driving face has moved since the reference frame, so the two
        faces need not match in pose or framing.

        With steady=True the keypoints are filtered over time, so a head that is
        still is drawn still; `at` is when the frame was taken, in seconds.
        With limit=True the head keeps to the range that looks right on a still
        body (POSE_RANGE_DEG): inside it nothing changes.
        """
        if self._source is None:
            raise RuntimeError("Call set_source() before drive().")
        w, s = self._wrapper, self._source

        start = self._now()
        driving, rotation = motion if motion is not None else self._read_motion(driving_face_bgr)
        if self._reference is None:
            self._reference = {"info": driving, "rotation": rotation}
        if limit:
            self._follow_posture(driving, at)
        ref = self._reference

        size = driving["scale"] / ref["info"]["scale"]
        if limit:
            turned = {axis: ref["info"][axis] + soft_limit(driving[axis] - ref["info"][axis], *self.pose_range_deg[axis])
                      for axis in self.pose_range_deg}
            rotation = self._rotation(turned["pitch"], turned["yaw"], turned["roll"])
            size = 1.0 + soft_limit(size - 1.0, *self.scale_range)
        rotation_new = (rotation @ ref["rotation"].permute(0, 2, 1)) @ s["rotation"]
        expression_new = s["info"]["exp"] + (driving["exp"] - ref["info"]["exp"])
        scale_new = s["info"]["scale"] * size
        shift_new = s["info"]["t"] + (driving["t"] - ref["info"]["t"])
        shift_new[..., 2] = 0
        # How far the driving face has moved from its neutral pose (for logs and limits).
        self.last_motion = {
            "shift_x": float(driving["t"][0, 0] - ref["info"]["t"][0, 0]),
            "shift_y": float(driving["t"][0, 1] - ref["info"]["t"][0, 1]),
            "scale": float(driving["scale"][0, 0] / ref["info"]["scale"][0, 0]),
            "pitch_deg": float(driving["pitch"][0, 0] - ref["info"]["pitch"][0, 0]),
            "yaw_deg": float(driving["yaw"][0, 0] - ref["info"]["yaw"][0, 0]),
        }
        kp_driving = scale_new * (s["canonical_kp"] @ rotation_new + expression_new) + shift_new
        kp_driving = w.stitching(s["kp"], kp_driving)
        if steady:
            held = self._steady(kp_driving.detach().float().cpu().numpy(), time.perf_counter() if at is None else at)
            kp_driving = torch.as_tensor(held, dtype=kp_driving.dtype, device=kp_driving.device)
        motion_done = self._now()

        out = w.warp_decode(s["feature"], s["kp"], kp_driving)["out"]
        render_done = self._now()

        self._last_out = out
        frame = self._compose(out) if out.is_cuda else self._paste_back(self.last_crop)
        self.last_timing_ms = {
            "motion": (motion_done - start) * 1000,
            "render": (render_done - motion_done) * 1000,
            "compose": (time.perf_counter() - render_done) * 1000,
        }
        return frame

    @staticmethod
    def _now():
        # GPU calls return before the work finishes; wait so timings are real.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def gpu_memory_gb(self):
        """GPU memory the engine holds, in GB, as the driver sees it.

        PyTorch's own count leaves out what TensorRT holds, so this is the drop in the card's free
        memory since the engine began loading. Other programs that start or stop meanwhile shift it.
        """
        if self._gpu_free_before is None:
            return 0.0
        return max(0.0, (self._gpu_free_before - torch.cuda.mem_get_info()[0]) / 1024**3)

    @staticmethod
    def peak_vram_gb():
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**3
