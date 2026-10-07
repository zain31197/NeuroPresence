"""Stage 3: animate the enrolled source face with the live driving face.

The neural networks are LivePortrait's (third_party/LivePortrait). This module
splits their use into "prepare the source once" and "drive one frame", which is
what a live loop needs, and times each part.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..capture.crop import MAX_PICTURE_DIM, limit_size
from ..capture.steady import OneEuro

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


class ReenactmentEngine:
    def __init__(self, liveportrait_dir=LIVEPORTRAIT_DIR, half_precision=True):
        CropConfig, InferenceConfig, Wrapper, self._rotation, Cropper = _load_liveportrait(
            Path(liveportrait_dir)
        )
        self._crop_cfg = CropConfig()
        self._wrapper = Wrapper(InferenceConfig(flag_use_half_precision=half_precision))
        # The cropper runs once per source, so the CPU is fast enough, and it
        # avoids ONNX Runtime's GPU build, which needs CUDA libraries PyTorch lacks.
        self._cropper = Cropper(crop_cfg=self._crop_cfg, flag_force_cpu=True)
        self._source = None
        self._reference = None
        self._paste = None
        self.source_frame = None  # the full source image, shown as the static fallback frame
        self.source_crop = None  # 512x512 BGR face crop that the networks animate
        self.last_crop = None  # 512x512 BGR network output for the latest driven frame
        self.last_timing_ms = {}
        self.last_motion = {}
        self._steady = OneEuro(KEYPOINT_MIN_CUTOFF_HZ, KEYPOINT_BETA)

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
        source = w.prepare_source(cv2.resize(crop_rgb, (INPUT_SIZE, INPUT_SIZE),
                                             interpolation=cv2.INTER_AREA))
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
        self.source_frame = self.source_crop = self.last_crop = None

    def reset_reference(self):
        """Forget the neutral driving pose; the next driven frame becomes the new neutral."""
        self._reference = None
        self._steady.reset()

    def set_reference(self, driving_face_bgr):
        """Declare this cropped driving face to be the neutral pose.

        When the source picture was taken from the camera, passing the same
        frame here makes the output start exactly on the source: every later
        movement is then measured from the pose the picture itself shows.
        """
        info, rotation = self._read_motion(driving_face_bgr)
        self._reference = {"info": info, "rotation": rotation}
        self._steady.reset()

    def _read_motion(self, driving_face_bgr):
        w = self._wrapper
        face = cv2.resize(driving_face_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        info = w.get_kp_info(w.prepare_source(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
        return info, self._rotation(info["pitch"], info["yaw"], info["roll"])

    def drive(self, driving_face_bgr, at=None, steady=False, limit=False):
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
        driving, rotation = self._read_motion(driving_face_bgr)
        if self._reference is None:
            self._reference = {"info": driving, "rotation": rotation}
        ref = self._reference

        size = driving["scale"] / ref["info"]["scale"]
        if limit:
            turned = {axis: ref["info"][axis] + soft_limit(driving[axis] - ref["info"][axis], *POSE_RANGE_DEG[axis])
                      for axis in POSE_RANGE_DEG}
            rotation = self._rotation(turned["pitch"], turned["yaw"], turned["roll"])
            size = 1.0 + soft_limit(size - 1.0, *SCALE_RANGE)
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
        crop_rgb = w.parse_output(out)[0]
        render_done = self._now()

        self.last_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        frame = self._paste_back(self.last_crop)
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

    @staticmethod
    def peak_vram_gb():
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**3
