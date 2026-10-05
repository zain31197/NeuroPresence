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

LIVEPORTRAIT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "LivePortrait"
INPUT_SIZE = 256  # LivePortrait network input
OUTPUT_SIZE = 512  # LivePortrait generator output
MAX_SOURCE_DIM = 1280  # larger source images are scaled down to this


class SourceError(ValueError):
    """The source image cannot be used (no face found)."""


def _limit_size(image, max_dim):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


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
        self._cropper = Cropper(crop_cfg=self._crop_cfg)
        self._source = None
        self._reference = None
        self._paste = None
        self.source_frame = None  # the full source image, shown as the static fallback frame
        self.source_crop = None  # 512x512 BGR face crop that the networks animate
        self.last_timing_ms = {}
        self.last_motion = {}

    def set_source(self, image_bgr):
        """Crop the source face and cache everything that does not change per frame."""
        image_bgr = _limit_size(image_bgr, MAX_SOURCE_DIM)
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

    def reset_reference(self):
        """Forget the neutral driving pose; the next driven frame becomes the new neutral."""
        self._reference = None

    def drive(self, driving_face_bgr):
        """Animate the source with one cropped driving face.

        Returns the source frame (same size as the source image, BGR) with the
        animated face blended in. Motion is relative: the source moves by how
        much the driving face has moved since the reference frame, so the two
        faces need not match in pose or framing.
        """
        if self._source is None:
            raise RuntimeError("Call set_source() before drive().")
        w, s = self._wrapper, self._source

        start = self._now()
        face = cv2.resize(driving_face_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        driving = w.get_kp_info(w.prepare_source(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
        rotation = self._rotation(driving["pitch"], driving["yaw"], driving["roll"])
        if self._reference is None:
            self._reference = {"info": driving, "rotation": rotation}
        ref = self._reference

        rotation_new = (rotation @ ref["rotation"].permute(0, 2, 1)) @ s["rotation"]
        expression_new = s["info"]["exp"] + (driving["exp"] - ref["info"]["exp"])
        scale_new = s["info"]["scale"] * (driving["scale"] / ref["info"]["scale"])
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
        motion_done = self._now()

        out = w.warp_decode(s["feature"], s["kp"], kp_driving)["out"]
        crop_rgb = w.parse_output(out)[0]
        render_done = self._now()

        frame = self._paste_back(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
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
