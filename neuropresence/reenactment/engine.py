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
        self._cropper = Cropper(crop_cfg=self._crop_cfg)
        self._source = None
        self._reference = None
        self.source_crop = None  # 512x512 BGR, shown as the static fallback frame
        self.last_timing_ms = {}

    def set_source(self, image_bgr):
        """Crop the source face and cache everything that does not change per frame."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop = self._cropper.crop_source_image(rgb, self._crop_cfg)
        if crop is None:
            raise SourceError("No face found in the source image.")
        w = self._wrapper
        source = w.prepare_source(crop["img_crop_256x256"])
        info = w.get_kp_info(source)
        self._source = {
            "info": info,
            "canonical_kp": info["kp"],
            "rotation": self._rotation(info["pitch"], info["yaw"], info["roll"]),
            "feature": w.extract_feature_3d(source),
            "kp": w.transform_keypoint(info),
        }
        self.source_crop = cv2.cvtColor(crop["img_crop"], cv2.COLOR_RGB2BGR)
        self.reset_reference()

    def reset_reference(self):
        """Forget the neutral driving pose; the next driven frame becomes the new neutral."""
        self._reference = None

    def drive(self, driving_face_bgr):
        """Animate the source with one cropped driving face. Returns a 512x512 BGR frame.

        Motion is relative: the source moves by how much the driving face has
        moved since the reference frame, so the two faces need not match in
        pose or framing.
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
        kp_driving = scale_new * (s["canonical_kp"] @ rotation_new + expression_new) + shift_new
        kp_driving = w.stitching(s["kp"], kp_driving)
        motion_done = self._now()

        out = w.warp_decode(s["feature"], s["kp"], kp_driving)["out"]
        frame_rgb = w.parse_output(out)[0]
        render_done = self._now()

        self.last_timing_ms = {
            "motion": (motion_done - start) * 1000,
            "render": (render_done - motion_done) * 1000,
        }
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

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
