"""ArcFace face embedding (InsightFace w600k_r50, run with ONNX Runtime)."""

from pathlib import Path

import cv2
import numpy as np

from .align import ALIGNED_SIZE

DEFAULT_ARCFACE = Path(__file__).resolve().parents[2] / "models" / "arcface" / "w600k_r50.onnx"


class ArcFaceEmbedder:
    """Turns an aligned 112x112 face into a 512-number, unit-length embedding.

    Runs on the CPU by default (about 35 ms per face on the development PC).
    ONNX Runtime's GPU build needs its own CUDA libraries, which are not the
    ones PyTorch ships, so the CPU is the setting that works on every machine.
    """

    def __init__(self, model_path=DEFAULT_ARCFACE, use_gpu=False):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {model_path}. Run: python scripts/download_models.py"
            )
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(str(model_path), options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        # ONNX Runtime falls back to the CPU if its CUDA libraries cannot be loaded.
        self.provider = self._session.get_providers()[0]

    def embed(self, aligned_bgr):
        if aligned_bgr.shape != (ALIGNED_SIZE, ALIGNED_SIZE, 3):
            raise ValueError(f"Expected an aligned {ALIGNED_SIZE}x{ALIGNED_SIZE} BGR face, got {aligned_bgr.shape}")
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis]
        embedding = self._session.run(None, {self._input_name: blob})[0][0]
        return embedding / np.linalg.norm(embedding)
