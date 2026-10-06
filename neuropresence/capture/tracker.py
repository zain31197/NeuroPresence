"""Stage 1-2: locate the face and extract the driving signal (pose, expression)."""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


class TrackStatus(Enum):
    OK = "ok"
    NO_FACE = "no_face"
    # The system animates one enrolled user, so a second face is rejected
    # rather than guessed between.
    MULTIPLE_FACES = "multiple_faces"


@dataclass
class TrackResult:
    status: TrackStatus
    latency_ms: float
    landmarks: np.ndarray | None = None  # (N, 2) pixel coordinates
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in pixels
    pose_deg: tuple[float, float, float] | None = None  # yaw, pitch, roll
    blendshapes: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self):
        return self.status is TrackStatus.OK


def rotation_to_euler_deg(rotation):
    """Convert a 3x3 rotation matrix to (yaw, pitch, roll) in degrees.

    Uses R = Rz(roll) @ Ry(yaw) @ Rx(pitch). Columns are normalised first
    because the tracker's matrix can carry a scale factor.
    """
    r = np.asarray(rotation, dtype=np.float64)
    r = r / np.linalg.norm(r, axis=0, keepdims=True)
    yaw = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    pitch = math.atan2(r[2, 1], r[2, 2])
    roll = math.atan2(r[1, 0], r[0, 0])
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def classify_face_count(count):
    if count == 0:
        return TrackStatus.NO_FACE
    if count > 1:
        return TrackStatus.MULTIPLE_FACES
    return TrackStatus.OK


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker and returns one TrackResult per frame.

    video=True (default) is for a live stream: the face is followed from one
    frame to the next. video=False treats every image on its own, which is
    what measurements on unrelated images need.
    """

    def __init__(self, model_path=DEFAULT_MODEL, video=True):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face model not found at {model_path}. Run: python scripts/download_models.py"
            )
        self._video = video
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=2,  # 2 so a second person is detected, not silently ignored
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr):
        start = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            # VIDEO mode requires strictly increasing timestamps.
            ts_ms = max(int(start * 1000), self._last_ts_ms + 1)
            self._last_ts_ms = ts_ms
            raw = self._landmarker.detect_for_video(image, ts_ms)
        else:
            raw = self._landmarker.detect(image)

        status = classify_face_count(len(raw.face_landmarks))
        if status is not TrackStatus.OK:
            return TrackResult(status, (time.perf_counter() - start) * 1000)

        h, w = frame_bgr.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in raw.face_landmarks[0]], dtype=np.float32)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        matrix = np.asarray(raw.facial_transformation_matrixes[0])
        blendshapes = {c.category_name: c.score for c in raw.face_blendshapes[0]}
        return TrackResult(
            status=status,
            latency_ms=(time.perf_counter() - start) * 1000,
            landmarks=points,
            bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            pose_deg=rotation_to_euler_deg(matrix[:3, :3]),
            blendshapes=blendshapes,
        )

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
