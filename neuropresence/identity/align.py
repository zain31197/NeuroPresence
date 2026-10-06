"""Move a face to the position ArcFace expects before it is embedded."""

import cv2
import numpy as np

ALIGNED_SIZE = 112

# Where ArcFace expects the two eyes, the nose tip and the two mouth corners in
# a 112x112 crop (the standard InsightFace template). Order: eye on the left of
# the image, eye on the right, nose, mouth corner on the left, on the right.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

# MediaPipe FaceMesh landmark numbers for the same five points, in the same order.
FIVE_POINT_INDICES = (468, 473, 4, 61, 291)


def five_points(landmarks):
    """Pick the five alignment points out of the 478 MediaPipe landmarks."""
    landmarks = np.asarray(landmarks)
    if landmarks.shape[0] <= max(FIVE_POINT_INDICES):
        raise ValueError(f"Expected 478 face landmarks, got {landmarks.shape[0]}")
    return landmarks[list(FIVE_POINT_INDICES)].astype(np.float64)


def similarity_transform(src, dst):
    """Best rotation + uniform scale + shift that maps src points onto dst.

    Least-squares solution (Umeyama, 1991). Returns a 2x3 matrix for
    cv2.warpAffine. It never mirrors the points.
    """
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 2:
        raise ValueError(f"Need two matching (N, 2) point sets, got {src.shape} and {dst.shape}")
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    variance = (src_c**2).sum() / len(src)
    if variance < 1e-12:
        raise ValueError("The source points all coincide.")

    u, singular, vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1] = -1  # without this the best fit would be a mirror image
    rotation = u @ np.diag(sign) @ vt
    scale = (singular * sign).sum() / variance
    shift = dst_mean - scale * rotation @ src_mean
    return np.hstack([scale * rotation, shift[:, None]])


def align_face(image_bgr, points, size=ALIGNED_SIZE):
    """Return the size x size crop in which the five points sit on the template."""
    matrix = similarity_transform(points, ARCFACE_TEMPLATE * (size / ALIGNED_SIZE))
    return cv2.warpAffine(image_bgr, matrix, (size, size), borderValue=0.0)
