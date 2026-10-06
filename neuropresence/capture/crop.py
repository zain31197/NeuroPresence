"""Cut a square face region out of a frame for the reenactment stage."""

import cv2
import numpy as np

MAX_PICTURE_DIM = 1280  # an enrolled picture is never stored larger than this on its longer side


def square_face_crop(frame, bbox, scale=2.0, out_size=256):
    """Return an out_size x out_size crop centred on the face.

    bbox is (x, y, w, h) in pixels. The crop side is scale * the larger bbox
    side, so the head keeps some margin. Where the square runs past the frame
    edge, the edge pixels are repeated outward: the face keeps its proportions
    and the network never sees a hard black band.
    """
    x, y, w, h = bbox
    side = int(round(max(w, h) * scale))
    if side <= 0:
        raise ValueError(f"Empty face box: {bbox}")
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side

    fh, fw = frame.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
    if sx1 <= sx0 or sy1 <= sy0:
        raise ValueError(f"Face box lies outside the frame: {bbox}")
    crop = cv2.copyMakeBorder(frame[sy0:sy1, sx0:sx1], sy0 - y0, y1 - sy1, sx0 - x0, x1 - sx1,
                              cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def limit_size(image, max_dim):
    """Shrink the image so that neither side is longer than max_dim. Never enlarges."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def letterbox(frame, size):
    """Fit a frame inside a size x size black panel without distorting it."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    panel[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return panel
