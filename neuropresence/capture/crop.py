"""Cut a square face region out of a frame for the reenactment stage."""

import cv2
import numpy as np


def square_face_crop(frame, bbox, scale=2.0, out_size=256):
    """Return an out_size x out_size crop centred on the face.

    bbox is (x, y, w, h) in pixels. The crop side is scale * the larger bbox
    side, so the head keeps some margin. Where the square runs past the frame
    edge it is padded with black instead of being squeezed, so the face keeps
    its proportions.
    """
    x, y, w, h = bbox
    side = int(round(max(w, h) * scale))
    if side <= 0:
        raise ValueError(f"Empty face box: {bbox}")
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side

    fh, fw = frame.shape[:2]
    canvas = np.zeros((side, side, 3), dtype=frame.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
