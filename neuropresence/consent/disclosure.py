"""The disclosure mark: every output frame says that it is not a camera picture.

A small label, "AI reenacted", in the lower left corner. It is drawn as the last step, on live
frames and on the still picture alike, and it has no switch: whoever sees the output, in a meeting
or in a recording of one, can tell what they are looking at.

The label is as tall as an eighteenth of the frame, and never smaller than 22 pixels, so that it
can still be read when a meeting application shows the picture small. It is made once for a frame
size and then only blended in, which takes a few hundredths of a millisecond a frame.
"""

from functools import lru_cache

import cv2
import numpy as np

LABEL = "AI reenacted"
HEIGHT_SHARE = 0.055  # the height of the label, as a share of the frame's height ...
MIN_HEIGHT_PX = 22  # ... and at least this, while the frame has the room
MARGIN_SHARE = 0.025  # its distance from the left and lower edges, as a share of the frame's height
BOX_BGR = (28, 24, 22)
BOX_OPACITY = 0.78
TEXT_BGR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DRAWN_LARGER = 4  # the label is drawn this many times too large and brought down, which keeps small letters clean


@lru_cache(maxsize=16)
def _label(frame_h, frame_w):
    """The label for a frame of this size: where it goes and what is blended in.

    Returns (x, y, drawn, keep, letters): the top left corner, what is added to each pixel, how
    much of the frame's own pixel is kept (0 to 1), and how much of each pixel is letter.
    """
    margin = max(2, round(MARGIN_SHARE * frame_h))
    height = max(MIN_HEIGHT_PX, round(HEIGHT_SHARE * frame_h))
    height = max(6, min(height, frame_h - 2 * margin))
    big = height * DRAWN_LARGER
    # The capital letters are a little under half as tall as the box. The scale is found from a trial size.
    (_, trial_h), _ = cv2.getTextSize(LABEL, FONT, 1.0, 2)
    scale = 0.46 * big / trial_h
    thickness = max(2, round(big / 14))
    (text_w, text_h), _ = cv2.getTextSize(LABEL, FONT, scale, thickness)
    pad = round(0.5 * big)
    big_w = text_w + 2 * pad
    box = np.zeros((big, big_w), dtype=np.uint8)
    radius = big // 3
    cv2.rectangle(box, (radius, 0), (big_w - 1 - radius, big - 1), 255, -1)
    cv2.rectangle(box, (0, radius), (big_w - 1, big - 1 - radius), 255, -1)
    for cx, cy in ((radius, radius), (big_w - 1 - radius, radius), (radius, big - 1 - radius), (big_w - 1 - radius, big - 1 - radius)):
        cv2.circle(box, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    text = np.zeros((big, big_w), dtype=np.uint8)
    cv2.putText(text, LABEL, (pad, (big + text_h) // 2), FONT, scale, 255, thickness, cv2.LINE_AA)
    width = max(4, min(round(big_w / DRAWN_LARGER), frame_w - 2 * margin))  # a frame too narrow for it: it is squeezed to fit
    box = cv2.resize(box, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    letters = cv2.resize(text, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cover = np.maximum(BOX_OPACITY * box, letters)[:, :, None]
    colour = (np.array(BOX_BGR, dtype=np.float32) * (1.0 - letters[:, :, None])
              + np.array(TEXT_BGR, dtype=np.float32) * letters[:, :, None])
    return margin, frame_h - margin - height, colour * cover, 1.0 - cover, letters


def mark(frame_bgr):
    """Draw the disclosure mark on the frame, in place, and return the frame."""
    frame_h, frame_w = frame_bgr.shape[:2]
    x, y, drawn, keep, _ = _label(frame_h, frame_w)
    height, width = keep.shape[:2]
    corner = frame_bgr[y:y + height, x:x + width]
    np.copyto(corner, (corner * keep + drawn).astype(np.uint8))
    return frame_bgr


def is_marked(frame_bgr):
    """Does this frame carry the mark? Bright letters on a dark box, where the label belongs."""
    frame_h, frame_w = frame_bgr.shape[:2]
    x, y, _, keep, letters = _label(frame_h, frame_w)
    height, width = keep.shape[:2]
    corner = frame_bgr[y:y + height, x:x + width].astype(np.float32).mean(axis=2)
    box = (letters < 0.02) & (keep[:, :, 0] < 0.3)
    if letters.sum() < 1.0 or not box.any():
        return False
    on_letters, on_box = float((corner * letters).sum() / letters.sum()), float(corner[box].mean())
    # The box lets a fifth of the picture through, so under it no pixel is brighter than 80 or so.
    return on_box <= 95.0 and on_letters - on_box >= 90.0
