"""Packs a camera frame, and the output made from it, into one message for the browser.

Layout of a frame message (binary):
    4 bytes   length N of the header, unsigned, little-endian
    N bytes   header as UTF-8 JSON: id, kind, live, status, camera_bytes, output_bytes, output_size
    then      the camera picture as JPEG (camera_bytes long)
    then      the output as JPEG (output_bytes long; zero in the enrolment preview)

In a live session both pictures travel together, so the app always shows a
camera frame beside the output that was made from that same frame.
"""

import json
import struct

import cv2

CAMERA_WIDTH = {"live": 640, "preview": 960}  # the preview is the main picture on its screen
# The output is sent no larger than the 16:9 monitor it is shown in. An upright picture of 960 x 1280
# was sent whole and shown a quarter of that size, which the browser had to decode and scale 20 times a second.
OUTPUT_BOX = (1280, 720)
JPEG_QUALITY = 82
OVERLAY_COLOR = (120, 235, 90)  # BGR


def _fit(image, width, height=None):
    """Shrink the image to at most this width, and this height if one is given. Returns the image and the scale used."""
    h, w = image.shape[:2]
    scale = min(1.0, width / w, 1.0 if height is None else height / h)
    if scale >= 1.0:
        return image, 1.0
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA), scale


def _jpeg(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("The frame could not be encoded as JPEG.")
    return encoded.tobytes()


def pack_frame(pair, overlay=False):
    camera, scale = _fit(pair.camera, CAMERA_WIDTH[pair.kind])
    if overlay and pair.landmarks is not None:
        camera = camera.copy()
        for x, y in pair.landmarks * scale:
            cv2.circle(camera, (int(x), int(y)), 1, OVERLAY_COLOR, -1, cv2.LINE_AA)
        if pair.crop_window is not None:  # the window the driving face crop is cut from
            x, y, side = (value * scale for value in pair.crop_window)
            cv2.rectangle(camera, (int(round(x)), int(round(y))), (int(round(x + side)), int(round(y + side))), OVERLAY_COLOR, 1, cv2.LINE_AA)
    camera_jpeg = _jpeg(camera)
    output_jpeg = b"" if pair.output is None else _jpeg(_fit(pair.output, *OUTPUT_BOX)[0])
    header = json.dumps({
        "id": pair.id,
        "kind": pair.kind,
        "live": pair.live,
        "status": pair.status,
        "camera_bytes": len(camera_jpeg),
        "output_bytes": len(output_jpeg),
        # The size of the output itself, which is not always the size it is sent at.
        "output_size": None if pair.output is None else [int(pair.output.shape[1]), int(pair.output.shape[0])],
    }).encode("utf-8")
    return struct.pack("<I", len(header)) + header + camera_jpeg + output_jpeg


def unpack_frame(message):
    """Inverse of pack_frame: returns (header dict, camera JPEG bytes, output JPEG bytes)."""
    (length,) = struct.unpack_from("<I", message, 0)
    header = json.loads(message[4:4 + length].decode("utf-8"))
    start = 4 + length
    camera = message[start:start + header["camera_bytes"]]
    output = message[start + header["camera_bytes"]:start + header["camera_bytes"] + header["output_bytes"]]
    return header, camera, output
