"""The virtual camera: the output as a camera that a meeting application can select.

A meeting application knows nothing of NeuroPresence. It lists the cameras of the computer, and
one of them is fed by this module: OBS Virtual Camera on Windows, a v4l2loopback device on Linux,
both through pyvirtualcam. Whatever application reads it gets ordinary video.

What is sent is the finished frame of a running session, the same one the Live Studio shows. So
it carries the disclosure mark, and when the system falls back to the still picture (a lost face,
an output that no longer looks like the user, a late picture, someone else at the camera) the
meeting sees that fallback too. There is no second path that could go around them.

Nothing of the user is sent before the consent check has passed. Outside a running camera session
the camera shows a plain card. If it showed the enrolled picture instead, anyone at the computer
could put that picture into a meeting without answering the liveness prompt.

The device has one size and one rate for as long as it is open, as a webcam has: 1280 x 720 at 30
frames a second. An upright picture is set in the middle with bars beside it. The newest frame
is sent again until the next is drawn, so the application gets an even feed.

Meeting applications crop the camera picture to fill their tiles. A wide tile takes off the top and
bottom: Google Meet's main tile was 1.97 to 1 in a call on 18 October 2026, and cut the disclosure
mark in half. So the picture is kept within the middle 84% of the height, which a tile up to 2.1 to
1 leaves whole, mark included. A narrow tile takes off the sides, which the bars and the mark's
place in the middle of the lower edge allow for.

Measured on 14 October 2026 (Windows 11, OBS Studio 32.2.2, pyvirtualcam 0.15.0): the device opens
as "OBS Virtual Camera" and takes 29.9 frames a second.
"""

import threading
import time

import cv2
import numpy as np

SIZE = (1280, 720)  # width, height
FPS = 30
SAFE_HEIGHT_SHARE = 0.84  # the picture stays within this much of the height: whole in a tile up to 2.1 to 1

NOT_RUNNING = "NeuroPresence is not running"
NOT_RUNNING_HINT = "The picture appears here once a session has started."

INSTALL = {
    "win32": "Install OBS Studio (obsproject.com), which brings the OBS Virtual Camera, and start the engine again.",
    "linux": "Load the v4l2loopback module (sudo modprobe v4l2loopback exclusive_caps=1 card_label=NeuroPresence) "
             "and start the engine again.",
}


def fit(frame, size=SIZE, safe=SAFE_HEIGHT_SHARE):
    """The frame at the size of the device: scaled to fit the middle of it, clear of the band a wide
    tile crops off, with black around it."""
    width, height = size
    h, w = frame.shape[:2]
    scale = min(width / w, safe * height / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def card(size=SIZE):
    """What the camera shows when there is no session: a plain card, and nothing of the user."""
    width, height = size
    image = np.full((height, width, 3), (20, 15, 13), dtype=np.uint8)
    for text, scale, weight, colour, y in ((NOT_RUNNING, 1.3, 2, (240, 240, 240), height // 2 - 10),
                                           (NOT_RUNNING_HINT, 0.7, 1, (165, 160, 155), height // 2 + 40)):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
        cv2.putText(image, text, ((width - text_w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, weight, cv2.LINE_AA)
    return image


def open_device(width, height, fps):
    """The camera device of this platform, through pyvirtualcam. Raises RuntimeError with what to do if there is none."""
    import sys

    try:
        import pyvirtualcam
    except ImportError as err:
        raise RuntimeError("The virtual camera needs the pyvirtualcam package: pip install pyvirtualcam.") from err
    try:
        return pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
    except Exception as err:  # no driver installed, or the device is taken by another program
        hint = INSTALL.get(sys.platform, "Install a virtual camera that pyvirtualcam supports.")
        raise RuntimeError(f"No virtual camera could be opened ({err}). {hint}") from err


class VirtualCamera:
    """Feeds the camera device from `picture`, a function that returns the frame to show now, or None
    when nothing of the user may be shown. It is called FPS times a second on a thread of its own."""

    def __init__(self, picture, opener=open_device, size=SIZE, fps=FPS):
        self._picture, self._opener = picture, opener
        self.size, self.fps = size, fps
        self._card = card(size)
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.on = False
        self.device = None  # the name the meeting application lists it under
        self.reason = None  # why it could not be switched on, in words for the person
        self.showing = "nothing"  # "session": the output of a session; "card": the plain card
        self.sent = 0

    def switch(self, on):
        """Switch the camera on or off. Returns the state, with the reason if it could not be opened."""
        with self._lock:
            if on and not self.on:
                self._start()
            elif not on and self.on:
                self._end()
        return self.snapshot()

    def close(self):
        with self._lock:
            if self.on:
                self._end()

    def snapshot(self):
        return {"on": self.on, "device": self.device, "reason": self.reason, "showing": self.showing if self.on else "nothing",
                "size": list(self.size), "fps": self.fps}

    def _start(self):
        try:
            device = self._opener(self.size[0], self.size[1], self.fps)
        except RuntimeError as err:
            self.reason = str(err)
            return
        self.reason, self.device = None, getattr(device, "device", None)
        self.on, self.sent = True, 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(device,), name="virtual-camera", daemon=True)
        self._thread.start()

    def _end(self):
        self._stop.set()
        self._thread.join(timeout=3.0)
        self.on, self._thread, self.showing = False, None, "nothing"

    def _run(self, device):
        due = time.perf_counter()
        last, shown = None, None  # the frame last fitted, and what was sent for it
        try:
            while not self._stop.is_set():
                frame = self._picture()
                if frame is None:
                    shown, last, self.showing = self._card, None, "card"
                else:
                    if frame is not last:  # a frame is fitted once, however often it is sent again
                        shown, last = fit(frame, self.size), frame
                    self.showing = "session"
                device.send(shown)
                self.sent += 1
                due += 1.0 / self.fps
                wait = due - time.perf_counter()
                if wait > 0:
                    self._stop.wait(wait)
                else:
                    due = time.perf_counter()  # fell behind: do not try to catch up
        except Exception as err:  # the device was taken away, or a frame could not be sent
            self.reason = f"The virtual camera stopped: {err}"
            self.on, self.showing = False, "nothing"
        finally:
            try:
                device.send(self._card)  # the last thing a meeting sees is the card, never a frozen face
            except Exception:
                pass
            closer = getattr(device, "close", None)
            if closer is not None:
                closer()
