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

The device has one size and one rate for as long as it is open, as a webcam has: 1280 x 720 at 22
frames a second. An upright picture is set in the middle with bars beside it.

Why 20, and why each frame is sent as it is finished (21 November 2026, scripts/study_meeting_cadence.py,
results/meeting_cadence_study.json). The engine finishes its frames evenly, 45 ms apart to within 1 ms. The
camera used to be opened at 30 and fed by a clock of its own, 30 times a second, with the newest frame. OBS
Virtual Camera's filter then picks frames up at the camera's rate on a clock of its own again. A frame drawn at
22 a second and picked up at 30 stays up for one pick up or two by turns: a third of the frames a meeting got
(34 and 37% in two runs) were held half again as long as their neighbours, so a steady turn of the head reached
it in short and long steps (16 ms of step error), and each frame waited 32 ms on average after it was finished.
Now 3.0 and 0.8% are held long, the step error is 6.7 and 2.0 ms, and the wait is 25 ms. Now each frame is sent once, as soon as it is
finished, and the camera is opened at the rate the drawing is held to while it is on (session.py: the
"even_delivery" switch): 22 since 22 November 2026. It was 20, what the engine reached during a call (20.0 to 20.2
on 23 October), until a Meet call on 22 November drew 17 to 18: the card is at its power limit and a call, the Live
Studio page and a screen recorder all take from it. Since then the heaviest layers of the networks run in 8 bit and
the studio holds its decoration still in a meeting, and a test call with both draws 23.6 a second where it drew 19.5.
Each frame is then picked up once, for the same 45 ms. Checked on the device itself: at 22 drawn on a camera at 22
no frame is held long, at 21.5 drawn 2.2% are, at 20 drawn 7.7%. On Linux v4l2loopback hands each frame written to
the reader, so there the frame sent as it is finished is the whole of it.

Checked on the device itself (scripts/study_virtual_camera.py): read back as a meeting application reads it,
41 and 38% of the frames were held long before, 0 and 2.1% now. It also found that a reader keeps the rate it
was offered when it opened the camera, and that with no writer OBS offers 30: the camera has to be on before a
meeting opens it, so the engine switches it back on at start if it was on when it stopped (runtime.py).

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
FPS = 22
KEEPALIVE_SECONDS = 0.5  # with no new frame for this long, the last one is sent again
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
    when nothing of the user may be shown.

    With `wait`, a function wait(seen, timeout) that returns as soon as a frame newer than number `seen` has been
    made (or the timeout passes) and gives the newest number, every new frame is sent once, as soon as it is
    finished, and the last one again only after KEEPALIVE_SECONDS without one. Without it, the newest frame is
    sent FPS times a second on a clock of the camera's own."""

    def __init__(self, picture, opener=open_device, size=SIZE, fps=FPS, wait=None):
        self._picture, self._opener, self._wait = picture, opener, wait
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
        self.fresh = 0  # of those, the frames of a session that had not been sent before

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

    def probe(self):
        """Can the device be opened? Returns the name meeting applications list it under. When the camera is on it
        is not opened a second time: it is in use, by this. Raises RuntimeError, with what to do, when it cannot."""
        with self._lock:
            if self.on:
                return self.device
            device = self._opener(self.size[0], self.size[1], self.fps)
            name = getattr(device, "device", None)
            closer = getattr(device, "close", None)
            if closer is not None:
                closer()
            return name

    def _start(self):
        try:
            device = self._opener(self.size[0], self.size[1], self.fps)
        except RuntimeError as err:
            self.reason = str(err)
            return
        self.reason, self.device = None, getattr(device, "device", None)
        self.on, self.sent, self.fresh = True, 0, 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(device,), name="virtual-camera", daemon=True)
        self._thread.start()

    def _end(self):
        self._stop.set()
        self._thread.join(timeout=3.0)
        self.on, self._thread, self.showing = False, None, "nothing"

    def _run(self, device):
        due = time.perf_counter()
        last, shown, sent_at, seen = None, None, None, None  # the frame last fitted, what was sent for it, when, and its number
        try:
            while not self._stop.is_set():
                frame = self._picture()
                if frame is None:
                    new = shown is not self._card
                    shown, last, self.showing = self._card, None, "card"
                else:
                    new = frame is not last
                    if new:  # a frame is fitted once, however often it is sent again
                        shown, last = fit(frame, self.size), frame
                        self.fresh += 1
                    self.showing = "session"
                now = time.perf_counter()
                if self._wait is None or new or sent_at is None or now - sent_at >= KEEPALIVE_SECONDS:
                    device.send(shown)
                    self.sent += 1
                    sent_at = now
                if self._wait is not None:
                    # Woken by the next frame as soon as it is finished; a frame made meanwhile is not missed,
                    # since the wait is for any frame after the number already seen.
                    seen = self._wait(seen, min(KEEPALIVE_SECONDS, 0.1))
                    continue
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
