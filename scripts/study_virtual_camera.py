"""What a meeting application really gets from OBS Virtual Camera, read back from the device itself (Windows).

scripts/study_meeting_cadence.py replays a session's frame times through a model of how the camera's frames are
picked up. This checks that model on the real device. Numbered test frames (a row of black and white blocks,
nothing of anyone) are written into OBS Virtual Camera with pyvirtualcam, the way the engine writes its output,
and read back by ffmpeg through DirectShow, the way a meeting application reads a camera. From which number
arrives when it measures, for each way of writing:

- new frames a second, and the share of frames written that never arrive;
- the share of new frames held half again as long as is typical, and the step error: from one new frame to the
  next, the time it stayed up less the time between their being written. Reading through ffmpeg's pipe adds a few
  ms of its own to every arrival, so about 8 ms of step error is the floor of this measurement;
- what rate a reader is offered with no writer, and whether a reader that is already reading follows a writer
  that is opened again at another rate.

Only the virtual device is opened, by its name ("OBS Virtual Camera"); no real camera and no microphone. The
engine's own virtual camera must be off while this runs, since the device takes one writer. Needs OBS Studio.
Writes results/virtual_camera_study.json.

Usage:
    python scripts/study_virtual_camera.py          # about three minutes
"""

import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "OBS Virtual Camera"
W, H = 1280, 720  # what the engine opens it at (server/virtual_camera.py)
SW, SH = 320, 180  # what is read back
BITS = 12
WARM_SECONDS, SECONDS = 3.0, 20.0

CASES = [
    # camera rate, rate frames are made at, how they are written, name
    (30, 22.0, "30 Hz clock", "Before: camera at 30, drawn at 22, sent on a 30 a second clock"),
    (20, 20.0, "as made", "Camera session now: camera at 20, drawn at 20, each sent as it is made"),
    (20, 22.0, "as made", "Even delivery off: camera at 20, drawn at 22, each sent as it is made"),
    (30, 16.5, "30 Hz clock", "Voice mode before: camera at 30, drawn at 16.5, 30 a second clock"),
    (20, 16.5, "as made", "Voice mode now: camera at 20, drawn at 16.5, each sent as it is made"),
    (30, 16.5, "as made", "Voice mode on a camera at 30, each sent as it is made"),
    (30, 15.0, "as made", "Voice mode held to 15, on a camera at 30"),
]


def ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def picture(number):
    frame = np.full((H, W, 3), 60, np.uint8)
    for bit in range(BITS):
        if number >> bit & 1:
            frame[300:400, 40 + bit * 100:140 + bit * 100] = 255
    return frame


def number_of(gray):
    value = 0
    for bit in range(BITS):
        if gray[350 * SH // H, (90 + bit * 100) * SW // W] > 160:
            value |= 1 << bit
    return value


def reader():
    return subprocess.Popen([ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "dshow", "-rtbufsize", "64M",
                             "-i", f"video={DEVICE}", "-vf", f"scale={SW}:{SH}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                            stdout=subprocess.PIPE)


def offered_rate():
    """The frame rate the device offers a reader now."""
    text = subprocess.run([ffmpeg(), "-hide_banner", "-f", "dshow", "-list_options", "true", "-i", f"video={DEVICE}"],
                          capture_output=True, text=True).stderr
    rates = {float(part.split("=")[1]) for line in text.splitlines() for part in line.split() if part.startswith("fps=")}
    return sorted(rates)


def run(camera_fps, made_fps, writing):
    import pyvirtualcam

    pictures = [picture(i) for i in range(1 << BITS)]
    made, arrived = [], []
    stop, opened = threading.Event(), threading.Event()

    def write():
        with pyvirtualcam.Camera(W, H, camera_fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            opened.set()
            start = time.perf_counter() + 0.2
            k, tick, newest = 0, start, None
            while not stop.is_set():
                if writing == "as made":
                    time.sleep(max(0.0, start + k / made_fps - time.perf_counter()))
                    made.append(time.perf_counter())
                    cam.send(pictures[k % len(pictures)])
                    k += 1
                else:  # a clock of the camera's own, sending the newest frame made by then
                    time.sleep(max(0.0, tick - time.perf_counter()))
                    now = time.perf_counter()
                    while start + k / made_fps <= now:
                        made.append(start + k / made_fps)
                        newest, k = k, k + 1
                    if newest is not None:
                        cam.send(pictures[newest % len(pictures)])
                    tick += 1.0 / camera_fps

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    if not opened.wait(5):
        raise SystemExit(f"{DEVICE} could not be opened for writing. Is the engine's virtual camera on, or OBS missing?")
    time.sleep(0.5)
    proc = reader()
    end = time.perf_counter() + WARM_SECONDS + SECONDS
    size = SW * SH
    while time.perf_counter() < end:
        data = proc.stdout.read(size)
        if len(data) < size:
            break
        arrived.append((time.perf_counter(), number_of(np.frombuffer(data, np.uint8).reshape(SH, SW))))
    proc.kill()
    stop.set()
    writer.join(3)
    if not arrived:
        raise SystemExit("Nothing was read back from the device.")

    shown = [(t, n) for t, n in arrived if t >= arrived[0][0] + WARM_SECONDS]
    firsts, last = [], None
    for t, n in shown:
        if n != last:
            firsts.append((t, n))
            last = n
    # The numbers wrap every 4096 frames: each is taken as the frame of that number made nearest before it arrived.
    def index(t, n):
        candidates = [i for i in range(n, len(made), 1 << BITS) if made[i] <= t]
        return candidates[-1] if candidates else None
    firsts = [(t, i) for t, i in ((t, index(t, n)) for t, n in firsts) if i is not None]
    holds = [(b[0] - a[0]) * 1000 for a, b in zip(firsts, firsts[1:])]
    steps = [((b[0] - a[0]) - (made[b[1]] - made[a[1]])) * 1000 for a, b in zip(firsts, firsts[1:])]
    late = [(t - made[i]) * 1000 for t, i in firsts]
    window = [i for i in range(len(made)) if firsts[0][0] <= made[i] <= firsts[-1][0] - 0.2]
    seen = {i for _, i in firsts}
    typical = statistics.median(holds)
    return {
        "camera_fps": camera_fps,
        "made_fps": made_fps,
        "written": writing,
        "read_per_second": round(len(shown) / (shown[-1][0] - shown[0][0]), 2),
        "new_per_second": round(len(firsts) / (firsts[-1][0] - firsts[0][0]), 2),
        "never_arrived": round(sum(1 for i in window if i not in seen) / max(1, len(window)), 4),
        "held_half_again_longer": round(sum(1 for h in holds if h > 1.5 * typical) / len(holds), 4),
        "step_error_ms_sd": round(statistics.pstdev(steps), 2),
        "late_ms_mean": round(statistics.mean(late), 1),
    }


def switch_rate():
    """A reader opened while the writer runs at 20; the writer then opened again at 30. What does the reader read?"""
    import pyvirtualcam

    arrived, stop = [], threading.Event()

    def write(rate, seconds, level):
        with pyvirtualcam.Camera(W, H, rate, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            end, k = time.perf_counter() + seconds, 0
            while time.perf_counter() < end:
                cam.send(np.full((H, W, 3), level + (k % 2) * 40, np.uint8))
                k += 1
                time.sleep(1.0 / rate)

    def read(proc):
        while not stop.is_set():
            data = proc.stdout.read(SW * SH)
            if len(data) < SW * SH:
                break
            arrived.append(time.perf_counter())

    first = threading.Thread(target=write, args=(20, 8.0, 60))
    first.start()
    time.sleep(1.0)
    proc = reader()
    threading.Thread(target=read, args=(proc,), daemon=True).start()
    first.join()
    switched = time.perf_counter()
    time.sleep(0.3)
    write(30, 8.0, 150)
    stop.set()
    proc.kill()

    def rate(a, b):
        return round(sum(1 for t in arrived if a <= t < b) / (b - a), 1)

    return {"read_while_writing_at_20": rate(arrived[0] + 1.0, switched - 0.2) if arrived else None,
            "read_after_the_writer_opened_again_at_30": rate(switched + 1.5, switched + 7.5)}


def main():
    if sys.platform != "win32":
        raise SystemExit("This reads the device back through DirectShow, so it runs on Windows. On Linux v4l2loopback "
                         "hands every frame written to the reader; scripts/study_meeting_cadence.py covers that case.")
    print(f"With no writer, {DEVICE} offers a reader {offered_rate()} frames a second.", flush=True)
    cases = []
    for camera_fps, made_fps, writing, name in CASES:
        result = {"case": name, **run(camera_fps, made_fps, writing)}
        cases.append(result)
        print(f"  {name:72s} new {result['new_per_second']:5.1f}/s  never arrived {result['never_arrived']:5.1%}  "
              f"held long {result['held_half_again_longer']:5.1%}  step error {result['step_error_ms_sd']:5.1f} ms", flush=True)
        time.sleep(1.0)
    switch = switch_rate()
    print(f"A reader already reading, the writer opened again at 30: {switch}", flush=True)
    idle = offered_rate()

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": {"os": platform.platform()},
        "device": DEVICE,
        "seconds_per_case": SECONDS,
        "offered_with_no_writer_fps": idle,
        "cases": cases,
        "rate_change_while_read": switch,
        "notes": [
            "Numbered test frames written with pyvirtualcam and read back by ffmpeg through DirectShow; arrival times "
            "include ffmpeg's pipe, which puts a floor of about 8 ms under the step error.",
            "A reader keeps the rate it was offered when it opened the camera, and with no writer the device offers 30. "
            "So the engine's camera has to be on before a meeting opens it; the engine switches it back on at start "
            "if it was on when it stopped.",
        ],
    }
    out = ROOT / "results" / "virtual_camera_study.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
