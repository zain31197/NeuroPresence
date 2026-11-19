"""Preflight: is this computer ready for a live session, and for the demonstration?

The FYP-1 final demonstration (9 and 10 December 2026) runs on a laptop that is not the computer the system was
built on, so everything it needs is checked in one go, each item with what to do when it is missing:

    python scripts/preflight.py              every check, camera 0 included
    python scripts/preflight.py --camera 1   another camera
    python scripts/preflight.py --no-camera  leave the camera alone
    python scripts/preflight.py --warm       also make the networks ready for this graphics card, warm the voice
                                             model, and measure how fast this computer draws over a minute

The report is printed and kept in logs/preflight.log. The exit code is 1 if anything failed.
"""

import importlib.util
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "ok", "warn", "fail"

# Packages without which nothing runs, and packages one feature needs (with the feature).
CORE_PACKAGES = ("numpy", "cv2", "torch", "mediapipe", "onnxruntime", "fastapi", "uvicorn")
FEATURE_PACKAGES = {"tensorrt": "the full drawing speed", "pyvirtualcam": "the virtual camera",
                    "sounddevice": "the voice in step and the camera-off voice mode", "librosa": "the camera-off voice mode"}

# Files the system reads, with what is missing without them, how to get them, and how bad their absence is.
FILES = (
    ("models/face_landmarker.task", "face tracking", "python scripts/download_models.py", FAIL),
    ("models/arcface/w600k_r50.onnx", "the identity checks", "python scripts/download_models.py", FAIL),
    ("third_party/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth", "reenactment",
     "python scripts/setup_liveportrait.py", FAIL),
    ("third_party/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth", "reenactment",
     "python scripts/setup_liveportrait.py", FAIL),
    ("third_party/ditto/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "the camera-off voice mode",
     "python scripts/setup_ditto.py", WARN),
    ("third_party/ditto/checkpoints/ditto_pytorch", "the camera-off voice mode", "python scripts/setup_ditto.py", WARN),
    ("web/dist/index.html", "the web app", "cd web, then: npm install, npm run build", FAIL),
)

MIN_CAMERA_FPS = 24.0  # a session draws no faster than the camera delivers, and draws about 20 to 22 a second
MIN_FREE_DISK_GB = 5.0
MIN_FREE_MEMORY_GB = 4.0
TARGET_FPS = 20.0  # what a live session should reach on the demonstration laptop (24 is the proposal's target)
INTERNET_URL = "https://meet.google.com"


# What the warm-up loaded is kept until the program ends. Freed earlier, the voice model hung the program: its
# compiled model holds CUDA graphs recorded on its own thread, which has ended by then, and releasing them from
# another thread never returned (found 20 November 2026). The server keeps the model for as long as it runs.
_KEPT = []


@dataclass
class Check:
    name: str
    level: str
    detail: str
    fix: str = ""


# ------------------------------------------------------------------ checks


def python_check():
    version = platform.python_version()
    if sys.version_info[:2] == (3, 12):
        return Check("Python", OK, version)
    return Check("Python", WARN, f"{version}; the system was built and tested on 3.12", "Use Python 3.12 for the .venv")


def packages_check(find=importlib.util.find_spec):
    missing_core = [name for name in CORE_PACKAGES if find(name) is None]
    missing_feature = [name for name in FEATURE_PACKAGES if find(name) is None]
    if missing_core:
        return Check("Python packages", FAIL, "missing: " + ", ".join(missing_core),
                     "pip install -r requirements.txt (and PyTorch first, as the README says)")
    if missing_feature:
        lost = "; ".join(f"{name} ({FEATURE_PACKAGES[name]})" for name in missing_feature)
        return Check("Python packages", WARN, "missing: " + lost, "pip install -r requirements.txt")
    return Check("Python packages", OK, "all present")


def gpu_check(torch=None):
    try:
        if torch is None:
            import torch
    except ImportError:
        return Check("Graphics card", FAIL, "PyTorch is not installed", "Install PyTorch as the README says")
    if not torch.cuda.is_available():
        return Check("Graphics card", FAIL, "PyTorch sees no NVIDIA graphics card",
                     "Install the NVIDIA driver and the CUDA build of PyTorch (README, Setup)")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    detail = f"{name}, {memory:.1f} GB"
    limit = _power_limit()
    if limit:
        detail += f", power limit {limit}"
    if memory < 6.0:
        return Check("Graphics card", WARN, detail + "; under 6 GB may not hold the voice model as well", "")
    return Check("Graphics card", OK, detail)


def _power_limit():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def power_check(battery=None, scheme=None):
    """A laptop on its battery slows its graphics card down; so may a power-saving plan."""
    if battery is None:
        try:
            import psutil
            battery = psutil.sensors_battery()
        except Exception:
            battery = None
    if scheme is None:
        scheme = _power_scheme()
    if battery is not None and not battery.power_plugged:
        return Check("Power", FAIL, f"on battery ({battery.percent:.0f}%)",
                     "Plug in the charger: on battery the graphics card runs much slower")
    detail = "no battery (a desktop)" if battery is None else f"plugged in ({battery.percent:.0f}%)"
    if scheme and "saver" in scheme.lower():
        return Check("Power", WARN, f"{detail}; power plan: {scheme}",
                     "Windows Settings, System, Power: choose Best performance")
    return Check("Power", OK, detail + (f"; power plan: {scheme}" if scheme else ""))


def _power_scheme():
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.split("(", 1)[1].rsplit(")", 1)[0] if "(" in out else None


def files_check(root=ROOT):
    checks = []
    for relative, needed_for, fix, level in FILES:
        if not (root / relative).exists():
            checks.append(Check(f"File: {Path(relative).name}", level, f"missing; needed for {needed_for}", fix))
    if not checks:
        checks.append(Check("Models and web app", OK, "all files present"))
    plans = sorted((root / "models" / "tensorrt").glob("*.plan"))
    if plans:
        checks.append(Check("TensorRT networks", OK, f"{len(plans)} converted networks on disk. Whether they fit this "
                            "graphics card shows when they are loaded: run with --warm"))
    else:
        checks.append(Check("TensorRT networks", WARN, "none converted yet; the first session would wait about "
                            "a minute and a half for the conversion", "python scripts/preflight.py --warm"))
    return checks


def enrolment_check(data_dir):
    folder = Path(data_dir) / "enrolment"
    picture, record = folder / "picture.png", folder / "record.json"
    if not picture.exists() or not (folder / "identity.npy").exists():
        return Check("Enrolment", WARN, "nobody is enrolled on this computer",
                     "Open the web app, Enrolment: verify the face, register the poses, upload the meeting picture")
    when = ""
    try:
        when = " on " + json.loads(record.read_text())["enrolled_at"].replace("T", " at ")
    except (OSError, ValueError, KeyError):
        pass
    poses = folder / "poses.json"
    count = len(json.loads(poses.read_text())) if poses.exists() else 0
    if count < 5:
        return Check("Enrolment", WARN, f"picture enrolled{when}, {count} of 5 poses registered",
                     "Register the five poses in Enrolment, so the head keeps to this person's own range")
    return Check("Enrolment", OK, f"picture enrolled{when}, 5 poses registered")


def virtual_camera_check(opener=None):
    if opener is None:
        from .server.virtual_camera import open_device as opener
    try:
        device = opener(1280, 720, 30)
    except RuntimeError as err:
        text = str(err)
        if "taken" in text.lower() or "in use" in text.lower():
            return Check("Virtual camera", WARN, text, "Switch the virtual camera off in the Live Studio, then run again")
        return Check("Virtual camera", FAIL, text, "Install OBS Studio (Windows) or v4l2loopback (Linux)")
    name = getattr(device, "device", "found")
    close = getattr(device, "close", None)
    if close:
        close()
    return Check("Virtual camera", OK, f"{name}; a meeting application lists it under this name")


def virtual_microphone_check(lister=None):
    if lister is None:
        try:
            from .server.virtual_microphone import list_devices as lister
        except ImportError:
            return Check("Virtual microphone", WARN, "sounddevice is not installed", "pip install -r requirements.txt")
    try:
        devices = lister()
    except Exception as err:  # no sound system, or PortAudio missing
        return Check("Virtual microphone", WARN, f"the sound devices could not be listed ({err})", "")
    if not devices.get("cable"):
        hint = ("Install VB-Audio Virtual Cable and restart" if sys.platform == "win32"
                else "Create a null sink named NeuroPresence (README, Virtual microphone)")
        return Check("Virtual microphone", WARN, "no virtual cable: the meeting would hear the voice ahead of the lips", hint)
    microphone = devices.get("default") or (devices.get("microphones") or [None])[0]
    return Check("Virtual microphone", OK, f"{devices['cable']}; microphone: {microphone or 'none found'}")


def camera_check(index, seconds=3.0, opener=None, clock=time.perf_counter):
    """Opens the camera as a session does (1280 x 720) and counts what it delivers."""
    if opener is None:
        from .capture.camera import FrameSource, FrameSourceError

        def opener(source):
            try:
                return FrameSource(source, 1280, 720)
            except FrameSourceError:
                return None
    camera = opener(index)
    if camera is None:
        return Check(f"Camera {index}", FAIL, "could not be opened",
                     "Plug it in, close other programs that use it (Meet, Zoom, Camo), or try --camera 1")
    try:
        first = None
        frames, shape = 0, None
        start = clock()
        while clock() - start < seconds + 1.0:
            frame = camera.read()
            if frame is None:
                continue
            if first is None:  # the first frame can take a moment; count from it
                first, frames = clock(), 0
            frames += 1
            shape = frame.shape
    finally:
        camera.release()
    if first is None or shape is None:
        return Check(f"Camera {index}", FAIL, "opened but delivered no picture", "Check the camera in the Camera app")
    rate = (frames - 1) / max(clock() - first, 1e-6)
    detail = f"{shape[1]} x {shape[0]}, {rate:.0f} frames a second"
    if rate < MIN_CAMERA_FPS:
        return Check(f"Camera {index}", WARN, detail + f"; a session draws no faster than the camera sends",
                     "More light helps (cameras slow down in the dark); a USB webcam is steadier than a phone")
    return Check(f"Camera {index}", OK, detail)


def internet_check(url=INTERNET_URL, opener=urllib.request.urlopen):
    start = time.perf_counter()
    try:
        with opener(urllib.request.Request(url, method="HEAD"), timeout=5):
            pass
    except Exception as err:
        return Check("Internet", WARN, f"{url} could not be reached ({err.__class__.__name__})",
                     "Google Meet needs the internet; check the network, or plan the local demo without a call")
    return Check("Internet", OK, f"{url} answered in {(time.perf_counter() - start) * 1000:.0f} ms")


def port_check(port=8000):
    with socket.socket() as s:
        s.settimeout(0.5)
        taken = s.connect_ex(("127.0.0.1", port)) == 0
    if not taken:
        return Check(f"Port {port}", OK, "free for the engine")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as r:
            json.loads(r.read())
        return Check(f"Port {port}", OK, "the engine is already running there")
    except Exception:
        return Check(f"Port {port}", WARN, "another program uses it",
                     f"Close it, or start the engine with --port {port + 1}")


def resources_check(root=ROOT, memory=None):
    free_disk = shutil.disk_usage(root).free / 2 ** 30
    if memory is None:
        try:
            import psutil
            memory = psutil.virtual_memory().available / 2 ** 30
        except Exception:
            memory = None
    detail = f"{free_disk:.0f} GB free on disk" + (f", {memory:.1f} GB memory free" if memory is not None else "")
    if free_disk < MIN_FREE_DISK_GB:
        return Check("Disk and memory", WARN, detail, "Free some disk space: logs and converted networks need room")
    if memory is not None and memory < MIN_FREE_MEMORY_GB:
        return Check("Disk and memory", WARN, detail, "Close other programs (browsers with many tabs, games)")
    return Check("Disk and memory", OK, detail)


def run_checks(camera=0, data_dir=None, port=8000):
    data_dir = ROOT / "data" if data_dir is None else Path(data_dir)
    checks = [python_check(), packages_check(), gpu_check(), power_check()]
    checks += files_check()
    checks += [enrolment_check(data_dir), virtual_camera_check(), virtual_microphone_check()]
    if camera is not None:
        checks.append(camera_check(camera))
    checks += [internet_check(), port_check(port), resources_check()]
    return checks


# ------------------------------------------------------------------ warm-up


def warm(data_dir=None, seconds=60.0, voice=True, say=print):
    """Load the engine as a session does (converting its networks for this card if needed), draw for `seconds`
    from a sample clip the way a session draws, and warm the voice model. Returns checks with the figures.

    The two halves are separate functions on purpose: what the drawing used is freed before the voice model
    is loaded. Freed after it, in one function, the program hung (20 November 2026)."""
    from .reenactment.engine import LIVEPORTRAIT_DIR

    data_dir = ROOT / "data" if data_dir is None else Path(data_dir)
    picture = data_dir / "enrolment" / "picture.png"
    source = picture if picture.exists() else LIVEPORTRAIT_DIR / "assets" / "examples" / "source" / "s9.jpg"
    checks = _warm_engine(source, seconds, say)
    if voice:
        checks += _warm_voice(source, say)
    return checks


def _warm_engine(source, seconds, say):
    import cv2
    import numpy as np

    from .capture import FaceTracker
    from .pipeline import Pipeline
    from .reenactment import ReenactmentEngine
    from .reenactment.engine import LIVEPORTRAIT_DIR

    checks = []
    say("Loading the engine. The first time on a graphics card the networks are converted: about 1.5 minutes.")
    start = time.perf_counter()
    engine = ReenactmentEngine(tensorrt=True, compile_networks=True)
    _KEPT.append(engine)
    loaded = time.perf_counter() - start
    level = OK if engine.accelerated == "tensorrt" else WARN
    checks.append(Check("Engine", level, f"ready in {loaded:.0f} s, drawing with {engine.accelerated or 'plain PyTorch'}",
                        "" if level == OK else "TensorRT did not load: pip install -r requirements.txt, then run again"))

    engine.set_source(cv2.imread(str(source)))
    clip = LIVEPORTRAIT_DIR / "assets" / "examples" / "driving" / "d0.mp4"
    frames = []
    cap = cv2.VideoCapture(str(clip))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    say(f"Drawing for {seconds:.0f} s, as a session does, to measure this computer.")
    times = []  # (when, how long one frame took to draw)
    with FaceTracker(video=True) as tracker:
        tracks = [tracker.process(f) for f in frames]
        pipeline = Pipeline(tracker, engine)
        began = time.perf_counter()
        i = 0
        while time.perf_counter() - began < seconds:
            frame, track = frames[i % len(frames)], tracks[i % len(frames)]
            at = time.perf_counter()
            pipeline.step(frame, at=at, track=track, prepared=pipeline.prepare(frame, track, at))
            times.append((at - began, time.perf_counter() - at))
            i += 1
    drawn = np.array([d for _, d in times])
    early = np.array([d for t, d in times if t < min(10.0, seconds / 2)])
    late = np.array([d for t, d in times if t >= seconds - min(10.0, seconds / 2)])
    rate, rate_early, rate_late = 1.0 / np.median(drawn), 1.0 / np.median(early), 1.0 / np.median(late)
    detail = (f"{rate:.1f} frames a second (first 10 s {rate_early:.1f}, last 10 s {rate_late:.1f}); "
              f"{np.median(drawn) * 1000:.0f} ms a frame, slowest 5% {np.percentile(drawn, 95) * 1000:.0f} ms")
    if rate_late < TARGET_FPS:
        fix = ("Plug in the charger, choose Best performance, set the laptop's own performance mode (its vendor app), "
               "close other programs, and run again")
        checks.append(Check("Drawing speed", WARN, detail + f"; under the {TARGET_FPS:.0f} a session should reach", fix))
    else:
        checks.append(Check("Drawing speed", OK, detail))
    if rate_late < 0.85 * rate_early:
        checks.append(Check("Heat", WARN, f"the card slowed from {rate_early:.1f} to {rate_late:.1f} frames a second "
                            "while drawing: it is getting hot", "Keep the laptop's vents free, on a hard surface"))
    return checks


def _warm_voice(source, say):
    try:
        from .speech.ditto import CONFIG, DittoStream
    except ImportError as err:
        return [Check("Voice model", WARN, f"cannot be loaded ({err})", "pip install -r requirements.txt")]
    if not CONFIG.exists():
        return [Check("Voice model", WARN, "Ditto is not set up", "python scripts/setup_ditto.py")]
    say("Warming the voice model (compiled for this card once; about a minute the first time).")
    start = time.perf_counter()
    stream = DittoStream()
    _KEPT.append(stream)
    stream.start(Path(source))
    ready = stream.ready_to_listen.wait(300)
    stream.stop()
    took = time.perf_counter() - start
    return [Check("Voice model", OK if ready else FAIL, f"ready in {took:.0f} s" if ready else "not ready after 5 minutes",
                  "" if ready else "Send logs/preflight.log to the team")]


# ------------------------------------------------------------------ report

MARKS = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}


def report(checks):
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        lines.append(f"[{MARKS[c.level]}] {c.name.ljust(width)}  {c.detail}")
        if c.fix and c.level != OK:
            lines.append(f"       {' ' * width}  -> {c.fix}")
    failed = sum(c.level == FAIL for c in checks)
    warned = sum(c.level == WARN for c in checks)
    if failed:
        lines.append(f"\n{failed} failed, {warned} to look at. Fix the failures before a session.")
    elif warned:
        lines.append(f"\nReady, with {warned} to look at.")
    else:
        lines.append("\nReady.")
    return "\n".join(lines)
