"""The System screen: the preflight checks (neuropresence/preflight.py) run by the engine itself, so that the
computer can be seen to be ready from inside the app, for example right before a demonstration.

Two differences from the command line. The graphics card is read from the driver, not from PyTorch, which may
still be loading on another thread. And the virtual camera, the microphone and the camera are reached through
the runtime's own devices, so a running session is never disturbed and the tests can stand in for them.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from .. import preflight
from ..preflight import FAIL, OK, WARN, Check
from .session import SessionError

CAMERA_TEST_SECONDS = 2.0

# The checks, in the groups the screen shows them in.
GROUPS = (
    ("computer", "This computer"),
    ("models", "Models and app"),
    ("meeting", "Meeting devices"),
    ("you", "Your enrolment"),
    ("network", "Network"),
)


def _virtual_camera(runtime):
    try:
        name = runtime.virtual_camera.probe()
    except RuntimeError as err:
        return Check("Virtual camera", FAIL, str(err), "Install OBS Studio (Windows) or v4l2loopback (Linux), then restart the engine")
    if runtime.virtual_camera.on:
        return Check("Virtual camera", OK, f"{name or 'on'}: switched on and sending to meeting applications")
    return Check("Virtual camera", OK, f"{name or 'found'}: ready; switch it on in the Live Studio for a call")


def _virtual_microphone(runtime):
    return preflight.virtual_microphone_check(lister=lambda: runtime.virtual_microphone.found(fresh=True))


def _engine(runtime):
    engine = runtime._engine  # read only: loading it is the first session's job
    if engine is None:
        return Check("Engine", OK, "not loaded yet: it loads with the first session, in a few seconds")
    accelerated = getattr(engine, "accelerated", None)
    if accelerated == "tensorrt":
        how = "TensorRT, its heaviest layers in 8 bit" if getattr(engine, "precision", None) == "int8" else "TensorRT"
        return Check("Engine", OK, f"loaded, drawing with {how}")
    return Check("Engine", WARN, f"loaded, drawing with {accelerated or 'plain PyTorch'}: slower than TensorRT",
                 "pip install -r requirements.txt, then restart the engine")


def system_report(runtime, internet=True):
    """Every check, grouped, with an overall verdict. The slow ones run side by side."""
    jobs = {
        "computer": [preflight.python_check, preflight.packages_check, preflight.gpu_smi_check, preflight.power_check,
                     preflight.resources_check],
        "models": [lambda: preflight.files_check(), lambda: _engine(runtime)],
        "meeting": [lambda: _virtual_camera(runtime), lambda: _virtual_microphone(runtime)],
        # The signed-in account's own folder: data/enrolment for the first account, data/users/<id> for the others.
        "you": [lambda: preflight.enrolment_check(None, folder=runtime.store.folder,
                                                  name=runtime.user["name"] if runtime.user else None)],
        "network": [preflight.internet_check] if internet else [],
    }
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {key: [pool.submit(job) for job in group] for key, group in jobs.items()}
        found = {}
        for key, group in futures.items():
            checks = []
            for future in group:
                try:
                    result = future.result(timeout=15)
                except Exception as err:  # a check that breaks is a finding, not a broken screen
                    result = Check("Check", WARN, f"could not be run: {err}", "")
                checks.extend(result if isinstance(result, list) else [result])
            found[key] = checks
    groups = [{"key": key, "title": title, "checks": [_as_dict(c) for c in found[key]]} for key, title in GROUPS if found[key]]
    every = [c for group in found.values() for c in group]
    failed = sum(c.level == FAIL for c in every)
    warned = sum(c.level == WARN for c in every)
    return {"groups": groups, "ready": failed == 0, "failed": failed, "warned": warned,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "took_ms": round((time.perf_counter() - started) * 1000)}


def camera_test(runtime, index):
    """Open the camera as a session does and count the pictures it sends for a moment. Only while nothing else
    has the camera: a session or the Enrolment preview."""
    if runtime.session.active:
        raise SessionError("Stop the session first: it has the camera.")
    if runtime.preview.state.value == "running":
        raise SessionError("Close the camera on the Enrolment screen first.")
    try:
        feed = runtime.make_feed(index)
    except Exception as err:
        return _as_dict(Check(f"Camera {index}", FAIL, f"could not be opened ({err})",
                              "Plug it in, close other programs that use it (Meet, Zoom, Camo), or try another camera"))
    try:
        last, frames, first, shape = -1, 0, None, None
        deadline = time.perf_counter() + CAMERA_TEST_SECONDS + 1.5
        while time.perf_counter() < deadline:
            item = feed.read(after_index=last, timeout=0.5)
            if item is None:
                if getattr(feed, "ended", False):
                    break
                continue
            last = item.index
            if first is None:
                first = time.perf_counter()  # count from the first picture: opening can take a moment
            frames += 1
            shape = item.image.shape
            if time.perf_counter() - first >= CAMERA_TEST_SECONDS:
                break
    finally:
        feed.close()
    if first is None or shape is None or frames < 2:
        return _as_dict(Check(f"Camera {index}", FAIL, "opened but sent no pictures", "Check the camera in the Camera app"))
    rate = (frames - 1) / max(time.perf_counter() - first, 1e-6)
    detail = f"{shape[1]} x {shape[0]}, {rate:.0f} pictures a second"
    if rate < preflight.MIN_CAMERA_FPS:
        return _as_dict(Check(f"Camera {index}", WARN, detail + "; a session draws no faster than the camera sends",
                              "More light helps (cameras slow down in the dark); a USB webcam is steadier than a phone"))
    return _as_dict(Check(f"Camera {index}", OK, detail))


def _as_dict(check):
    return {"name": check.name, "level": check.level, "detail": check.detail, "fix": check.fix}
