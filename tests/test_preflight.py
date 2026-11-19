"""The preflight checks, with stand-ins for every device: no camera, virtual camera or network is touched."""

import json
from types import SimpleNamespace

import numpy as np

from neuropresence import preflight
from neuropresence.preflight import FAIL, OK, WARN, Check


def test_missing_core_packages_fail_and_missing_feature_packages_only_warn():
    assert preflight.packages_check(find=lambda name: None if name == "torch" else object()).level == FAIL
    check = preflight.packages_check(find=lambda name: None if name == "pyvirtualcam" else object())
    assert check.level == WARN and "the virtual camera" in check.detail
    assert preflight.packages_check(find=lambda name: object()).level == OK


def test_a_laptop_on_battery_fails_and_a_power_saving_plan_warns():
    on_battery = SimpleNamespace(power_plugged=False, percent=64.0)
    check = preflight.power_check(battery=on_battery, scheme="Balanced")
    assert check.level == FAIL and "charger" in check.fix
    plugged = SimpleNamespace(power_plugged=True, percent=90.0)
    assert preflight.power_check(battery=plugged, scheme="Power saver").level == WARN
    assert preflight.power_check(battery=plugged, scheme="High performance").level == OK


def test_missing_files_say_how_to_get_them(tmp_path):
    checks = {c.name: c for c in preflight.files_check(root=tmp_path)}
    assert checks["File: face_landmarker.task"].level == FAIL
    assert checks["File: face_landmarker.task"].fix == "python scripts/download_models.py"
    assert checks["File: warping_module.pth"].fix == "python scripts/setup_liveportrait.py"
    assert checks["File: index.html"].level == FAIL and "npm run build" in checks["File: index.html"].fix
    assert checks["File: ditto_pytorch"].level == WARN  # only the voice mode is lost
    assert checks["TensorRT networks"].level == WARN and "--warm" in checks["TensorRT networks"].fix


def test_all_files_present_is_one_line(tmp_path):
    for relative, *_ in preflight.FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / "models" / "tensorrt").mkdir(parents=True)
    (tmp_path / "models" / "tensorrt" / "generator-abc.plan").write_text("x")
    checks = preflight.files_check(root=tmp_path)
    assert [c.level for c in checks] == [OK, OK]


def test_enrolment_needs_the_picture_and_the_five_poses(tmp_path):
    assert preflight.enrolment_check(tmp_path).level == WARN
    folder = tmp_path / "enrolment"
    folder.mkdir()
    for name in ("picture.png", "identity.npy"):
        (folder / name).write_bytes(b"x")
    (folder / "record.json").write_text(json.dumps({"enrolled_at": "2026-11-20T10:00:00"}))
    (folder / "poses.json").write_text(json.dumps({"front": {}, "left": {}, "right": {}}))
    check = preflight.enrolment_check(tmp_path)
    assert check.level == WARN and "3 of 5 poses" in check.detail
    (folder / "poses.json").write_text(json.dumps({k: {} for k in ("front", "left", "right", "up", "down")}))
    check = preflight.enrolment_check(tmp_path)
    assert check.level == OK and "2026-11-20 at 10:00:00" in check.detail


def test_the_virtual_camera_is_opened_and_let_go():
    closed = []
    device = SimpleNamespace(device="OBS Virtual Camera", close=lambda: closed.append(True))
    check = preflight.virtual_camera_check(opener=lambda w, h, fps: device)
    assert check.level == OK and "OBS Virtual Camera" in check.detail and closed

    def missing(w, h, fps):
        raise RuntimeError("No virtual camera could be opened (no driver). Install OBS Studio.")

    assert preflight.virtual_camera_check(opener=missing).level == FAIL


def test_without_the_cable_the_meeting_would_hear_the_voice_early():
    check = preflight.virtual_microphone_check(lister=lambda: {"microphones": ["Headset"], "default": "Headset", "cable": None})
    assert check.level == WARN and "ahead of the lips" in check.detail
    check = preflight.virtual_microphone_check(lister=lambda: {"microphones": ["Headset"], "default": "Headset",
                                                                "cable": "CABLE Input (VB-Audio Virtual Cable)"})
    assert check.level == OK and "Headset" in check.detail


class FakeCamera:
    """Delivers a frame every `period` seconds of a fake clock."""

    def __init__(self, clock, period, frames=True):
        self.clock, self.period, self.frames, self.released = clock, period, frames, False

    def read(self):
        self.clock.now += self.period
        return np.zeros((720, 1280, 3), np.uint8) if self.frames else None

    def release(self):
        self.released = True


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_the_camera_is_counted_and_a_slow_one_warns():
    clock = Clock()
    camera = FakeCamera(clock, 1 / 30)
    check = preflight.camera_check(0, seconds=2.0, opener=lambda i: camera, clock=clock)
    assert check.level == OK and "1280 x 720, 30 frames a second" in check.detail and camera.released
    clock = Clock()
    check = preflight.camera_check(0, seconds=2.0, opener=lambda i: FakeCamera(clock, 1 / 15), clock=clock)
    assert check.level == WARN and "15 frames a second" in check.detail


def test_a_camera_that_cannot_be_opened_or_gives_nothing_fails():
    assert preflight.camera_check(1, opener=lambda i: None).level == FAIL
    clock = Clock()
    check = preflight.camera_check(0, seconds=1.0, opener=lambda i: FakeCamera(clock, 0.1, frames=False), clock=clock)
    assert check.level == FAIL and "no picture" in check.detail


def test_no_internet_warns_and_says_what_it_means():
    def offline(request, timeout):
        raise OSError("no route")

    check = preflight.internet_check(opener=offline)
    assert check.level == WARN and "Google Meet" in check.fix


def test_the_report_lists_fixes_for_what_is_not_ok_and_sums_up():
    text = preflight.report([Check("Graphics card", OK, "RTX 5050"), Check("Power", FAIL, "on battery", "Plug in"),
                             Check("Internet", WARN, "unreachable", "Check the network")])
    lines = text.splitlines()
    assert lines[0].startswith("[OK  ] Graphics card")
    assert "-> Plug in" in text and "-> Check the network" in text
    assert "1 failed, 1 to look at" in text
    assert preflight.report([Check("Graphics card", OK, "RTX 5050")]).endswith("Ready.")
