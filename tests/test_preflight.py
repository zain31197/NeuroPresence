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


# ------------------------------------------------------------------ accounts and the demonstration


def make_accounts(data_dir, *accounts):
    """A database with the users table of accounts.py: (name, folder, last signed in) for each account."""
    import sqlite3

    connection = sqlite3.connect(data_dir / "neuropresence.db")
    connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, folder TEXT, created_at REAL, "
                       "last_login_at REAL)")
    connection.executemany("INSERT INTO users (name, folder, created_at, last_login_at) VALUES (?, ?, 1.0, ?)", accounts)
    connection.commit()
    connection.close()


def enrol_in(folder, poses=("front", "left", "right", "up", "down")):
    folder.mkdir(parents=True)
    for name in ("picture.png", "identity.npy"):
        (folder / name).write_bytes(b"x")
    (folder / "poses.json").write_text(json.dumps({pose: {} for pose in poses}))


def test_the_account_signed_in_last_is_the_one_checked(tmp_path):
    assert preflight.account_check(tmp_path).level == WARN  # no database: no account yet
    enrol_in(tmp_path / "enrolment")  # the first account's, from before accounts
    make_accounts(tmp_path, ("Zain", "enrolment", 100.0), ("Sana", "users/2", 200.0))
    check = preflight.account_check(tmp_path)
    assert check.level == OK and check.detail == "Sana signed in last (2 accounts on this computer)"
    check = preflight.enrolment_check(tmp_path)
    assert check.level == WARN and check.detail == "Sana has no meeting picture enrolled"  # not Zain's picture
    enrol_in(tmp_path / "users" / "2")
    assert preflight.enrolment_check(tmp_path).level == OK
    assert preflight.enrolment_folder(tmp_path) == tmp_path / "users" / "2"


def test_the_engine_names_the_account_whose_enrolment_is_checked(tmp_path):
    # The System screen: the signed-in account may be any of them, not only the one signed in last.
    enrol_in(tmp_path / "users" / "3")
    assert preflight.enrolment_check(None, folder=tmp_path / "users" / "3", name="Talha").level == OK
    check = preflight.enrolment_check(None, folder=tmp_path / "users" / "4", name="Sana")
    assert check.level == WARN and check.detail == "Sana has no meeting picture enrolled"


def test_a_database_from_before_accounts_is_the_old_folder(tmp_path):
    import sqlite3

    sqlite3.connect(tmp_path / "neuropresence.db").close()  # no users table
    enrol_in(tmp_path / "enrolment")
    assert preflight.last_account(tmp_path) is None
    assert preflight.enrolment_check(tmp_path).level == OK


def rehearsal(tmp_path, cases, code=0, lines=("ok   A case: figures",)):
    out = tmp_path / "logs" / "demo_rehearsal.json"
    said = []

    def run(command, say):
        assert "--out" in command and str(out) in command
        if cases is not None:
            out.write_text(json.dumps({"results": cases}))
        return code, list(lines)

    return preflight.rehearse(say=said.append, out=out, run=run), said, out


def test_a_rehearsal_that_went_as_expected_is_one_line(tmp_path):
    cases = [{"case": "A photograph", "expected": "refused", "as_expected": True, "seen": []},
             {"case": "Someone else", "expected": "the still picture", "as_expected": True}]
    checks, said, _ = rehearsal(tmp_path, cases)
    assert [(c.name, c.level) for c in checks] == [("Demo rehearsal", OK)]
    assert checks[0].detail.startswith("2 of 2 cases as expected") and "four minutes" in said[0]


def test_each_case_that_did_not_go_as_expected_is_named_with_its_figures(tmp_path):
    cases = [{"case": "A photograph", "expected": "refused", "as_expected": True},
             {"case": "Someone else sits down", "expected": "the still picture within about 3 s", "as_expected": False,
              "still_after_s": None, "seen": [{"t": 0.1}], "events": ["a long list"]}]
    checks, _, _ = rehearsal(tmp_path, cases)
    assert checks[0].level == FAIL and checks[0].detail.startswith("1 of 2")
    assert checks[1].level == FAIL and "Someone else sits down" in checks[1].name
    assert checks[1].detail == "still_after_s None" and "within about 3 s" in checks[1].fix  # not the long lists
    assert "-> Expected: the still picture" in preflight.report(checks, demo=True)
    assert preflight.report(checks, demo=True).endswith("Fix the failures before the demonstration.")


def test_a_rehearsal_that_did_not_finish_keeps_its_last_lines(tmp_path):
    checks, _, out = rehearsal(tmp_path, None, code=1, lines=("loading", "Traceback (most recent call last):", "RuntimeError: no GPU"))
    assert [(c.name, c.level) for c in checks] == [("Demo rehearsal", FAIL)]
    assert "exit code 1" in checks[0].detail and "no GPU" in checks[0].detail
    assert "RuntimeError: no GPU" in out.with_suffix(".log").read_text()


def test_the_report_says_ready_for_the_demonstration():
    assert preflight.report([Check("Graphics card", OK, "RTX 5050")], demo=True).endswith("Ready for the demonstration.")
