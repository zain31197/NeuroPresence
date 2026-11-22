"""The accounts: secrets, signing up and in, what each account may see, and the data it keeps."""

import sqlite3
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_server import FRAME, enrol, events, make_runtime, quick_hashes, sign_up, start, status, wait_for  # noqa: F401
from test_server import consent_at_once, neutral_pose_at_once, parts  # noqa: F401  (fixtures)

from neuropresence import accounts as module
from neuropresence.accounts import (FREE_TRIES, TRY_WINDOW_SECONDS, AccountError, Accounts, check_password, clean_email,
                                    device_of, hash_code, hash_secret, new_code, verify_secret)
from neuropresence.database import MIGRATIONS, TABLES, Database
from neuropresence.server import create_app
from neuropresence.server.account_routes import COOKIE

PASSWORD = "correct horse battery"


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def made(tmp_path):
    """A database and its accounts, on a clock the tests move."""
    clock = Clock()
    db = Database(tmp_path / "np.db", clock=clock)
    found = Accounts(db, tmp_path, clock=clock)
    found.db, found.clock = db, clock
    yield found
    db.close()


# ------------------------------------------------------------------ secrets


def test_a_password_is_kept_as_a_salted_hash_that_only_it_matches():
    one, two = hash_secret(PASSWORD), hash_secret(PASSWORD)
    assert one.startswith("scrypt$") and PASSWORD not in one
    assert one != two  # each with a salt of its own: the same password twice does not look the same
    assert verify_secret(PASSWORD, one) and verify_secret(PASSWORD, two)
    assert not verify_secret("Correct horse battery", one)
    assert not verify_secret(PASSWORD, "not a hash") and not verify_secret(PASSWORD, "md5$1$1$1$aa$bb")


def test_the_cost_is_kept_with_the_hash_so_it_can_be_raised_later(monkeypatch):
    old = hash_secret(PASSWORD)
    monkeypatch.setattr(module, "SCRYPT_COST", (2 ** 6, 8, 1))
    assert verify_secret(PASSWORD, old) and hash_secret(PASSWORD).startswith("scrypt$64$")


def test_the_real_cost_is_the_recommended_one():
    import importlib
    source = importlib.util.find_spec("neuropresence.accounts").loader.get_source("neuropresence.accounts")
    assert "SCRYPT_COST = (2 ** 15, 8, 3)" in source  # the tests lower it; what ships must not be lowered


def test_a_recovery_code_is_long_random_and_forgiving_to_type():
    codes = {new_code() for _ in range(200)}
    assert len(codes) == 200 and all(len(code.replace("-", "")) == 20 for code in codes)
    code = new_code()
    assert hash_code(code) == hash_code(code.lower().replace("-", "  ")) != hash_code(new_code())


@pytest.mark.parametrize("password", ["short", "password", "12345678", "aaaaaaaa", "zain@example.com", "x" * 200])
def test_easy_passwords_are_refused(password):
    with pytest.raises(AccountError):
        check_password(password, "zain@example.com", "Zain")


@pytest.mark.parametrize("email", ["", "zain", "zain@", "@example.com", "a b@example.com", "zain@example", "x" * 300 + "@a.com"])
def test_what_is_not_an_email_is_refused(email):
    with pytest.raises(AccountError):
        clean_email(email)


def test_the_device_is_read_from_the_browser():
    assert device_of("Mozilla/5.0 (Windows NT 10.0) AppleWebKit Chrome/130 Safari/537 Edg/130") == "Edge on Windows"
    assert device_of("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko Firefox/130.0") == "Firefox on Linux"
    assert device_of(None) == "A browser"


# ------------------------------------------------------------------ signing up and in


def test_signing_up_and_in(made):
    user, code = made.sign_up("  Zain@Example.com ", " Zain  Shahid ", PASSWORD)
    assert user["email"] == "zain@example.com" and user["name"] == "Zain Shahid" and len(code) == 24
    assert "password_hash" not in user and "recovery_hash" not in user
    assert made.check("ZAIN@example.com", PASSWORD)["id"] == user["id"]
    token, _ = made.new_login(user["id"], "Edge on Windows")
    assert made.user_of(token)["id"] == user["id"]
    assert made.user_of("another token") is None and made.user_of(None) is None and made.user_of("") is None
    made.sign_out(token)
    assert made.user_of(token) is None


def test_nothing_secret_is_in_the_file(made, tmp_path):
    user, code = made.sign_up("zain@example.com", "Zain", PASSWORD)
    token, _ = made.new_login(user["id"])
    made.db.flush()
    made.db._writer_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    contents = (tmp_path / "np.db").read_bytes()
    for secret in (PASSWORD, code, code.replace("-", ""), token):
        assert secret.encode() not in contents


def test_one_email_has_one_account_however_it_is_written(made):
    made.sign_up("zain@example.com", "Zain", PASSWORD)
    with pytest.raises(AccountError) as err:
        made.sign_up("ZAIN@EXAMPLE.COM", "Another", PASSWORD)
    assert err.value.status == 409
    assert made.db.read("SELECT COUNT(*) AS n FROM users")[0]["n"] == 1


def test_a_wrong_password_and_an_unknown_email_get_the_same_answer(made):
    made.sign_up("zain@example.com", "Zain", PASSWORD)
    answers = []
    for email, password in (("zain@example.com", "not this one"), ("nobody@example.com", PASSWORD)):
        with pytest.raises(AccountError) as err:
            made.check(email, password)
        answers.append((str(err.value), err.value.status))
    assert answers[0] == answers[1]  # so trying does not tell which emails have an account


def test_guessing_is_limited_and_time_or_the_right_password_clears_it(made):
    user, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    for _ in range(FREE_TRIES - 1):
        with pytest.raises(AccountError):
            made.check("zain@example.com", "wrong")
    assert made.check("zain@example.com", PASSWORD)["id"] == user["id"]  # within the free tries: and it forgets them
    for _ in range(FREE_TRIES):
        with pytest.raises(AccountError):
            made.check("zain@example.com", "wrong")
    with pytest.raises(AccountError) as err:
        made.check("zain@example.com", PASSWORD)  # even the right one waits now
    assert err.value.status == 429 and "15 minutes" in str(err.value)
    made.clock.now += TRY_WINDOW_SECONDS + 1
    assert made.check("zain@example.com", PASSWORD)["id"] == user["id"]
    tries = made.attempts(user["id"], 50)
    assert tries[0]["succeeded"] is True and sum(not t["succeeded"] for t in tries) == 2 * FREE_TRIES - 1


def test_the_limit_on_guessing_survives_a_restart(made, tmp_path):
    made.sign_up("zain@example.com", "Zain", PASSWORD)
    for _ in range(FREE_TRIES):
        with pytest.raises(AccountError):
            made.check("zain@example.com", "wrong")
    again = Accounts(made.db, tmp_path, clock=made.clock)
    with pytest.raises(AccountError) as err:
        again.check("zain@example.com", PASSWORD)
    assert err.value.status == 429


def test_a_sign_in_ends_on_its_own(made):
    user, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    short, _ = made.new_login(user["id"], remember=False)
    long, _ = made.new_login(user["id"], remember=True)
    made.clock.now += 13 * 3600
    assert made.user_of(short) is None and made.user_of(long)["id"] == user["id"]
    made.clock.now += 31 * 24 * 3600
    assert made.user_of(long) is None and made.last_user() is None


def test_signing_in_to_another_account_signs_the_one_before_out(made):
    zain, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    sana, _ = made.sign_up("sana@example.com", "Sana", PASSWORD)
    first, _ = made.new_login(zain["id"])
    second, _ = made.new_login(sana["id"])
    assert made.user_of(first) is None and made.user_of(second)["id"] == sana["id"]
    assert made.last_user()["id"] == sana["id"]


# ------------------------------------------------------------------ keeping it


def test_a_new_password_signs_other_browsers_out(made):
    user, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    here, _ = made.new_login(user["id"])
    there, _ = made.new_login(user["id"])
    with pytest.raises(AccountError) as err:
        made.change_password(user["id"], "wrong", "another good one", here)
    assert err.value.status == 403
    with pytest.raises(AccountError):
        made.change_password(user["id"], PASSWORD, "short", here)
    with pytest.raises(AccountError):
        made.change_password(user["id"], PASSWORD, PASSWORD, here)
    made.change_password(user["id"], PASSWORD, "another good one", here)
    assert made.user_of(here) is not None and made.user_of(there) is None
    assert made.check("zain@example.com", "another good one")
    with pytest.raises(AccountError):
        made.check("zain@example.com", PASSWORD)


def test_a_forgotten_password_is_answered_by_the_recovery_code_once(made):
    user, code = made.sign_up("zain@example.com", "Zain", PASSWORD)
    token, _ = made.new_login(user["id"])
    with pytest.raises(AccountError):
        made.reset_password("zain@example.com", new_code(), "another good one")
    found, next_code = made.reset_password("zain@example.com", code.lower(), "another good one")
    assert found["id"] == user["id"] and next_code != code
    assert made.user_of(token) is None  # whoever was signed in is not any more
    assert made.check("zain@example.com", "another good one")
    with pytest.raises(AccountError):
        made.reset_password("zain@example.com", code, "a third good one")  # spent
    made.reset_password("zain@example.com", next_code, "a third good one")


def test_guessing_recovery_codes_is_limited_too(made):
    made.sign_up("zain@example.com", "Zain", PASSWORD)
    for _ in range(FREE_TRIES):
        with pytest.raises(AccountError) as err:
            made.reset_password("zain@example.com", new_code(), "another good one")
        assert err.value.status == 401
    with pytest.raises(AccountError) as err:
        made.reset_password("zain@example.com", new_code(), "another good one")
    assert err.value.status == 429


def test_the_name_is_changed_freely_and_the_email_with_the_password(made):
    user, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    made.sign_up("sana@example.com", "Sana", PASSWORD)
    assert made.update(user["id"], name="Zain S")["name"] == "Zain S"
    with pytest.raises(AccountError) as err:
        made.update(user["id"], email="new@example.com", password="wrong")
    assert err.value.status == 403
    with pytest.raises(AccountError) as err:
        made.update(user["id"], email="sana@example.com", password=PASSWORD)
    assert err.value.status == 409
    assert made.update(user["id"], email="new@example.com", password=PASSWORD)["email"] == "new@example.com"
    assert made.check("new@example.com", PASSWORD)


def test_deleting_an_account_takes_every_row_and_file_of_its(made, tmp_path):
    zain, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    sana, _ = made.sign_up("sana@example.com", "Sana", PASSWORD)
    db = made.db
    for user in (zain, sana):
        folder = tmp_path / user["folder"]
        folder.mkdir(parents=True)
        (folder / "identity.npy").write_bytes(b"face")
        session = db.start_session("camera", "Camo", user_id=user["id"])
        db.add_figures(session, fps=22.0)
        db.add_event("info", "hello", session_id=session, user_id=user["id"])
        db.add_consent_check("session", True, session_id=session, user_id=user["id"])
        db.add_enrolment("face_verified", user_id=user["id"])
        db.set_user_setting(user["id"], "ui.input", "camera:0")
        db.keep_picture(user["id"], "Formal", b"png", b"jpg", f"sha{user['id']}", 2, 2, 0.8, 12)
        made.new_login(user["id"])
    with pytest.raises(AccountError):
        made.check("zain@example.com", "wrong")
    made.delete(zain["id"])
    overview = {table["name"]: table["rows"] for table in db.overview(zain["id"])["tables"]}
    assert set(overview.values()) == {0}, overview  # nothing of his is left, in any table
    assert not (tmp_path / zain["folder"]).exists()
    kept = {table["name"]: table["rows"] for table in db.overview(sana["id"])["tables"]}
    assert all(kept[name] >= 1 for name in ("users", "sessions", "session_figures", "events", "consent_checks",
                                            "enrolment_history", "user_settings", "pictures")), kept
    assert (tmp_path / sana["folder"] / "identity.npy").exists()
    assert db.read("SELECT COUNT(*) AS n FROM login_attempts WHERE email = 'zain@example.com'")[0]["n"] == 0
    assert db.read("PRAGMA foreign_key_check") == []


def test_a_number_is_never_given_to_someone_else(made):
    first, _ = made.sign_up("a@example.com", "A", PASSWORD)
    second, _ = made.sign_up("b@example.com", "B", PASSWORD)
    made.delete(second["id"])
    third, _ = made.sign_up("c@example.com", "C", PASSWORD)
    assert third["id"] > second["id"] > first["id"]


# ------------------------------------------------------------------ what was there before


def test_the_first_account_takes_over_what_was_there_and_the_second_does_not(made):
    db = made.db
    session = db.start_session("camera", "Camo")
    db.add_event("info", "from before")
    db.add_consent_check("verify", True)
    db.add_enrolment("face_verified")
    db.set_setting("feature.even_motion", False)
    db.set_setting("ui.input", "camera:1")
    db.set_setting("virtual_camera.on", True)
    zain, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    sana, _ = made.sign_up("sana@example.com", "Sana", PASSWORD)
    assert zain["folder"] == "enrolment" and sana["folder"] == f"users/{sana['id']}"
    assert [s["id"] for s in db.sessions(user_id=zain["id"])] == [session] and db.sessions(user_id=sana["id"]) == []
    assert len(db.security(user_id=zain["id"])) == 2 and db.security(user_id=sana["id"]) == []
    assert db.user_settings(zain["id"]) == {"feature.even_motion": False, "ui.input": "camera:1"}
    assert db.user_settings(sana["id"]) == {}
    assert db.settings() == {"virtual_camera.on": True}  # the computer's, not a person's


def test_a_file_from_before_accounts_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    for version in (1, 2):
        old.executescript(f"BEGIN; {MIGRATIONS[version]} PRAGMA user_version = {version}; COMMIT;")
    old.execute("INSERT INTO sessions (id, started_at, ended_at, kind, input_label, outcome) VALUES (7, 100, 200, 'camera', 'Camo', 'stopped')")
    old.commit()
    old.close()
    db = Database(path)
    try:
        zain, _ = Accounts(db, tmp_path).sign_up("zain@example.com", "Zain", PASSWORD)
        found = db.session(7, zain["id"])
        assert found["input_label"] == "Camo" and found["starred"] is False and found["title"] is None
        assert db.session(7, None) is None  # it has an owner now
    finally:
        db.close()


# ------------------------------------------------------------------ the database, per account


def test_a_session_is_named_noted_starred_and_found(made):
    zain, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    sana, _ = made.sign_up("sana@example.com", "Sana", PASSWORD)
    db = made.db
    one, two = (db.start_session("camera", "Camo", user_id=zain["id"]) for _ in range(2))
    for session in (one, two):
        db.end_session(session, "stopped")
    assert db.update_session(one, zain["id"], title="  Supervisor meeting ", note="100% of the lips_ followed", starred=True)
    found = db.session(one, zain["id"])
    assert (found["title"], found["starred"]) == ("Supervisor meeting", True)
    assert [s["id"] for s in db.sessions(user_id=zain["id"], search="supervisor")] == [one]
    assert [s["id"] for s in db.sessions(user_id=zain["id"], search="100%")] == [one]  # taken as written, not as a pattern
    assert db.sessions(user_id=zain["id"], search="lipsX") == [] and len(db.sessions(user_id=zain["id"], search="camo")) == 2
    assert [s["id"] for s in db.sessions(user_id=zain["id"], starred=True)] == [one]
    assert not db.update_session(one, sana["id"], title="mine now") and db.session(one, sana["id"]) is None
    assert not db.delete_session(one, sana["id"])
    db.clear_sessions(zain["id"])
    assert [s["id"] for s in db.sessions(user_id=zain["id"])] == [one]  # what is starred is kept
    assert db.delete_session(one, zain["id"]) and db.sessions(user_id=zain["id"]) == []
    with pytest.raises(ValueError):
        db.update_session(two, zain["id"], outcome="error")  # only what a user may write


def test_a_picture_is_kept_once_and_one_is_in_use(made):
    zain, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    sana, _ = made.sign_up("sana@example.com", "Sana", PASSWORD)
    db = made.db
    formal = db.keep_picture(zain["id"], "Formal", b"one", b"t", "a", 960, 1280, 0.81, 12)
    made.clock.now += 1
    casual = db.keep_picture(zain["id"], "Casual", b"two", b"t", "b", 960, 1280, 0.79, 12)
    assert [(p["name"], p["in_use"]) for p in db.pictures(zain["id"])] == [("Casual", True), ("Formal", False)]
    made.clock.now += 1
    assert db.keep_picture(zain["id"], "Formal again", b"one", b"t", "a", 960, 1280, 0.82, 12) == formal  # the same picture
    assert [(p["name"], p["in_use"]) for p in db.pictures(zain["id"])] == [("Formal", True), ("Casual", False)]
    session = db.start_session("camera", "Camo", user_id=zain["id"])
    assert db.session(session, zain["id"])["picture"] == "Formal"  # which picture a session animated is kept with it
    assert db.picture(zain["id"], formal) == b"one" and db.picture(sana["id"], formal) is None
    assert not db.rename_picture(sana["id"], formal, "x") and not db.delete_picture(sana["id"], formal)
    assert db.delete_picture(zain["id"], formal)
    assert db.session(session, zain["id"])["picture"] is None  # the session stays, without it
    assert [p["id"] for p in db.pictures(zain["id"])] == [casual]
    db.keep_picture(zain["id"], "Old", b"three", b"t", "c", 960, 1280, 0.8, 12, created_at=1000.0)
    assert db.pictures(zain["id"])[0]["created_at"] == 1000.0  # one from before accounts keeps its own day


def test_a_full_library_lets_the_one_used_longest_ago_go(made):
    zain, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    for n in range(4):
        made.clock.now += 1
        made.db.keep_picture(zain["id"], f"P{n}", b"x", b"t", str(n), 1, 1, None, 3)
    assert [p["name"] for p in made.db.pictures(zain["id"])] == ["P3", "P2", "P1"]


def test_a_piece_of_work_that_fails_leaves_nothing_behind_and_the_others_stand(made):
    def half(c):
        c.execute("INSERT INTO settings (key, value, updated_at) VALUES ('half', '1', 0)")
        raise RuntimeError("stopped halfway")

    with pytest.raises(RuntimeError):
        made.db.write(half)
    made.db.set_setting("whole", 1)
    assert made.db.settings() == {"whole": 1}


def test_the_overview_is_read_from_the_file_itself(made):
    zain, _ = made.sign_up("zain@example.com", "Zain", PASSWORD)
    made.db.start_session("camera", "Camo", user_id=zain["id"])
    found = made.db.overview(zain["id"])
    assert found["schema"] == 3 and found["journal"] == "wal" and found["foreign_keys"] and found["integrity"] == "ok"
    assert found["orphans"] == 0 and found["bytes"] > 0
    tables = {table["name"]: table for table in found["tables"]}
    assert set(tables) == set(TABLES)
    assert tables["sessions"]["rows"] == 1 and tables["pictures"]["rows"] == 0
    columns = {column["name"]: column for column in tables["sessions"]["columns"]}
    assert columns["id"]["key"] and columns["user_id"]["references"] == {"table": "users", "column": "id", "on_delete": "CASCADE"}


def test_an_export_holds_the_data_and_no_secret(made):
    zain, code = made.sign_up("zain@example.com", "Zain", PASSWORD)
    sana, _ = made.sign_up("sana@example.com", "Sana", PASSWORD)
    made.new_login(zain["id"], "Edge on Windows")
    made.db.keep_picture(zain["id"], "Formal", b"one", b"t", "a", 960, 1280, 0.81, 12)
    made.db.start_session("camera", "Sana's camera", user_id=sana["id"])
    found = made.db.export(zain["id"])
    assert found["users"][0]["email"] == "zain@example.com" and found["pictures"][0]["name"] == "Formal"
    assert found["sessions"] == []  # only his
    text = repr(found)
    for secret in ("password_hash", "recovery_hash", "token_hash", "scrypt", "sana@example.com"):
        assert secret not in text


# ------------------------------------------------------------------ over HTTP


def another_browser(parts):
    return TestClient(parts["client"].app)


@pytest.mark.parametrize("method, path", [
    ("get", "/api/status"), ("get", "/api/history"), ("get", "/api/security"), ("get", "/api/pictures"),
    ("get", "/api/account/data"), ("get", "/api/account/export"), ("get", "/api/account/devices"), ("get", "/api/enrolment/picture"),
    ("get", "/api/system"), ("post", "/api/session/start"), ("post", "/api/session/stop"), ("post", "/api/virtual-camera"),
    ("delete", "/api/history"), ("delete", "/api/enrolment/face"), ("patch", "/api/preferences"), ("post", "/api/account/delete"),
])
def test_nothing_of_the_studio_answers_without_signing_in(parts, method, path):
    stranger = another_browser(parts)
    assert getattr(stranger, method)(path).status_code == 401
    stranger.cookies.set(COOKIE, "made-up")
    assert getattr(stranger, method)(path).status_code == 401


@pytest.mark.parametrize("path", ["/api/health", "/api/engine", "/api/capabilities", "/api/account"])
def test_the_landing_page_and_signing_in_need_no_account(parts, path):
    assert another_browser(parts).get(path).status_code == 200


def test_what_is_public_says_nothing_of_the_account(parts):
    enrol(parts)
    stranger = another_browser(parts)
    assert stranger.get("/api/account").json() == {"user": None, "first": False, "claims": False}
    engine = stranger.get("/api/engine").json()
    assert set(engine) == {"session", "virtual_camera", "virtual_microphone", "gpu", "targets", "identity"}
    assert "zain" not in str(engine).lower()


def test_every_api_path_is_private_unless_it_is_listed(parts):
    """A path added later is behind the sign-in without anyone having to remember it."""
    from neuropresence.server.account_routes import PUBLIC

    stranger = another_browser(parts)
    paths = {route.path for route in parts["client"].app.routes if route.path.startswith("/api/") and "GET" in (getattr(route, "methods", None) or ())}
    for path in sorted(paths - PUBLIC):
        assert stranger.get(path.replace("{session_id}", "1").replace("{picture_id}", "1")).status_code == 401, path


def test_the_cookie_is_out_of_reach_of_scripts_and_other_sites(parts):
    reply = another_browser(parts).post("/api/account/sign-in", json={"email": "zain@example.com", "password": PASSWORD})
    cookie = reply.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "max-age=2592000" in cookie
    reply = another_browser(parts).post("/api/account/sign-in", json={"email": "zain@example.com", "password": PASSWORD, "remember": False})
    assert "max-age" not in reply.headers["set-cookie"].lower()  # gone with the browser
    assert COOKIE.lower() in cookie and PASSWORD not in reply.text


def test_the_stream_is_closed_to_whoever_is_not_signed_in(parts):
    with pytest.raises(WebSocketDisconnect) as err:
        with another_browser(parts).websocket_connect("/api/stream") as socket:
            socket.receive_text()
    assert err.value.code == 4401


def test_the_stream_ends_when_the_account_signs_out(parts):
    with parts["client"].websocket_connect("/api/stream") as socket:
        socket.receive_text()
        parts["client"].post("/api/account/sign-out")
        with pytest.raises(WebSocketDisconnect) as err:
            for _ in range(50):
                socket.receive_text()
    assert err.value.code == 4401


def test_signing_out_stops_the_session_and_closes_the_account(parts):
    start(parts)
    runtime = parts["runtime"]
    assert runtime.enrolment is not None
    assert parts["client"].post("/api/account/sign-out").json() == {"ok": True}
    assert not runtime.session.active and runtime.user is None and runtime.enrolment is None and runtime.face is None
    assert parts["engines"][0].source_frame is None  # the model forgets the picture
    assert parts["client"].get("/api/status").status_code == 401
    assert parts["client"].post("/api/account/sign-in", json={"email": "zain@example.com", "password": PASSWORD}).status_code == 200
    assert status(parts)["enrolment"]["record"] is not None  # and it is all there again


def test_two_accounts_see_nothing_of_each_other(parts):
    enrol(parts)
    client, runtime = parts["client"], parts["runtime"]
    start(parts)
    client.post("/api/session/stop")
    client.patch("/api/features/even_motion", json={"enabled": False})
    mine = client.get("/api/history").json()["sessions"]
    assert mine and client.get("/api/pictures").json()["pictures"]

    sign_up(client, "sana@example.com", "Sana")  # the same browser, another account
    found = status(parts)
    assert found["enrolment"]["face"] is None and found["enrolment"]["record"] is None
    assert {f["key"]: f["enabled"] for f in found["features"]}["even_motion"] is True
    assert client.get("/api/history").json()["sessions"] == [] and client.get("/api/pictures").json()["pictures"] == []
    assert client.get(f"/api/history/{mine[0]['id']}").status_code == 404
    assert client.patch(f"/api/history/{mine[0]['id']}", json={"title": "mine"}).status_code == 404
    assert client.get("/api/pictures/1/image").status_code == 404 and client.post("/api/pictures/1/use").status_code == 404
    assert [e for e in events(parts) if "enrolled" in e] == []  # nor what the event list said
    assert client.post("/api/session/start", json={"input": "camera:0"}).status_code == 409  # she has no picture
    assert client.delete("/api/history").status_code == 200

    client.post("/api/account/sign-in", json={"email": "zain@example.com", "password": PASSWORD})
    assert [s["id"] for s in client.get("/api/history").json()["sessions"]] == [s["id"] for s in mine]
    assert status(parts)["enrolment"]["record"] is not None
    assert {f["key"]: f["enabled"] for f in status(parts)["features"]}["even_motion"] is False
    assert runtime.store.folder == parts["tmp"] / "data" / "enrolment"


def test_the_engine_does_not_change_hands_while_a_session_runs(parts):
    start(parts)
    other = another_browser(parts)
    Accounts(parts["runtime"].db, parts["tmp"] / "data").sign_up("sana@example.com", "Sana", PASSWORD)
    reply = other.post("/api/account/sign-in", json={"email": "sana@example.com", "password": PASSWORD})
    assert reply.status_code == 409 and "Zain" in reply.json()["detail"]
    assert status(parts)["session"]["state"] == "running"  # and he is still signed in


def test_a_restart_opens_the_account_that_was_signed_in(parts):
    enrol(parts)
    cookie = parts["client"].cookies.get(COOKIE)
    parts["runtime"].shutdown()
    again = make_runtime(parts["tmp"], parts)
    try:
        assert again.user["email"] == "zain@example.com" and again.enrolment is not None
        assert again.accounts.user_of(cookie)["id"] == again.user_id  # the browser is still signed in
    finally:
        again.shutdown()


def test_signed_out_after_a_restart_opens_no_ones_data(parts):
    enrol(parts)
    parts["client"].post("/api/account/sign-out")
    parts["runtime"].shutdown()
    again = make_runtime(parts["tmp"], parts)
    try:
        assert again.user is None and again.enrolment is None and again.face is None
        assert not (parts["tmp"] / "data" / "signed-out").exists()
    finally:
        again.shutdown()


def test_a_session_is_named_over_http(parts):
    start(parts)
    client = parts["client"]
    client.post("/api/session/stop")
    session = client.get("/api/history").json()["sessions"][0]
    reply = client.patch(f"/api/history/{session['id']}", json={"title": "Supervisor meeting", "starred": True})
    assert reply.status_code == 200 and reply.json()["title"] == "Supervisor meeting" and reply.json()["starred"] is True
    assert reply.json()["picture"] == "me"
    assert client.patch(f"/api/history/{session['id']}", json={"note": "Went well."}).json()["title"] == "Supervisor meeting"
    assert client.get("/api/history?search=supervisor").json()["sessions"][0]["id"] == session["id"]
    assert client.get("/api/history?search=nothing").json()["sessions"] == []
    assert client.delete(f"/api/history/{session['id']}").json() == {"ok": True}
    assert client.delete(f"/api/history/{session['id']}").status_code == 404


def test_the_library_keeps_every_enrolled_picture_and_brings_one_back(parts):
    enrol(parts)
    client = parts["client"]
    first = client.get("/api/pictures").json()["pictures"]
    assert [(p["name"], p["in_use"]) for p in first] == [("me", True)] and first[0]["bytes"] > 0
    image = client.get(f"/api/pictures/{first[0]['id']}/image")
    assert image.headers["content-type"] == "image/png" and image.headers["cache-control"] == "no-store"
    thumbnail = cv2.imdecode(np.frombuffer(client.get(f"/api/pictures/{first[0]['id']}/image?thumbnail=true").content, np.uint8), cv2.IMREAD_COLOR)
    assert max(thumbnail.shape[:2]) == 320

    other = cv2.imencode(".png", cv2.flip(FRAME, 1))[1].tobytes()  # another picture of the same face
    assert client.post("/api/enrolment/upload", files={"file": ("My formal picture.png", other, "image/png")}).json()["passed"]
    client.post("/api/enrolment/confirm")
    both = client.get("/api/pictures").json()["pictures"]
    assert [(p["name"], p["in_use"]) for p in both] == [("My formal picture", True), ("me", False)]

    before = status(parts)["enrolment"]["record"]["id"]
    used = client.post(f"/api/pictures/{first[0]['id']}/use")
    assert used.status_code == 200 and {p["name"]: p["in_use"] for p in used.json()["pictures"]} == {"me": True, "My formal picture": False}
    found = status(parts)["enrolment"]
    assert found["record"]["id"] != before and found["candidate"] is None

    assert client.patch(f"/api/pictures/{first[0]['id']}", json={"name": "  Work  "}).json()["pictures"][0]["name"] == "Work"
    assert client.patch(f"/api/pictures/{first[0]['id']}", json={"name": " "}).status_code == 409
    client.delete(f"/api/pictures/{first[0]['id']}")  # the one in use: it stops being the meeting picture too
    assert status(parts)["enrolment"]["record"] is None
    assert [p["name"] for p in client.get("/api/pictures").json()["pictures"]] == ["My formal picture"]
    client.delete("/api/enrolment/face")
    assert client.get("/api/pictures").json()["pictures"] == []  # forgetting the face forgets every picture of it


def test_the_library_cannot_bring_back_a_picture_of_someone_else(parts):
    enrol(parts)
    client = parts["client"]
    picture = client.get("/api/pictures").json()["pictures"][0]
    client.delete("/api/enrolment")
    parts["runtime"].scorer().alike = 0.1  # the verified face is now another person's
    parts["runtime"].scorer().unlike = 0.1
    reply = client.post(f"/api/pictures/{picture['id']}/use")
    assert reply.status_code == 409
    assert status(parts)["enrolment"]["record"] is None and status(parts)["enrolment"]["candidate"] is None


def test_the_account_over_http(parts):
    client = parts["client"]
    assert client.get("/api/account").json()["user"]["name"] == "Zain"
    assert "folder" not in client.get("/api/account").json()["user"]
    assert client.patch("/api/account", json={"name": "Zain Shahid"}).json()["user"]["name"] == "Zain Shahid"
    assert client.get("/api/account").json()["user"]["name"] == "Zain Shahid"
    assert client.post("/api/account/password", json={"current": "wrong", "new": "another good one"}).status_code == 403
    assert client.post("/api/account/password", json={"current": PASSWORD, "new": "another good one"}).json() == {"ok": True}
    assert client.post("/api/account/recovery-code", json={"password": PASSWORD}).status_code == 403
    code = client.post("/api/account/recovery-code", json={"password": "another good one"}).json()["recovery_code"]
    stranger = another_browser(parts)
    reset = stranger.post("/api/account/reset", json={"email": "zain@example.com", "code": code, "password": "a third good one"})
    assert reset.status_code == 200 and stranger.get("/api/status").status_code == 200
    assert client.get("/api/status").status_code == 401  # a reset signs everyone else out
    devices = stranger.get("/api/account/devices").json()
    assert [d["this"] for d in devices["devices"]] == [True] and any(not a["succeeded"] for a in devices["attempts"])


def test_the_data_and_the_export_over_http(parts):
    client = parts["client"]
    found = client.get("/api/account/data").json()
    assert found["schema"] == 3 and {t["name"] for t in found["tables"]} == set(TABLES)
    export = client.get("/api/account/export")
    assert "attachment" in export.headers["content-disposition"] and export.json()["account"]["email"] == "zain@example.com"
    assert "scrypt" not in export.text


def test_deleting_the_account_over_http(parts):
    enrol(parts)
    client = parts["client"]
    assert client.post("/api/account/delete", json={"password": "wrong"}).status_code == 403
    assert client.post("/api/account/delete", json={"password": PASSWORD}).json() == {"ok": True}
    assert client.get("/api/status").status_code == 401
    assert parts["runtime"].enrolment is None and not (parts["tmp"] / "data" / "enrolment").exists()
    found = client.get("/api/account").json()
    assert found["user"] is None and found["first"] is True
    assert client.post("/api/account/sign-in", json={"email": "zain@example.com", "password": PASSWORD}).status_code == 401


def test_the_first_account_is_told_that_there_is_something_to_take_over(tmp_path):
    found = {"feeds": [], "sources": [], "engines": [], "gpu_reads": [], "tmp": tmp_path}
    from test_server import FakeTracker
    found["tracker"] = FakeTracker()
    found["runtime"] = make_runtime(tmp_path, found)
    with TestClient(create_app(found["runtime"], web_dist=tmp_path / "no-web-build")) as client:
        assert client.get("/api/account").json() == {"user": None, "first": True, "claims": False}
        found["runtime"].db.start_session("camera", "Camo")
        assert client.get("/api/account").json()["claims"] is True
        sign_up(client)
        assert len(client.get("/api/history").json()["sessions"]) == 1


def test_a_stranger_cannot_tell_how_long_hashing_took(made):
    """An unknown email costs a hash too, so the time of an answer does not tell whether it has an account."""
    made.sign_up("zain@example.com", "Zain", PASSWORD)
    hashed = []
    real = module.verify_secret

    def counted(secret, stored):
        hashed.append(stored)
        return real(secret, stored)

    module.verify_secret = counted
    try:
        for email in ("zain@example.com", "nobody@example.com"):
            with pytest.raises(AccountError):
                made.check(email, "wrong")
    finally:
        module.verify_secret = real
    assert len(hashed) == 2 and all(stored.startswith("scrypt$") for stored in hashed)


def test_the_event_list_of_one_browser_is_not_behind_after_a_switch(parts):
    """The numbers of the events go on over a change of account, so a page that is open misses nothing."""
    runtime = parts["runtime"]
    last = runtime.events.last_id
    runtime.close_account()
    runtime.events.add("info", "after")
    assert runtime.events.since(last)[-1]["message"] == "after"
    assert wait_for(lambda: True) and time.time() > 0
