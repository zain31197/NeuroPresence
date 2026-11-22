"""The app's database (neuropresence/database.py), on a file in a test folder and a clock the tests move."""

import re
import sqlite3
from pathlib import Path

import pytest

from neuropresence.database import SCHEMA_VERSION, TABLES, Database


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def db(tmp_path):
    clock = Clock()
    database = Database(tmp_path / "data" / "neuropresence.db", clock=clock)
    database.clock = clock
    yield database
    database.close()


def test_a_new_file_gets_every_table_at_the_newest_schema(tmp_path):
    database = Database(tmp_path / "np.db")
    database.close()
    connection = sqlite3.connect(tmp_path / "np.db")
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"sessions", "session_figures", "events", "consent_checks", "enrolment_history", "settings"} <= tables
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    connection.close()
    Database(tmp_path / "np.db").close()  # opening it again changes nothing


def test_a_session_from_start_to_end_with_its_figures_and_events(db):
    session = db.start_session("camera", "Camera 0")
    for fps, delay in ((20.0, 96.0), (22.0, 94.0), (21.0, 95.0)):
        db.clock.now += 2
        db.add_figures(session, fps=fps, delay_ms=delay, render_ms=38.0, identity=0.95, live_share=1.0)
    db.add_event("info", "Session started on Camera 0.", session_id=session)
    db.clock.now += 1
    db.end_session(session, "stopped", frames=130, virtual_camera=True)
    found = db.session(session)
    assert found["outcome"] == "stopped" and found["kind"] == "camera" and found["frames"] == 130
    assert found["fps"] == 21.0 and found["delay_ms"] == 95.0 and found["identity"] == 0.95  # medians
    assert found["seconds"] == 7.0 and found["virtual_camera"] is True and found["voice_in_step"] is False
    assert [f["fps"] for f in found["figures"]] == [20.0, 22.0, 21.0]
    assert [e["message"] for e in found["events"]] == ["Session started on Camera 0."]
    assert db.session(999) is None


def test_sessions_come_newest_first_and_a_running_one_counts_up(db):
    first = db.start_session("camera", "Camera 0")
    db.end_session(first, "stopped")
    db.clock.now += 60
    second = db.start_session("voice", "Voice, camera off")
    db.clock.now += 30
    listed = db.sessions()
    assert [s["id"] for s in listed] == [second, first]
    assert listed[0]["outcome"] == "running" and listed[0]["seconds"] == 30.0


def test_a_session_left_running_when_the_app_stopped_is_closed_as_interrupted(tmp_path):
    clock = Clock()
    database = Database(tmp_path / "np.db", clock=clock)
    session = database.start_session("camera", "Camera 0")
    clock.now += 5
    database.add_figures(session, fps=21.0)
    database.flush()
    database.close()  # the app stops without ending the session
    again = Database(tmp_path / "np.db", clock=clock)
    found = again.session(session)
    assert found["outcome"] == "interrupted" and found["seconds"] == 5.0 and found["fps"] == 21.0
    again.close()


def test_totals_leave_out_sample_clips(db):
    camera = db.start_session("camera", "Camera 0")
    db.clock.now += 120
    db.end_session(camera, "stopped")
    sample = db.start_session("sample", "Sample clip d0")
    db.clock.now += 30
    db.end_session(sample, "stopped")
    totals = db.totals(since=db.clock.now - 10_000)
    assert totals["all"] == {"sessions": 1, "seconds": 120.0} and totals["since"]["sessions"] == 1


def test_only_failures_after_the_last_pass_count_against_the_next_check(db):
    db.add_consent_check("session", False, "late", "Your head did not turn")
    db.clock.now += 10
    db.add_consent_check("session", True)
    db.clock.now += 10
    db.add_consent_check("verify", False, "identity", "Not the verified face")
    db.clock.now += 10
    db.add_consent_check("session", False, "late", "Missed")
    assert db.failures_since_last_pass(within=3600) == [db.clock.now - 10, db.clock.now]
    db.clock.now += 4000
    assert db.failures_since_last_pass(within=3600) == []  # an hour later they are forgotten


def test_the_security_timeline_mixes_checks_and_enrolment_changes_newest_first(db):
    db.add_enrolment("face_verified")
    db.clock.now += 1
    db.add_consent_check("session", True, lowest_match=0.81)
    db.clock.now += 1
    db.add_enrolment("pose_registered", "left")
    timeline = db.security()
    assert [item["kind"] for item in timeline] == ["enrolment", "check", "enrolment"]
    assert timeline[0]["detail"] == "left" and timeline[1]["passed"] is True
    db.clear_security()
    assert db.security() == []


def test_settings_are_kept_as_json_and_read_by_prefix(db):
    db.set_setting("feature.steady_crop", False)
    db.set_setting("feature.even_motion", True)
    db.set_setting("ui.input", "camera:1")
    db.set_setting("feature.steady_crop", True)  # changed again
    assert db.settings("feature.") == {"feature.even_motion": True, "feature.steady_crop": True}
    assert db.settings("ui.") == {"ui.input": "camera:1"}


def test_clearing_the_history_keeps_a_running_session(db):
    done = db.start_session("camera", "Camera 0")
    db.add_event("info", "old", session_id=done)
    db.end_session(done, "stopped")
    running = db.start_session("camera", "Camera 0")
    db.clear_sessions()
    assert [s["id"] for s in db.sessions()] == [running]
    assert db.events() == []


def test_writes_after_close_are_dropped_quietly(tmp_path):
    database = Database(tmp_path / "np.db")
    database.close()
    database.add_event("info", "late")  # the app is shutting down: nothing to do, nothing to raise
    assert database.start_session("camera", "Camera 0") is None


def test_a_file_from_the_first_version_gains_the_lips_and_keeps_its_figures(tmp_path):
    """The database is upgraded in place, so a history kept by an older engine is not lost."""
    import sqlite3

    from neuropresence.database import MIGRATIONS, SCHEMA_VERSION

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("BEGIN;" + MIGRATIONS[1] + "PRAGMA user_version = 1; COMMIT;")
    old.execute("INSERT INTO sessions (id, started_at, ended_at, kind, input_label, outcome) VALUES (7, 100, 200, 'camera', 'Camo', 'stopped')")
    old.execute("INSERT INTO session_figures (session_id, at, fps) VALUES (7, 110, 20.5)")
    old.commit()
    old.close()
    db = Database(path)
    try:
        assert db._reader.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        figures = db.session(7)["figures"]
        assert figures[0]["fps"] == 20.5 and figures[0]["lips_camera"] is None and figures[0]["lips_output"] is None
        db.add_figures(7, fps=19.0, lips_camera=0.01, lips_output=0.02)
        db.flush()
        assert db.session(7)["figures"][-1]["lips_output"] == 0.02
    finally:
        db.close()


def test_the_document_names_every_table_and_column_as_they_are(tmp_path):
    """docs/DATABASE.md is read by people who do not read the code: it may not fall behind the schema."""
    Database(tmp_path / "np.db").close()
    connection = sqlite3.connect(tmp_path / "np.db")
    names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
    real = {name: sorted((row[1], row[2].lower()) for row in connection.execute(f"PRAGMA table_info({name})")) for name in names}
    connection.close()
    document = (Path(__file__).parent.parent / "docs" / "DATABASE.md").read_text(encoding="utf-8")

    # In the data dictionary: a heading for each table, and a row for each column with its type.
    sections = re.split(r"^### ", document, flags=re.MULTILINE)[1:]
    told = {section.split(":")[0]: re.findall(r"^\| `(\w+)` \| (\w+) \|", section, flags=re.MULTILINE) for section in sections}
    assert {name: sorted((column, kind.lower()) for column, kind in rows) for name, rows in told.items()} == real

    # And the same in the diagram.
    diagram = document.split("```mermaid")[1].split("```")[0]
    boxes = re.findall(r"^    (\w+) \{\n(.*?)^    \}", diagram, flags=re.MULTILINE | re.DOTALL)
    drawn = {name: re.findall(r"^ +(\w+) (\w+)", body, flags=re.MULTILINE) for name, body in boxes}
    assert {name: sorted((column, kind) for kind, column in rows) for name, rows in drawn.items()} == real

    # The Data screen speaks of every table that has an owner.
    assert set(TABLES) == set(real) - {"settings"}
