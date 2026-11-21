"""The app's database: what happened, kept on this computer.

SQLite, a single file in the data folder (`data/neuropresence.db`, which git ignores). Chosen over a database
server such as PostgreSQL on 21 November 2026, for this use: one person on their own computer, nothing to
install or start on the demonstration laptop or on Kubuntu (it is part of Python), and no network service, in
keeping with "nothing leaves your computer". It is a real SQL database all the same: tables, keys, transactions.

What it holds, and what each serves:
  sessions            one row a session: when, how it was driven, how it ended, its typical figures (Sessions)
  session_figures     the figures every few seconds during a session (the charts of a past session)
  events              everything the event list says, kept past a restart (Sessions, Live Studio)
  consent_checks      every liveness check and how it ended (Security; the limit on retries survives restarts)
  enrolment_history   face verified, poses registered, picture enrolled or removed, face forgotten (Security)
  settings            the feature switches and the app's choices, remembered (Live Studio)

The face signatures and the meeting picture stay as files in data/enrolment, as before: "Forget my face"
deletes them outright, which a row in a database file would not guarantee.

Writes go through one background thread, so the live session never waits on the disk; the screens read
through their own connection, after the writes queued before them have landed (WAL mode lets both run).
"""

import json
import queue
import sqlite3
import statistics
import threading
import time
from concurrent.futures import Future
from pathlib import Path

SCHEMA_VERSION = 1

MIGRATIONS = {
    1: """
    CREATE TABLE sessions (
        id           INTEGER PRIMARY KEY,
        started_at   REAL    NOT NULL,          -- seconds since 1970
        ended_at     REAL,                      -- NULL while it runs
        kind         TEXT    NOT NULL CHECK (kind IN ('camera', 'voice', 'sample')),
        input_label  TEXT    NOT NULL,
        outcome      TEXT    NOT NULL DEFAULT 'running'
                     CHECK (outcome IN ('running', 'stopped', 'refused', 'error', 'interrupted')),
        message      TEXT,
        frames       INTEGER NOT NULL DEFAULT 0,
        fps          REAL,                      -- the medians of its figures
        delay_ms     REAL,
        render_ms    REAL,
        identity     REAL,
        virtual_camera INTEGER NOT NULL DEFAULT 0,  -- was the meeting sent the picture
        voice_in_step  INTEGER NOT NULL DEFAULT 0   -- and the voice held back to match
    );
    CREATE INDEX sessions_by_start ON sessions (started_at DESC);

    CREATE TABLE session_figures (
        session_id  INTEGER NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
        at          REAL    NOT NULL,
        fps         REAL,
        delay_ms    REAL,
        render_ms   REAL,
        identity    REAL,
        live_share  REAL,
        PRIMARY KEY (session_id, at)
    );

    CREATE TABLE events (
        id          INTEGER PRIMARY KEY,
        at          REAL    NOT NULL,
        level       TEXT    NOT NULL,
        message     TEXT    NOT NULL,
        session_id  INTEGER REFERENCES sessions (id) ON DELETE CASCADE
    );
    CREATE INDEX events_by_session ON events (session_id, at);

    CREATE TABLE consent_checks (
        id            INTEGER PRIMARY KEY,
        at            REAL    NOT NULL,
        purpose       TEXT    NOT NULL CHECK (purpose IN ('session', 'verify')),
        passed        INTEGER NOT NULL,
        failure       TEXT,
        reason        TEXT,
        lowest_match  REAL,
        session_id    INTEGER REFERENCES sessions (id) ON DELETE SET NULL
    );
    CREATE INDEX consent_checks_by_time ON consent_checks (at DESC);

    CREATE TABLE enrolment_history (
        id      INTEGER PRIMARY KEY,
        at      REAL NOT NULL,
        action  TEXT NOT NULL CHECK (action IN ('face_verified', 'pose_registered', 'picture_enrolled',
                                                'picture_removed', 'face_forgotten')),
        detail  TEXT
    );

    CREATE TABLE settings (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,             -- JSON
        updated_at  REAL NOT NULL
    );
    """,
}

FLUSH_SECONDS = 5.0  # how long a read waits for the writes queued before it


class Database:
    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._writes = queue.Queue()
        self._read_lock = threading.Lock()
        self._writer_connection = self._connect()
        self._migrate(self._writer_connection)
        self._close_unfinished(self._writer_connection)
        self._reader = self._connect()
        self._writer = threading.Thread(target=self._write_loop, name="database-writer", daemon=True)
        self._writer.start()
        self.closed = False

    # ---------------------------------------------------------------- plumbing

    def _connect(self):
        connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")  # with WAL: safe after a crash, without a disk sync per write
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _migrate(connection):
        """Bring the file to the newest schema, one numbered step at a time; a step is never edited once released."""
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        for step in range(version + 1, SCHEMA_VERSION + 1):
            connection.executescript(f"BEGIN; {MIGRATIONS[step]} PRAGMA user_version = {step}; COMMIT;")

    def _close_unfinished(self, connection):
        """A session still marked as running when the app starts ended with the app: say so, and when."""
        rows = connection.execute("SELECT id, started_at FROM sessions WHERE ended_at IS NULL").fetchall()
        for row in rows:
            last = connection.execute("SELECT MAX(at) FROM session_figures WHERE session_id = ?", (row["id"],)).fetchone()[0]
            connection.execute(
                "UPDATE sessions SET ended_at = ?, outcome = 'interrupted', message = ? WHERE id = ?",
                (last or row["started_at"], "The app stopped while this session was running.", row["id"]))
            self._summarise(connection, row["id"])

    def _write_loop(self):
        while True:
            item = self._writes.get()
            if item is None:
                self._writes.task_done()
                return
            batch = [item]
            while True:  # whatever else is waiting goes into the same transaction
                try:
                    more = self._writes.get_nowait()
                except queue.Empty:
                    break
                if more is None:
                    self._writes.put(None)  # stop after this batch
                    self._writes.task_done()
                    break
                batch.append(more)
            connection = self._writer_connection
            try:
                connection.execute("BEGIN")
                for work, future in batch:
                    try:
                        result = work(connection)
                        if future is not None:
                            future.set_result(result)
                    except Exception as err:  # one bad write must not lose the others
                        if future is not None:
                            future.set_exception(err)
                connection.execute("COMMIT")
            except sqlite3.Error:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            finally:
                for _ in batch:
                    self._writes.task_done()

    def _write(self, work, wait=False):
        if self.closed:
            return None
        future = Future() if wait else None
        self._writes.put((work, future))
        return future.result(timeout=FLUSH_SECONDS) if wait else None

    def flush(self):
        """Wait until every write queued so far has landed."""
        if not self.closed:
            self._writes.join()

    def _read(self, sql, params=()):
        self.flush()
        with self._read_lock:
            return [dict(row) for row in self._reader.execute(sql, params).fetchall()]

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._writes.put(None)
        self._writer.join(timeout=FLUSH_SECONDS)
        for connection in (self._writer_connection, self._reader):
            try:
                connection.close()
            except sqlite3.Error:
                pass

    # ---------------------------------------------------------------- sessions

    def start_session(self, kind, input_label):
        """A session begins. Returns its id."""
        at = self._clock()
        return self._write(lambda c: c.execute(
            "INSERT INTO sessions (started_at, kind, input_label) VALUES (?, ?, ?)", (at, kind, input_label)).lastrowid,
            wait=True)

    def add_figures(self, session_id, fps=None, delay_ms=None, render_ms=None, identity=None, live_share=None):
        at = self._clock()
        self._write(lambda c: c.execute(
            "INSERT OR REPLACE INTO session_figures (session_id, at, fps, delay_ms, render_ms, identity, live_share) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", (session_id, at, fps, delay_ms, render_ms, identity, live_share)))

    def end_session(self, session_id, outcome, message="", frames=0, virtual_camera=False, voice_in_step=False):
        at = self._clock()

        def work(c):
            c.execute("UPDATE sessions SET ended_at = ?, outcome = ?, message = ?, frames = ?, virtual_camera = ?, "
                      "voice_in_step = ? WHERE id = ?",
                      (at, outcome, message or None, frames, int(virtual_camera), int(voice_in_step), session_id))
            self._summarise(c, session_id)
        self._write(work)

    @staticmethod
    def _summarise(connection, session_id):
        """The session's typical figures: the median of each over the session."""
        rows = connection.execute("SELECT fps, delay_ms, render_ms, identity FROM session_figures WHERE session_id = ?",
                                  (session_id,)).fetchall()
        median = {}
        for name in ("fps", "delay_ms", "render_ms", "identity"):
            values = [row[name] for row in rows if row[name] is not None]
            median[name] = round(statistics.median(values), 3) if values else None
        connection.execute("UPDATE sessions SET fps = ?, delay_ms = ?, render_ms = ?, identity = ? WHERE id = ?",
                           (median["fps"], median["delay_ms"], median["render_ms"], median["identity"], session_id))

    def _rounded(self, rows):
        now = self._clock()
        for row in rows:
            row["seconds"] = round((row["ended_at"] or now) - row["started_at"], 1)
            row["virtual_camera"], row["voice_in_step"] = bool(row["virtual_camera"]), bool(row["voice_in_step"])
        return rows

    def sessions(self, limit=100):
        return self._rounded(self._read("SELECT * FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?", (limit,)))

    def session(self, session_id):
        """One session with its figures over time, its events and its consent check, or None."""
        found = self._rounded(self._read("SELECT * FROM sessions WHERE id = ?", (session_id,)))
        if not found:
            return None
        session = found[0]
        session["figures"] = self._read(
            "SELECT at, fps, delay_ms, render_ms, identity, live_share FROM session_figures WHERE session_id = ? ORDER BY at",
            (session_id,))
        session["events"] = self._read("SELECT at, level, message FROM events WHERE session_id = ? ORDER BY at, id", (session_id,))
        session["checks"] = self._read(
            "SELECT at, passed, failure, reason, lowest_match FROM consent_checks WHERE session_id = ? ORDER BY at", (session_id,))
        return session

    def totals(self, since=None):
        """How much the app has been used: sessions and time live, in all and since a moment."""
        def count(where, params):
            row = self._read(
                f"SELECT COUNT(*) AS sessions, COALESCE(SUM(COALESCE(ended_at, ?) - started_at), 0) AS seconds "
                f"FROM sessions WHERE kind != 'sample' {where}", (self._clock(), *params))[0]
            return {"sessions": row["sessions"], "seconds": round(row["seconds"], 1)}
        return {"all": count("", ()), "since": count("AND started_at >= ?", (since,)) if since is not None else None}

    def clear_sessions(self):
        """Forget the session history: the sessions, their figures and their events."""
        self._write(lambda c: (c.execute("DELETE FROM sessions WHERE ended_at IS NOT NULL"),
                               c.execute("DELETE FROM events WHERE session_id IS NULL")), wait=True)

    # ---------------------------------------------------------------- events

    def add_event(self, level, message, session_id=None, at=None):
        at = self._clock() if at is None else at
        self._write(lambda c: c.execute("INSERT INTO events (at, level, message, session_id) VALUES (?, ?, ?, ?)",
                                        (at, level, message, session_id)))

    def events(self, limit=200):
        return list(reversed(self._read("SELECT at, level, message, session_id FROM events ORDER BY at DESC, id DESC LIMIT ?",
                                        (limit,))))

    # ---------------------------------------------------------------- security

    def add_consent_check(self, purpose, passed, failure=None, reason=None, lowest_match=None, session_id=None):
        at = self._clock()
        self._write(lambda c: c.execute(
            "INSERT INTO consent_checks (at, purpose, passed, failure, reason, lowest_match, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", (at, purpose, int(bool(passed)), failure, reason or None, lowest_match, session_id)))

    def failures_since_last_pass(self, within):
        """When the checks that did not pass ended, after the last one that passed and within `within` seconds."""
        rows = self._read(
            "SELECT at FROM consent_checks WHERE passed = 0 AND at >= ? AND at > COALESCE("
            "(SELECT MAX(at) FROM consent_checks WHERE passed = 1), 0) ORDER BY at", (self._clock() - within,))
        return [row["at"] for row in rows]

    def add_enrolment(self, action, detail=None):
        at = self._clock()
        self._write(lambda c: c.execute("INSERT INTO enrolment_history (at, action, detail) VALUES (?, ?, ?)",
                                        (at, action, detail)))

    def security(self, limit=200):
        """The consent checks and the changes to the enrolment, newest first, as one timeline."""
        checks = self._read("SELECT 'check' AS kind, at, purpose, passed, failure, reason, lowest_match, session_id "
                            "FROM consent_checks ORDER BY at DESC LIMIT ?", (limit,))
        changes = self._read("SELECT 'enrolment' AS kind, at, action, detail FROM enrolment_history ORDER BY at DESC LIMIT ?",
                             (limit,))
        for check in checks:
            check["passed"] = bool(check["passed"])
        return sorted(checks + changes, key=lambda item: item["at"], reverse=True)[:limit]

    def clear_security(self):
        self._write(lambda c: (c.execute("DELETE FROM consent_checks"), c.execute("DELETE FROM enrolment_history")), wait=True)

    # ---------------------------------------------------------------- settings

    def set_setting(self, key, value):
        at = self._clock()
        text = json.dumps(value)
        self._write(lambda c: c.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at", (key, text, at)))

    def settings(self, prefix=""):
        rows = self._read("SELECT key, value FROM settings WHERE key LIKE ? ORDER BY key", (prefix.replace("%", "") + "%",))
        found = {}
        for row in rows:
            try:
                found[row["key"]] = json.loads(row["value"])
            except ValueError:
                continue
        return found
