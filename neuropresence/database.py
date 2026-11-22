"""The app's database: what happened, kept on this computer.

SQLite, a single file in the data folder (`data/neuropresence.db`, which git ignores). Chosen over a database
server such as PostgreSQL on 21 November 2026, for this use: one person on their own computer, nothing to
install or start on the demonstration laptop or on Kubuntu (it is part of Python), and no network service, in
keeping with "nothing leaves your computer". It is a real SQL database all the same: tables, keys, transactions.

What it holds, and what each serves (docs/DATABASE.md has the diagram and every column):
  users               the accounts on this computer (accounts.py)
  logins              the sign-ins that are still good, as hashes of their tokens
  login_attempts      every try at signing in, for the limit on guessing and the Account screen
  user_settings       a user's feature switches and choices (Live Studio)
  pictures            a user's meeting pictures, the picture itself included (Enrolment: Your pictures)
  sessions            one row a session: whose, when, how it was driven, how it ended, its typical figures, and the
                      name and note the user gave it (Sessions)
  session_figures     the figures every few seconds during a session (the charts of a past session)
  events              everything the event list says, kept past a restart (Sessions, Live Studio)
  consent_checks      every liveness check and how it ended (Security; the limit on retries survives restarts)
  enrolment_history   face verified, poses registered, picture enrolled or removed, face forgotten (Security)
  settings            what belongs to the computer and not to a user, such as the virtual camera being on

Every row of a user's carries their id, and every query here that reads or changes one is given the user's id
(None is the time before the first account). Deleting an account deletes its rows with it, by the foreign keys.

The face signatures and the picture a session animates stay as files in the user's own folder: "Forget my face"
deletes them outright. The database overwrites what is deleted from it (secure_delete), the pictures included.

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

SCHEMA_VERSION = 3

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
    # 22 November 2026: how open the lips are at the camera and in the output, to see whether the mouth follows; and
    # the longest time between two frames, which an average hides.
    2: """
    ALTER TABLE session_figures ADD COLUMN lips_camera REAL;
    ALTER TABLE session_figures ADD COLUMN lips_output REAL;
    ALTER TABLE session_figures ADD COLUMN longest_gap_ms REAL;
    """,
    # 22 November 2026: accounts. Every row gets an owner; a history from before the first account has none, and goes
    # to the first account that is made (accounts.py: Accounts.sign_up).
    3: """
    CREATE TABLE users (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,   -- never given to someone else after a deletion
        email          TEXT    NOT NULL UNIQUE COLLATE NOCASE,
        name           TEXT    NOT NULL CHECK (length(name) BETWEEN 1 AND 60),
        password_hash  TEXT    NOT NULL,                    -- scrypt, with its own salt and cost
        recovery_hash  TEXT    NOT NULL,                    -- of the code shown once, for a forgotten password
        folder         TEXT    NOT NULL UNIQUE,             -- where the face signature and the picture files are
        created_at     REAL    NOT NULL,
        last_login_at  REAL
    );

    CREATE TABLE logins (
        token_hash    TEXT    PRIMARY KEY,                  -- the token itself is only in the browser's cookie
        user_id       INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        created_at    REAL    NOT NULL,
        last_seen_at  REAL    NOT NULL,
        expires_at    REAL    NOT NULL,
        device        TEXT
    );
    CREATE INDEX logins_by_user ON logins (user_id);

    CREATE TABLE login_attempts (
        id         INTEGER PRIMARY KEY,
        at         REAL    NOT NULL,
        email      TEXT    NOT NULL COLLATE NOCASE,
        user_id    INTEGER REFERENCES users (id) ON DELETE CASCADE,
        succeeded  INTEGER NOT NULL
    );
    CREATE INDEX login_attempts_by_email ON login_attempts (email, at DESC);

    CREATE TABLE user_settings (
        user_id     INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        key         TEXT    NOT NULL,
        value       TEXT    NOT NULL,                       -- JSON
        updated_at  REAL    NOT NULL,
        PRIMARY KEY (user_id, key)
    );

    CREATE TABLE pictures (
        id            INTEGER PRIMARY KEY,
        user_id       INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        name          TEXT    NOT NULL CHECK (length(name) BETWEEN 1 AND 60),
        created_at    REAL    NOT NULL,
        last_used_at  REAL,
        width         INTEGER NOT NULL,
        height        INTEGER NOT NULL,
        face_match    REAL,                                 -- how alike its face is to the verified face
        sha256        TEXT    NOT NULL,                     -- of the picture, so the same one is not kept twice
        image         BLOB    NOT NULL,                     -- PNG
        thumbnail     BLOB    NOT NULL,                     -- JPEG
        in_use        INTEGER NOT NULL DEFAULT 0,           -- the one a session animates
        UNIQUE (user_id, sha256)
    );
    CREATE UNIQUE INDEX one_picture_in_use ON pictures (user_id) WHERE in_use = 1;

    ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users (id) ON DELETE CASCADE;
    ALTER TABLE sessions ADD COLUMN picture_id INTEGER REFERENCES pictures (id) ON DELETE SET NULL;
    ALTER TABLE sessions ADD COLUMN title TEXT;
    ALTER TABLE sessions ADD COLUMN note TEXT;
    ALTER TABLE sessions ADD COLUMN starred INTEGER NOT NULL DEFAULT 0;
    CREATE INDEX sessions_by_user ON sessions (user_id, started_at DESC);
    ALTER TABLE events ADD COLUMN user_id INTEGER REFERENCES users (id) ON DELETE CASCADE;
    CREATE INDEX events_by_user ON events (user_id, at DESC);
    ALTER TABLE consent_checks ADD COLUMN user_id INTEGER REFERENCES users (id) ON DELETE CASCADE;
    CREATE INDEX consent_checks_by_user ON consent_checks (user_id, at DESC);
    ALTER TABLE enrolment_history ADD COLUMN user_id INTEGER REFERENCES users (id) ON DELETE CASCADE;
    CREATE INDEX enrolment_history_by_user ON enrolment_history (user_id, at DESC);
    """,
}

# What each table is for, in words for the screen that shows a user their data (the Data screen).
TABLES = {
    "users": "Your account: name, email, and a hash of your password.",
    "logins": "Where you are signed in. Only a hash of each token is kept.",
    "login_attempts": "Every try at signing in to your account, for the limit on guessing.",
    "user_settings": "Your switches and choices, restored when you sign in.",
    "pictures": "Your meeting pictures, each with how well it matched your face.",
    "sessions": "Every session you ran, with its typical figures, its name and your note.",
    "session_figures": "A reading every two seconds of each session: frame rate, delay, identity, lips.",
    "events": "What happened, as the event list said it.",
    "consent_checks": "Every liveness check: what it was for, whether it passed, and why not.",
    "enrolment_history": "Every change to your enrolment: face, poses, picture.",
}
OWNED_THROUGH = {"session_figures": "session_id IN (SELECT id FROM sessions WHERE user_id IS ?)", "users": "id IS ?"}
TEXT_FIELDS = {"title": 80, "note": 2000}  # what a user may write on a session, and how long

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
        connection.execute("PRAGMA secure_delete = ON")  # what is deleted is overwritten: pictures are in here too
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
                done = []
                for work, future in batch:
                    # All of one piece of work or none of it; and one that fails must not lose the others.
                    connection.execute("SAVEPOINT piece")
                    try:
                        done.append((future, work(connection)))
                        connection.execute("RELEASE piece")
                    except Exception as err:
                        connection.execute("ROLLBACK TO piece")
                        connection.execute("RELEASE piece")
                        if future is not None:
                            future.set_exception(err)
                connection.execute("COMMIT")
                for future, result in done:  # told only once it is really in the file
                    if future is not None:
                        future.set_result(result)
            except sqlite3.Error as err:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                for _, future in batch:
                    if future is not None and not future.done():
                        future.set_exception(err)
            finally:
                for _ in batch:
                    self._writes.task_done()

    def _write(self, work, wait=False):
        if self.closed:
            return None
        future = Future() if wait else None
        self._writes.put((work, future))
        return future.result(timeout=FLUSH_SECONDS) if wait else None

    def write(self, work):
        """Do `work(connection)` on the writing thread, all of it or none, and return what it returns."""
        if self.closed:
            raise RuntimeError("The database is closed.")
        return self._write(work, wait=True)

    def read(self, sql, params=()):
        """Rows, as dicts, after every write queued so far has landed."""
        return self._read(sql, params)

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

    def start_session(self, kind, input_label, user_id=None):
        """A session begins, with the picture the user has in use if it animates one. Returns its id."""
        at = self._clock()

        def work(c):
            picture = None
            if kind != "sample" and user_id is not None:
                row = c.execute("SELECT id FROM pictures WHERE user_id = ? AND in_use = 1", (user_id,)).fetchone()
                picture = None if row is None else row["id"]
            return c.execute("INSERT INTO sessions (started_at, kind, input_label, user_id, picture_id) VALUES (?, ?, ?, ?, ?)",
                             (at, kind, input_label, user_id, picture)).lastrowid
        return self._write(work, wait=True)

    def add_figures(self, session_id, fps=None, delay_ms=None, render_ms=None, identity=None, live_share=None,
                    lips_camera=None, lips_output=None, longest_gap_ms=None):
        at = self._clock()
        self._write(lambda c: c.execute(
            "INSERT OR REPLACE INTO session_figures (session_id, at, fps, delay_ms, render_ms, identity, live_share, "
            "lips_camera, lips_output, longest_gap_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, at, fps, delay_ms, render_ms, identity, live_share, lips_camera, lips_output, longest_gap_ms)))

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
            for flag in ("virtual_camera", "voice_in_step", "starred"):
                row[flag] = bool(row[flag])
            row.pop("user_id", None)
        return rows

    def sessions(self, limit=100, user_id=None, search="", starred=False):
        """A user's sessions, newest first; those whose name, note or input holds `search`, if given."""
        where, params = ["user_id IS ?"], [user_id]
        if search.strip():
            pattern = "%" + search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where.append("(title LIKE ? ESCAPE '\\' OR note LIKE ? ESCAPE '\\' OR input_label LIKE ? ESCAPE '\\')")
            params += [pattern] * 3
        if starred:
            where.append("starred = 1")
        return self._rounded(self._read(
            f"SELECT * FROM sessions WHERE {' AND '.join(where)} ORDER BY started_at DESC, id DESC LIMIT ?", (*params, limit)))

    def update_session(self, session_id, user_id, **changes):
        """Give a session a name or a note, or a star. Returns False if it is not one of this user's."""
        fields, values = [], []
        for key, value in changes.items():
            if key in TEXT_FIELDS:
                fields.append(f"{key} = ?")
                values.append((value or "").strip()[:TEXT_FIELDS[key]] or None)
            elif key == "starred":
                fields.append("starred = ?")
                values.append(int(bool(value)))
            else:
                raise ValueError(f"a session has no {key}")
        if not fields:
            return bool(self._read("SELECT id FROM sessions WHERE id = ? AND user_id IS ?", (session_id, user_id)))
        return self._write(lambda c: c.execute(
            f"UPDATE sessions SET {', '.join(fields)} WHERE id = ? AND user_id IS ?", (*values, session_id, user_id)).rowcount > 0,
            wait=True)

    def delete_session(self, session_id, user_id):
        """Forget one finished session, its figures and its events with it. False if it is not this user's, or runs."""
        return self._write(lambda c: c.execute(
            "DELETE FROM sessions WHERE id = ? AND user_id IS ? AND ended_at IS NOT NULL", (session_id, user_id)).rowcount > 0,
            wait=True)

    def session(self, session_id, user_id=None):
        """One of a user's sessions with its figures over time, its events and its consent check, or None."""
        found = self._rounded(self._read("SELECT * FROM sessions WHERE id = ? AND user_id IS ?", (session_id, user_id)))
        if not found:
            return None
        session = found[0]
        session["figures"] = self._read(
            "SELECT at, fps, delay_ms, render_ms, identity, live_share, lips_camera, lips_output, longest_gap_ms "
            "FROM session_figures WHERE session_id = ? ORDER BY at",
            (session_id,))
        session["events"] = self._read("SELECT at, level, message FROM events WHERE session_id = ? ORDER BY at, id", (session_id,))
        session["checks"] = self._read(
            "SELECT at, passed, failure, reason, lowest_match FROM consent_checks WHERE session_id = ? ORDER BY at", (session_id,))
        picture = self._read("SELECT name FROM pictures WHERE id IS ?", (session["picture_id"],))
        session["picture"] = picture[0]["name"] if picture else None
        return session

    def totals(self, since=None, user_id=None):
        """How much a user has used the app: sessions and time live, in all and since a moment."""
        def count(where, params):
            row = self._read(
                f"SELECT COUNT(*) AS sessions, COALESCE(SUM(COALESCE(ended_at, ?) - started_at), 0) AS seconds "
                f"FROM sessions WHERE kind != 'sample' AND user_id IS ? {where}", (self._clock(), user_id, *params))[0]
            return {"sessions": row["sessions"], "seconds": round(row["seconds"], 1)}
        return {"all": count("", ()), "since": count("AND started_at >= ?", (since,)) if since is not None else None}

    def clear_sessions(self, user_id=None):
        """Forget a user's session history: the sessions, their figures and their events. Starred ones are kept."""
        self._write(lambda c: (
            c.execute("DELETE FROM sessions WHERE ended_at IS NOT NULL AND starred = 0 AND user_id IS ?", (user_id,)),
            c.execute("DELETE FROM events WHERE session_id IS NULL AND user_id IS ?", (user_id,))), wait=True)

    # ---------------------------------------------------------------- events

    def add_event(self, level, message, session_id=None, at=None, user_id=None):
        at = self._clock() if at is None else at
        self._write(lambda c: c.execute("INSERT INTO events (at, level, message, session_id, user_id) VALUES (?, ?, ?, ?, ?)",
                                        (at, level, message, session_id, user_id)))

    def events(self, limit=200, user_id=None):
        return list(reversed(self._read(
            "SELECT at, level, message, session_id FROM events WHERE user_id IS ? ORDER BY at DESC, id DESC LIMIT ?",
            (user_id, limit))))

    # ---------------------------------------------------------------- security

    def add_consent_check(self, purpose, passed, failure=None, reason=None, lowest_match=None, session_id=None, user_id=None):
        at = self._clock()
        self._write(lambda c: c.execute(
            "INSERT INTO consent_checks (at, purpose, passed, failure, reason, lowest_match, session_id, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (at, purpose, int(bool(passed)), failure, reason or None, lowest_match, session_id, user_id)))

    def failures_since_last_pass(self, within, user_id=None):
        """When a user's checks that did not pass ended, after the last that passed and within `within` seconds."""
        rows = self._read(
            "SELECT at FROM consent_checks WHERE passed = 0 AND user_id IS ? AND at >= ? AND at > COALESCE("
            "(SELECT MAX(at) FROM consent_checks WHERE passed = 1 AND user_id IS ?), 0) ORDER BY at",
            (user_id, self._clock() - within, user_id))
        return [row["at"] for row in rows]

    def add_enrolment(self, action, detail=None, user_id=None):
        at = self._clock()
        self._write(lambda c: c.execute("INSERT INTO enrolment_history (at, action, detail, user_id) VALUES (?, ?, ?, ?)",
                                        (at, action, detail, user_id)))

    def security(self, limit=200, user_id=None):
        """A user's consent checks and changes to the enrolment, newest first, as one timeline."""
        checks = self._read("SELECT 'check' AS kind, at, purpose, passed, failure, reason, lowest_match, session_id "
                            "FROM consent_checks WHERE user_id IS ? ORDER BY at DESC LIMIT ?", (user_id, limit))
        changes = self._read("SELECT 'enrolment' AS kind, at, action, detail FROM enrolment_history WHERE user_id IS ? "
                             "ORDER BY at DESC LIMIT ?", (user_id, limit))
        for check in checks:
            check["passed"] = bool(check["passed"])
        return sorted(checks + changes, key=lambda item: item["at"], reverse=True)[:limit]

    def clear_security(self, user_id=None):
        self._write(lambda c: (c.execute("DELETE FROM consent_checks WHERE user_id IS ?", (user_id,)),
                               c.execute("DELETE FROM enrolment_history WHERE user_id IS ?", (user_id,))), wait=True)

    # ---------------------------------------------------------------- settings

    def set_setting(self, key, value):
        at = self._clock()
        text = json.dumps(value)
        self._write(lambda c: c.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at", (key, text, at)))

    def set_user_setting(self, user_id, key, value):
        """Keep a choice of a user's. Without one (before the first account) it is the computer's."""
        if user_id is None:
            self.set_setting(key, value)
            return
        at, text = self._clock(), json.dumps(value)
        self._write(lambda c: c.execute(
            "INSERT INTO user_settings (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (user_id, key, text, at)))

    def user_settings(self, user_id, prefix=""):
        if user_id is None:
            return self.settings(prefix)
        return self._parsed(self._read("SELECT key, value FROM user_settings WHERE user_id = ? AND key LIKE ? ORDER BY key",
                                       (user_id, prefix.replace("%", "") + "%")))

    def settings(self, prefix=""):
        return self._parsed(self._read("SELECT key, value FROM settings WHERE key LIKE ? ORDER BY key",
                                       (prefix.replace("%", "") + "%",)))

    @staticmethod
    def _parsed(rows):
        found = {}
        for row in rows:
            try:
                found[row["key"]] = json.loads(row["value"])
            except ValueError:
                continue
        return found

    # ---------------------------------------------------------------- pictures

    def keep_picture(self, user_id, name, image, thumbnail, sha256, width, height, face_match, limit, created_at=None):
        """Keep a meeting picture in the user's library and mark it the one in use. The same picture is kept once.
        Returns its id. When the library is full the one used longest ago makes room. created_at is for a picture
        that was enrolled before now."""
        at = self._clock()

        def work(c):
            c.execute("UPDATE pictures SET in_use = 0 WHERE user_id = ?", (user_id,))
            row = c.execute("SELECT id FROM pictures WHERE user_id = ? AND sha256 = ?", (user_id, sha256)).fetchone()
            if row is not None:
                c.execute("UPDATE pictures SET in_use = 1, last_used_at = ?, face_match = ? WHERE id = ?", (at, face_match, row["id"]))
                return row["id"]
            kept = c.execute("SELECT COUNT(*) FROM pictures WHERE user_id = ?", (user_id,)).fetchone()[0]
            if kept >= limit:
                c.execute("DELETE FROM pictures WHERE id IN (SELECT id FROM pictures WHERE user_id = ? "
                          "ORDER BY COALESCE(last_used_at, created_at) LIMIT ?)", (user_id, kept - limit + 1))
            return c.execute(
                "INSERT INTO pictures (user_id, name, created_at, last_used_at, width, height, face_match, sha256, image, "
                "thumbnail, in_use) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (user_id, name, at if created_at is None else created_at, at, width, height, face_match, sha256, image,
                 thumbnail)).lastrowid
        return self.write(work)

    def pictures(self, user_id):
        """A user's pictures without the pictures themselves, the newest used first."""
        rows = self._read(
            "SELECT p.id, p.name, p.created_at, p.last_used_at, p.width, p.height, p.face_match, p.in_use, "
            "length(p.image) AS bytes, COUNT(s.id) AS sessions FROM pictures p LEFT JOIN sessions s ON s.picture_id = p.id "
            "WHERE p.user_id = ? GROUP BY p.id ORDER BY COALESCE(p.last_used_at, p.created_at) DESC, p.id DESC", (user_id,))
        for row in rows:
            row["in_use"] = bool(row["in_use"])
        return rows

    def picture(self, user_id, picture_id, column="image"):
        """One picture of a user's, or its thumbnail, as bytes; None if it is not theirs."""
        assert column in ("image", "thumbnail", "name")
        rows = self._read(f"SELECT {column} FROM pictures WHERE id = ? AND user_id = ?", (picture_id, user_id))
        return rows[0][column] if rows else None

    def rename_picture(self, user_id, picture_id, name):
        return self.write(lambda c: c.execute("UPDATE pictures SET name = ? WHERE id = ? AND user_id = ?",
                                              (name, picture_id, user_id)).rowcount > 0)

    def delete_picture(self, user_id, picture_id):
        return self.write(lambda c: c.execute("DELETE FROM pictures WHERE id = ? AND user_id = ?",
                                              (picture_id, user_id)).rowcount > 0)

    def no_picture_in_use(self, user_id):
        self._write(lambda c: c.execute("UPDATE pictures SET in_use = 0 WHERE user_id IS ?", (user_id,)))

    def delete_pictures(self, user_id):
        self._write(lambda c: c.execute("DELETE FROM pictures WHERE user_id IS ?", (user_id,)), wait=True)

    # ---------------------------------------------------------------- what a user has in here

    def overview(self, user_id):
        """The database as a user's data: every table with its columns, keys and how many of its rows are theirs.

        The structure is read from the file itself, so what the Data screen draws is what the database is."""
        tables = []
        for name, purpose in TABLES.items():
            columns = self._read(f"PRAGMA table_info({name})")
            references = {row["from"]: {"table": row["table"], "column": row["to"], "on_delete": row["on_delete"]}
                          for row in self._read(f"PRAGMA foreign_key_list({name})")}
            mine = OWNED_THROUGH.get(name, "user_id IS ?")
            tables.append({
                "name": name, "purpose": purpose,
                "rows": self._read(f"SELECT COUNT(*) AS n FROM {name} WHERE {mine}", (user_id,))[0]["n"],
                "columns": [{"name": c["name"], "type": c["type"], "key": bool(c["pk"]), "required": bool(c["notnull"] or c["pk"]),
                             "references": references.get(c["name"])} for c in columns],
            })
        files = [self.path, self.path.with_name(self.path.name + "-wal")]
        pragma = {name: self._read(f"PRAGMA {name}")[0] for name in ("user_version", "journal_mode", "foreign_keys", "quick_check")}
        value = {name: next(iter(row.values())) for name, row in pragma.items()}
        return {
            "file": self.path.name,
            "bytes": sum(file.stat().st_size for file in files if file.exists()),
            "pictures_bytes": self._read("SELECT COALESCE(SUM(length(image) + length(thumbnail)), 0) AS n FROM pictures "
                                         "WHERE user_id IS ?", (user_id,))[0]["n"],
            "sqlite": sqlite3.sqlite_version,
            "schema": value["user_version"],
            "journal": value["journal_mode"],
            "foreign_keys": bool(value["foreign_keys"]),
            "integrity": value["quick_check"],
            "orphans": len(self._read("PRAGMA foreign_key_check")),
            "tables": tables,
        }

    def export(self, user_id):
        """Everything the database holds about a user, as plain values: no hash, no token, and no picture's bytes."""
        found = {}
        hidden = {"password_hash", "recovery_hash", "token_hash", "image", "thumbnail", "folder"}
        for name in TABLES:
            shown = ", ".join(c["name"] for c in self._read(f"PRAGMA table_info({name})") if c["name"] not in hidden)
            mine = OWNED_THROUGH.get(name, "user_id IS ?")
            found[name] = self._read(f"SELECT {shown} FROM {name} WHERE {mine}", (user_id,))
        return found
