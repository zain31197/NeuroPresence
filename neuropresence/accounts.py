"""The accounts on this computer.

An account is who the data in the database belongs to: a face and its poses, meeting pictures, sessions and their
figures, checks, notes, switches. It is made and kept on this computer only. There is no server anywhere that knows
about it and no email is ever sent, so a forgotten password is answered by a recovery code instead, which is shown
once when the account is made.

What is kept of the secrets:
  a password       its scrypt hash with a random salt. scrypt is made to be slow and to need memory (32 MB and about
                   a fifth of a second a try here), so a stolen database file does not give passwords away quickly.
  a recovery code  its SHA-256. It is 100 random bits, so it cannot be guessed and needs no slow hash.
  a sign-in        a random token of 256 bits, in a cookie scripts cannot read. The database has its SHA-256 only,
                   so the file does not let anyone in either.

Guessing is limited: after five tries in a row that did not match, an email has to wait a quarter of an hour. An
email that has no account takes the same time and gets the same answer, so trying does not tell which emails exist.

One account uses the engine at a time, since there is one camera and one graphics card: signing in to another
account signs the one before out, and is refused while a session runs.
"""

import base64
import hashlib
import hmac
import re
import secrets
import shutil
import sqlite3
import threading
import time
from pathlib import Path

SCRYPT_COST = (2 ** 15, 8, 3)  # N, r, p: what OWASP recommends for scrypt at 32 MB
SCRYPT_MEMORY = 128 * 1024 * 1024
REMEMBERED_SECONDS = 30 * 24 * 3600.0  # how long a sign-in lasts with "Keep me signed in"
NOT_REMEMBERED_SECONDS = 12 * 3600.0  # and without: a working day at most, and gone with the browser
SEEN_EVERY_SECONDS = 60.0  # a sign-in's last use is written at most this often
CACHE_SECONDS = 20.0  # how long a token that was found good is taken as good without asking the database again

FREE_TRIES, TRY_WINDOW_SECONDS = 5, 15 * 60.0
LEAST_PASSWORD, MOST_PASSWORD = 8, 128
MOST_NAME, MOST_EMAIL = 60, 254
EMAIL = re.compile(r"[^@\s]+@[^@\s.]+(\.[^@\s.]+)+")
TOO_COMMON = {"password", "password1", "passw0rd", "12345678", "123456789", "1234567890", "qwertyui", "qwerty123",
              "11111111", "00000000", "iloveyou", "letmein1", "welcome1", "admin123", "abcd1234", "abc12345"}
CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # without I, O, 0 and 1, which are mistaken for one another
CODE_GROUPS, CODE_GROUP = 5, 4

NO_MATCH = "That email and password do not match."


class AccountError(Exception):
    """Something about an account could not be done. The message is for the person; `status` is the HTTP one."""

    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------------- secrets


def hash_secret(secret):
    """The scrypt hash of a password, with its salt and cost beside it so that the cost can be raised later."""
    n, r, p = SCRYPT_COST
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(secret.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32, maxmem=SCRYPT_MEMORY)
    return "$".join(["scrypt", str(n), str(r), str(p), base64.b64encode(salt).decode(), base64.b64encode(key).decode()])


def verify_secret(secret, stored):
    """Is this the password of that hash? Compared in constant time."""
    try:
        kind, n, r, p, salt, key = stored.split("$")
        if kind != "scrypt":
            return False
        wanted = base64.b64decode(key)
        given = hashlib.scrypt(secret.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                               dklen=len(wanted), maxmem=SCRYPT_MEMORY)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(given, wanted)


def new_code():
    """A recovery code such as 7KQF-M2XP-9RTD-HV4B-WN8C."""
    return "-".join("".join(secrets.choice(CODE_LETTERS) for _ in range(CODE_GROUP)) for _ in range(CODE_GROUPS))


def hash_code(code):
    """The hash of a recovery code, however it was typed: the case, the dashes and the spaces do not matter."""
    plain = "".join(letter for letter in code.upper() if letter in CODE_LETTERS)
    return hashlib.sha256(plain.encode("ascii")).hexdigest()


def hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------- what may be typed


def clean_email(email):
    email = (email or "").strip().lower()
    if len(email) > MOST_EMAIL or not EMAIL.fullmatch(email):
        raise AccountError("That does not look like an email address.")
    return email


def clean_name(name):
    name = " ".join((name or "").split())
    if not name:
        raise AccountError("Enter your name.")
    if len(name) > MOST_NAME:
        raise AccountError(f"A name can have {MOST_NAME} letters at most.")
    return name


def check_password(password, email="", name=""):
    """Raise if the password is too easy. Length is what counts; what a guesser tries first is refused."""
    password = password or ""
    if len(password) < LEAST_PASSWORD:
        raise AccountError(f"A password needs at least {LEAST_PASSWORD} characters.")
    if len(password) > MOST_PASSWORD:
        raise AccountError(f"A password can have {MOST_PASSWORD} characters at most.")
    plain = password.casefold()
    if plain in TOO_COMMON or len(set(plain)) < 4:
        raise AccountError("That password is one of the first anyone would try. Choose another.")
    if plain in {email.casefold(), email.split("@")[0].casefold(), name.casefold()}:
        raise AccountError("A password cannot be your name or your email.")


def device_of(user_agent):
    """A browser and a system from a User-Agent, such as "Edge on Windows", for the list of sign-ins."""
    text = user_agent or ""
    browser = next((label for mark, label in (("Edg/", "Edge"), ("OPR/", "Opera"), ("Firefox/", "Firefox"), ("Chrome/", "Chrome"),
                                              ("Safari/", "Safari")) if mark in text), "A browser")
    system = next((label for mark, label in (("Windows", "Windows"), ("Android", "Android"), ("iPhone", "iPhone"), ("iPad", "iPad"),
                                             ("Mac OS X", "macOS"), ("Linux", "Linux")) if mark in text), None)
    return f"{browser} on {system}" if system else browser


def _minutes(seconds):
    minutes = max(1, int(-(-seconds // 60)))
    return "a minute" if minutes == 1 else f"{minutes} minutes"


class Accounts:
    """Signs people up and in, on the tables users, logins and login_attempts of the database."""

    def __init__(self, db, data_dir, clock=time.time):
        self._db, self._data_dir, self._clock = db, Path(data_dir), clock
        self._lock = threading.Lock()
        self._good = {}  # tokens found good lately: token hash -> (user, until when it is taken as good)
        self._nobody = None  # a hash to spend the same time on for an email that has no account

    # ----------------------------------------------------------- looking up

    def any(self):
        return bool(self._db.read("SELECT 1 FROM users LIMIT 1"))

    def _row(self, where, params):
        rows = self._db.read(f"SELECT * FROM users WHERE {where}", params)
        return rows[0] if rows else None

    @staticmethod
    def public(row):
        """What of a user may be shown to them: never a hash."""
        return {key: row[key] for key in ("id", "email", "name", "folder", "created_at", "last_login_at")}

    def last_user(self):
        """The account whose sign-in was used last and is still good: whose data the engine opens when it starts."""
        rows = self._db.read("SELECT u.* FROM logins l JOIN users u ON u.id = l.user_id WHERE l.expires_at > ? "
                             "ORDER BY l.last_seen_at DESC LIMIT 1", (self._clock(),))
        return self.public(rows[0]) if rows else None

    def user_of(self, token):
        """The user a token signs in, or None."""
        if not token:
            return None
        hashed, now = hash_token(token), self._clock()
        with self._lock:
            kept = self._good.get(hashed)
        if kept is not None and now < kept[1]:
            return kept[0]
        rows = self._db.read("SELECT l.last_seen_at, l.expires_at, u.* FROM logins l JOIN users u ON u.id = l.user_id "
                             "WHERE l.token_hash = ?", (hashed,))
        if not rows or rows[0]["expires_at"] <= now:
            self._forget(hashed)
            return None
        if now - rows[0]["last_seen_at"] > SEEN_EVERY_SECONDS:
            self._db.write(lambda c: c.execute("UPDATE logins SET last_seen_at = ? WHERE token_hash = ?", (now, hashed)))
        user = self.public(rows[0])
        with self._lock:
            self._good[hashed] = (user, min(now + CACHE_SECONDS, rows[0]["expires_at"]))
        return user

    def _forget(self, hashed=None):
        """Ask the database again about a token, or about all of them."""
        with self._lock:
            if hashed is None:
                self._good.clear()
            else:
                self._good.pop(hashed, None)

    # ----------------------------------------------------------- making and entering

    def sign_up(self, email, name, password):
        """Make an account. Returns (user, recovery code); the code is shown once and kept only as a hash.

        The first account on a computer takes over what was there before there were accounts: the enrolled face and
        picture, the history and the switches. Nothing is moved or copied; its rows get an owner."""
        email, name = clean_email(email), clean_name(name)
        check_password(password, email, name)
        code, now = new_code(), self._clock()
        hashed, recovery = hash_secret(password), hash_code(code)

        def work(c):
            first = c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            user_id = c.execute("INSERT INTO users (email, name, password_hash, recovery_hash, folder, created_at) "
                                "VALUES (?, ?, ?, ?, ?, ?)", (email, name, hashed, recovery, f"new {secrets.token_hex(8)}", now)).lastrowid
            c.execute("UPDATE users SET folder = ? WHERE id = ?", ("enrolment" if first else f"users/{user_id}", user_id))
            if first:
                for table in ("sessions", "events", "consent_checks", "enrolment_history"):
                    c.execute(f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL", (user_id,))
                c.execute("INSERT INTO user_settings (user_id, key, value, updated_at) SELECT ?, key, value, updated_at "
                          "FROM settings WHERE key LIKE 'ui.%' OR key LIKE 'feature.%'", (user_id,))
                c.execute("DELETE FROM settings WHERE key LIKE 'ui.%' OR key LIKE 'feature.%'")
            return user_id
        try:
            user_id = self._db.write(work)
        except sqlite3.IntegrityError:
            raise AccountError("There is already an account with that email. Sign in instead.", 409) from None
        return self.public(self._row("id = ?", (user_id,))), code

    def check(self, email, password):
        """The user of this email and password. Raises AccountError if they do not match, or too many tries did not."""
        email = (email or "").strip().lower()[:MOST_EMAIL]
        self._refuse_guessing(email)
        row = self._row("email = ?", (email,))
        if row is None:
            if self._nobody is None:
                self._nobody = hash_secret(secrets.token_hex(16))
            verify_secret(password or "", self._nobody)  # the same time as for an email that has an account
            self._tried(email, None, False)
            raise AccountError(NO_MATCH, 401)
        if not verify_secret(password or "", row["password_hash"]):
            self._tried(email, row["id"], False)
            raise AccountError(NO_MATCH, 401)
        self._tried(email, row["id"], True)
        return self.public(row)

    def _tried(self, email, user_id, succeeded):
        now = self._clock()
        self._db.write(lambda c: c.execute("INSERT INTO login_attempts (at, email, user_id, succeeded) VALUES (?, ?, ?, ?)",
                                           (now, email, user_id, int(succeeded))))

    def _refuse_guessing(self, email):
        """Too many tries in a row that did not match: the email waits."""
        now = self._clock()
        rows = self._db.read(
            "SELECT at FROM login_attempts WHERE email = ? AND succeeded = 0 AND at > ? AND id > COALESCE("
            "(SELECT MAX(id) FROM login_attempts WHERE email = ? AND succeeded = 1), 0) ORDER BY id DESC LIMIT ?",
            (email, now - TRY_WINDOW_SECONDS, email, FREE_TRIES))
        if len(rows) >= FREE_TRIES:
            wait = rows[-1]["at"] + TRY_WINDOW_SECONDS - now
            raise AccountError(f"Too many tries that did not match. Try again in {_minutes(wait)}.", 429)

    def new_login(self, user_id, device="", remember=True):
        """Sign a user in. Returns the token for the browser's cookie and how long it is good."""
        token, now = secrets.token_urlsafe(32), self._clock()
        seconds = REMEMBERED_SECONDS if remember else NOT_REMEMBERED_SECONDS

        def work(c):
            c.execute("DELETE FROM logins WHERE expires_at <= ?", (now,))
            c.execute("INSERT INTO logins (token_hash, user_id, created_at, last_seen_at, expires_at, device) VALUES (?, ?, ?, ?, ?, ?)",
                      (hash_token(token), user_id, now, now, now + seconds, device))
            c.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))
            # One account uses the engine at a time: whoever else was signed in here no longer is.
            c.execute("DELETE FROM logins WHERE user_id != ?", (user_id,))
        self._db.write(work)
        self._forget()
        return token, seconds

    def sign_out(self, token):
        hashed = hash_token(token or "")
        self._db.write(lambda c: c.execute("DELETE FROM logins WHERE token_hash = ?", (hashed,)))
        self._forget(hashed)

    # ----------------------------------------------------------- keeping it

    def update(self, user_id, name=None, email=None, password=None):
        """Change the name, or the email, which needs the password."""
        row = self._row("id = ?", (user_id,))
        name = row["name"] if name is None else clean_name(name)
        wanted = row["email"] if email is None else clean_email(email)
        if wanted != row["email"].lower():
            self._same_person(row, password)
        try:
            self._db.write(lambda c: c.execute("UPDATE users SET name = ?, email = ? WHERE id = ?", (name, wanted, user_id)))
        except sqlite3.IntegrityError:
            raise AccountError("There is already an account with that email.", 409) from None
        self._forget()
        return self.public(self._row("id = ?", (user_id,)))

    def _same_person(self, row, password):
        """Something that needs the password again. Wrong ones count as tries at signing in."""
        self._refuse_guessing(row["email"].lower())
        if not verify_secret(password or "", row["password_hash"]):
            self._tried(row["email"].lower(), row["id"], False)
            raise AccountError("That is not your password.", 403)

    def change_password(self, user_id, current, new, token=None):
        """Set a new password, and sign every other browser out."""
        row = self._row("id = ?", (user_id,))
        self._same_person(row, current)
        check_password(new, row["email"], row["name"])
        if current == new:
            raise AccountError("The new password is the same as the old one.")
        self._set_password(user_id, new, keep=token)

    def _set_password(self, user_id, password, keep=None, code=None):
        hashed, kept = hash_secret(password), hash_token(keep) if keep else ""
        recovery = None if code is None else hash_code(code)

        def work(c):
            c.execute("UPDATE users SET password_hash = ?, recovery_hash = COALESCE(?, recovery_hash) WHERE id = ?",
                      (hashed, recovery, user_id))
            c.execute("DELETE FROM logins WHERE user_id = ? AND token_hash != ?", (user_id, kept))
        self._db.write(work)
        self._forget()

    def reset_password(self, email, code, new):
        """A forgotten password: the recovery code sets a new one. Returns (user, the next code); the old one is spent."""
        email = (email or "").strip().lower()[:MOST_EMAIL]
        self._refuse_guessing(email)
        row = self._row("email = ?", (email,))
        if row is None or not hmac.compare_digest(hash_code(code or ""), row["recovery_hash"]):
            self._tried(email, None if row is None else row["id"], False)
            raise AccountError("That email and recovery code do not match.", 401)
        check_password(new, row["email"], row["name"])
        next_code = new_code()
        self._set_password(row["id"], new, code=next_code)
        self._tried(email, row["id"], True)
        return self.public(row), next_code

    def new_recovery_code(self, user_id, password):
        """A fresh recovery code for someone who lost theirs but knows the password. The old one no longer works."""
        row = self._row("id = ?", (user_id,))
        self._same_person(row, password)
        code = new_code()
        self._db.write(lambda c: c.execute("UPDATE users SET recovery_hash = ? WHERE id = ?", (hash_code(code), user_id)))
        return code

    def devices(self, user_id, token=None):
        """Where the account is signed in, the one asking first."""
        mine = hash_token(token) if token else ""
        rows = self._db.read("SELECT token_hash, created_at, last_seen_at, expires_at, device FROM logins "
                             "WHERE user_id = ? AND expires_at > ? ORDER BY last_seen_at DESC", (user_id, self._clock()))
        found = [{"device": row["device"], "created_at": row["created_at"], "last_seen_at": row["last_seen_at"],
                  "expires_at": row["expires_at"], "this": row["token_hash"] == mine} for row in rows]
        return sorted(found, key=lambda device: not device["this"])

    def sign_out_others(self, user_id, token):
        kept = hash_token(token or "")
        self._db.write(lambda c: c.execute("DELETE FROM logins WHERE user_id = ? AND token_hash != ?", (user_id, kept)))
        self._forget()

    def attempts(self, user_id, limit=10):
        """The latest tries at signing in to the account."""
        rows = self._db.read("SELECT at, succeeded FROM login_attempts WHERE user_id = ? ORDER BY at DESC, id DESC LIMIT ?",
                             (user_id, limit))
        return [{"at": row["at"], "succeeded": bool(row["succeeded"])} for row in rows]

    def confirm(self, user_id, password):
        """Raise unless this is the user's password."""
        self._same_person(self._row("id = ?", (user_id,)), password)

    def delete(self, user_id):
        """Delete an account and everything of its: its rows go with it, and the files of its face and picture."""
        row = self._row("id = ?", (user_id,))

        def work(c):
            c.execute("DELETE FROM login_attempts WHERE email = ?", (row["email"],))
            c.execute("DELETE FROM users WHERE id = ?", (user_id,))  # the foreign keys take the rest
        self._db.write(work)
        self._forget()
        folder = (self._data_dir / row["folder"]).resolve()
        if folder != self._data_dir.resolve() and self._data_dir.resolve() in folder.parents:
            shutil.rmtree(folder, ignore_errors=True)
