"""A live session: frames from the feed go through the pipeline on a background thread."""

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ..capture import TrackStatus
from ..capture.crop import square_face_crop
from ..consent import liveness
from ..consent.disclosure import mark
from ..pipeline import Pipeline, at_rest
from ..targets import TARGETS
from .metrics import MetricsWindow
from .pacing import MotionClock
from .watchdog import DelayWatchdog

log = logging.getLogger("neuropresence.server")

SAMPLE_SOURCE_SECONDS = 10.0
# After this long, a camera that delivers clearly fewer frames than could be drawn is pointed out, once.
SLOW_CAMERA_AFTER_SECONDS = 5.0
SLOW_CAMERA_SHARE = 0.9
MAX_FRONTAL_ANGLE_DEG = 20.0
# The watchdog starts reading this long after the first frame, when the session has found its pace.
WATCHDOG_AFTER_SECONDS = 2.0
# A camera silent this long is pointed out while the feed reconnects to it (capture/feed.py).
CAMERA_QUIET_SECONDS = 1.0
# How often a running session writes its figures to the log (logs/server.log). The Meet recording of 9 November
# 2026 showed 9 new pictures a second where the engine draws about 21, and nothing said where the rest were lost:
# the drawing, the virtual camera, or the meeting application. With these lines the log tells.
FIGURES_EVERY_SECONDS = 30.0
# How often a running session's figures are kept in the database, for the charts of a past session.
KEEP_FIGURES_SECONDS = 2.0

STATUS_EVENTS = {
    TrackStatus.NO_FACE: ("warning", "No face in view. Showing the enrolled picture."),
    TrackStatus.MULTIPLE_FACES: ("warning", "More than one face in view. Showing the enrolled picture."),
    TrackStatus.OK: ("info", "Face found. Reenactment resumed."),
}


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class SessionError(RuntimeError):
    """A request that cannot be carried out right now. The message is shown to the user."""


class _Stopped(Exception):
    """Stop was requested while the session was still starting."""


class _Newest:
    """Holds the newest tracked frame for the drawing loop. An older one that was never taken is dropped.

    Tracking a face takes the CPU about 12 ms and drawing takes the GPU several times that. Doing
    the tracking on its own thread, while the GPU draws the frame before, takes it out of the time
    between two output frames.
    """

    def __init__(self):
        self._ready = threading.Condition()
        self._held = None
        self._dropped = 0
        self._error = None

    def put(self, item, track, skipped, prepared=None):
        with self._ready:
            if self._held is not None:
                self._dropped += 1  # tracked, but a newer frame arrived before it could be drawn
            self._dropped += skipped
            self._held = (item, track, prepared)
            self._ready.notify_all()

    def wait_until_taken(self, timeout):
        """Wait until the drawing loop has taken what is held, so no frame is prepared only to be dropped."""
        with self._ready:
            if self._held is not None:
                self._ready.wait(timeout)
            return self._held is None

    def fail(self, error):
        with self._ready:
            self._error = error
            self._ready.notify()

    def take(self, timeout):
        """The newest tracked frame as (item, track, frames dropped since the last one), or None after the timeout."""
        with self._ready:
            if self._held is None and self._error is None:
                self._ready.wait(timeout)
            if self._held is None:
                if self._error is not None:
                    raise self._error
                return None
            (item, track, prepared), self._held = self._held, None
            dropped, self._dropped = self._dropped, 0
            self._ready.notify_all()
            return item, track, dropped, prepared


@dataclass
class FramePair:
    """A camera frame and, in a live session, the output made from it."""

    id: int
    camera: np.ndarray
    output: np.ndarray | None  # None in the enrolment preview, which has no output
    live: bool
    status: str
    landmarks: np.ndarray | None
    kind: str = "live"  # "live" or "preview"
    crop_window: tuple | None = None  # where the driving face crop was cut, as (x, y, side) in the camera frame


def camera_off_card(width=640, height=360):
    """What the camera monitor shows while the voice drives the picture: a plain card, the camera being off."""
    card = np.full((height, width, 3), (24, 20, 18), dtype=np.uint8)
    for text, scale, weight, colour, y in (("Camera off", 1.1, 2, (235, 235, 235), height // 2 - 8),
                                           ("Your voice drives the picture", 0.6, 1, (170, 165, 160), height // 2 + 30)):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
        cv2.putText(card, text, ((width - text_w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, weight, cv2.LINE_AA)
    return card


def is_frontal(track):
    return track.ok and all(abs(angle) <= MAX_FRONTAL_ANGLE_DEG for angle in track.pose_deg)


class LiveSession:
    """Owns the camera and runs the pipeline. One session at a time.

    The runtime supplies the engine, the feed, the tracker, the feature
    switches and the event log, so tests can run a session without a camera
    or a GPU.
    """

    def __init__(self, runtime):
        self._rt = runtime
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._commands = queue.SimpleQueue()
        self._thread = None
        self.state = SessionState.IDLE
        self.message = ""
        self.input_label = None
        self.uses_enrolment = True  # False for a sample clip, which animates its own first frame
        self.voice = False  # True when the voice drives the picture: the camera only for the check, then off
        self._started_at = None
        self._pair = None
        self._tracking = None
        self._metrics = MetricsWindow()
        self._draw_seconds = 0.0  # a running average of how long one frame takes to draw
        self._cycle_seconds = 0.0  # how long from one frame drawn to the next: the drawing, or the pace if longer
        self._next_draw = None  # when the next frame is due while the drawing is paced
        self.pace_fps = None  # a rate to hold the drawing to, set by a study or a test; otherwise the runtime says
        self._made = 0  # how many frames have been made, so that a reader can wait for the next one
        self._made_ready = threading.Condition()
        self.run = 0  # counts the sessions started, so a watcher can tell a new one from the last
        # Two reasons to show the still picture although a face is being tracked. The identity
        # monitor sets the first (see runtime.py); the session sets the second itself.
        self.identity_hold = False
        self.delay_hold = False
        # A third, set by the consent watch (see runtime.py): the person at the camera is not the registered one.
        self.consent_hold = False
        self._consent = None  # how the check before the session stands or ended, for the app to show
        self._pair_id = 0
        self._watchdog = DelayWatchdog(TARGETS["end_to_end_ms"])
        self._clock = MotionClock()  # which moment each frame's movement is of (see pacing.py)
        self._feed = None  # the camera or clip of the running session
        self._drawn_seconds = 0.0  # how long a frame that is really drawn takes, as a running average
        self._slow_camera_told = False
        self._anchor = threading.Event()  # a fresh neutral pose has been asked for
        self._figures_at = None  # when the figures were last written to the log, and the virtual camera's counts then
        self.record_id = None  # this session's row in the database (database.py: sessions)
        self._kept_at = None  # when its figures were last kept there
        self._used_camera = self._used_voice = False  # whether the meeting was sent the picture, and the voice
        # Since when the person at the camera has been at rest (facing it, mouth closed, eyes open), or None. The
        # identity monitor resumes after a fallback by itself once they have been (runtime.py, IdentityMonitor).
        self.resting_since = None

    @property
    def active(self):
        return self.state in (SessionState.STARTING, SessionState.RUNNING)

    def request_anchor(self):
        """Take a fresh neutral pose with the next frame at rest, and start the filters again."""
        self._anchor.set()

    def hold_for_identity(self, hold):
        """Show the still picture (True) until this is called again with False."""
        self.identity_hold = bool(hold)

    def hold_for_consent(self, hold):
        """Show the still picture (True) while the person at the camera is not the registered one."""
        self.consent_hold = bool(hold)

    # ------------------------------------------------------------ control

    def start(self, source, label, uses_enrolment, voice=False):
        with self._lock:
            if self.active:
                raise SessionError("A session is already running.")
            self._stop.clear()
            self._metrics = MetricsWindow()
            self._pair = self._tracking = self._started_at = self._figures_at = None
            self.resting_since = None
            self.run += 1
            self.identity_hold = self.delay_hold = self.consent_hold = False
            self._consent = None
            self._watchdog.reset()
            self._clock.reset()
            self._drawn_seconds, self._slow_camera_told = 0.0, False
            self._anchor.clear()
            self.uses_enrolment = uses_enrolment
            self.voice = voice
            self._kept_at, self._used_camera, self._used_voice = None, False, False
            database = getattr(self._rt, "db", None)
            kind = "voice" if voice else "camera" if uses_enrolment else "sample"
            self.record_id = database.start_session(kind, label) if database is not None else None
            opening = "Opening the camera" if isinstance(source, int) else "Opening the clip"
            self.state, self.message, self.input_label = SessionState.STARTING, opening, label
            self._thread = threading.Thread(target=self._run, args=(source,), name="live-session", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        with self._lock:
            self.state, self.message = SessionState.IDLE, ""
            self._pair = self._tracking = self._consent = None

    def call(self, action, timeout=5.0):
        """Run action(frame, track) on the session thread between two frames.

        Returns its result or raises what it raised. The engine is only ever
        touched from the session thread while a session runs.
        """
        if self.state is not SessionState.RUNNING:
            raise SessionError("No session is running.")
        future = Future()
        self._commands.put((action, future))
        return future.result(timeout=timeout)

    # ------------------------------------------------------------- reading

    def wait_for_frame(self, seen, timeout):
        """Wait until a frame after number `seen` has been made, or `timeout` seconds. Returns the newest number.
        The virtual camera waits here, so that it sends each frame as soon as it is finished."""
        with self._made_ready:
            if self._made == seen:
                self._made_ready.wait(timeout)
            return self._made

    def _frame_made(self):
        with self._made_ready:
            self._made += 1
            self._made_ready.notify_all()

    def _pace(self):
        """The rate the drawing is held to now, or None to draw as fast as the card allows."""
        if self.pace_fps is not None:
            return self.pace_fps
        pace = getattr(self._rt, "output_pace", None)
        return pace() if pace is not None else None

    def _wait_for_turn(self):
        """While the drawing is paced, wait for the next frame's turn, so frames come at the camera's rate, each one
        for the same time. A frame that comes late is drawn at once; after one more than a frame late the turns start
        afresh rather than hurry to catch up."""
        pace = self._pace()
        if not pace:
            self._next_draw, self._cycle_seconds = None, self._draw_seconds
            return
        interval = 1.0 / pace
        now = time.perf_counter()
        if self._next_draw is None or now - self._next_draw > interval:
            self._next_draw = now
        elif now < self._next_draw:
            time.sleep(self._next_draw - now)  # precise to well under a millisecond on Windows since Python 3.11
        self._next_draw += interval
        self._cycle_seconds = max(self._draw_seconds, interval)

    def latest_pair(self):
        with self._lock:
            return self._pair

    def delay_ms(self):
        """The delay from camera to output over the last seconds, or None when no session is running."""
        if self.state is not SessionState.RUNNING:
            return None
        return self._metrics.summary()["end_to_end_ms"]

    def snapshot(self):
        with self._lock:
            state, message, label = self.state, self.message, self.input_label
            started, tracking, consent, feed = self._started_at, self._tracking, self._consent, self._feed
        running = state is SessionState.RUNNING
        camera_fps = getattr(feed, "rate", None) if running else None
        return {
            "state": state.value,
            "message": message,
            "input": label,
            # What the output is made from: the enrolled picture, or a sample clip's own frame.
            "source": ("enrolment" if self.uses_enrolment else "sample") if state is not SessionState.IDLE else None,
            "uptime_s": round(time.perf_counter() - started, 1) if running and started else None,
            "frames": self._metrics.total_frames if running else 0,
            "metrics": self._metrics.summary() if running else None,
            # What the camera itself delivers. The output cannot be faster than this.
            "camera_fps": None if camera_fps is None else round(camera_fps, 1),
            "tracking": tracking if running else None,
            # Why the still picture is shown although a face is in view, if it is.
            "holds": {"identity": running and self.identity_hold, "delay": running and self.delay_hold,
                      "consent": running and self.consent_hold},
            # The check a camera session starts with: what is being asked for, or how it ended.
            "consent": consent if state is not SessionState.IDLE else None,
            # True while the voice drives the picture and the camera is off.
            "voice": self.voice if state is not SessionState.IDLE else False,
        }

    # ---------------------------------------------------------- the thread

    def _run(self, source):
        rt = self._rt
        feed = tracker = None
        try:
            feed = rt.make_feed(source)
            with self._lock:
                self._feed = feed
            self._progress("Loading the models and tuning them for this GPU. A minute or two, the first time")
            engine = rt.engine()
            tracker = rt.make_tracker()
            if self.uses_enrolment:
                rt.load_enrolment()
                self._check_consent(feed, tracker, engine)
                if self.voice:  # the check is passed: the camera's work is done, and the voice takes over
                    with self._lock:
                        self._feed = None
                    feed.close()
                    feed = None
                    self._voice_session(engine)
                    rt.events.add("info", "Session stopped.")
                    self._finish(SessionState.IDLE, "")
                    return
            else:
                self._progress("Waiting for a front-facing frame")
                self._borrow_source(feed, tracker)
            pipeline = Pipeline(tracker, engine)
            if self.uses_enrolment:
                # The picture may come from another camera: measure movement from how the person sits at this one.
                pipeline.wait_for_neutral()
            with self._lock:
                self.state, self.message = SessionState.RUNNING, ""
                self._started_at = time.perf_counter()
            rt.events.add("info", f"Session started on {self.input_label}.")
            self._loop(feed, pipeline)
            rt.events.add("info", "Session stopped.")
            self._finish(SessionState.IDLE, "")
        except _Stopped:
            self._finish(SessionState.IDLE, "")
        except Exception as err:  # shown to the user, so the session never dies silently
            message = str(err) or type(err).__name__
            log.error("Session on %s ended: %s", self.input_label, message,
                      exc_info=not isinstance(err, SessionError))  # the trace, for anything not foreseen
            rt.events.add("error", message)
            self._finish(SessionState.ERROR, message)
        finally:
            with self._lock:
                self._feed = None
            if feed is not None:
                feed.close()
            if tracker is not None:
                tracker.close()
            self._fail_pending_commands()

    def _progress(self, message):
        if self._stop.is_set():
            raise _Stopped
        with self._lock:
            self.message = message

    def _finish(self, state, message):
        refused = state is SessionState.ERROR and (self._consent or {}).get("stage") == "refused"
        with self._lock:
            self.state, self.message = state, message
            self._pair = self._tracking = None
        database = getattr(self._rt, "db", None)
        if database is not None and self.record_id is not None:
            outcome = "refused" if refused else "error" if state is SessionState.ERROR else "stopped"
            database.end_session(self.record_id, outcome, message, frames=self._metrics.total_frames,
                                 virtual_camera=self._used_camera, voice_in_step=self._used_voice)

    def _keep_figures(self, now):
        """Every few seconds, the session's figures into the database, for its charts later."""
        database = getattr(self._rt, "db", None)
        if database is None or self.record_id is None or (self._kept_at is not None and now - self._kept_at < KEEP_FIGURES_SECONDS):
            return
        self._kept_at = now
        camera, voice = getattr(self._rt, "virtual_camera", None), getattr(self._rt, "virtual_microphone", None)
        self._used_camera |= bool(camera is not None and camera.on)
        self._used_voice |= bool(voice is not None and voice.on)
        m = self._metrics.summary(now)
        if m["fps"] is None:
            return
        identity = None
        monitor = getattr(self._rt, "identity", None)
        if monitor is not None:
            identity = monitor.snapshot().get("csim")
        database.add_figures(self.record_id, fps=m["fps"], delay_ms=m["end_to_end_ms"], render_ms=m["render_ms"],
                             identity=identity, live_share=m["live_share"])

    def _check_consent(self, feed, tracker, engine):
        """Before anything is animated: is a living person at the camera, and is it the person the
        picture belongs to? Two actions are asked for and the face is matched all the way through
        (consent/check.py). Raises SessionError, with the reason, if the check is not passed."""
        rt = self._rt
        check, scorer = rt.consent_check()
        still = mark(engine.source_frame.copy())  # what the output shows in the meantime
        self._progress("Checking that it is you")
        last = -1
        while not check.done:
            if self._stop.is_set():
                raise _Stopped
            item = feed.read(after_index=last, timeout=0.5)
            if item is None:
                if feed.ended:
                    raise SessionError("The camera stopped delivering frames.")
                continue
            last = item.index
            track = tracker.process(item.image)
            check.frame(liveness.read(track), item.captured_at)
            if track.ok and check.wants_face(item.captured_at):
                check.face(scorer.embed(item.image, track.landmarks), item.captured_at)
            self._pair_id += 1
            with self._lock:
                self._pair = FramePair(self._pair_id, item.image, still, False, track.status.value,
                                       track.landmarks if track.ok else None)
                self._frame_made()
                self._consent = check.snapshot(item.captured_at)
        with self._lock:
            self._pair = None  # the frames of the check are not frames of the session
        if not check.passed:
            raise SessionError(check.reason)
        rt.events.add("info", "The liveness check was passed, and the face matched the verified face all the way "
                              f"through ({check.lowest:.2f} at the lowest).")

    def _borrow_source(self, feed, tracker):
        """A sample clip animates its own first front-facing frame. Nothing is enrolled or stored."""
        deadline = time.perf_counter() + SAMPLE_SOURCE_SECONDS
        last = -1
        while time.perf_counter() < deadline:
            if self._stop.is_set():
                raise _Stopped
            item = feed.read(after_index=last, timeout=0.5)
            if item is None:
                if feed.ended:
                    raise SessionError("The clip ended before a front-facing frame was found.")
                continue
            last = item.index
            track = tracker.process(item.image)
            if is_frontal(track):
                self._rt.use_sample_source(item.image, square_face_crop(item.image, track.bbox))
                return
        raise SessionError("The clip shows no single front-facing face to animate.")

    def _voice_session(self, engine):
        """The voice drives the picture (speech/live.py): the frame whose speech was heard a moment ago is
        drawn, with the mark, the identity guard and the virtual camera as in a camera session. The delay
        watchdog does not act here: the picture is late on purpose, and the voice is held back to match."""
        rt = self._rt
        drive = rt.voice()
        try:
            drive.start(rt.picture_path(), microphone=rt.voice_microphone(), progress=self._progress)
            neutral, rotation = engine.source_reading()
            with rt.engine_lock:
                engine.reset_reference()
                engine.drive(None, motion=(neutral, rotation))
            from ..speech.ditto import DittoReadings
            readings = DittoReadings(neutral, engine.rotation, blink=drive.blink)
            still = mark(engine.source_frame.copy())
            camera = camera_off_card()
            with self._lock:
                self.state, self.message = SessionState.RUNNING, ""
                self._started_at = time.perf_counter()
            rt.events.add("info", f"Session started with the camera off: your voice from {drive.microphone} drives "
                                  "the picture. Switch on \"Voice in step\" so that the meeting hears you in step with it.")
            pair_id = self._pair_id
            while not self._stop.is_set():
                got = drive.due(time.perf_counter())
                if got is None:
                    time.sleep(0.003)
                    continue
                self._anchor.clear()  # no neutral pose to take: the picture's own is the neutral one
                drawing = time.perf_counter()
                with rt.engine_lock:
                    reading = readings.between(got.frame, got.after, got.share, got.voice, got.number)
                    live = rt.features.enabled("reenactment") and not self.identity_hold
                    output = mark(engine.drive(None, motion=reading).copy()) if live else still
                    timing = dict(engine.last_timing_ms) if live else {}
                    self._run_commands(camera, None)
                finished = time.perf_counter()
                timing["total"] = (finished - drawing) * 1000
                # From the speech being heard to its frame drawn: what the voice is held back by.
                self._metrics.add(finished, timing, (finished - got.heard) * 1000, live, got.skipped)
                self._log_figures(finished)
                self._keep_figures(finished)
                pair_id += 1
                with self._lock:
                    self._pair = FramePair(pair_id, camera, output, live, TrackStatus.OK.value, None)
                    self._frame_made()
                    self._tracking = {"status": TrackStatus.OK.value, "pose_deg": None, "mouth_open": round(float(got.voice), 3)}
        finally:
            drive.stop()

    def _loop(self, feed, pipeline):
        rt = self._rt
        pair_id, status = self._pair_id, None
        held, neutrals = rt.engine_holds, pipeline.neutral_taken
        newest, tracking_over = _Newest(), threading.Event()

        def track_frames():
            last, quiet = -1, False
            try:
                while not self._stop.is_set() and not tracking_over.is_set():
                    item = feed.read(after_index=last, timeout=0.5)
                    if item is None:
                        if feed.ended:
                            raise SessionError("The camera stopped delivering frames.")
                        # The feed opens a silent camera again by itself (capture/feed.py); say so meanwhile.
                        if not quiet and getattr(feed, "silent_seconds", 0.0) >= CAMERA_QUIET_SECONDS:
                            quiet = True
                            rt.events.add("warning", "The camera stopped sending pictures. Reconnecting to it; "
                                                     "the session goes on as soon as it is back.")
                        continue
                    if quiet:
                        quiet = False
                        rt.events.add("info", "The camera is back.")
                    skipped = item.index - last - 1 if last >= 0 else 0
                    last = item.index
                    started = time.perf_counter()
                    track = pipeline.tracker.process(item.image)
                    # The face crop too: it is CPU work and does not need the GPU's turn.
                    newest.put(item, track, skipped, pipeline.prepare(item.image, track, item.captured_at))
                    ahead = time.perf_counter() - started
                    # One frame is prepared per frame drawn. It is started late enough to be ready just as
                    # the GPU comes free, so it is as fresh as it can be when it is drawn.
                    while not newest.wait_until_taken(0.2):
                        if self._stop.is_set() or tracking_over.is_set():
                            return
                    time.sleep(max(0.0, self._cycle_seconds - ahead - 0.004))
            except Exception as err:  # handed to the drawing loop, which ends the session with it
                newest.fail(err)

        tracker_thread = threading.Thread(target=track_frames, name="live-tracking", daemon=True)
        tracker_thread.start()
        try:
            self._draw(pipeline, newest, pair_id, status, held, neutrals)
        finally:
            tracking_over.set()
            tracker_thread.join(timeout=5.0)

    def _draw(self, pipeline, newest, pair_id, status, held, neutrals):
        rt = self._rt
        self._next_draw = None
        while not self._stop.is_set():
            self._wait_for_turn()
            got = newest.take(timeout=0.5)
            if got is None:
                self._next_draw = None
                continue
            item, tracked, dropped, prepared = got
            drawing = time.perf_counter()
            with rt.engine_lock:
                pipeline.steady_crop = rt.features.enabled("steady_crop")
                pipeline.steady_keypoints = rt.features.enabled("steady_keypoints")
                pipeline.natural_range = rt.features.enabled("natural_range")
                if self._anchor.is_set():
                    self._anchor.clear()
                    pipeline.wait_for_neutral()
                reenact = rt.features.enabled("reenactment") and not self.identity_hold and not self.consent_hold
                # The movement is shown as it was a fixed moment ago, so that it is shown evenly (see pacing.py).
                motion_at = self._clock.moment(drawing, item.captured_at) if rt.features.enabled("even_motion") else None
                result = pipeline.step(item.image, reenact=reenact, at=item.captured_at, track=tracked,
                                       prepared=prepared, show=not self.delay_hold, motion_at=motion_at)
                self._run_commands(item.image, result.track)
                if rt.engine_holds != held:  # the picture was replaced while running: a new picture, a new neutral pose
                    held = rt.engine_holds
                    pipeline.wait_for_neutral()
            if pipeline.neutral_taken != neutrals:
                neutrals = pipeline.neutral_taken
                rt.events.add("info", "Neutral pose taken from the camera. Movement is measured from how you sit now.")
            finished = time.perf_counter()
            self._draw_seconds = 0.8 * self._draw_seconds + 0.2 * (finished - drawing)  # how long the GPU's turn takes
            # How old the movement in the picture is, which is not always how old the camera frame is.
            shown = item.captured_at if result.motion_at is None else result.motion_at
            delay_ms = (finished - shown) * 1000
            self._metrics.add(finished, result.timing_ms, delay_ms, result.live, dropped)
            self._log_figures(finished)
            self._keep_figures(finished)
            self._watch_delay(delay_ms, finished, drawn="render" in result.timing_ms)
            if "render" in result.timing_ms:
                turn = finished - drawing
                self._drawn_seconds = turn if self._drawn_seconds == 0.0 else 0.9 * self._drawn_seconds + 0.1 * turn
                self._watch_camera(finished)

            if result.status is not status:
                if status is not None or result.status is not TrackStatus.OK:
                    rt.events.add(*STATUS_EVENTS[result.status])
                status = result.status
            pair_id += 1
            track = result.track
            # Tracked even while the still picture is shown, so the identity monitor can tell when to resume.
            self.resting_since = (self.resting_since or finished) if track.ok and at_rest(track) else None
            tracking = {"status": result.status.value, "pose_deg": None, "mouth_open": None}
            if track.ok:
                tracking["pose_deg"] = [round(float(angle), 1) for angle in track.pose_deg]
                tracking["mouth_open"] = round(float(track.blendshapes.get("jawOpen", 0.0)), 3)
            with self._lock:
                self._pair = FramePair(pair_id, item.image, result.output, result.live,
                                       result.status.value, track.landmarks if track.ok else None,
                                       crop_window=result.crop_window)
                self._frame_made()
                self._tracking = tracking

    def _log_figures(self, now):
        """Every FIGURES_EVERY_SECONDS, one line in the log with how the session is running: frames drawn a second,
        the delay, the render time, the share of camera frames skipped and of frames reenacted, and what the
        virtual camera sent: frames a second in all, and how many of them were new pictures."""
        camera = getattr(self._rt, "virtual_camera", None)
        counts = (camera.sent, camera.fresh) if camera is not None and camera.on else None
        if self._figures_at is None:
            self._figures_at = (now, counts)
            return
        then, before = self._figures_at
        if now - then < FIGURES_EVERY_SECONDS:
            return
        self._figures_at = (now, counts)
        m = self._metrics.summary(now)
        if m["fps"] is None:
            return
        line = (f"Session on {self.input_label}: {m['fps']:.1f} frames a second drawn, {m['end_to_end_ms']:.0f} ms delay, "
                f"render {m['render_ms'] or 0:.0f} ms, {100 * m['dropped_share']:.0f}% of camera frames skipped, "
                f"{100 * m['live_share']:.0f}% reenacted")
        if counts is not None and before is not None and counts[0] >= before[0]:
            span = now - then
            line += (f"; virtual camera {(counts[0] - before[0]) / span:.1f} frames a second, "
                     f"{(counts[1] - before[1]) / span:.1f} of them new")
        log.info(line)

    def _watch_delay(self, delay_ms, finished, drawn):
        """Hold the live picture back while it is arriving too late (see watchdog.py)."""
        rt = self._rt
        if not rt.features.enabled("delay_watchdog"):
            self.delay_hold = False
            self._watchdog.reset()
            return
        # Only frames that were drawn say how late the picture is, and not the first ones of a session.
        if not drawn or finished - self._started_at < WATCHDOG_AFTER_SECONDS:
            return
        held = self._watchdog.sample(delay_ms, finished)
        if held == self.delay_hold:
            return
        self.delay_hold = held
        limit, mean = self._watchdog.limit_ms, self._watchdog.mean_ms
        if held:
            rt.events.add("warning", f"The picture is arriving late: {mean:.0f} ms on average, above the limit of "
                                     f"{limit:.0f} ms. Showing the still picture until it recovers.")
        else:
            rt.events.add("info", f"The delay is back to {mean:.0f} ms. Reenactment resumed.")

    def _watch_camera(self, now):
        """Say so, once, when it is the camera that holds the frame rate down and not the drawing."""
        if self._slow_camera_told or now - self._started_at < SLOW_CAMERA_AFTER_SECONDS:
            return
        rate = getattr(self._feed, "rate", None)
        if rate is None or not getattr(self._feed, "is_camera", False) or rate >= SLOW_CAMERA_SHARE / self._drawn_seconds:
            return
        self._slow_camera_told = True
        self._rt.events.add("warning", f"The camera is delivering {rate:.0f} frames a second, and the output cannot be "
                                       f"faster than the camera; {1.0 / self._drawn_seconds:.0f} could be drawn. "
                                       "More light usually brings a webcam back to 30.")

    def _run_commands(self, frame, track):
        while True:
            try:
                action, future = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                future.set_result(action(frame, track))
            except Exception as err:
                future.set_exception(err)

    def _fail_pending_commands(self):
        while True:
            try:
                _, future = self._commands.get_nowait()
            except queue.Empty:
                return
            future.set_exception(SessionError("The session ended."))
