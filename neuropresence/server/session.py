"""A live session: frames from the feed go through the pipeline on a background thread."""

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..capture import TrackStatus
from ..capture.crop import square_face_crop
from ..consent import liveness
from ..consent.disclosure import mark
from ..pipeline import Pipeline
from ..targets import TARGETS
from .metrics import MetricsWindow
from .watchdog import DelayWatchdog

SAMPLE_SOURCE_SECONDS = 10.0
MAX_FRONTAL_ANGLE_DEG = 20.0
# The watchdog starts reading this long after the first frame, when the session has found its pace.
WATCHDOG_AFTER_SECONDS = 2.0

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
        self._started_at = None
        self._pair = None
        self._tracking = None
        self._metrics = MetricsWindow()
        self._draw_seconds = 0.0  # a running average of how long one frame takes to draw
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
        self._anchor = threading.Event()  # a fresh neutral pose has been asked for

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

    def start(self, source, label, uses_enrolment):
        with self._lock:
            if self.active:
                raise SessionError("A session is already running.")
            self._stop.clear()
            self._metrics = MetricsWindow()
            self._pair = self._tracking = self._started_at = None
            self.run += 1
            self.identity_hold = self.delay_hold = self.consent_hold = False
            self._consent = None
            self._watchdog.reset()
            self._anchor.clear()
            self.uses_enrolment = uses_enrolment
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

    def latest_pair(self):
        with self._lock:
            return self._pair

    def snapshot(self):
        with self._lock:
            state, message, label = self.state, self.message, self.input_label
            started, tracking, consent = self._started_at, self._tracking, self._consent
        running = state is SessionState.RUNNING
        return {
            "state": state.value,
            "message": message,
            "input": label,
            # What the output is made from: the enrolled picture, or a sample clip's own frame.
            "source": ("enrolment" if self.uses_enrolment else "sample") if state is not SessionState.IDLE else None,
            "uptime_s": round(time.perf_counter() - started, 1) if running and started else None,
            "frames": self._metrics.total_frames if running else 0,
            "metrics": self._metrics.summary() if running else None,
            "tracking": tracking if running else None,
            # Why the still picture is shown although a face is in view, if it is.
            "holds": {"identity": running and self.identity_hold, "delay": running and self.delay_hold,
                      "consent": running and self.consent_hold},
            # The check a camera session starts with: what is being asked for, or how it ended.
            "consent": consent if state is not SessionState.IDLE else None,
        }

    # ---------------------------------------------------------- the thread

    def _run(self, source):
        rt = self._rt
        feed = tracker = None
        try:
            feed = rt.make_feed(source)
            self._progress("Loading the models and tuning them for this GPU. A minute or two, the first time")
            engine = rt.engine()
            tracker = rt.make_tracker()
            if self.uses_enrolment:
                rt.load_enrolment()
                self._check_consent(feed, tracker, engine)
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
            rt.events.add("error", message)
            self._finish(SessionState.ERROR, message)
        finally:
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
        with self._lock:
            self.state, self.message = state, message
            self._pair = self._tracking = None

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

    def _loop(self, feed, pipeline):
        rt = self._rt
        pair_id, status = self._pair_id, None
        held, neutrals = rt.engine_holds, pipeline.neutral_taken
        newest, tracking_over = _Newest(), threading.Event()

        def track_frames():
            last = -1
            try:
                while not self._stop.is_set() and not tracking_over.is_set():
                    item = feed.read(after_index=last, timeout=0.5)
                    if item is None:
                        if feed.ended:
                            raise SessionError("The camera stopped delivering frames.")
                        continue
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
                    time.sleep(max(0.0, self._draw_seconds - ahead - 0.004))
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
        while not self._stop.is_set():
            got = newest.take(timeout=0.5)
            if got is None:
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
                result = pipeline.step(item.image, reenact=reenact, at=item.captured_at, track=tracked,
                                       prepared=prepared, show=not self.delay_hold)
                self._run_commands(item.image, result.track)
                if rt.engine_holds != held:  # the picture was replaced while running: a new picture, a new neutral pose
                    held = rt.engine_holds
                    pipeline.wait_for_neutral()
            if pipeline.neutral_taken != neutrals:
                neutrals = pipeline.neutral_taken
                rt.events.add("info", "Neutral pose taken from the camera. Movement is measured from how you sit now.")
            finished = time.perf_counter()
            self._draw_seconds = 0.8 * self._draw_seconds + 0.2 * (finished - drawing)  # how long the GPU's turn takes
            delay_ms = (finished - item.captured_at) * 1000
            self._metrics.add(finished, result.timing_ms, delay_ms, result.live, dropped)
            self._watch_delay(delay_ms, finished, drawn="render" in result.timing_ms)

            if result.status is not status:
                if status is not None or result.status is not TrackStatus.OK:
                    rt.events.add(*STATUS_EVENTS[result.status])
                status = result.status
            pair_id += 1
            track = result.track
            tracking = {"status": result.status.value, "pose_deg": None, "mouth_open": None}
            if track.ok:
                tracking["pose_deg"] = [round(float(angle), 1) for angle in track.pose_deg]
                tracking["mouth_open"] = round(float(track.blendshapes.get("jawOpen", 0.0)), 3)
            with self._lock:
                self._pair = FramePair(pair_id, item.image, result.output, result.live,
                                       result.status.value, track.landmarks if track.ok else None,
                                       crop_window=result.crop_window)
                self._tracking = tracking

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
