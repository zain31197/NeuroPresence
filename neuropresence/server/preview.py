"""The enrolment preview: the camera with the enrolment checks, live.

It runs the tracker and the checks on every frame, but not the reenactment
engine, so it starts at once and needs no GPU.
"""

import threading
import time
from concurrent.futures import Future

from ..consent import liveness
from ..consent.check import THE_FACE_CHANGED
from ..enrolment import make_candidate, make_pose_candidate, prepare
from ..enrolment.checks import outline_for
from .session import FramePair, SessionError, SessionState

READY_SECONDS = 0.4  # the checks must hold this long before a picture can be taken
BURST_SECONDS = 0.5  # from the first good frame, the best frame of this long is kept
TAKE_TIMEOUT_SECONDS = 4.0  # give up if no frame passes in this time
CHECK_TIMEOUT_SECONDS = 45.0  # the liveness prompt before a face is verified ends by itself well within this
UNWATCHED_SECONDS = 15.0  # nobody has fetched a frame for this long: close the camera


class NotGoodEnough(ValueError):
    """No frame passed the checks. The message says what to change."""


class EnrolmentPreview:
    def __init__(self, runtime):
        self._rt = runtime
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.state = SessionState.IDLE
        self.message = ""
        self.input_label = None
        self._pair = None
        self._checks = None  # of the newest frame, as dicts
        self._hint = ""
        self._tip = ""  # with every check passed: what would make the picture better still
        self._ready = False
        self._outline = None  # where the face should be, as shares of the frame, for the app to draw
        self._taking = None  # the picture request in progress, if any
        self._liveness = None  # how the liveness prompt of that request stands or ended, for the app to show
        self._watched_at = 0.0  # when a frame was last fetched for showing
        self._pose = None  # None for the ordinary frontal capture; "left"/"right"/"up"/"down" for a pose capture

    @property
    def active(self):
        return self.state in (SessionState.STARTING, SessionState.RUNNING)

    def start(self, source, label, pose=None):
        """pose, if given, switches every frame to the pose-capture checks (see checks.py:
        evaluate_pose) instead of the ordinary frontal ones: the picture is judged on being
        turned the right way, not on facing the camera, and there is no outline to draw."""
        with self._lock:
            if self.active:
                raise SessionError("The camera is already open.")
            self._stop.clear()
            self._pair = self._checks = self._taking = self._outline = self._liveness = None
            self._hint, self._tip, self._ready = "", "", False
            self._watched_at = time.perf_counter()
            self._pose = pose
            self.state, self.message, self.input_label = SessionState.STARTING, "Opening the camera", label
            self._thread = threading.Thread(target=self._run, args=(source,), name="enrolment-preview", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        with self._lock:
            self.state, self.message = SessionState.IDLE, ""
            self._pair = self._checks = self._outline = None
            self._hint, self._tip, self._ready = "", "", False

    def take(self):
        """Take the picture: the sharpest, most open-eyed frame of a short burst that passes every check.

        The picture a face is verified with is taken only after the liveness prompt has been answered
        (consent/check.py), so that a photograph held to the camera cannot become the user's face. A
        pose is registered for a face that has been verified already and has no prompt of its own.

        Blocks until it is done. Raises NotGoodEnough, saying what to change, if the prompt is not
        answered or no frame passes.
        """
        check = scorer = None
        if self._pose is None:
            check, scorer = self._rt.verification_check()
        with self._lock:
            if self.state is not SessionState.RUNNING:
                raise SessionError("Open the camera first.")
            if self._taking is not None:
                raise SessionError("A picture is already being taken.")
            self._watched_at = time.perf_counter()
            request = {"future": Future(), "best": None, "first_good_at": None,
                       "deadline": time.perf_counter() + TAKE_TIMEOUT_SECONDS,
                       "check": check, "scorer": scorer, "checked": check is None}
            self._taking, self._liveness = request, None
        return request["future"].result(timeout=(CHECK_TIMEOUT_SECONDS if check else TAKE_TIMEOUT_SECONDS) + 5.0)

    def latest_pair(self):
        """The newest frame, for showing. Fetching it is what keeps the camera open."""
        with self._lock:
            self._watched_at = time.perf_counter()
            return self._pair

    def snapshot(self):
        with self._lock:
            running = self.state is SessionState.RUNNING
            return {
                "state": self.state.value,
                "message": self.message,
                "input": self.input_label,
                "pose": self._pose,
                "checks": self._checks if running else None,
                "hint": self._hint if running else "",
                "tip": self._tip if running else "",
                # True once every check has held steadily: a picture can be taken now.
                "ready": self._ready if running else False,
                # A face that fills this box passes the size and framing checks with room to spare.
                # Only set for the ordinary frontal capture: a pose capture has no fixed target box.
                "outline": self._outline if running else None,
                "taking": self._taking is not None,
                # While a face is being verified: what the liveness prompt asks for, or how it ended.
                "liveness": self._liveness if running else None,
            }

    # ---------------------------------------------------------- the thread

    def _run(self, source):
        rt = self._rt
        feed = tracker = None
        try:
            feed = rt.make_feed(source)
            tracker = rt.make_tracker()
            with self._lock:
                self.state, self.message = SessionState.RUNNING, ""
            self._loop(feed, tracker)
            self._finish(SessionState.IDLE, "")
        except Exception as err:
            self._finish(SessionState.ERROR, str(err) or type(err).__name__)
        finally:
            if feed is not None:
                feed.close()
            if tracker is not None:
                tracker.close()
            self._settle(None, SessionError("The camera was closed."))

    def _finish(self, state, message):
        with self._lock:
            self.state, self.message = state, message
            self._pair = self._checks = self._outline = None
            self._hint, self._tip, self._ready = "", "", False

    def _loop(self, feed, tracker):
        last, pair_id, good_since, shape, outline = -1, 0, None, None, None
        while not self._stop.is_set():
            item = feed.read(after_index=last, timeout=0.5)
            if item is None:
                if feed.ended:
                    raise SessionError("The camera stopped delivering frames.")
                continue
            last = item.index
            frame = prepare(item.image)
            pose = self._pose
            if pose is None and frame.shape[:2] != shape:  # the outline depends only on the shape of the camera's picture
                shape = frame.shape[:2]
                outline = outline_for(frame.shape[1], frame.shape[0])
            track = tracker.process(frame)
            candidate = make_pose_candidate(frame, track, pose) if pose else make_candidate(frame, track, "camera")
            now = time.perf_counter()
            good_since = (good_since or now) if candidate.passed else None
            summary = candidate.summary()
            pair_id += 1
            with self._lock:
                self._pair = FramePair(pair_id, frame, None, False, track.status.value,
                                       track.landmarks if track.ok else None, kind="preview")
                self._checks, self._hint, self._tip = summary["checks"], summary["hint"], summary["tip"]
                self._ready = good_since is not None and now - good_since >= READY_SECONDS
                self._outline = outline
                request = self._taking
                unwatched = now - self._watched_at > UNWATCHED_SECONDS
            if request is not None:
                if self._answered(request, candidate, frame, track, item.captured_at, now):
                    self._serve(request, candidate, now)
            elif unwatched:
                # A camera left open with no page showing it is a camera filming for nothing.
                self._rt.events.add("info", "The camera was closed because the Enrolment screen is no longer open.")
                return

    def _answered(self, request, candidate, frame, track, at, now):
        """Has the liveness prompt of this request been answered, so that the picture may be taken?"""
        check = request.get("check")
        if check is None or request["checked"]:
            return True
        if not request.get("began"):
            # The prompt begins once the picture itself is good. Until then the checks say what to change.
            if not candidate.passed:
                if now >= request["deadline"]:
                    self._settle(None, NotGoodEnough(candidate.summary()["hint"] or "The picture did not pass the checks."))
                return False
            request["began"] = True
        check.frame(liveness.read(track), at)
        if track.ok and check.wants_face(at):
            check.face(request["scorer"].embed(frame, track.landmarks), at)
        with self._lock:
            self._liveness = check.snapshot(at)
            self._watched_at = now  # the person is busy answering: the camera is being watched
        if check.passed is False:
            self._settle(None, NotGoodEnough(check.reason))
        elif check.passed:
            request["checked"], request["deadline"] = True, now + TAKE_TIMEOUT_SECONDS
        return False  # the picture is taken from the frames that follow

    def _serve(self, request, candidate, now):
        if candidate.passed:
            if request["best"] is None or candidate.score > request["best"].score:
                request["best"] = candidate
            request["first_good_at"] = request["first_good_at"] or now
        burst_over = request["first_good_at"] is not None and now - request["first_good_at"] >= BURST_SECONDS
        if burst_over or now >= request["deadline"]:
            best, check = request["best"], request.get("check")
            if best is not None and check is not None and not check.same_face(
                    request["scorer"].embed(best.image, best.landmarks)):
                self._settle(None, NotGoodEnough(THE_FACE_CHANGED))  # not the face that answered the prompt
            elif best is not None:
                self._settle(best, None)
            else:
                self._settle(None, NotGoodEnough(candidate.summary()["hint"] or "The picture did not pass the checks."))

    def _settle(self, candidate, error):
        with self._lock:
            request, self._taking = self._taking, None
        if request is None:
            return
        if error is not None:
            request["future"].set_exception(error)
        else:
            request["future"].set_result(candidate)
