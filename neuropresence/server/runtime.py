"""What the server keeps alive between requests: the engine, the enrolment, the session, the switches."""

import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from dataclasses import replace

from ..capture.crop import MAX_PICTURE_DIM, limit_size, square_face_crop
from ..consent import LivenessChallenge
from ..consent.check import WATCH_SECONDS, ConsentCheck, ConsentWatch
from ..enrolment import POSE_KEYS, Check, EnrolmentStore, make_candidate, prepare
from ..enrolment.checks import MIN_FACE_HEIGHT_PX, TARGET_FACE_HEIGHT_SHARE
from ..identity import SAME_PERSON_CSIM, GuardState, IdentityGuard
from ..targets import TARGETS
from .benchmarks import latest_benchmark
from .features import FeatureSet
from .guide import enrolment_guide
from .preview import EnrolmentPreview
from .session import LiveSession, SessionError, SessionState

log = logging.getLogger("neuropresence.server")
ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLIPS = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving"
SAMPLE_NAMES = ("d0", "d3", "d6", "d9", "d10", "d13", "d18")
CAMERA_INDEXES = (0, 1, 2)
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


class EventLog:
    """The last few things that happened, for the app's event list."""

    def __init__(self, limit=200):
        self._events = deque(maxlen=limit)
        self._lock = threading.Lock()
        self.last_id = 0

    def add(self, level, message):
        with self._lock:
            self.last_id += 1
            event = {"id": self.last_id, "at": time.time(), "level": level, "message": message}
            self._events.append(event)
            return event

    def since(self, last_id=0):
        with self._lock:
            return [event for event in self._events if event["id"] > last_id]


class IdentityMonitor:
    """Scores the live output against the picture it is made from, twice a second, and acts on it.

    It runs on its own thread so the pipeline never waits for it: one score takes the CPU about
    50 ms, and scoring four times a second left the frame rate and the delay where they were
    (results/delay_study.json). What a low score leads to is decided by the guard (identity/guard.py):
    nothing for a short dip, a fresh neutral pose for a lasting one, and the still picture if that
    does not help. The still picture then stays until the person resumes.

    The same thread keeps the consent watch (consent/check.py): once a second the face in the camera
    picture is compared with the signatures registered for the user, and while it is someone else's
    the still picture is shown.
    """

    def __init__(self, runtime, interval=0.5):
        self._rt = runtime
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = {"available": False, "csim": None, "reason": "Starts with the first session."}
        self.guard = IdentityGuard()
        self._guard_lock = threading.Lock()
        self._watching = None  # the session and the picture the readings of the guard belong to
        self.watch = ConsentWatch()
        self._next_watch = 0.0

    def ensure_started(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="identity-monitor", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def snapshot(self):
        with self._lock:
            state = dict(self._state)
        with self._guard_lock:
            guard = self.guard
            state["guard"] = {
                "enabled": self._rt.features.enabled("identity_guard"),
                "state": guard.state.value,
                # How long the still picture has been shown, for the prompt that offers to resume.
                "seconds": round(time.perf_counter() - guard.since, 1) if guard.state is GuardState.FALLBACK else None,
                "mean": None if guard.mean is None else round(guard.mean, 3),
                "low": guard.low,
            }
            # The consent watch: how alike the camera face last was to the registered one.
            state["consent"] = {"held": self.watch.held,
                                "match": None if self.watch.match is None else round(self.watch.match, 3)}
        return state

    def resume(self):
        """Go live again after a fallback: a fresh neutral pose, and the next seconds have to prove it."""
        session = self._rt.session
        with self._guard_lock:
            if not session.identity_hold:
                raise SessionError("The output is not on hold.")
            self.guard.resume(time.perf_counter())
            session.request_anchor()
            session.hold_for_identity(False)
        self._rt.events.add("info", "Reenactment resumed with a fresh neutral pose.")

    def _set(self, **state):
        with self._lock:
            self._state = state

    def _follow(self, session):
        """A new session, a new picture or the switch turned off: what the guard knew no longer holds."""
        watching = (session.run, self._rt.engine_holds)
        enabled = self._rt.features.enabled("identity_guard")
        with self._guard_lock:
            changed = watching != self._watching
            self._watching = watching
            if changed or (not enabled and (self.guard.state is not GuardState.STEADY or session.identity_hold)):
                self.guard.reset()
                session.hold_for_identity(False)
            if changed:
                self.watch.reset()
                self._next_watch = 0.0  # a new session is looked at straight away
                session.hold_for_consent(False)
        return enabled

    def _watch_camera(self, session, scorer, pair):
        """Once a second: is the person at the camera still the one the picture belongs to?"""
        if (pair is None or pair.kind != "live" or pair.landmarks is None or not session.uses_enrolment
                or session.state is not SessionState.RUNNING):
            return
        now = time.perf_counter()
        if now < self._next_watch:
            return
        self._next_watch = now + WATCH_SECONDS
        signatures = self._rt.face_signatures()
        if not signatures:
            return
        face = scorer.embed(pair.camera, pair.landmarks)
        match = max(scorer.similarity(known, face) for known in signatures)
        with self._guard_lock:
            action = self.watch.sample(match)
            if action is not None:
                # What the output scored while someone else drove it says nothing about the picture:
                # the identity guard starts again, and this hold takes the place of any it had put up.
                self.guard.reset()
                session.hold_for_identity(False)
            if action == "hold":
                session.hold_for_consent(True)
            elif action == "release":
                session.request_anchor()  # the neutral pose may be the other person's: measure movement afresh
                session.hold_for_consent(False)
        if action == "hold":
            self._rt.events.add("warning", f"The face at the camera is not the verified face (match {match:.2f}). "
                                           "Showing the still picture until the verified face is back.")
        elif action == "release":
            self._rt.events.add("info", f"The verified face is back (match {match:.2f}). Reenactment resumed.")

    def _act(self, score):
        session, events = self._rt.session, self._rt.events
        with self._guard_lock:
            if session.identity_hold:
                return
            action = self.guard.sample(score, time.perf_counter())
            mean, reason = self.guard.mean, self.guard.reason
            if action == "anchor":
                session.request_anchor()
            elif action == "fallback":
                session.hold_for_identity(True)
        if action == "anchor" and reason == "changed":
            events.add("info", f"The match has stepped down since the last neutral pose (identity {mean:.2f}). "
                               "Taking the neutral pose again.")
        elif action == "anchor":
            events.add("warning", f"The output has stopped matching your picture (identity {mean:.2f} over the last "
                                  "seconds). Taking a fresh neutral pose.")
        elif action == "fallback":
            events.add("warning", f"The output does not match your picture (identity {mean:.2f}). Showing the still "
                                  "picture until you resume.")
        elif action == "recovered":
            events.add("info", f"Identity match is back ({mean:.2f}).")

    def _run(self):
        try:
            scorer = self._rt.scorer()
        except Exception as err:  # most often the ArcFace model has not been downloaded
            self._set(available=False, csim=None, reason=str(err))
            return
        held, source_embedding = None, None
        while not self._stop.wait(self._interval):
            session = self._rt.session
            guarding = self._follow(session)
            pair = session.latest_pair()
            try:
                self._watch_camera(session, scorer, pair)
            except Exception as err:
                self._set(available=False, csim=None, reason=f"The face at the camera could not be compared: {err}")
                return
            if pair is None or not pair.live:
                self._set(available=True, csim=None, reason=None)
                continue
            try:
                if self._rt.engine_holds != held:
                    held = self._rt.engine_holds
                    source_embedding = self._rt.source_signature(scorer)
                if source_embedding is None:
                    self._set(available=True, csim=None, reason=None)
                    continue
                embedding = scorer.embed(pair.output)
                score = None if embedding is None else scorer.similarity(source_embedding, embedding)
                self._set(available=True, csim=None if score is None else round(score, 3), reason=None)
            except Exception as err:
                self._set(available=False, csim=None, reason=f"Identity scoring failed: {err}")
                return
            if guarding:
                self._act(score)


def _default_engine():
    from ..reenactment import ReenactmentEngine

    # TensorRT if it is installed, which is the fastest; otherwise the networks compiled by PyTorch;
    # otherwise as they are. The first start on a machine takes a minute or two for the conversion.
    return ReenactmentEngine(tensorrt=True, compile_networks=True)


def _default_feed(source):
    from ..capture import FrameSourceError
    from ..capture.feed import LiveFeed

    try:
        return LiveFeed(source)
    except FrameSourceError:
        if isinstance(source, int):
            raise SessionError(f"Camera {source} could not be opened. Check that it is connected "
                               "and not in use by another app.") from None
        raise SessionError(f"The clip could not be opened: {source}") from None


def _default_tracker(video=True):
    from ..capture import FaceTracker

    return FaceTracker(video=video)


def _default_scorer():
    from ..identity import IdentityScorer

    return IdentityScorer()


def _default_gpu():
    """Name and memory of the GPU, or None if there is none."""
    import torch

    if not torch.cuda.is_available():
        return None
    device = torch.cuda.get_device_properties(0)
    gb = 1024**3
    return {
        "name": device.name,
        "total_gb": round(device.total_memory / gb, 2),
        "allocated_gb": round(torch.cuda.memory_allocated() / gb, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated() / gb, 2),
    }


def _jpeg(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes() if ok else None


class Runtime:
    def __init__(self, engine_factory=_default_engine, feed_factory=_default_feed,
                 tracker_factory=_default_tracker, scorer_factory=_default_scorer, gpu_probe=_default_gpu,
                 data_dir=DATA_DIR, results_dir=RESULTS_DIR, sample_dir=SAMPLE_CLIPS):
        self._engine_factory = engine_factory
        self.make_feed = feed_factory
        self.make_tracker = tracker_factory
        self._scorer_factory = scorer_factory
        self._gpu_probe = gpu_probe
        self._results_dir = Path(results_dir)
        self._sample_dir = Path(sample_dir)
        self._engine = None
        self._scorer = None
        # Re-entrant: the session thread holds it for a frame and may swap the picture inside it.
        self.engine_lock = threading.RLock()
        self._scorer_lock = threading.Lock()
        # Which picture the engine currently animates: ("enrolment", id), ("sample", n) or None.
        self.engine_holds = None
        self._samples_used = 0
        self._gpu_peak = 0.0

        self.store = EnrolmentStore(Path(data_dir) / "enrolment")
        self.enrolment = self.store.load()  # the meeting picture
        self.face = self.store.load_identity()  # who the user is: a signature taken live from the camera
        self.candidate = None  # a checked picture waiting to be confirmed or discarded
        self._candidate_id = 0

        self.features = FeatureSet()
        self.events = EventLog()
        self.session = LiveSession(self)
        self.preview = EnrolmentPreview(self)
        self.identity = IdentityMonitor(self)

    # ------------------------------------------------------------- engine

    def engine(self):
        """The reenactment engine, loaded on first use (this takes a few seconds)."""
        with self.engine_lock:
            if self._engine is None:
                self._engine = self._engine_factory()
                self._apply_pose_calibration(self._engine)
            return self._engine

    def _apply_pose_calibration(self, engine):
        """Fit the engine's pose-range clamp to this person's own registered poses, if any
        (see reenactment/engine.py: calibrate_pose_range). Call with self.engine_lock held."""
        poses = {} if self.face is None else self.face.poses
        yaw = [abs(poses[side]["angle_deg"]) for side in ("left", "right") if side in poses]
        up = poses.get("up", {}).get("angle_deg")
        down = poses.get("down", {}).get("angle_deg")
        if yaw or up is not None or down is not None:
            engine.calibrate_pose_range(yaw_extreme_deg=max(yaw) if yaw else None,
                                        pitch_up_extreme_deg=up, pitch_down_extreme_deg=down)

    def scorer(self):
        """The identity scorer, loaded on first use. Raises if its model is not installed."""
        with self._scorer_lock:
            if self._scorer is None:
                self._scorer = self._scorer_factory()
            return self._scorer

    def load_enrolment(self):
        """Give the engine the meeting picture. Called when a camera session starts."""
        with self.engine_lock:
            enrolment = self.enrolment
            if enrolment is None:
                raise SessionError("Enrol a picture first.")
            engine = self.engine()
            if self.engine_holds != ("enrolment", enrolment.id):
                engine.set_source(enrolment.picture)
                self.engine_holds = ("enrolment", enrolment.id)
            # The neutral pose is not set here. It is the person's own resting face at the camera,
            # which the session takes from the first calm frame (see pipeline.py).

    def use_sample_source(self, frame, neutral_face):
        """A sample clip animates one of its own frames. It is never stored as an enrolment."""
        with self.engine_lock:
            engine = self.engine()
            engine.set_source(frame)
            engine.set_reference(neutral_face)
            self._samples_used += 1
            self.engine_holds = ("sample", self._samples_used)

    def source_signature(self, scorer):
        """The face signature of the picture the engine holds, for the identity check."""
        enrolment = self.enrolment
        if enrolment is not None and self.engine_holds == ("enrolment", enrolment.id) and enrolment.signature is not None:
            return enrolment.signature
        engine = self._engine
        if engine is None or engine.source_frame is None:
            return None
        return scorer.embed(engine.source_frame)

    def face_signatures(self):
        """Every signature registered for the user: the verified face, and each pose registered after it.

        A face is compared with all of them and the best match decides: a face turned to the side
        looks more like the pose registered at that angle than like the capture facing the camera.
        """
        face = self.face
        return [] if face is None else [face.signature] + [pose["signature"] for pose in face.poses.values()]

    def consent_check(self):
        """The check a camera session starts with (consent/check.py), and the scorer it matches faces with."""
        signatures = self.face_signatures()
        if not signatures:
            raise SessionError("Verify your face with the camera first. A session starts only for the verified face.")
        return ConsentCheck(LivenessChallenge(), signatures, self._face_scorer().similarity), self._face_scorer()

    def verification_check(self):
        """The check a face is verified with: the same prompt, and the face has to stay the same face
        from its first frame to the picture that is taken."""
        return ConsentCheck(LivenessChallenge(), None, self._face_scorer().similarity), self._face_scorer()

    def _face_scorer(self):
        try:
            return self.scorer()
        except FileNotFoundError:
            raise SessionError("The identity model is not installed, so it cannot be checked that it is you. "
                               "Run scripts/download_models.py.") from None

    def reset_neutral(self):
        self.session.call(lambda frame, track: self.engine().reset_reference())
        self.events.add("info", "Neutral pose reset.")

    def resume_reenactment(self):
        """Go live again after the identity monitor put the still picture up."""
        self.identity.resume()

    # ---------------------------------------------------------- enrolment

    def start_preview(self, input_id):
        if self.session.active:
            raise SessionError("Stop the live session first. The camera is in use.")
        entry = self._input(input_id)
        if entry["kind"] != "camera":
            raise SessionError("Your face can only be verified with a camera.")
        self.preview.start(self._source_of(entry), entry["label"])

    def start_pose_preview(self, pose, input_id):
        """Open the camera to register one pose ("left", "right", "up" or "down"): turned to
        the side or tilted up/down, used both for a richer identity signature and to calibrate
        how far this person's own head actually moves (see enrolment/checks.py and
        reenactment/engine.py: calibrate_pose_range)."""
        if pose not in POSE_KEYS:
            raise SessionError(f"Unknown pose: {pose}")
        if self.face is None:
            raise SessionError("Verify your face with the camera first.")
        if self.session.active:
            raise SessionError("Stop the live session first. The camera is in use.")
        entry = self._input(input_id)
        if entry["kind"] != "camera":
            raise SessionError("A pose is registered with a camera.")
        self.preview.start(self._source_of(entry), entry["label"], pose=pose)

    def stop_preview(self):
        self.preview.stop()

    def _hold(self, candidate):
        self._candidate_id += 1
        self.candidate = candidate
        return self.candidate_summary()

    def take_picture(self):
        """Take a picture from the open camera. Raises NotGoodEnough if no frame passes the checks."""
        candidate = self.preview.take()
        if candidate.origin.startswith("pose:"):
            # Every signature kept for the user is one a session can later be started with, so a pose
            # has to be of the verified face. On the one face registered so far, poses turned 40 degrees
            # matched the capture facing the camera at 0.67 and 0.79, against a limit of 0.35.
            candidate.checks.append(self._same_person(candidate, self.face, "Only you can register a pose."))
        return self._hold(candidate)

    def check_upload(self, image):
        """Check an uploaded meeting picture: the usual checks, and that it shows the verified face.

        It becomes the candidate whether or not it passes, so the result can be shown.
        """
        face = self.face
        if face is None:
            raise SessionError("Verify your face with the camera first. An uploaded picture is compared with it.")
        missing = [pose for pose in POSE_KEYS if pose not in face.poses]
        if missing:
            raise SessionError("Register these poses first, facing the camera, turned to each side, and tilted "
                               "up and down: " + ", ".join(missing) + ".")
        tracker = self.make_tracker(video=False)
        try:
            candidate = self._upload_candidate(image, tracker)
        finally:
            tracker.close()
        identity = self._same_person(candidate, face)
        candidate.checks.append(identity)
        if identity.passed:
            self._accept_small_face_once_identity_is_confirmed(candidate)
        return self._hold(candidate)

    @staticmethod
    def _accept_small_face_once_identity_is_confirmed(candidate):
        """Once the picture is confirmed to be the enrolled person, a small face no longer
        refuses it: identity is who it is of, size is only how sharp the animation looks, and
        those are different questions. The size check becomes a warning instead of a block,
        carried in `tip` the same way an already-passing check suggests a better picture
        elsewhere. It is left blocking when anything else also fails: a small face on a photo
        that is also blurred or badly lit is still worth refusing, on that other ground.
        """
        checks = candidate.checks
        size = next((c for c in checks if c.key == "size"), None)
        if size is None or size.passed:
            return
        if any(not c.passed for c in checks if c.key not in ("size", "identity")):
            return
        note = (f"Your face is about {round(size.value)} px tall here, below the {MIN_FACE_HEIGHT_PX} px that "
                "gives the sharpest result. Since this is confirmed to be you, the picture is accepted anyway; "
                "a closer or higher-resolution photo would animate more sharply.")
        checks[checks.index(size)] = replace(size, passed=True, hint="", tip=note)

    def _upload_candidate(self, image, tracker):
        """Judge the picture as given; if only its framing is at fault, try again cropped
        tighter around the face before giving up on it.

        A casual workplace photo (a desk, a wall, a person sitting back from the
        camera) often has a small face only because of how it is framed, not
        because the file lacks detail: the original upload, before it is brought
        down to the stored size, usually has far more resolution on the face than
        a crop of the whole scene would suggest. Cropping first and bringing down
        only that crop keeps that detail; downscaling the whole photo first, as
        the straightforward path does, would have thrown most of it away on
        background the output never uses. If the file truly does not have the
        detail (a small or heavily compressed upload), the crop does not invent
        any, and the picture is still refused, now for a reason that is actually
        about the photo rather than a camera instruction that does not apply.
        """
        picture = prepare(image)
        track = tracker.process(picture)
        candidate = make_candidate(picture, track, "upload")
        if not track.ok:
            return candidate
        too_small = any(check.key == "size" and not check.passed for check in candidate.checks)
        if not too_small:
            return candidate
        reframed = self._reframe_on_face(image, picture.shape[:2], track.bbox)
        if reframed is None:
            return candidate
        retrack = tracker.process(reframed)
        if not retrack.ok:
            return candidate
        tighter = make_candidate(reframed, retrack, "upload")
        if self._face_px(tighter) >= self._face_px(candidate):
            return tighter  # at least as much real detail on the face, now better framed
        return candidate  # the crop made it worse (rare); keep the original's honest report

    @staticmethod
    def _face_px(candidate):
        check = next((c for c in candidate.checks if c.key == "size"), None)
        return check.value if check is not None and check.value is not None else 0.0

    @staticmethod
    def _reframe_on_face(image, picture_shape, bbox):
        """Crop `image` (the upload as given, before it was brought down to the stored size)
        tightly around bbox (found in the already-downscaled `picture`), so the face fills
        about the same share of the frame the live camera outline asks for. Returns None if
        the picture is already at or below the stored size, since there is then nothing a
        tighter crop could recover that downscaling had not already kept.
        """
        ph, pw = picture_shape
        ih, iw = image.shape[:2]
        if max(ih, iw) <= max(ph, pw):
            return None
        scale = iw / pw  # == ih / ph: prepare() keeps the aspect ratio
        x, y, w, h = (value * scale for value in bbox)
        side = max(w, h) / TARGET_FACE_HEIGHT_SHARE
        crop = square_face_crop(image, (x, y, w, h), scale=side / max(w, h), out_size=round(side))
        return limit_size(crop, MAX_PICTURE_DIM)

    def _same_person(self, candidate, face, remedy="Upload a picture of yourself."):
        """The check an uploaded picture and a pose get: is this the verified face?

        Compared against every signature registered for this face, not only the front-on one: a
        meeting picture taken at a slight angle, or a live face turned to the side, can look more
        like one of the registered poses than like the straight-on capture. The best match of the
        five (or just the one, before any pose is registered) decides it.
        """
        label = "Same person as your face"
        if candidate.landmarks is None:
            return Check("identity", label, False, "")  # no single face to compare
        try:
            scorer = self.scorer()
        except FileNotFoundError:
            raise SessionError("The identity model is not installed, so the picture cannot be compared with your face. "
                               "Run scripts/download_models.py.") from None
        candidate.signature = scorer.embed(candidate.image, candidate.landmarks)
        alike = max(scorer.similarity(reference, candidate.signature) for reference in self.face_signatures())
        hint = "" if alike >= SAME_PERSON_CSIM else f"This is not the face you verified with the camera. {remedy}"
        return Check("identity", label, not hint, hint, round(alike, 3))

    def discard_candidate(self):
        self.candidate = None

    def candidate_summary(self):
        candidate = self.candidate
        return None if candidate is None else {"id": self._candidate_id, **candidate.summary()}

    def confirm_candidate(self):
        """Keep the candidate.

        A picture taken with the camera becomes the user's face: its signature is
        stored and the picture itself is dropped. An uploaded picture becomes the
        meeting picture. Raises SessionError if it cannot be kept, and ValueError
        if the animation model cannot use an uploaded picture.
        """
        candidate = self.candidate
        if candidate is None:
            raise SessionError("There is no picture to keep. Take or upload one first.")
        if not candidate.passed:
            raise SessionError("This picture did not pass every check, so it cannot be used.")
        if candidate.origin == "camera":
            self._verify_face(candidate)
        elif candidate.origin.startswith("pose:"):
            self._register_pose(candidate)
        else:
            self._enrol_picture(candidate)
        self.candidate = None
        self.preview.stop()
        return {"face": None if self.face is None else self.face.summary(),
                "record": None if self.enrolment is None else self.enrolment.summary()}

    def _verify_face(self, candidate):
        """The camera picture says who the user is. Its signature is kept; the picture is not."""
        if self.session.active:
            raise SessionError("Stop the live session first.")
        signature = self._signature(candidate)
        if signature is None:
            raise SessionError("Your face could not be verified, because its signature could not be made. "
                               "Check that the identity model is installed: python scripts/download_models.py.")
        self.face = self.store.save_identity(signature, candidate.summary()["checks"])
        self.events.add("info", "Face verified from the camera. Only its signature was kept.")
        # A meeting picture enrolled earlier has to be of this face too.
        enrolment = self.enrolment
        if enrolment is not None and enrolment.signature is not None:
            if self.scorer().similarity(signature, enrolment.signature) < SAME_PERSON_CSIM:
                self._drop_picture()
                self.events.add("warning", "The meeting picture does not match the newly verified face and was removed.")

    def _register_pose(self, candidate):
        """A pose capture (turned to one side, or tilted up/down) that passed its checks:
        add it to the verified face's registered poses."""
        pose = candidate.origin.split(":", 1)[1]
        signature = self._signature(candidate)
        if signature is None:
            raise SessionError("This pose could not be registered, because its signature could not be made. "
                               "Check that the identity model is installed: python scripts/download_models.py.")
        angle = next(check.value for check in candidate.checks if check.key == "pose")
        self.store.save_pose(pose, signature, angle, candidate.summary()["checks"])
        self.face = self.store.load_identity()
        with self.engine_lock:
            if self._engine is not None:
                self._apply_pose_calibration(self._engine)
        self.events.add("info", f"{pose.capitalize()} pose registered.")

    def _enrol_picture(self, candidate):
        """An uploaded picture that passed every check, the face match among them, becomes the meeting picture."""
        if self.face is None:
            raise SessionError("Verify your face with the camera first.")
        if self.session.active and not self.session.uses_enrolment:
            # A sample clip is using the engine for its own picture; do not swap it mid-session.
            raise SessionError("Stop the sample session first.")
        if self.session.active:
            # The session thread owns the engine while it runs; swap the picture between two frames.
            self.session.call(lambda frame, track: self._enrol(candidate), timeout=30.0)
        else:
            self._enrol(candidate)
        self.events.add("info", "Meeting picture enrolled. Its face matches the verified face.")

    def _enrol(self, candidate):
        with self.engine_lock:
            engine = self.engine()
            engine.set_source(candidate.image)  # raises SourceError if the model finds no face
            match = next((check.value for check in candidate.checks if check.key == "identity"), None)
            self.enrolment = self.store.save(engine.source_frame, candidate.neutral_face, candidate.signature,
                                             candidate.origin, candidate.summary()["checks"], match)
            self.engine_holds = ("enrolment", self.enrolment.id)

    def _signature(self, candidate):
        """The face signature of a picture, or None if the identity model is not installed or fails."""
        try:
            return self.scorer().embed(candidate.image, candidate.landmarks)
        except FileNotFoundError:
            self.events.add("warning", "The identity model is not installed, so no face signature could be made. "
                                       "Run scripts/download_models.py.")
        except Exception:
            log.exception("The face signature could not be computed.")
            self.events.add("warning", "The face signature could not be computed.")
        return None

    def remove_enrolment(self):
        """Remove the meeting picture. The verified face stays."""
        if self.session.active:
            raise SessionError("Stop the live session first.")
        self._drop_picture()
        self.events.add("info", "Meeting picture removed.")

    def forget_face(self):
        """Remove everything kept about the user: the face signature and the meeting picture."""
        if self.session.active:
            raise SessionError("Stop the live session first.")
        self._drop_picture()
        self.store.remove_identity()
        self.face = None
        self.candidate = None
        self.events.add("info", "Face signature and meeting picture removed.")

    def _drop_picture(self):
        with self.engine_lock:
            self.store.remove()
            self.enrolment = None
            if self._engine is not None and self.engine_holds is not None and self.engine_holds[0] == "enrolment":
                self._engine.clear_source()
                self.engine_holds = None

    def enrolment_jpeg(self):
        enrolment = self.enrolment
        return None if enrolment is None else _jpeg(enrolment.picture)

    def candidate_jpeg(self):
        candidate = self.candidate
        return None if candidate is None else _jpeg(candidate.image)

    # ------------------------------------------------------------ session

    def inputs(self):
        """What a session can be driven from: cameras, and sample clips if installed."""
        inputs = [{"id": f"camera:{index}", "kind": "camera", "label": f"Camera {index}"}
                  for index in CAMERA_INDEXES]
        for name in SAMPLE_NAMES:
            if (self._sample_dir / f"{name}.mp4").exists():
                inputs.append({"id": f"sample:{name}", "kind": "sample", "label": f"Sample clip {name}"})
        return inputs

    def _input(self, input_id):
        entry = next((item for item in self.inputs() if item["id"] == input_id), None)
        if entry is None:
            raise SessionError(f"Unknown input: {input_id}")
        return entry

    def _source_of(self, entry):
        kind, _, value = entry["id"].partition(":")
        return int(value) if kind == "camera" else str(self._sample_dir / f"{value}.mp4")

    def start_session(self, input_id):
        entry = self._input(input_id)
        uses_enrolment = entry["kind"] == "camera"
        if uses_enrolment and self.enrolment is None:
            raise SessionError("Enrol a picture first.")
        if uses_enrolment and self.face is None:
            raise SessionError("Verify your face with the camera first. A session starts only for the verified face.")
        if self.session.active:
            raise SessionError("A session is already running.")
        self.preview.stop()  # the camera can only be open in one place
        self.session.start(self._source_of(entry), entry["label"], uses_enrolment)
        self.identity.ensure_started()

    def stop_session(self):
        self.session.stop()

    def latest_pair(self):
        """The newest frames to show: from the live session, or else from the enrolment preview."""
        return self.session.latest_pair() or self.preview.latest_pair()

    def shutdown(self):
        self.session.stop()
        self.preview.stop()
        self.identity.stop()

    # ------------------------------------------------------------- status

    def gpu(self):
        """GPU name and memory, once the engine has loaded. None before that.

        PyTorch must not be touched earlier: while the engine is loading on the
        session thread, the module is only half imported.
        """
        if self._engine is None:
            return None
        info = self._gpu_probe()
        # PyTorch's count leaves out TensorRT's memory, so the engine's own reading from the driver replaces it.
        measure = getattr(self._engine, "gpu_memory_gb", None)
        if info is not None and measure is not None:
            used = round(measure(), 2)
            self._gpu_peak = max(self._gpu_peak, used)
            info = {**info, "allocated_gb": used, "peak_gb": self._gpu_peak}
        return info

    def status(self):
        enrolment = self.enrolment
        return {
            "session": self.session.snapshot(),
            "enrolment": {
                "face": None if self.face is None else self.face.summary(),
                "record": None if enrolment is None else enrolment.summary(),
                "candidate": self.candidate_summary(),
                "preview": self.preview.snapshot(),
            },
            "identity": self.identity.snapshot(),
            "features": self.features.as_list(),
            "gpu": self.gpu(),
            "targets": TARGETS,
            "inputs": self.inputs(),
        }

    def latest_benchmark(self):
        return latest_benchmark(self._results_dir)

    def enrolment_guide(self):
        """The limits the enrolment checks apply, and the study they were set from if it has been run."""
        return enrolment_guide(self._results_dir)
