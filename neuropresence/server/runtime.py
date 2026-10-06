"""What the server keeps alive between requests: the engine, the enrolment, the session, the switches."""

import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from ..enrolment import EnrolmentStore, make_candidate, prepare
from ..targets import TARGETS
from .benchmarks import latest_benchmark
from .features import FeatureSet
from .guide import enrolment_guide
from .preview import EnrolmentPreview
from .session import LiveSession, SessionError

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
    """Scores the live output against the picture it is made from, about once a second.

    It runs on its own thread so the pipeline never waits for it. This is the
    measuring half of the identity stage; reacting to a low score is not built.
    """

    def __init__(self, runtime, interval=1.0):
        self._rt = runtime
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = {"available": False, "csim": None, "reason": "Starts with the first session."}

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
            return dict(self._state)

    def _set(self, **state):
        with self._lock:
            self._state = state

    def _run(self):
        try:
            scorer = self._rt.scorer()
        except Exception as err:  # most often the ArcFace model has not been downloaded
            self._set(available=False, csim=None, reason=str(err))
            return
        held, source_embedding = None, None
        while not self._stop.wait(self._interval):
            pair = self._rt.session.latest_pair()
            if pair is None or not pair.live:
                self._set(available=True, csim=None, reason=None)
                continue
            try:
                if self._rt.engine_holds != held:
                    held = self._rt.engine_holds
                    source_embedding = self._rt.source_signature(scorer)
                embedding = None if source_embedding is None else scorer.embed(pair.output)
                csim = None if embedding is None else round(scorer.similarity(source_embedding, embedding), 3)
                self._set(available=True, csim=csim, reason=None)
            except Exception as err:
                self._set(available=False, csim=None, reason=f"Identity scoring failed: {err}")
                return


def _default_engine():
    from ..reenactment import ReenactmentEngine

    return ReenactmentEngine()


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

        self.store = EnrolmentStore(Path(data_dir) / "enrolment")
        self.enrolment = self.store.load()
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
            return self._engine

    def scorer(self):
        """The identity scorer, loaded on first use. Raises if its model is not installed."""
        with self._scorer_lock:
            if self._scorer is None:
                self._scorer = self._scorer_factory()
            return self._scorer

    def load_enrolment(self):
        """Give the engine the enrolled picture and its neutral pose. Called when a camera session starts."""
        with self.engine_lock:
            enrolment = self.enrolment
            if enrolment is None:
                raise SessionError("Enrol a picture first.")
            engine = self.engine()
            if self.engine_holds != ("enrolment", enrolment.id):
                engine.set_source(enrolment.picture)
                self.engine_holds = ("enrolment", enrolment.id)
            # The enrolled picture's own face is the neutral pose: when the person sits
            # as they did in the picture, the output is the picture.
            engine.set_reference(enrolment.neutral_face)

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

    def reset_neutral(self):
        self.session.call(lambda frame, track: self.engine().reset_reference())
        self.events.add("info", "Neutral pose reset.")

    # ---------------------------------------------------------- enrolment

    def start_preview(self, input_id):
        if self.session.active:
            raise SessionError("Stop the live session first. The camera is in use.")
        entry = self._input(input_id)
        if entry["kind"] != "camera":
            raise SessionError("A picture can only be enrolled from a camera or an uploaded file.")
        self.preview.start(self._source_of(entry), entry["label"])

    def stop_preview(self):
        self.preview.stop()

    def _hold(self, candidate):
        self._candidate_id += 1
        self.candidate = candidate
        return self.candidate_summary()

    def take_picture(self):
        """Take a picture from the open camera. Raises NotGoodEnough if no frame passes the checks."""
        return self._hold(self.preview.take())

    def check_upload(self, image):
        """Check an uploaded picture. It becomes the candidate whether or not it passes, so it can be shown."""
        picture = prepare(image)
        tracker = self.make_tracker(video=False)
        try:
            track = tracker.process(picture)
        finally:
            tracker.close()
        return self._hold(make_candidate(picture, track, "upload"))

    def discard_candidate(self):
        self.candidate = None

    def candidate_summary(self):
        candidate = self.candidate
        return None if candidate is None else {"id": self._candidate_id, **candidate.summary()}

    def confirm_candidate(self):
        """Enrol the candidate.

        Raises SessionError if there is none or it failed a check, and
        ValueError if the animation model cannot use the picture.
        """
        candidate = self.candidate
        if candidate is None:
            raise SessionError("There is no picture to enrol. Take or upload one first.")
        if not candidate.passed:
            raise SessionError("This picture did not pass every check, so it cannot be enrolled.")
        if self.session.active and not self.session.uses_enrolment:
            # A sample clip is using the engine for its own picture; do not swap it mid-session.
            raise SessionError("Stop the sample session first.")
        if self.session.active:
            # The session thread owns the engine while it runs; swap the picture between two frames.
            self.session.call(lambda frame, track: self._enrol(candidate), timeout=30.0)
        else:
            self._enrol(candidate)
        self.candidate = None
        self.preview.stop()
        origin = "the camera" if candidate.origin == "camera" else "an uploaded file"
        self.events.add("info", f"Picture enrolled from {origin}.")
        return self.enrolment.summary()

    def _enrol(self, candidate):
        with self.engine_lock:
            engine = self.engine()
            engine.set_source(candidate.image)  # raises SourceError if the model finds no face
            engine.set_reference(candidate.neutral_face)
            self.enrolment = self.store.save(engine.source_frame, candidate.neutral_face, self._signature(candidate),
                                             candidate.origin, candidate.summary()["checks"])
            self.engine_holds = ("enrolment", self.enrolment.id)

    def _signature(self, candidate):
        """The picture's face signature, or None if the identity model is not installed."""
        try:
            return self.scorer().embed(candidate.image, candidate.landmarks)
        except FileNotFoundError:
            self.events.add("warning", "The identity model is not installed, so no face signature was saved. "
                                       "Run scripts/download_models.py and enrol again.")
        except Exception:
            log.exception("The face signature could not be computed.")
            self.events.add("warning", "The face signature could not be computed, so none was saved.")
        return None

    def remove_enrolment(self):
        if self.session.active:
            raise SessionError("Stop the live session first.")
        with self.engine_lock:
            self.store.remove()
            self.enrolment = None
            if self._engine is not None and self.engine_holds is not None and self.engine_holds[0] == "enrolment":
                self._engine.clear_source()
                self.engine_holds = None
        self.events.add("info", "Enrolled picture removed.")

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
        return self._gpu_probe()

    def status(self):
        enrolment = self.enrolment
        return {
            "session": self.session.snapshot(),
            "enrolment": {
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
