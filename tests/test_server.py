"""The web server, run with stand-ins for the camera, the tracker and the engine."""

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuropresence.capture import TrackResult, TrackStatus
from neuropresence.capture.feed import FeedFrame
from neuropresence.enrolment import checks as limits
from neuropresence.enrolment import POSE_KEYS, make_candidate, prepare
from neuropresence.identity import SAME_PERSON_CSIM
from neuropresence.server import create_app
from neuropresence.server.metrics import MetricsWindow
from neuropresence.server.runtime import Runtime
from neuropresence.server.session import FramePair, SessionError
from neuropresence.server.stream import pack_frame, unpack_frame

FACE_BOX = (220, 75, 200, 215)  # x, y, w, h: one well-placed face in a 640 x 360 picture


def well_lit_picture(seed=0):
    """A stand-in for a good camera frame: broad light and shade with fine detail on top."""
    ys, xs = np.mgrid[0:360, 0:640]
    grey = 128 + 45 * np.sin(xs / 10.0) * np.cos(ys / 12.0) + np.random.default_rng(seed).integers(-20, 21, (360, 640))
    return np.repeat(np.clip(grey, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


FRAME = well_lit_picture()
FLAT = np.full((360, 640, 3), 120, dtype=np.uint8)  # no detail at all: fails the light and sharpness checks


class FakeFeed:
    """Hands out a new frame every few milliseconds."""

    is_camera = True

    def __init__(self):
        self.index = -1
        self.ended = False
        self.closed = False

    def read(self, after_index=-1, timeout=1.0):
        time.sleep(0.005)
        self.index += 2  # every other frame is "dropped"
        return FeedFrame(FRAME.copy(), self.index, time.perf_counter())

    def close(self):
        self.closed = True


class FakeTracker:
    """Always finds the same well-placed face, unless told otherwise."""

    status = TrackStatus.OK
    yaw = 0.0
    pitch = 0.0
    blink = 0.1

    def process(self, frame):
        if self.status is not TrackStatus.OK:
            return TrackResult(self.status, 1.0)
        x, y, w, h = FACE_BOX
        rng = np.random.default_rng(1)
        points = np.column_stack([rng.uniform(x, x + w, 478), rng.uniform(y, y + h, 478)]).astype(np.float32)
        points[:4] = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        middle = x + w / 2
        points[limits.UPPER_LIP], points[limits.LOWER_LIP] = (middle, y + 120), (middle, y + 122)
        points[limits.MOUTH_LEFT], points[limits.MOUTH_RIGHT] = (middle - 30, y + 121), (middle + 30, y + 121)
        return TrackResult(self.status, 1.0, landmarks=points, bbox=FACE_BOX, pose_deg=(self.yaw, self.pitch, 0.0),
                           blendshapes={"jawOpen": 0.25, "eyeBlinkLeft": self.blink, "eyeBlinkRight": self.blink})

    def close(self):
        pass


class FakeEngine:
    reject = False  # set to make the animation model refuse the next picture

    def __init__(self):
        self.source_frame = None
        self.source_crop = None
        self.last_timing_ms = {"motion": 2.0, "render": 3.0, "compose": 1.0}
        self.resets = 0
        self.neutral_faces = []
        self.pose_calibration = None  # the kwargs calibrate_pose_range was last called with

    def calibrate_pose_range(self, **kwargs):
        self.pose_calibration = kwargs

    def set_source(self, image):
        if self.reject:
            raise ValueError("No face found in the source image.")
        self.source_frame = image.copy()
        self.source_crop = image[:64, :64].copy()

    def clear_source(self):
        self.source_frame = self.source_crop = None

    def drive(self, face, at=None, steady=False, limit=False, motion=None):
        return np.full_like(self.source_frame, 200)

    def reset_reference(self):
        self.resets += 1

    def set_reference(self, face):
        self.neutral_faces.append(face.shape)


class FakeScorer:
    """Gives every face the same signature. `alike` is how alike it says any two faces are."""

    alike = 0.9

    def embed(self, image, landmarks=None):
        return np.array([1.0, 0.0])

    def similarity(self, a, b):
        return self.alike


def make_runtime(tmp_path, parts):
    def make_feed(source):
        parts["sources"].append(source)
        parts["feeds"].append(FakeFeed())
        return parts["feeds"][-1]

    def make_engine():
        parts["engines"].append(FakeEngine())
        return parts["engines"][-1]

    def read_gpu():
        parts["gpu_reads"].append(len(parts["engines"]))
        return {"name": "Test GPU", "total_gb": 8.0, "allocated_gb": 0.5, "peak_gb": 1.0}

    samples = tmp_path / "samples"
    samples.mkdir(exist_ok=True)
    (samples / "d0.mp4").touch()  # the fake feed never opens it; it only has to be listed
    return Runtime(engine_factory=make_engine, feed_factory=make_feed,
                   tracker_factory=lambda video=True: parts["tracker"], scorer_factory=FakeScorer,
                   gpu_probe=read_gpu, data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                   sample_dir=samples)


@pytest.fixture
def parts(tmp_path):
    parts = {"feeds": [], "sources": [], "engines": [], "gpu_reads": [], "tracker": FakeTracker(), "tmp": tmp_path}
    parts["runtime"] = make_runtime(tmp_path, parts)
    with TestClient(create_app(parts["runtime"], web_dist=tmp_path / "no-web-build")) as client:
        parts["client"] = client
        yield parts


@pytest.fixture(autouse=True)
def neutral_pose_at_once(monkeypatch):
    """The stand-in face talks all the time, so do not wait to see it at rest: take the first frame."""
    monkeypatch.setattr("neuropresence.pipeline.NEUTRAL_WAIT_SECONDS", 0.0)


def wait_for(condition, seconds=3.0):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def status(parts):
    return parts["client"].get("/api/status").json()


def png(image):
    return cv2.imencode(".png", image)[1].tobytes()


def upload(parts, image=FRAME):
    return parts["client"].post("/api/enrolment/upload", files={"file": ("me.png", png(image), "image/png")})


def verify_face(parts, camera="camera:0"):
    """Verify the face with the camera: open it, take a picture, keep it. Only its signature stays."""
    client = parts["client"]
    assert client.post("/api/enrolment/preview/start", json={"input": camera}).status_code == 200
    assert wait_for(lambda: status(parts)["enrolment"]["preview"]["ready"])
    assert client.post("/api/enrolment/take").json()["passed"]
    reply = client.post("/api/enrolment/confirm")
    assert reply.status_code == 200, reply.text
    return reply.json()["face"]


POSE_ANGLES = {"front": (0.0, 0.0), "left": (-60.0, 0.0), "right": (60.0, 0.0), "up": (0.0, -25.0), "down": (0.0, 25.0)}


def register_all_poses(parts, camera="camera:0"):
    """Register every pose (front, left, right, up, down) for the verified face: the real flow
    requires all five before a meeting picture can be uploaded (see runtime.py: check_upload)."""
    client, tracker = parts["client"], parts["tracker"]
    for pose, (yaw, pitch) in POSE_ANGLES.items():
        tracker.yaw, tracker.pitch = yaw, pitch
        assert client.post(f"/api/enrolment/pose/{pose}/start", json={"input": camera}).status_code == 200
        assert wait_for(lambda: status(parts)["enrolment"]["preview"]["ready"])
        assert client.post("/api/enrolment/take").json()["passed"]
        reply = client.post("/api/enrolment/confirm")
        assert reply.status_code == 200, reply.text
    tracker.yaw, tracker.pitch = 0.0, 0.0


def enrol(parts, image=FRAME):
    """Verify the face and register every pose if that has not been done, then upload a meeting
    picture and keep it."""
    if status(parts)["enrolment"]["face"] is None:
        verify_face(parts)
        register_all_poses(parts)
        parts["feeds"].clear()  # the camera opened for that is not what the tests go on to look at
        parts["sources"].clear()
    assert upload(parts, image).json()["passed"]
    reply = parts["client"].post("/api/enrolment/confirm")
    assert reply.status_code == 200, reply.text
    return reply.json()["record"]


def start(parts, input_id="camera:0"):
    """Start a session and wait until frames flow. A camera session needs an enrolled picture."""
    if input_id.startswith("camera") and status(parts)["enrolment"]["record"] is None:
        enrol(parts)
    reply = parts["client"].post("/api/session/start", json={"input": input_id})
    assert reply.status_code == 200, reply.text
    assert wait_for(lambda: status(parts)["session"]["state"] == "running")
    assert wait_for(lambda: parts["runtime"].session.latest_pair() is not None)


def events(parts):
    return [event["message"] for event in parts["runtime"].events.since(0)]


# ------------------------------------------------------------------ status


def test_status_when_nothing_has_happened(parts):
    found = status(parts)
    assert found["session"]["state"] == "idle" and found["session"]["metrics"] is None
    assert found["enrolment"] == {"face": None, "record": None, "candidate": None,
                                  "preview": {"state": "idle", "message": "", "input": None, "pose": None, "checks": None,
                                              "hint": "", "tip": "", "ready": False, "outline": None,
                                              "taking": False}}
    assert {f["key"]: f["enabled"] for f in found["features"]} == {"reenactment": True, "steady_crop": True,
                                                                "steady_keypoints": True, "natural_range": True,
                                                                "tracking_overlay": False}
    assert [i["id"] for i in found["inputs"]] == ["camera:0", "camera:1", "camera:2", "sample:d0"]
    assert found["targets"]["fps"] == 24.0
    assert parts["engines"] == []  # the models are not loaded until they are needed
    # The GPU is not asked about before the engine exists: PyTorch may still be loading.
    assert found["gpu"] is None and parts["gpu_reads"] == []


def test_capabilities_list_the_six_stages(parts):
    reply = parts["client"].get("/api/capabilities").json()
    stages = reply["stages"]
    assert [s["key"] for s in stages] == ["capture", "motion", "reenactment", "identity", "consent",
                                          "virtual_camera"]
    assert {s["status"] for s in stages} <= {"working", "partial", "planned"}
    assert reply["enrolment"]["key"] == "enrolment" and reply["enrolment"]["status"] == "working"


def test_the_enrolment_guide_gives_the_limits_and_the_study_behind_them(parts):
    client = parts["client"]
    guide = client.get("/api/enrolment/guide").json()
    assert guide["study"] is None  # the study has not been run in this results folder
    assert guide["limits"]["min_face_height_px"] == limits.MIN_FACE_HEIGHT_PX
    assert guide["limits"]["good_face_height_px"] == limits.GOOD_FACE_HEIGHT_PX
    assert guide["limits"]["min_sharpness"] == limits.MIN_SHARPNESS

    results = parts["tmp"] / "results"
    results.mkdir(exist_ok=True)
    (results / "enrolment_study.json").write_text(json.dumps({
        "face_size_reference_px": 260,
        "source_faults_summary": {
            "neutral": {"clips": 6, "csim_to_real": 0.85, "csim_to_real_change": 0.0, "detail_kept": 0.41},
            "dark": {"clips": 6, "csim_to_real": 0.81, "csim_to_real_change": -0.04, "detail_kept": 0.14},
            "blurred": {"clips": 6, "csim_to_real": 0.78, "csim_to_real_change": -0.07, "detail_kept": 0.12},
        },
        "face_size_summary": {"source/s7.jpg": {"458": 1.7, "148": 0.33, "258": 1.0}},
    }))
    study = client.get("/api/enrolment/guide").json()["study"]
    assert study["clips"] == 6 and study["good_picture"] == {"csim": 0.85, "detail_kept": 0.41}
    assert [fault["name"] for fault in study["faults"]] == ["blurred", "dark"]  # the costliest first
    assert study["faults"][0] == {"name": "blurred", "clips": 6, "csim_change": -0.07, "detail_kept": 0.12}
    assert study["face_size"] == {"reference_px": 260, "pictures": [{"name": "s7", "points": [
        {"face_height_px": 148, "detail": 0.33}, {"face_height_px": 258, "detail": 1.0},
        {"face_height_px": 458, "detail": 1.7}]}]}

    (results / "enrolment_study.json").write_text("not json")
    assert client.get("/api/enrolment/guide").json()["study"] is None  # unreadable is the same as not run


def test_latest_benchmark(parts):
    assert parts["client"].get("/api/benchmarks/latest").status_code == 404
    results = parts["tmp"] / "results"
    results.mkdir()
    (results / "benchmark_old.json").write_text(json.dumps({"date": "2026-10-01T10:00:00", "summary": {"fps": 5}}))
    (results / "benchmark_new.json").write_text(json.dumps({"date": "2026-10-06T10:00:00", "summary": {"fps": 7.5, "pairs": 8}}))
    (results / "benchmark_broken.json").write_text("{not json")
    latest = parts["client"].get("/api/benchmarks/latest").json()
    assert latest["file"] == "benchmark_new.json"
    assert latest["summary"]["fps"] == 7.5
    assert latest["clips"] == 8


# --------------------------------------------------------------- enrolment


def test_the_camera_picture_becomes_the_face_signature_and_nothing_else(parts):
    client = parts["client"]
    early = upload(parts)  # there is no verified face to compare an upload with yet
    assert early.status_code == 409 and early.json()["detail"].startswith("Verify your face with the camera first.")
    face = verify_face(parts)
    found = status(parts)["enrolment"]
    assert found["face"]["id"] == face["id"] and found["record"] is None and found["candidate"] is None
    assert [check["key"] for check in face["checks"]] == ["face", "facing", "size", "framing", "light", "sharp", "expression"]
    stored = parts["tmp"] / "data" / "enrolment"
    assert sorted(p.name for p in stored.iterdir()) == ["identity.json", "identity.npy"]  # no picture of the face is kept
    assert parts["engines"] == []  # verifying a face does not need the animation model
    assert client.get("/api/enrolment/picture").status_code == 404
    assert client.post("/api/session/start", json={"input": "camera:0"}).status_code == 409  # a face is not a meeting picture
    assert "Face verified from the camera. Only its signature was kept." in events(parts)


def test_an_uploaded_picture_of_the_same_person_becomes_the_meeting_picture(parts):
    client = parts["client"]
    verify_face(parts)
    register_all_poses(parts)
    candidate = upload(parts).json()
    assert candidate["passed"] and candidate["origin"] == "upload" and candidate["hint"] == ""
    assert (candidate["width"], candidate["height"]) == (640, 360)
    assert [check["key"] for check in candidate["checks"]] == ["face", "facing", "size", "framing", "light", "sharp",
                                                               "expression", "identity"]
    same = candidate["checks"][-1]
    assert same["label"] == "Same person as your face" and same["passed"] and same["value"] == 0.9
    assert status(parts)["enrolment"]["record"] is None  # checked, not yet enrolled
    shown = client.get("/api/enrolment/candidate/picture")
    assert cv2.imdecode(np.frombuffer(shown.content, np.uint8), cv2.IMREAD_COLOR).shape == FRAME.shape
    assert parts["engines"] == []  # checking a picture does not need the animation model

    record = client.post("/api/enrolment/confirm").json()["record"]
    assert record["origin"] == "upload" and record["has_signature"] is True and record["match"] == 0.9
    assert (record["width"], record["height"]) == (640, 360)
    found = status(parts)["enrolment"]
    assert found["record"]["id"] == record["id"] and found["candidate"] is None
    stored = parts["tmp"] / "data" / "enrolment"
    assert sorted(p.name for p in stored.iterdir()) == ["identity.json", "identity.npy", "neutral.png", "picture.png",
                                                        "pose_down.npy", "pose_front.npy", "pose_left.npy",
                                                        "pose_right.npy", "pose_up.npy", "poses.json", "record.json",
                                                        "signature.npy"]
    assert parts["engines"][0].source_frame.shape == FRAME.shape
    assert parts["engines"][0].neutral_faces == []  # the neutral pose comes from the camera, when a session starts
    assert client.get("/api/enrolment/picture").headers["content-type"] == "image/jpeg"
    assert client.get("/api/enrolment/candidate/picture").status_code == 404
    assert "Meeting picture enrolled. Its face matches the verified face." in events(parts)


def test_a_picture_of_someone_else_is_refused(parts, monkeypatch):
    client = parts["client"]
    verify_face(parts)
    register_all_poses(parts)
    monkeypatch.setattr(FakeScorer, "alike", 0.2)  # a different person: no two different people scored above 0.27
    candidate = upload(parts).json()
    assert candidate["passed"] is False
    assert [check["key"] for check in candidate["checks"] if not check["passed"]] == ["identity"]
    assert candidate["checks"][-1]["value"] == 0.2
    assert candidate["hint"] == "This is not the face you verified with the camera. Upload a picture of yourself."
    assert client.get("/api/enrolment/candidate/picture").status_code == 200  # shown, with the reason
    assert client.post("/api/enrolment/confirm").status_code == 409
    assert status(parts)["enrolment"]["record"] is None and parts["engines"] == []
    monkeypatch.setattr(FakeScorer, "alike", SAME_PERSON_CSIM)  # exactly at the limit counts as the same person
    assert upload(parts).json()["passed"]


def test_verifying_another_face_removes_a_meeting_picture_that_no_longer_matches(parts, monkeypatch):
    record = enrol(parts)
    verify_face(parts)  # the same person verifies again: the picture stays
    assert status(parts)["enrolment"]["record"]["id"] == record["id"]
    monkeypatch.setattr(FakeScorer, "alike", 0.1)  # then someone else sits down and verifies theirs
    verify_face(parts)
    assert status(parts)["enrolment"]["record"] is None
    assert parts["engines"][0].source_frame is None  # and the model no longer holds it
    assert "The meeting picture does not match the newly verified face and was removed." in events(parts)


def test_a_picture_that_fails_a_check_is_shown_but_cannot_be_enrolled(parts):
    client = parts["client"]
    verify_face(parts)
    register_all_poses(parts)
    candidate = upload(parts, FLAT).json()
    assert candidate["passed"] is False
    failed = {check["key"] for check in candidate["checks"] if not check["passed"]}
    assert failed == {"light", "sharp"}
    assert candidate["hint"] == "There is not enough light on the face. Choose a better-lit photo."
    assert client.get("/api/enrolment/candidate/picture").status_code == 200  # so the person can see why
    refused = client.post("/api/enrolment/confirm")
    assert refused.status_code == 409
    assert refused.json()["detail"] == "This picture did not pass every check, so it cannot be used."
    assert client.delete("/api/enrolment/candidate").json() == {"ok": True}
    assert status(parts)["enrolment"] == {**status(parts)["enrolment"], "record": None, "candidate": None}
    assert client.post("/api/enrolment/confirm").status_code == 409  # nothing left to confirm


def test_uploads_that_are_not_usable_pictures(parts):
    client = parts["client"]
    verify_face(parts)
    register_all_poses(parts)
    not_image = client.post("/api/enrolment/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert not_image.status_code == 422 and "not a picture" in not_image.json()["detail"]
    parts["tracker"].status = TrackStatus.NO_FACE
    no_face = upload(parts).json()
    assert no_face["passed"] is False
    # An uploaded picture cannot be told to sit in front of the camera: it asks for a different photo instead.
    assert no_face["hint"] == "No face was found. Choose a different photo."
    assert no_face["checks"][-1] == {"key": "identity", "label": "Same person as your face", "passed": False, "hint": "",
                                     "value": None, "tip": ""}  # with no face there is nothing to compare
    parts["tracker"].status = TrackStatus.MULTIPLE_FACES
    assert upload(parts).json()["hint"] == "More than one face was found. Choose a photo with only you in it."


def _face_track(bbox, frame_shape):
    """A believable track for a face at bbox: random points inside the box, so the convex
    hull and brightness measurements stay within it, with a closed mouth and open eyes."""
    x, y, w, h = bbox
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.uniform(x, x + w, 478), rng.uniform(y, y + h, 478)]).astype(np.float32)
    points[:4] = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    middle, mouth_y = x + w / 2, y + h * 0.55
    points[limits.UPPER_LIP], points[limits.LOWER_LIP] = (middle, mouth_y), (middle, mouth_y + 2)
    points[limits.MOUTH_LEFT], points[limits.MOUTH_RIGHT] = (middle - w * 0.15, mouth_y), (middle + w * 0.15, mouth_y)
    return TrackResult(TrackStatus.OK, 1.0, landmarks=points, bbox=bbox, pose_deg=(0.0, 0.0, 0.0),
                       blendshapes={"jawOpen": 0.05, "eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1,
                                    "mouthSmileLeft": 0.1, "mouthSmileRight": 0.1})


def textured_at(shape):
    """A stand-in for a well-lit, detailed picture: broad light and shade with fine detail on top."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    shade = 45 * np.sin(xs / 10.0) * np.cos(ys / 12.0)
    detail = np.random.default_rng(0).integers(-20, 21, shape[:2])
    grey = np.clip(128 + shade + detail, 0, 255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def test_an_upload_with_a_small_face_in_a_large_casual_photo_is_reframed_tighter(tmp_path):
    """A wide, casual photo (a desk, a wall, the subject sitting back) can still have plenty of
    native resolution on the face; downscaling the whole photo to the stored size first throws
    most of that away on background the output never uses. Cropping around the face first, in
    the original upload, keeps it, and a picture that looked unusable at a glance passes."""
    image = textured_at((3000, 4000))  # a big original upload: plenty of resolution, badly framed
    picture_shape = (960, 1280)  # what prepare() brings a 3000x4000 upload down to
    small_box_in_picture = (600, 400, 100, 100)  # a small, distant face once the whole photo is downscaled

    class FramingAwareTracker:
        """Finds the same face wherever it is asked to look: a small box in the downscaled
        whole photo, and a box filling the outline's own share in a tighter crop of it."""

        def process(self, frame):
            h, w = frame.shape[:2]
            if (h, w) == picture_shape:
                box = small_box_in_picture
            else:
                side = h * limits.TARGET_FACE_HEIGHT_SHARE
                box = ((w - side) / 2, (h - side) / 2, side, side)
            return _face_track(box, frame.shape)

        def close(self):
            pass

    tracker = FramingAwareTracker()
    naive = make_candidate(prepare(image), tracker.process(prepare(image)), "upload")
    naive_size = next(c for c in naive.checks if c.key == "size")
    assert not naive_size.passed  # confirms the premise: downscaling the whole photo first loses the face

    runtime = Runtime(tracker_factory=lambda video=True: tracker, data_dir=tmp_path / "data")
    reframed = runtime._upload_candidate(image, tracker)
    reframed_size = next(c for c in reframed.checks if c.key == "size")
    assert reframed_size.passed  # cropping around the face first kept enough of its native detail
    assert reframed_size.value > naive_size.value


class _MatchingScorer:
    """Says any two faces it is given are the enrolled person: for testing the identity gate
    in isolation from the real ArcFace model."""

    def embed(self, image, landmarks=None):
        return np.array([1.0, 0.0])

    def similarity(self, a, b):
        return 0.9  # comfortably above SAME_PERSON_CSIM


def _register_dummy_poses(store):
    """Fill in all four registered poses directly, for a unit test that is not about pose
    capture itself: check_upload refuses an upload until they are all registered."""
    for pose in POSE_KEYS:
        store.save_pose(pose, np.array([1.0, 0.0]), 60.0, [{"key": "pose", "passed": True}])


def test_a_small_face_is_accepted_once_identity_is_confirmed(tmp_path):
    """The size check protects how sharp the output looks, not who it is of. Once the ArcFace
    identity check confirms the uploaded picture is the enrolled person, a small face no
    longer blocks it, only warns about it in the tip a passing check already carries."""
    small_box = (220, 75, 100, 90)  # well under MIN_FACE_HEIGHT_PX, and too small to be fixed by reframing

    class SmallFaceTracker:
        def process(self, frame):
            return _face_track(small_box, frame.shape)

        def close(self):
            pass

    tracker = SmallFaceTracker()
    runtime = Runtime(tracker_factory=lambda video=True: tracker, scorer_factory=lambda: _MatchingScorer(),
                      data_dir=tmp_path / "data")
    runtime.store.save_identity(np.array([1.0, 0.0]), [{"key": "face", "passed": True}])
    _register_dummy_poses(runtime.store)
    runtime.face = runtime.store.load_identity()

    result = runtime.check_upload(textured_at((360, 640)))  # already at the stored size

    checks = {check["key"]: check for check in result["checks"]}
    assert checks["identity"]["passed"] is True
    assert checks["size"]["passed"] is True  # no longer a block ...
    assert checks["size"]["hint"] == ""
    assert "confirmed to be you" in checks["size"]["tip"]  # ... but still visibly a warning
    assert result["passed"] is True


def test_a_small_dark_face_is_still_refused_even_once_identity_is_confirmed(tmp_path):
    """Size is forgiven once identity is confirmed; the other checks are not. A photo that is
    both too small and too dark is still refused, on the lighting ground."""
    small_box = (220, 75, 100, 90)

    class SmallFaceTracker:
        def process(self, frame):
            return _face_track(small_box, frame.shape)

        def close(self):
            pass

    tracker = SmallFaceTracker()
    runtime = Runtime(tracker_factory=lambda video=True: tracker, scorer_factory=lambda: _MatchingScorer(),
                      data_dir=tmp_path / "data")
    runtime.store.save_identity(np.array([1.0, 0.0]), [{"key": "face", "passed": True}])
    _register_dummy_poses(runtime.store)
    runtime.face = runtime.store.load_identity()

    dark = textured_at((360, 640)) // 5  # the whole picture, including the face, is too dark
    result = runtime.check_upload(dark)

    checks = {check["key"]: check for check in result["checks"]}
    assert checks["identity"]["passed"] is True
    assert checks["light"]["passed"] is False  # the other fault still blocks ...
    assert checks["size"]["passed"] is False  # ... so size is not softened either
    assert result["passed"] is False


def test_an_upload_already_at_the_stored_size_is_not_reframed(tmp_path):
    """Nothing is gained by cropping a picture that was never going to be downscaled: there is
    no extra native resolution past the stored size for a tighter crop to recover."""
    image = textured_at((360, 640))  # well within the stored-size cap: prepare() leaves it alone
    box = (220, 75, 100, 90)  # small enough to fail "size" on its own

    class SmallFaceTracker:
        def process(self, frame):
            return _face_track(box, frame.shape)

        def close(self):
            pass

    tracker = SmallFaceTracker()
    runtime = Runtime(tracker_factory=lambda video=True: tracker, data_dir=tmp_path / "data")
    result = runtime._upload_candidate(image, tracker)
    size_check = next(c for c in result.checks if c.key == "size")
    assert not size_check.passed  # an honest refusal: the file itself does not have the detail
    assert size_check.value == pytest.approx(90, abs=1)


def test_a_picture_the_animation_model_cannot_use_is_refused_and_nothing_changes(parts):
    enrol(parts)
    before = status(parts)["enrolment"]["record"]["id"]
    upload(parts, well_lit_picture(seed=5))
    parts["engines"][0].reject = True
    refused = parts["client"].post("/api/enrolment/confirm")
    assert refused.status_code == 422
    assert refused.json()["detail"] == "No face found in the source image. Try a clearer, front-facing picture."
    assert status(parts)["enrolment"]["record"]["id"] == before


def test_large_uploads_are_stored_at_the_working_size(parts):
    verify_face(parts)
    register_all_poses(parts)
    big = cv2.resize(FRAME, (2560, 1440), interpolation=cv2.INTER_NEAREST)
    # The stand-in tracker reports the face for a 640 x 360 picture, so only the size is checked here.
    assert (upload(parts, big).json()["width"], upload(parts, big).json()["height"]) == (1280, 720)


def test_preview_shows_the_checks_and_takes_the_picture(parts):
    client = parts["client"]
    reply = client.post("/api/enrolment/preview/start", json={"input": "camera:1"})
    assert reply.status_code == 200
    assert parts["sources"] == [1]
    assert wait_for(lambda: status(parts)["enrolment"]["preview"]["ready"])
    preview = status(parts)["enrolment"]["preview"]
    assert preview["state"] == "running" and preview["input"] == "Camera 1" and preview["hint"] == ""
    assert all(check["passed"] for check in preview["checks"])
    assert preview["outline"] == limits.outline_for(640, 360)  # where the face should be, for the app to draw
    assert parts["engines"] == []  # the preview needs no animation model

    with client.websocket_connect("/api/stream") as socket:
        frame = None
        for _ in range(50):
            message = socket.receive()
            if message.get("bytes"):
                frame = message["bytes"]
                break
    header, camera, output = unpack_frame(frame)
    assert header["kind"] == "preview" and header["output_bytes"] == 0 and output == b""
    assert cv2.imdecode(np.frombuffer(camera, np.uint8), cv2.IMREAD_COLOR).shape == FRAME.shape

    taken = client.post("/api/enrolment/take").json()
    assert taken["passed"] and taken["origin"] == "camera"
    assert status(parts)["enrolment"]["preview"]["state"] == "running"  # still open, for a retake
    kept = client.post("/api/enrolment/confirm").json()
    assert kept["face"] is not None and kept["record"] is None  # the face signature is kept, the picture is not
    assert status(parts)["enrolment"]["preview"]["state"] == "idle"  # the camera is released
    assert parts["feeds"][0].closed
    assert "Face verified from the camera. Only its signature was kept." in events(parts)


def test_a_picture_is_not_taken_while_a_check_fails(parts, monkeypatch):
    monkeypatch.setattr("neuropresence.server.preview.TAKE_TIMEOUT_SECONDS", 0.3)
    client = parts["client"]
    assert client.post("/api/enrolment/take").status_code == 409  # the camera is not open
    parts["tracker"].yaw = 40.0
    client.post("/api/enrolment/preview/start", json={"input": "camera:0"})
    assert wait_for(lambda: status(parts)["enrolment"]["preview"]["checks"] is not None)
    preview = status(parts)["enrolment"]["preview"]
    assert preview["ready"] is False and preview["hint"] == "The head is turned to one side. Face the camera."
    refused = client.post("/api/enrolment/take")
    assert refused.status_code == 422
    assert refused.json()["detail"] == "The head is turned to one side. Face the camera."
    assert status(parts)["enrolment"]["candidate"] is None
    assert status(parts)["enrolment"]["preview"]["taking"] is False
    parts["tracker"].yaw = 0.0
    assert client.post("/api/enrolment/take").json()["passed"]  # once the fault is fixed it works


def test_the_preview_closes_the_camera_when_nobody_is_watching(parts, monkeypatch):
    monkeypatch.setattr("neuropresence.server.preview.UNWATCHED_SECONDS", 0.3)
    client = parts["client"]
    client.post("/api/enrolment/preview/start", json={"input": "camera:0"})
    assert wait_for(lambda: status(parts)["enrolment"]["preview"]["state"] == "running")
    watched_until = time.perf_counter() + 0.8
    while time.perf_counter() < watched_until:  # a page is showing the preview: frames are being fetched
        parts["runtime"].latest_pair()
        time.sleep(0.02)
    assert status(parts)["enrolment"]["preview"]["state"] == "running"
    assert not parts["feeds"][0].closed
    assert wait_for(lambda: parts["feeds"][0].closed)  # the page is gone: the camera is released by itself
    assert wait_for(lambda: status(parts)["enrolment"]["preview"]["state"] == "idle")
    assert "The camera was closed because the Enrolment screen is no longer open." in events(parts)


def test_the_best_frame_of_the_burst_is_kept():
    from neuropresence.server.preview import BURST_SECONDS, EnrolmentPreview

    class Frame:
        def __init__(self, passed, score):
            self.passed, self.score = passed, score

        def summary(self):
            return {"hint": "Hold still."}

    def served(frames):
        preview = EnrolmentPreview(runtime=None)
        request = {"future": __import__("concurrent.futures").futures.Future(), "best": None, "first_good_at": None,
                   "deadline": 100.0}
        preview._taking = request
        for at, frame in frames:
            if preview._taking is not None:
                preview._serve(request, frame, at)
        return request["future"]

    blink, sharp, late = Frame(True, 10.0), Frame(True, 50.0), Frame(True, 99.0)
    result = served([(0.0, Frame(False, 0)), (0.1, blink), (0.2, sharp), (0.1 + BURST_SECONDS, blink), (5.0, late)])
    assert result.result(timeout=0) is sharp  # the best of the burst, not the first and not a later one
    with pytest.raises(ValueError, match="Hold still."):
        served([(0.0, Frame(False, 0)), (100.0, Frame(False, 0))]).result(timeout=0)  # nothing passed in time


def test_only_cameras_can_verify_a_face_and_only_one_thing_uses_the_camera(parts):
    client = parts["client"]
    sample = client.post("/api/enrolment/preview/start", json={"input": "sample:d0"})
    assert sample.status_code == 409 and sample.json()["detail"] == "Your face can only be verified with a camera."
    enrol(parts)
    client.post("/api/enrolment/preview/start", json={"input": "camera:0"})
    assert client.post("/api/enrolment/preview/start", json={"input": "camera:0"}).status_code == 409  # already open
    start(parts)  # starting a session takes the camera over
    assert status(parts)["enrolment"]["preview"]["state"] == "idle" and parts["feeds"][0].closed
    busy = client.post("/api/enrolment/preview/start", json={"input": "camera:0"})
    assert busy.status_code == 409 and busy.json()["detail"] == "Stop the live session first. The camera is in use."


def test_removing_the_picture_keeps_the_face_and_forgetting_the_face_removes_both(parts):
    client = parts["client"]
    start(parts)
    assert client.delete("/api/enrolment").status_code == 409  # not while it is being animated
    assert client.delete("/api/enrolment/face").status_code == 409
    client.post("/api/session/stop")
    assert client.delete("/api/enrolment").json() == {"ok": True}
    found = status(parts)["enrolment"]
    assert found["record"] is None and found["face"] is not None  # the face stays: another picture can be uploaded
    stored = parts["tmp"] / "data" / "enrolment"
    # Removing the meeting picture does not touch the registered poses: they are the face's, not the picture's.
    assert sorted(p.name for p in stored.iterdir()) == ["identity.json", "identity.npy", "pose_down.npy",
                                                        "pose_front.npy", "pose_left.npy", "pose_right.npy",
                                                        "pose_up.npy", "poses.json"]
    assert parts["engines"][0].source_frame is None  # the model forgets it too
    assert client.get("/api/enrolment/picture").status_code == 404
    assert client.post("/api/session/start", json={"input": "camera:0"}).status_code == 409
    assert "Meeting picture removed." in events(parts)

    enrol(parts)
    assert client.delete("/api/enrolment/face").json() == {"ok": True}
    found = status(parts)["enrolment"]
    assert found["face"] is None and found["record"] is None
    assert list(stored.iterdir()) == []  # nothing about the person is left on disk
    assert upload(parts).status_code == 409  # and an upload has nothing to be compared with again
    assert "Face signature and meeting picture removed." in events(parts)


def test_the_enrolment_survives_a_restart(parts):
    record = enrol(parts)
    again = {"feeds": [], "sources": [], "engines": [], "gpu_reads": [], "tracker": FakeTracker()}
    restarted = make_runtime(parts["tmp"], again)
    found = restarted.status()["enrolment"]["record"]
    assert found == record  # known at once, before any model is loaded
    assert again["engines"] == []
    restarted.load_enrolment()
    assert again["engines"][0].source_frame.shape == FRAME.shape
    assert again["engines"][0].neutral_faces == []  # the neutral pose is not stored: a session takes it from the camera


# ----------------------------------------------------------------- session


def test_a_camera_session_needs_an_enrolled_picture(parts):
    refused = parts["client"].post("/api/session/start", json={"input": "camera:0"})
    assert refused.status_code == 409 and refused.json()["detail"] == "Enrol a picture first."
    assert status(parts)["session"]["state"] == "idle" and parts["feeds"] == []


def test_a_camera_session_animates_the_enrolled_picture(parts):
    start(parts)
    client = parts["client"]
    # Enough frames that the averages no longer depend on the very first one.
    assert wait_for(lambda: status(parts)["session"]["frames"] >= 30)
    found = status(parts)
    session = found["session"]
    assert session["input"] == "Camera 0" and session["source"] == "enrolment"
    assert session["metrics"]["stages_ms"] == {"tracker": 1.0, "motion": 2.0, "render": 3.0, "compose": 1.0}
    assert session["metrics"]["dropped_share"] == pytest.approx(0.5, abs=0.1)
    assert session["metrics"]["live_share"] == 1.0
    assert session["tracking"] == {"status": "ok", "pose_deg": [0.0, 0.0, 0.0], "mouth_open": 0.25}
    assert found["gpu"]["name"] == "Test GPU"
    # The neutral pose is the live face at the camera, taken once when the session began: not the picture's own.
    assert parts["engines"][0].neutral_faces == [(256, 256, 3)]
    assert "Neutral pose taken from the camera. Movement is measured from how you sit now." in events(parts)
    assert len(parts["engines"]) == 1
    assert client.get("/api/enrolment/picture").status_code == 200


def test_a_sample_clip_animates_its_own_frame_and_enrols_nothing(parts):
    verify_face(parts)
    register_all_poses(parts)  # poses are registered before the session, since the camera cannot open during one
    parts["feeds"].clear()
    parts["sources"].clear()
    start(parts, "sample:d0")
    assert status(parts)["session"]["source"] == "sample"
    assert parts["sources"] == [str(parts["tmp"] / "samples" / "d0.mp4")]
    assert parts["runtime"].engine_holds[0] == "sample"
    assert status(parts)["enrolment"]["record"] is None
    stored = parts["tmp"] / "data" / "enrolment"
    # The face signature, and the four registered poses; still no meeting picture.
    assert sorted(p.name for p in stored.iterdir()) == ["identity.json", "identity.npy", "pose_down.npy",
                                                        "pose_front.npy", "pose_left.npy", "pose_right.npy",
                                                        "pose_up.npy", "poses.json"]
    assert "Session started on Sample clip d0." in events(parts)
    upload(parts)
    busy = parts["client"].post("/api/enrolment/confirm")
    assert busy.status_code == 409 and busy.json()["detail"] == "Stop the sample session first."
    parts["client"].post("/api/session/stop")
    start(parts)  # a camera session afterwards puts the enrolled picture back into the model
    assert parts["runtime"].engine_holds[0] == "enrolment"


def test_the_picture_can_be_replaced_during_a_camera_session(parts):
    start(parts)
    first = status(parts)["enrolment"]["record"]["id"]
    brighter = np.clip(well_lit_picture(seed=7).astype(np.int16) + 30, 0, 255).astype(np.uint8)
    record = enrol(parts, brighter)
    assert record["id"] != first
    assert status(parts)["session"]["state"] == "running"
    assert parts["engines"][0].source_frame.mean() == pytest.approx(brighter.mean(), abs=0.5)


def test_identity_is_scored_while_live(parts):
    start(parts)
    parts["runtime"].identity._interval = 0.02
    assert wait_for(lambda: status(parts)["identity"]["csim"] == 0.9, seconds=4.0)


def test_starting_twice_and_unknown_inputs_are_refused(parts):
    start(parts)
    twice = parts["client"].post("/api/session/start", json={"input": "camera:0"})
    assert twice.status_code == 409 and twice.json()["detail"] == "A session is already running."
    assert parts["client"].post("/api/session/start", json={"input": "camera:9"}).status_code == 409


def test_stop_returns_to_idle_and_releases_the_camera(parts):
    start(parts)
    assert parts["client"].post("/api/session/stop").json()["session"]["state"] == "idle"
    assert parts["feeds"][0].closed
    assert parts["runtime"].latest_pair() is None
    start(parts)  # and it can be started again
    assert len(parts["feeds"]) == 2 and len(parts["engines"]) == 1  # the models are loaded once


def test_camera_failure_is_reported_and_recoverable(parts):
    enrol(parts)
    runtime = parts["runtime"]
    good_feed = runtime.make_feed

    def broken(source):
        raise SessionError("Camera 0 could not be opened. Check that it is connected and not in use by another app.")

    runtime.make_feed = broken
    parts["client"].post("/api/session/start", json={"input": "camera:0"})
    assert wait_for(lambda: status(parts)["session"]["state"] == "error")
    assert status(parts)["session"]["message"].startswith("Camera 0 could not be opened")
    runtime.make_feed = good_feed
    start(parts)


def test_a_sample_clip_without_a_front_facing_face_is_an_error(parts, monkeypatch):
    monkeypatch.setattr("neuropresence.server.session.SAMPLE_SOURCE_SECONDS", 0.2)
    parts["tracker"].yaw = 45.0
    parts["client"].post("/api/session/start", json={"input": "sample:d0"})
    assert wait_for(lambda: status(parts)["session"]["state"] == "error")
    assert "front-facing" in status(parts)["session"]["message"]
    assert parts["feeds"][0].closed


def test_lost_face_falls_back_and_is_logged(parts):
    start(parts)
    parts["tracker"].status = TrackStatus.NO_FACE
    assert wait_for(lambda: parts["runtime"].session.latest_pair().live is False)
    pair = parts["runtime"].session.latest_pair()
    assert pair.status == "no_face"
    # Not at once: the last live frame is held for a moment, then fades into the enrolled picture.
    assert wait_for(lambda: parts["runtime"].session.latest_pair().output.mean() == pytest.approx(FRAME.mean()))
    parts["tracker"].status = TrackStatus.OK
    assert wait_for(lambda: parts["runtime"].session.latest_pair().live is True)
    assert "No face in view. Showing the enrolled picture." in events(parts)
    assert "Face found. Reenactment resumed." in events(parts)


def test_neutral_reset(parts):
    assert parts["client"].post("/api/session/neutral").status_code == 409
    start(parts)
    assert parts["engines"][0].resets == 1  # the session began by waiting for a neutral pose
    assert parts["client"].post("/api/session/neutral").json() == {"ok": True}
    assert parts["engines"][0].resets == 2  # and the button takes a new one


# ---------------------------------------------------------------- features


def test_reenactment_switch(parts):
    start(parts)
    client = parts["client"]
    reply = client.patch("/api/features/reenactment", json={"enabled": False})
    assert reply.json()["enabled"] is False
    assert wait_for(lambda: parts["runtime"].session.latest_pair().live is False)
    assert parts["runtime"].session.latest_pair().status == "ok"  # still tracked, just not animated
    client.patch("/api/features/reenactment", json={"enabled": True})
    assert wait_for(lambda: parts["runtime"].session.latest_pair().live is True)
    assert "Reenactment turned off." in events(parts)


def test_unknown_feature(parts):
    assert parts["client"].patch("/api/features/teleport", json={"enabled": True}).status_code == 404


# ------------------------------------------------------------------ stream


def test_stream_sends_status_then_frames(parts):
    start(parts)
    with parts["client"].websocket_connect("/api/stream") as socket:
        first = json.loads(socket.receive_text())
        assert first["status"]["session"]["state"] == "running"
        assert any(e["message"] == "Session started on Camera 0." for e in first["events"])
        frame = None
        for _ in range(50):
            message = socket.receive()
            if message.get("bytes"):
                frame = message["bytes"]
                break
        assert frame is not None
    header, camera, output = unpack_frame(frame)
    assert header["kind"] == "live" and header["live"] is True and header["status"] == "ok"
    assert cv2.imdecode(np.frombuffer(camera, np.uint8), cv2.IMREAD_COLOR).shape == FRAME.shape
    assert cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR).mean() == pytest.approx(200, abs=2)


def test_stream_survives_a_failing_status_reading(parts):
    start(parts)
    runtime = parts["runtime"]
    real_status, calls = runtime.status, []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise AttributeError("partially initialized module")
        return real_status()

    runtime.status = flaky
    with parts["client"].websocket_connect("/api/stream") as socket:
        texts = 0
        for _ in range(200):
            message = socket.receive()
            if message.get("text"):
                texts += 1
                break
        assert texts == 1  # the stream is still open and sent the next reading
    assert len(calls) >= 2


def test_frame_packing_round_trip_and_overlay():
    camera = np.zeros((720, 1280, 3), dtype=np.uint8)
    output = np.full((720, 1280, 3), 90, dtype=np.uint8)
    landmarks = np.array([[640.0, 360.0]], dtype=np.float32)
    pair = FramePair(7, camera, output, True, "ok", landmarks)
    header, plain, _ = unpack_frame(pack_frame(pair, overlay=False))
    assert header == {"id": 7, "kind": "live", "live": True, "status": "ok", "camera_bytes": len(plain),
                      "output_bytes": header["output_bytes"]}
    plain = cv2.imdecode(np.frombuffer(plain, np.uint8), cv2.IMREAD_COLOR)
    assert plain.shape == (360, 640, 3)  # the camera preview is shrunk; the output is not
    assert plain.max() < 10
    _, marked, big = unpack_frame(pack_frame(pair, overlay=True))
    marked = cv2.imdecode(np.frombuffer(marked, np.uint8), cv2.IMREAD_COLOR)
    assert marked[180, 320].max() > 100  # the landmark, at half scale
    assert cv2.imdecode(np.frombuffer(big, np.uint8), cv2.IMREAD_COLOR).shape == (720, 1280, 3)
    assert camera.max() == 0  # the frame itself is never drawn on


def test_a_preview_frame_has_no_output_and_a_larger_camera_picture():
    pair = FramePair(3, np.zeros((720, 1280, 3), dtype=np.uint8), None, False, "ok", None, kind="preview")
    header, camera, output = unpack_frame(pack_frame(pair))
    assert header["kind"] == "preview" and header["output_bytes"] == 0 and output == b""
    assert cv2.imdecode(np.frombuffer(camera, np.uint8), cv2.IMREAD_COLOR).shape == (540, 960, 3)


# ----------------------------------------------------------------- metrics


def test_metrics_window_averages_recent_frames_only():
    window = MetricsWindow(seconds=2.0)
    assert window.summary(now=0.0)["fps"] is None
    timing = {"tracker": 10.0, "motion": 20.0, "render": 90.0, "compose": 5.0, "total": 125.0}
    window.add(0.0, {"tracker": 500.0, "total": 500.0}, 600.0, False, 0)  # will be too old
    for i in range(11):
        window.add(10.0 + i * 0.1, timing, 140.0, True, 2)
    summary = window.summary(now=11.0)
    assert summary["fps"] == pytest.approx(10.0)
    assert summary["pipeline_ms"] == 125.0
    assert summary["end_to_end_ms"] == 140.0
    assert summary["stages_ms"] == {"tracker": 10.0, "motion": 20.0, "render": 90.0, "compose": 5.0}
    assert summary["render_ms"] == 90.0
    assert summary["dropped_share"] == pytest.approx(2 / 3, abs=0.001)
    assert summary["live_share"] == 1.0
    assert window.total_frames == 12
    assert window.summary(now=20.0)["fps"] is None  # nothing recent


def test_metrics_separate_render_time_from_the_average_frame():
    """With half the frames not reenacted, the render time per reenacted frame stays the
    same while the average frame spends half as long rendering."""
    window = MetricsWindow(seconds=5.0)
    live = {"tracker": 10.0, "motion": 20.0, "render": 90.0, "compose": 5.0, "total": 125.0}
    fallback = {"tracker": 10.0, "total": 10.0}
    for i in range(10):
        window.add(i * 0.1, live if i % 2 == 0 else fallback, 100.0, i % 2 == 0, 0)
    summary = window.summary(now=1.0)
    assert summary["render_ms"] == 90.0
    assert summary["stages_ms"] == {"tracker": 10.0, "motion": 10.0, "render": 45.0, "compose": 2.5}
    assert summary["live_share"] == 0.5
    for i in range(10, 20):
        window.add(i * 0.1, fallback, 100.0, False, 0)
    assert window.summary(now=7.0)["render_ms"] is None  # nothing was reenacted lately
