"""A covered mouth at the camera (capture/cover.py), and what the engine and the pipeline do about it."""

import json

import cv2
import numpy as np
import pytest

from neuropresence.capture import TrackResult, TrackStatus
from neuropresence.capture import cover as cover_module
from neuropresence.capture.cover import CHEEKS, EYE_CORNERS, INNER_LIPS, OUTER_LIPS, MouthCover, lip_redness

SKIN = (140, 160, 205)  # BGR
LIPS = (95, 90, 185)


def mesh():
    """478 landmarks of a face: eyes 120 px apart, lips a ring under them, a point on each cheek."""
    points = np.full((478, 2), 320.0, dtype=np.float32)
    points[EYE_CORNERS[0]], points[EYE_CORNERS[1]] = (260, 200), (380, 200)
    for ring, (rx, ry) in ((OUTER_LIPS, (45, 20)), (INNER_LIPS, (30, 6))):
        for k, index in enumerate(ring):
            angle = 2 * np.pi * k / len(ring)
            points[index] = (320 + rx * np.cos(angle), 300 + ry * np.sin(angle))
    for index, at in zip(CHEEKS, ((250, 260), (390, 260), (255, 285), (385, 285))):
        points[index] = at
    return points


def picture(lips=LIPS, covered=False):
    frame = np.full((480, 640, 3), SKIN, dtype=np.uint8)
    points = mesh()
    if not covered:
        cv2.fillPoly(frame, [points[list(OUTER_LIPS)].astype(np.int32)], lips)
    return frame


def feed(detector, frame, count, start=0.0):
    return [detector.update(frame, mesh(), start + i / 22) for i in range(count)]


def test_lips_read_redder_than_the_cheeks_and_a_cover_takes_it_away():
    assert lip_redness(picture(), mesh()) > 10
    assert abs(lip_redness(picture(covered=True), mesh())) < 1
    assert lip_redness(picture(), np.zeros((68, 2))) is None  # not the face mesh's landmarks
    small = mesh() * 0.1  # a face too small to read
    assert lip_redness(picture(), small) is None


def test_a_cover_is_found_in_a_few_frames_and_its_end_in_a_few_more():
    detector = MouthCover()
    assert feed(detector, picture(), cover_module.LEARN_FRAMES + 10) == [False] * (cover_module.LEARN_FRAMES + 10)
    assert detector.usable
    said = feed(detector, picture(covered=True), 10, start=2.0)
    assert said.index(True) == cover_module.FRAMES_TO_COVER - 1  # about a fifth of a second
    said = feed(detector, picture(), 10, start=3.0)
    assert said.index(False) == cover_module.FRAMES_TO_UNCOVER - 1


def test_what_the_detector_says_can_be_written_as_json():
    """The session puts these in the status. A numpy bool among them once cut off the Live Studio's stream."""
    detector = MouthCover()
    feed(detector, picture(), cover_module.LEARN_FRAMES + 10)  # learned, then followed for ten frames
    feed(detector, picture(covered=True), 10, start=2.0)
    assert type(detector.usable) is bool and type(detector.covered) is bool
    assert all(type(value) is float for value in (detector.base, detector.noise, detector.share))
    json.dumps({"usable": detector.usable, "covered": detector.covered, "share": detector.share})


def test_one_odd_frame_is_not_a_cover():
    detector = MouthCover()
    feed(detector, picture(), cover_module.LEARN_FRAMES + 5)
    for i in range(30):
        frame = picture(covered=True) if i % cover_module.FRAMES_TO_COVER == 0 else picture()
        assert detector.update(frame, mesh(), 2.0 + i / 22) is False


def test_lips_the_colour_of_the_skin_cannot_be_told_from_a_cover_so_nothing_is_ever_called_covered():
    detector = MouthCover()
    faint = picture(lips=(138, 158, 207))
    assert feed(detector, faint, 40) == [False] * 40
    assert not detector.usable
    assert feed(detector, picture(covered=True), 20, start=2.0) == [False] * 20


def test_a_mouth_covered_from_the_start_is_learned_once_it_shows():
    detector = MouthCover()
    feed(detector, picture(covered=True), 30)
    assert not detector.usable  # nothing to learn from yet
    feed(detector, picture(), cover_module.LEARN_FRAMES, start=2.0)
    assert detector.usable
    assert feed(detector, picture(covered=True), 10, start=4.0)[-1] is True


# ------------------------------------------------------------------ the engine


def engine_with_reference(mouth=0.0):
    import torch

    from neuropresence.reenactment.engine import LIP_KEYPOINTS, ReenactmentEngine

    def face(mouth=0.0, brows=0.0):
        expression = torch.full((1, 21, 3), brows)
        expression[:, LIP_KEYPOINTS] = mouth
        return {"exp": expression, "pitch": torch.zeros(1, 1), "yaw": torch.zeros(1, 1), "roll": torch.zeros(1, 1),
                "t": torch.zeros(1, 3), "scale": torch.ones(1, 1)}

    engine = ReenactmentEngine.__new__(ReenactmentEngine)  # only the logic: no model is loaded
    engine._reference = {"info": face(mouth=mouth), "rotation": None}
    engine._rotation = lambda pitch, yaw, roll: None
    return engine, face, LIP_KEYPOINTS


def test_a_covered_mouth_is_eased_to_the_neutral_one_quickly_and_back_a_little_slower():
    engine, _, _ = engine_with_reference()
    engine.mouth_covered = False
    assert engine._covering(0.0) == 0.0
    engine.mouth_covered = True
    held = [engine._covering(k / 22) for k in range(1, 12)]
    assert held[0] > 0.3 and held[-1] == 1.0  # a third at the first frame, all of it within half a second
    engine.mouth_covered = False
    back = [engine._covering(0.5 + k / 22) for k in range(1, 24)]
    assert back[0] > 0.6 and back[-1] == 0.0


def test_a_covered_mouth_is_not_taken_into_the_neutral_pose():
    engine, face, lips = engine_with_reference(mouth=0.0)
    engine.mouth_covered, engine.resting, engine.anchor_lips, engine.lips_closed = True, True, True, True
    covered = face(mouth=0.07, brows=0.0)  # how the cover reads: a mouth that is not there
    for step in range(60):
        engine._follow_rest(covered, at=step / 30)
        engine._follow_lips(covered, at=step / 30)
    assert float(engine._reference["info"]["exp"][:, lips].abs().max()) == 0.0
    engine.mouth_covered = False  # uncovered and at rest: the neutral pose follows the person again, as before
    for step in range(60, 200):
        engine._follow_rest(covered, at=step / 30)
    assert float(engine._reference["info"]["exp"][:, lips].abs().max()) > 0.0


# ------------------------------------------------------------------ the pipeline


class Engine:
    def __init__(self):
        self.source_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        self.last_timing_ms = {"motion": 2.0, "render": 3.0, "compose": 1.0}
        self.neutral_faces = []
        self.covered = []

    def reset_reference(self):
        pass

    def set_reference(self, face):
        self.neutral_faces.append(face.shape)

    def drive(self, face, at=None, steady=False, limit=False, motion=None):
        self.covered.append(self.mouth_covered)
        return np.full((480, 640, 3), 200, dtype=np.uint8)


class Tracker:
    def process(self, frame):
        return TrackResult(TrackStatus.OK, 1.0, bbox=(250, 180, 140, 160), landmarks=mesh(), pose_deg=(0.0, -9.0, 0.0),
                           blendshapes={"jawOpen": 0.02, "eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1})


def test_the_pipeline_tells_the_engine_and_never_takes_a_neutral_pose_from_a_covered_mouth(monkeypatch):
    from neuropresence import pipeline as pipeline_module
    from neuropresence.pipeline import Pipeline

    engine = Engine()
    pipeline = Pipeline(Tracker(), engine)
    for k in range(cover_module.LEARN_FRAMES + 2):
        pipeline.step(picture(), at=k / 22)
    for k in range(6):
        result = pipeline.step(picture(covered=True), at=2.0 + k / 22)
    assert result.mouth_covered and engine.covered[-1] is True
    pipeline.wait_for_neutral(seconds=0.0)  # the guard asks for a fresh neutral pose while the mouth is covered
    monkeypatch.setattr(pipeline_module.time, "perf_counter", lambda: 1e12)  # long after the wait has run out
    assert pipeline.step(picture(covered=True), at=3.0).live is False and engine.neutral_faces == []
    for k in range(cover_module.FRAMES_TO_UNCOVER):
        result = pipeline.step(picture(), at=3.1 + k / 22)
    assert result.live and engine.neutral_faces and not result.mouth_covered


def test_with_the_switch_off_nothing_is_called_covered():
    from neuropresence.pipeline import Pipeline

    engine = Engine()
    pipeline = Pipeline(Tracker(), engine)
    pipeline.watch_cover = False
    for k in range(cover_module.LEARN_FRAMES + 2):
        pipeline.step(picture(), at=k / 22)
    for k in range(8):
        result = pipeline.step(picture(covered=True), at=2.0 + k / 22)
    assert not result.mouth_covered and engine.covered[-1] is False
    assert pipeline.cover.covered  # it still followed the frames, so turning it on works at once


@pytest.mark.parametrize("key", ["mouth_cover"])
def test_the_switch_is_on_by_default(key):
    from neuropresence.server.features import FeatureSet

    assert FeatureSet().enabled(key)
