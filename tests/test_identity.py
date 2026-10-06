import math

import cv2
import numpy as np
import pytest

from neuropresence.identity.align import (
    ARCFACE_TEMPLATE,
    FIVE_POINT_INDICES,
    align_face,
    five_points,
    similarity_transform,
)
from neuropresence.identity.arcface import DEFAULT_ARCFACE, ArcFaceEmbedder
from neuropresence.reenactment.engine import LIVEPORTRAIT_DIR

SAMPLES = LIVEPORTRAIT_DIR / "assets" / "examples"
needs_arcface = pytest.mark.skipif(not DEFAULT_ARCFACE.exists(), reason="run scripts/download_models.py first")
needs_samples = pytest.mark.skipif(not SAMPLES.exists(), reason="run scripts/setup_liveportrait.py first")


def apply(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


def make_matrix(angle_deg, scale, shift):
    a = math.radians(angle_deg)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.hstack([scale * rotation, np.array(shift, dtype=np.float64)[:, None]])


def test_similarity_transform_recovers_known_motion():
    truth = make_matrix(30, 2.5, (40, -15))
    estimate = similarity_transform(ARCFACE_TEMPLATE, apply(truth, ARCFACE_TEMPLATE))
    assert estimate == pytest.approx(truth, abs=1e-9)


def test_similarity_transform_never_mirrors():
    mirrored = ARCFACE_TEMPLATE * np.array([-1.0, 1.0])
    estimate = similarity_transform(ARCFACE_TEMPLATE, mirrored)
    assert np.linalg.det(estimate[:, :2]) > 0


def test_similarity_transform_rejects_bad_input():
    with pytest.raises(ValueError):
        similarity_transform(np.zeros((5, 2)), ARCFACE_TEMPLATE)  # all points coincide
    with pytest.raises(ValueError):
        similarity_transform(ARCFACE_TEMPLATE[:3], ARCFACE_TEMPLATE)  # counts differ


def test_five_points_picks_the_documented_landmarks():
    landmarks = np.arange(478 * 2, dtype=np.float32).reshape(478, 2)
    assert five_points(landmarks).tolist() == landmarks[list(FIVE_POINT_INDICES)].tolist()
    with pytest.raises(ValueError):
        five_points(landmarks[:400])


def test_align_face_puts_the_points_on_the_template():
    # Draw five dots where a rotated, enlarged, shifted face would have its
    # eyes, nose and mouth corners; after alignment they must sit on the template.
    placed = apply(make_matrix(-20, 3.0, (150, 220)), ARCFACE_TEMPLATE)
    image = np.zeros((700, 700, 3), dtype=np.uint8)
    for x, y in placed:
        cv2.circle(image, (int(round(x)), int(round(y))), 6, (255, 255, 255), -1)
    aligned = align_face(image, placed)
    assert aligned.shape == (112, 112, 3)
    for x, y in ARCFACE_TEMPLATE:
        assert aligned[int(round(y)), int(round(x))].min() > 200
    assert aligned[5, 5].max() == 0


def test_missing_arcface_model_gives_download_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        ArcFaceEmbedder(tmp_path / "missing.onnx")


@pytest.fixture(scope="module")
def scorer():
    from neuropresence.identity import IdentityScorer

    with IdentityScorer() as scorer:
        yield scorer


def video_frame(name, index):
    capture = cv2.VideoCapture(str(SAMPLES / "driving" / name))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    assert ok
    return frame


@needs_arcface
def test_embedder_rejects_unaligned_input():
    with pytest.raises(ValueError):
        ArcFaceEmbedder().embed(np.zeros((200, 200, 3), dtype=np.uint8))


@needs_arcface
def test_no_face_gives_no_embedding(scorer):
    assert scorer.embed(np.zeros((480, 640, 3), dtype=np.uint8)) is None


@needs_arcface
@needs_samples
def test_same_person_scores_high_and_strangers_score_low(scorer):
    person_a = scorer.embed(video_frame("d0.mp4", 0))
    person_a_smiling = scorer.embed(video_frame("d0.mp4", 40))
    person_b = scorer.embed(video_frame("d3.mp4", 0))
    assert person_a.shape == (512,)
    assert np.linalg.norm(person_a) == pytest.approx(1.0, abs=1e-5)
    assert scorer.similarity(person_a, person_a) == pytest.approx(1.0, abs=1e-5)
    assert scorer.similarity(person_a, person_a_smiling) > 0.5
    assert scorer.similarity(person_a, person_b) < 0.3


@needs_arcface
@needs_samples
def test_supplied_landmarks_give_the_same_embedding(scorer):
    from neuropresence.capture import FaceTracker

    frame = video_frame("d0.mp4", 0)
    with FaceTracker(video=False) as tracker:
        landmarks = tracker.process(frame).landmarks
    assert scorer.similarity(scorer.embed(frame), scorer.embed(frame, landmarks)) == pytest.approx(1.0, abs=1e-4)
