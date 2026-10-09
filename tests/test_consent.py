"""The consent check, the watch during a session, and the disclosure mark."""

import numpy as np
import pytest

from neuropresence.capture import TrackResult, TrackStatus
from neuropresence.consent import ConsentCheck, ConsentWatch, LivenessChallenge, Reading, is_marked, mark
from neuropresence.consent import check as limits
from neuropresence.consent import disclosure as label
from neuropresence.pipeline import Pipeline

FPS = 30.0
REST = Reading(TrackStatus.OK, yaw=0.0, jaw=0.05, eyes=0.1)
TURNED = Reading(TrackStatus.OK, yaw=45.0, jaw=0.05, eyes=0.1)
BLINKING = Reading(TrackStatus.OK, yaw=0.0, jaw=0.05, eyes=0.9)
GONE = Reading(TrackStatus.NO_FACE)

# Face signatures, made so that the user's own faces score 0.9 or so against each other and a stranger's 0.1.
USER = np.array([1.0, 0.0, 0.0])
USER_TURNED = np.array([0.9, 0.0, 0.436])
STRANGER = np.array([0.1, 0.995, 0.0])

ANSWER = [(1.0, REST), (0.5, TURNED), (1.0, REST), (0.15, BLINKING), (1.2, REST)]  # to "turn left, then blink"


def consent(signatures=(USER,)):
    return ConsentCheck(LivenessChallenge(("turn_left", "blink"), waits=(0.0, 0.0)), list(signatures) if signatures else None)


def play(check, script, who=USER, start=0.0, swap=None):
    """Run the check over a script of (seconds, reading). `who` is whose face is in every frame;
    `swap` can name another face for a stretch of the time, as (from, to, signature)."""
    at, scored = start, 0
    for seconds, reading in script:
        for _ in range(max(1, int(round(seconds * FPS)))):
            if check.done:
                return at, scored
            check.frame(reading, at)
            if reading.ok and check.wants_face(at):
                shown = swap[2] if swap and swap[0] <= at < swap[1] else who
                check.face(shown, at)
                scored += 1
            at += 1.0 / FPS
    return at, scored


def test_the_registered_person_answering_the_prompt_passes():
    check = consent()
    _, scored = play(check, ANSWER)
    assert check.passed is True and check.reason == "" and check.failure is None
    assert check.match == pytest.approx(1.0) and check.lowest == pytest.approx(1.0)
    assert 10 <= scored <= 20  # matched about four times a second, not in every frame


def test_another_person_answering_the_prompt_is_refused():
    check = consent()
    at, scored = play(check, ANSWER, who=STRANGER)
    assert check.passed is False and check.failure == "identity" and check.reason == limits.NOT_THE_VERIFIED_FACE
    assert scored == 2 and at < 0.2  # looked at again in the next frame, and refused at once
    assert check.match == pytest.approx(0.1)


def test_a_photograph_held_up_for_the_match_while_someone_else_moves_is_refused():
    check = consent()
    play(check, ANSWER, who=USER, swap=(1.1, 1.4, STRANGER))  # the stranger's own face, only for the turn
    assert check.passed is False and check.failure == "identity"


def test_one_face_that_does_not_match_is_looked_at_again():
    check = consent()
    check.frame(REST, 0.0), check.face(USER, 0.0)
    check.frame(REST, 0.3), check.face(STRANGER, 0.3)  # one frame that does not match: a blurred one, say
    assert check.passed is None and check.wants_face(0.31)  # so the very next frame is matched too
    check.frame(REST, 0.33), check.face(USER, 0.33)  # and it is the user after all
    assert check.passed is None and not check.wants_face(0.36)
    play(check, ANSWER, start=0.4)
    assert check.passed is True and check.lowest == pytest.approx(0.1)


def test_the_best_of_the_registered_signatures_decides():
    assert float(USER @ USER_TURNED) == pytest.approx(0.9)
    check = consent(signatures=(STRANGER * 0 + np.array([0.0, 0.0, 1.0]), USER))  # a pose that looks nothing like it, and the face
    play(check, ANSWER, who=USER_TURNED)
    assert check.passed is True and check.match == pytest.approx(0.9)


def test_the_prompt_alone_is_not_enough_and_neither_is_the_match():
    check = consent()
    play(check, [(12.0, REST)])  # the right face, doing nothing: a photograph of the user
    assert check.passed is False and check.failure == "late"
    assert check.reason == "Your head did not turn to your left within 3 seconds."

    check = consent()
    at = 0.0
    for seconds, reading in ANSWER:  # the prompt answered, and no face ever given to match
        for _ in range(int(round(seconds * FPS))):
            check.frame(reading, at)
            at += 1.0 / FPS
    assert check.passed is None
    for _ in range(int(2.5 * FPS)):
        check.frame(REST, at)
        at += 1.0 / FPS
    assert check.passed is False and check.failure == "identity"


def test_the_face_is_matched_once_more_after_the_last_action():
    check = consent()
    at, _ = play(check, ANSWER[:-1] + [(0.6, REST)])
    assert check.challenge.done and check.passed is True  # the match that passes it was made after the prompt was passed
    check = consent()
    play(check, ANSWER, swap=(3.0, 9.0, STRANGER))  # someone else slips in as the prompt ends
    assert check.passed is False


def test_whoever_is_there_after_the_face_was_lost_is_matched_at_once():
    check = consent()
    at, _ = play(check, [(0.4, REST)])
    assert not check.wants_face(at)  # matched a moment ago
    check.frame(GONE, at)
    check.frame(REST, at + 0.04)
    assert check.wants_face(at + 0.04)


def test_a_face_being_verified_has_to_stay_the_same_face():
    check = consent(signatures=None)  # no signature yet: the first face seen is the one to match
    play(check, ANSWER, who=USER)
    assert check.passed is True and check.same_face(USER_TURNED) and not check.same_face(STRANGER)

    check = consent(signatures=None)
    play(check, ANSWER, who=USER, swap=(1.1, 1.5, STRANGER))
    assert check.passed is False and check.reason == limits.THE_FACE_CHANGED


def test_what_the_app_is_told_about_the_check():
    check = consent()
    at, _ = play(check, [(1.0, REST)])
    told = check.snapshot(at)
    assert told["stage"] == "act" and told["prompt"] == "Turn your head to your left" and told["match"] == 1.0
    at, _ = play(check, [(0.5, TURNED), (1.0, REST), (0.15, BLINKING), (0.45, REST)], start=at)
    assert check.challenge.stage.value == "settle"
    at, _ = play(check, [(1.0, REST)], start=at)
    assert check.snapshot(at)["stage"] == "passed" and check.snapshot(at)["reason"] == ""
    refused = consent()
    at, _ = play(refused, ANSWER, who=STRANGER)
    told = refused.snapshot(at)
    assert told["stage"] == "refused" and told["failure"] == "identity" and told["match"] == 0.1


# ------------------------------------------------------------------ the watch


def test_another_person_at_the_camera_is_held_back_after_three_readings():
    watch = ConsentWatch()
    assert [watch.sample(score) for score in (0.8, 0.7, 0.1, 0.2, 0.8)] == [None] * 5  # two that do not match: nothing
    assert watch.held is False
    assert [watch.sample(score) for score in (0.1, 0.2, 0.15)] == [None, None, "hold"]
    assert watch.held is True and watch.match == 0.15
    assert [watch.sample(score) for score in (0.1, 0.8, 0.2, 0.7, 0.9)] == [None, None, None, None, "release"]
    assert watch.held is False
    watch.sample(0.1), watch.sample(0.1)
    watch.reset()
    assert watch.sample(0.1) is None and watch.held is False  # a new session starts counting again
    assert ConsentWatch().sample(0.35) is None  # exactly at the limit is the same person


# ------------------------------------------------------------------- the mark


@pytest.mark.parametrize("size", [(720, 1280), (1080, 1920), (360, 640), (512, 512), (240, 320), (1280, 720)])
def test_the_mark_is_drawn_in_the_lower_left_corner_and_found_again(size):
    height, width = size
    rng = np.random.default_rng(3)
    for frame in (np.full((height, width, 3), 235, np.uint8), np.zeros((height, width, 3), np.uint8),
                  rng.integers(0, 256, (height, width, 3), dtype=np.uint8)):
        plain = frame.copy()
        assert not is_marked(frame)
        assert mark(frame) is frame and is_marked(frame)  # drawn in place
        changed = np.argwhere((frame != plain).any(axis=2))
        tall = changed[:, 0].max() - changed[:, 0].min() + 1
        assert changed[:, 0].min() > 0.9 * height - 20 and changed[:, 1].max() < 0.5 * width  # low and to the left
        assert tall >= max(label.MIN_HEIGHT_PX, round(label.HEIGHT_SHARE * height)) - 1
        assert (frame != plain).any(axis=2).mean() < 0.03  # and small: under 3% of the picture
        assert is_marked(mark(frame))  # drawing it twice leaves it readable


def test_a_frame_too_small_for_the_label_still_gets_one():
    tiny = np.full((48, 64, 3), 200, np.uint8)
    assert is_marked(mark(tiny))


class Tracker:
    def __init__(self):
        self.status = TrackStatus.OK

    def process(self, frame):
        return TrackResult(self.status, 1.0, bbox=(100, 100, 80, 80)) if self.status is TrackStatus.OK else TrackResult(self.status, 1.0)


class Engine:
    def __init__(self):
        self.source_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
        self.last_timing_ms = {"motion": 2.0, "render": 3.0, "compose": 1.0}

    def reset_reference(self):
        pass

    def set_reference(self, face):
        pass

    def drive(self, face, at=None, steady=False, limit=False, motion=None):
        return np.full((720, 1280, 3), 200, dtype=np.uint8)


def test_every_kind_of_output_frame_carries_the_mark():
    camera = np.zeros((480, 640, 3), np.uint8)
    tracker, engine = Tracker(), Engine()
    pipeline = Pipeline(tracker, engine)
    kinds = {"live": pipeline.step(camera, at=0.0)}
    tracker.status = TrackStatus.NO_FACE
    kinds["held"] = pipeline.step(camera, at=0.1)  # the last live frame, for a moment
    kinds["fading out"] = pipeline.step(camera, at=0.3)
    kinds["still"] = pipeline.step(camera, at=2.0)
    tracker.status = TrackStatus.OK
    kinds["fading in"] = pipeline.step(camera, at=2.1)
    kinds["switched off"] = Pipeline(tracker, engine).step(camera, reenact=False, at=0.0)
    late = Pipeline(tracker, engine)
    late.step(camera, at=0.0)
    kinds["held back"] = late.step(camera, at=2.0, show=False)  # drawn, but arriving too late to show
    tracker.status = TrackStatus.MULTIPLE_FACES
    kinds["two faces"] = Pipeline(tracker, engine).step(camera, at=0.0)
    assert [kind for kind, result in kinds.items() if not is_marked(result.output)] == []
    assert kinds["live"].live and not kinds["still"].live and 50 < kinds["fading out"].output[:300].mean() < 200
    assert not is_marked(engine.source_frame)  # the enrolled picture itself is left as it is
    assert is_marked(pipeline.still_picture())


def test_a_new_picture_gets_its_own_marked_still():
    tracker, engine = Tracker(), Engine()
    tracker.status = TrackStatus.NO_FACE
    pipeline = Pipeline(tracker, engine)
    first = pipeline.step(np.zeros((480, 640, 3), np.uint8), at=0.0).output
    engine.source_frame = np.full((360, 640, 3), 90, dtype=np.uint8)  # the picture is replaced during a session
    second = pipeline.step(np.zeros((480, 640, 3), np.uint8), at=0.1).output
    assert first.shape == (720, 1280, 3) and second.shape == (360, 640, 3)
    assert is_marked(first) and is_marked(second) and second[:100].mean() == 90
