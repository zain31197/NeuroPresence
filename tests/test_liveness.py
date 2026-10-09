"""The liveness prompt, run over readings written out by hand: no camera and no tracker."""

import itertools
import random

import pytest

from neuropresence.capture import TrackStatus
from neuropresence.consent import liveness
from neuropresence.consent.liveness import (ACTIONS, TURNS, LivenessChallenge, Reading, Stage, at_rest, pick_actions,
                                            pick_waits)

FPS = 30.0
REST = Reading(TrackStatus.OK, yaw=1.0, jaw=0.05, eyes=0.10)
GONE = Reading(TrackStatus.NO_FACE)
TWO = Reading(TrackStatus.MULTIPLE_FACES)


def face(**changed):
    return Reading(TrackStatus.OK, **{"yaw": REST.yaw, "jaw": REST.jaw, "eyes": REST.eyes, **changed})


# What a person does, as (seconds, reading).
SIT = (1.0, REST)
TURN_LEFT = (0.5, face(yaw=45.0))
TURN_RIGHT = (0.5, face(yaw=-45.0))
BLINK = (0.15, face(eyes=0.85))
OPEN_MOUTH = (0.6, face(jaw=0.8))
DOES = {"turn_left": TURN_LEFT, "turn_right": TURN_RIGHT, "blink": BLINK, "open_mouth": OPEN_MOUTH}


def challenge(*actions, wait=0.0):
    return LivenessChallenge(actions, waits=[wait] * len(actions))


def play(check, *script, start=0.0):
    """Feed the challenge what the script says, frame by frame. Returns the time it ended at."""
    at = start
    for seconds, reading in script:
        for _ in range(max(1, int(round(seconds * FPS)))):
            check.sample(reading, at)
            at += 1.0 / FPS
    return at


def test_the_actions_asked_for_done_in_order_pass():
    check = challenge("turn_left", "blink")
    end = play(check, SIT, TURN_LEFT, SIT, BLINK, (0.1, REST))
    assert check.stage is Stage.SETTLE and check.step == 2  # both seen: the face has to come to rest once more
    play(check, SIT, start=end)
    assert check.stage is Stage.PASSED and check.reason == ""


@pytest.mark.parametrize("order", [pair for pair in itertools.permutations(ACTIONS, 2) if set(pair) & set(TURNS)])
def test_every_order_that_can_be_asked_for_can_be_answered(order):
    check = challenge(*order, wait=0.4)
    play(check, SIT, DOES[order[0]], SIT, DOES[order[1]], SIT)
    assert check.stage is Stage.PASSED


def test_a_photograph_never_passes():
    check = challenge("turn_left", "blink")
    play(check, (20.0, REST))  # a face that does nothing at all
    assert check.stage is Stage.FAILED and check.failure == "late" and check.step == 0
    assert check.reason == "Your head did not turn to your left within 3 seconds."


def test_a_tilted_photograph_does_not_read_as_a_turn():
    # The largest turn read from a photograph, tilted and held as near as the camera can see it: 32.5 degrees.
    check = challenge("turn_left", "blink")
    play(check, SIT, (5.0, face(yaw=REST.yaw + 32.5)))
    assert check.stage is Stage.FAILED and check.failure == "late"
    assert liveness.TURN_DEG > 32.5


def test_the_actions_in_the_wrong_order_are_refused():
    check = challenge("turn_left", "open_mouth")
    play(check, SIT, OPEN_MOUTH)  # the second action, done first
    assert check.stage is Stage.FAILED and check.failure == "wrong"
    assert check.reason == "Your head did not turn to your left: something else was done."


def test_a_turn_the_wrong_way_is_refused():
    check = challenge("turn_right", "blink")
    play(check, SIT, TURN_LEFT)
    assert check.stage is Stage.FAILED and check.failure == "wrong"


def test_a_blink_is_never_the_wrong_answer():
    check = challenge("turn_left", "open_mouth")
    play(check, SIT, BLINK, (0.3, REST), TURN_LEFT, SIT, BLINK, (0.2, REST), OPEN_MOUTH, SIT)  # people blink all the time
    assert check.stage is Stage.PASSED


def test_each_action_has_to_come_within_three_seconds():
    check = challenge("blink", "turn_left")
    end = play(check, SIT, (2.0, REST), BLINK, SIT)  # the blink comes in time, 2.5 s after it was asked for
    assert check.stage is Stage.ACT and check.action == "turn_left"
    play(check, (3.2, REST), TURN_LEFT, start=end)  # the turn does not
    assert check.stage is Stage.FAILED and check.failure == "late" and check.step == 1


def test_something_done_before_it_is_asked_for_does_not_count():
    check = challenge("turn_left", "blink", wait=1.0)
    end = play(check, (0.3, REST), TURN_LEFT)  # a turn, before anything was asked
    assert check.stage is Stage.SETTLE and check.step == 0
    play(check, (5.0, REST), start=end)
    assert check.stage is Stage.FAILED and check.step == 0  # the turn asked for after that never came


def test_nothing_is_asked_for_until_the_face_is_at_rest_and_the_wait_is_over():
    check = challenge("turn_left", "blink", wait=1.0)
    end = play(check, (1.4, REST))  # the rest and the wait together take 1.5 s
    assert check.stage is Stage.SETTLE and check.snapshot(end)["prompt"] == "Face the camera, with your mouth closed"
    end = play(check, (0.2, REST), start=end)
    assert check.stage is Stage.ACT and check.snapshot(end)["prompt"] == "Turn your head to your left"


def test_a_face_never_seen_at_rest_is_refused():
    check = challenge("turn_left", "blink")
    play(check, (9.0, face(jaw=0.5)))  # talking all the while
    assert check.stage is Stage.FAILED and check.failure == "rest"
    assert at_rest(REST) and not at_rest(face(jaw=0.5)) and not at_rest(face(yaw=20.0)) and not at_rest(face(eyes=0.7))
    assert not at_rest(GONE)


def test_a_word_is_not_an_open_mouth_and_shut_eyes_are_not_a_blink():
    check = challenge("open_mouth", "turn_left")
    end = play(check, SIT, (0.2, face(jaw=0.8)), (0.3, REST), (0.2, face(jaw=0.8)), (0.3, REST))  # opened for a moment, twice
    assert check.stage is Stage.ACT and check.step == 0
    play(check, OPEN_MOUTH, start=end)  # held open
    assert check.step == 1

    check = challenge("blink", "turn_left")
    end = play(check, SIT, (1.5, face(eyes=0.9)), (0.2, REST))  # eyes shut for a second and a half
    assert check.stage is Stage.ACT and check.step == 0
    play(check, BLINK, (0.1, REST), start=end)
    assert check.step == 1


def test_eyes_are_measured_from_how_they_are_at_rest():
    narrow = face(eyes=0.42)  # open eyes scored up to 0.43 in the sample faces
    check = challenge("blink", "turn_left")
    end = play(check, (1.0, narrow), (0.15, face(eyes=0.55)), (0.3, narrow))  # barely more closed than at rest
    assert check.step == 0
    play(check, (0.15, face(eyes=0.8)), (0.2, narrow), start=end)
    assert check.step == 1
    assert liveness.OPEN_JAW >= 2 * liveness.REST_JAW  # a mouth wide open is far from any mouth at rest


def test_a_lost_face_ends_the_challenge_and_one_missed_frame_does_not():
    check = challenge("turn_left", "blink")
    end = play(check, SIT, (0.1, GONE), (0.3, face(yaw=20.0)), (0.1, GONE), TURN_LEFT)  # the tracker misses a frame now and then
    assert check.step == 1
    play(check, (0.7, GONE), start=end)
    assert check.stage is Stage.FAILED and check.failure == "no_face"
    assert check.reason == "Your face left the picture. Stay in view of the camera."

    check = challenge("turn_left", "blink")
    play(check, SIT, (0.7, TWO))
    assert check.stage is Stage.FAILED and check.failure == "faces"
    assert check.reason == "More than one face is in the picture. Only you should be in view."


def test_the_face_has_to_come_back_to_rest_after_every_action():
    check = challenge("turn_left", "blink")
    end = play(check, SIT, TURN_LEFT)
    assert check.stage is Stage.SETTLE and check.snapshot(end)["prompt"] == "Good. Face the camera again"
    play(check, (9.0, face(yaw=45.0)), start=end)  # the head stays turned
    assert check.stage is Stage.FAILED and check.failure == "rest" and check.step == 1


def test_a_finished_challenge_stays_as_it_ended():
    check = challenge("turn_left", "blink")
    end = play(check, SIT, TURN_LEFT, SIT, BLINK, SIT)
    assert check.done and check.stage is Stage.PASSED
    play(check, (5.0, GONE), start=end)
    assert check.stage is Stage.PASSED


def test_what_the_app_is_told():
    check = challenge("turn_left", "blink")
    end = play(check, SIT)
    told = check.snapshot(end)
    assert told["stage"] == "act" and told["action"] == "turn_left" and (told["step"], told["steps"]) == (1, 2)
    assert told["seconds"] == 3.0 and 2.3 < told["seconds_left"] <= 3.0 and told["progress"] == 0.0
    end = play(check, (0.2, face(yaw=REST.yaw + 17.5)), start=end)
    assert check.snapshot(end)["progress"] == 0.5  # half of the turn has been seen
    end = play(check, TURN_LEFT, SIT, start=end)
    told = check.snapshot(end)
    assert told["action"] == "blink" and told["step"] == 2 and told["progress"] is None
    end = play(check, (4.0, REST), start=end)
    told = check.snapshot(end)
    assert told["stage"] == "failed" and told["reason"] == "No blink was seen within 3 seconds." and told["action"] is None


def test_what_is_asked_for_is_chosen_at_random_and_always_holds_a_turn():
    rng = random.Random(7)
    picked = [pick_actions(rng) for _ in range(600)]
    assert all(len(pair) == 2 and pair[0] != pair[1] and set(pair) & set(TURNS) for pair in picked)
    counts = {pair: picked.count(pair) for pair in set(picked)}
    assert len(counts) == 10  # every order with a turn in it, and no other
    assert min(counts.values()) > 30  # and none of them rare
    waits = pick_waits(200, rng)
    assert min(waits) >= liveness.WAIT_SECONDS[0] and max(waits) <= liveness.WAIT_SECONDS[1] and len(set(waits)) > 150
    assert pick_actions(rng, count=0) == ()
    unseeded = LivenessChallenge()  # as the server makes it
    assert len(unseeded.actions) == 2 and set(unseeded.actions) & set(TURNS) and len(unseeded.waits) == 2
