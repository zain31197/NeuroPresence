"""The identity guard: what a series of identity readings leads to, one level at a time."""

from neuropresence.identity import GuardState, IdentityGuard
from neuropresence.identity.guard import FOLLOW_STEP, LOW_CSIM, RETRY_SECONDS

EVERY = 0.5  # the live monitor takes a reading twice a second


def feed(guard, scores, start=0.0):
    """Give the guard one reading every half second. Returns what it asked for, as (time, action)."""
    asked = []
    for index, score in enumerate(scores):
        at = start + index * EVERY
        action = guard.sample(score, at)
        if action is not None:
            asked.append((at, action))
    return asked


def actions(asked):
    return [action for _, action in asked]


def test_ordinary_readings_lead_to_nothing():
    guard = IdentityGuard()
    # What normal use looks like in the study: around 0.9, a few frames just under 0.8, none under 0.7.
    assert feed(guard, [0.92, 0.88, 0.79, 0.90, 0.72, 0.91, 0.85, 0.78, 0.93, 0.90] * 6) == []
    assert guard.state is GuardState.STEADY


def test_a_short_dip_is_ignored_but_seen():
    guard = IdentityGuard()
    assert feed(guard, [0.9] * 8) == []
    assert feed(guard, [0.6, 0.6], start=4.0) == []  # a second below the level: the mean of three seconds is still fine
    assert guard.state is GuardState.DIPPING
    assert feed(guard, [0.9] * 8, start=5.0) == []
    assert guard.state is GuardState.STEADY


def test_a_lasting_drop_takes_a_fresh_neutral_pose_once():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    asked = feed(guard, [0.6] * 5, start=4.0)
    assert actions(asked) == ["anchor"]
    assert asked[0][0] - 4.0 <= 2.5  # within a few seconds of the drop, not at once and not much later
    assert guard.state is GuardState.ANCHORING
    assert guard.mean < LOW_CSIM


def test_the_fresh_neutral_pose_helped():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    feed(guard, [0.6] * 5, start=4.0)
    assert actions(feed(guard, [0.9] * 6, start=10.0)) == ["recovered"]
    assert guard.state is GuardState.STEADY


def test_the_fresh_neutral_pose_did_not_help_so_the_still_picture_is_shown():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    feed(guard, [0.6] * 5, start=4.0)
    assert actions(feed(guard, [0.62] * 6, start=10.0)) == ["fallback"]
    assert guard.state is GuardState.FALLBACK
    assert feed(guard, [0.95] * 20, start=20.0) == []  # and it stays there: only the person resumes


def test_a_collapse_does_not_wait_for_the_mean():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    assert feed(guard, [0.3] + [0.9] * 7, start=4.0) == []  # one reading alone is not a collapse
    asked = feed(guard, [0.3, 0.3], start=8.0)
    assert asked == [(8.5, "fallback")]  # the second one in a row is, half a second later
    assert guard.state is GuardState.FALLBACK


def test_an_output_with_no_face_in_it_is_the_lowest_reading_there_is():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    assert actions(feed(guard, [None, None], start=4.0)) == ["fallback"]


def test_a_second_drop_soon_after_goes_straight_to_the_still_picture():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    feed(guard, [0.6] * 5, start=4.0)
    assert actions(feed(guard, [0.9] * 6, start=10.0)) == ["recovered"]
    assert actions(feed(guard, [0.6] * 8, start=13.0)) == ["fallback"]  # the fresh neutral pose was not the cure


def test_a_drop_long_after_an_anchor_gets_a_fresh_neutral_pose_again():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    feed(guard, [0.6] * 5, start=4.0)
    feed(guard, [0.9] * 6, start=10.0)
    later = 10.0 + RETRY_SECONDS + 10.0
    feed(guard, [0.9] * 8, start=later)
    assert actions(feed(guard, [0.6] * 5, start=later + 4.0)) == ["anchor"]


def test_a_step_down_after_an_anchor_that_worked_takes_the_neutral_pose_once_more():
    """Seen live: a neutral pose taken while the mouth was covered. The output matched while the
    cover lasted (0.94) and was off once it was gone (0.80), because the cover was in the neutral pose."""
    guard = IdentityGuard()
    feed(guard, [0.86] * 8)
    assert actions(feed(guard, [0.72] * 6, start=4.0)) == ["anchor"] and guard.reason == "low"
    assert actions(feed(guard, [0.94] * 6, start=8.0)) == ["recovered"]
    asked = feed(guard, [0.80] * 8, start=11.0)  # still above the level, but well under what the anchor reached
    assert actions(asked) == ["anchor"] and guard.reason == "changed"
    assert actions(feed(guard, [0.86] * 6, start=16.0)) == ["recovered"]
    # Only once: the second neutral pose is not watched, or two of them could chase each other.
    assert feed(guard, [0.86 - FOLLOW_STEP - 0.02] * 12, start=19.0) == []
    assert guard.state is GuardState.STEADY


def test_ordinary_variation_after_an_anchor_does_not_take_it_again():
    guard = IdentityGuard()
    feed(guard, [0.86] * 8)
    feed(guard, [0.72] * 6, start=4.0)
    assert actions(feed(guard, [0.88] * 6, start=8.0)) == ["recovered"]
    assert feed(guard, [0.84, 0.90, 0.83, 0.88, 0.85, 0.82, 0.87, 0.86] * 3, start=11.0) == []


def test_without_an_anchor_a_step_down_is_just_variation():
    guard = IdentityGuard()
    feed(guard, [0.95] * 8)
    assert feed(guard, [0.80] * 12, start=4.0) == []  # a calm start and then talking: nothing to do


def test_resuming_gives_the_output_a_few_seconds_to_prove_itself():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    feed(guard, [0.2, 0.2], start=4.0)
    assert guard.state is GuardState.FALLBACK
    guard.resume(20.0)
    assert guard.state is GuardState.ANCHORING
    assert actions(feed(guard, [0.9] * 6, start=20.5)) == ["recovered"]

    guard.resume(40.0)  # resumed again, and this time the output is still wrong
    assert actions(feed(guard, [0.6] * 6, start=40.5)) == ["fallback"]


def test_a_few_readings_decide_nothing():
    guard = IdentityGuard()
    assert feed(guard, [0.6, 0.6, 0.6]) == []  # a second and a half into a session is too early to say
    assert guard.state is GuardState.DIPPING


def test_reset_forgets_everything():
    guard = IdentityGuard()
    feed(guard, [0.9] * 8)
    feed(guard, [0.2, 0.2], start=4.0)
    guard.reset()
    assert guard.state is GuardState.STEADY and guard.mean is None
    assert feed(guard, [0.9] * 8, start=100.0) == []


def test_readings_carry_their_own_time():
    """The same readings four times a second lead to the same decision: the guard counts seconds, not readings."""
    slow, fast = IdentityGuard(), IdentityGuard()
    feed(slow, [0.9] * 8)
    for index in range(16):
        fast.sample(0.9, index * 0.25)
    asked_slow = feed(slow, [0.6] * 6, start=4.0)
    asked_fast = [(4.0 + index * 0.25, fast.sample(0.6, 4.0 + index * 0.25)) for index in range(12)]
    asked_fast = [(at, action) for at, action in asked_fast if action]
    assert actions(asked_slow) == actions(asked_fast) == ["anchor"]
    assert abs(asked_slow[0][0] - asked_fast[0][0]) <= 0.5
