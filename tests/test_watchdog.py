"""The delay watchdog: when the live picture is held back for arriving too late."""

from neuropresence.server.watchdog import DelayWatchdog

LIMIT = 150.0


def run(watchdog, delays, start=0.0, every=0.05):
    """One frame every 50 ms with the given delays. Returns whether the picture was held after each."""
    return [watchdog.sample(delay, start + index * every) for index, delay in enumerate(delays)]


def test_a_session_that_keeps_time_is_never_held():
    watchdog = DelayWatchdog(LIMIT)
    assert not any(run(watchdog, [80.0, 95.0, 78.0, 102.0] * 30))


def test_single_late_frames_do_not_count():
    """Measured beside a GPU that was busy elsewhere 75% of the time: frames up to 163 ms, the mean under 140."""
    watchdog = DelayWatchdog(LIMIT)
    assert not any(run(watchdog, [130.0, 128.0, 163.0, 125.0, 135.0, 155.0, 127.0, 131.0] * 15))
    assert watchdog.mean_ms < LIMIT


def test_a_session_that_stays_late_is_held():
    watchdog = DelayWatchdog(LIMIT)
    run(watchdog, [80.0] * 60)
    held = run(watchdog, [170.0] * 60, start=3.0)
    assert held[-1] is True
    first = held.index(True) * 0.05
    assert 0.5 <= first <= 2.0  # not on the first late frame, and within two seconds


def test_it_is_released_once_the_delay_has_stayed_down():
    watchdog = DelayWatchdog(LIMIT)
    run(watchdog, [170.0] * 60)
    assert watchdog.held
    held = run(watchdog, [85.0] * 80, start=3.0)
    assert held[0] is True  # not on the first good frame
    assert held[-1] is False
    assert watchdog.mean_ms < 0.9 * LIMIT


def test_a_delay_that_hovers_at_the_limit_does_not_switch_back_and_forth():
    watchdog = DelayWatchdog(LIMIT)
    run(watchdog, [170.0] * 60)
    # Back under the limit, but only just: the live picture stays held.
    assert all(run(watchdog, [142.0] * 100, start=3.0))
    # And a session that was never held is not held at the same delay.
    fresh = DelayWatchdog(LIMIT)
    assert not any(run(fresh, [142.0] * 100))


def test_a_few_frames_decide_nothing():
    watchdog = DelayWatchdog(LIMIT)
    assert run(watchdog, [400.0, 400.0, 400.0]) == [False, False, False]
    assert watchdog.mean_ms is None


def test_reset_forgets_everything():
    watchdog = DelayWatchdog(LIMIT)
    run(watchdog, [170.0] * 60)
    watchdog.reset()
    assert watchdog.held is False and watchdog.mean_ms is None
