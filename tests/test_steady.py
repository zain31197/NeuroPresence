"""The One Euro filter that steadies the driving signal."""

import numpy as np

from neuropresence.capture.steady import OneEuro, SteadyCrop, square_crop_at


def run(filter_, values, rate=30.0):
    return np.array([filter_(value, index / rate) for index, value in enumerate(values)])


def test_a_still_signal_with_noise_comes_out_steadier():
    # A still head as the tracker sees it: its position, in face sizes, with an error of about a third of a percent.
    noisy = 5.0 + np.random.default_rng(0).normal(0, 0.003, 200)
    held = run(OneEuro(0.3, 4.0), noisy)
    assert held[50:].std() < 0.3 * noisy[50:].std()
    assert abs(held[50:].mean() - 5.0) < 0.002  # steadier, and still in the right place


def test_fast_movement_passes_through_with_little_delay():
    ramp = np.linspace(0.0, 3.0, 60)  # three units in two seconds: fast for a signal measured in face sizes
    held = run(OneEuro(0.3, 4.0), ramp)
    assert abs(held[-1] - ramp[-1]) < 0.06  # it has caught up, where a plain low-pass at 0.3 Hz would trail far behind
    slow = run(OneEuro(0.3, 0.0), ramp)  # the same filter without the speed term
    assert abs(slow[-1] - ramp[-1]) > 5 * abs(held[-1] - ramp[-1])


def test_the_filter_does_not_depend_on_the_frame_rate():
    def settle(rate):
        f = OneEuro(1.0, 0.0)
        f(0.0, 0.0)
        return [float(f(1.0, (index + 1) / rate)) for index in range(int(rate))][-1]  # one second after a step
    assert abs(settle(30.0) - settle(7.0)) < 0.02  # the live loop runs at about 7 frames a second, clips at 30


def test_after_a_gap_the_filter_starts_again():
    f = OneEuro(0.3, 0.0)
    f(0.0, 0.0)
    assert float(f(10.0, 0.03)) < 1.0  # smoothed toward the old value
    assert float(f(10.0, 5.0)) == 10.0  # five seconds later the old value means nothing
    f.reset()
    assert float(f(3.0, 5.1)) == 3.0


def test_the_crop_is_cut_where_asked_in_fractions_of_a_pixel():
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[90:110, 140:160] = 255  # a white square in the middle
    crop = square_crop_at(frame, 150.0, 100.0, 100.0, out_size=100)
    assert crop.shape == (100, 100, 3)
    ys, xs = np.nonzero(crop[:, :, 0] > 127)
    assert abs(xs.mean() - 49.5) < 0.6 and abs(ys.mean() - 49.5) < 0.6
    shifted = square_crop_at(frame, 150.5, 100.0, 100.0, out_size=100)
    assert not np.array_equal(crop, shifted)  # half a pixel is not rounded away
    edge = square_crop_at(frame, 5.0, 5.0, 100.0, out_size=64)  # mostly outside the frame: edges are repeated, no error
    assert edge.shape == (64, 64, 3)


def test_the_steady_crop_reports_its_window_and_follows_a_moving_face():
    crop = SteadyCrop()
    frame = np.zeros((480, 640, 3), np.uint8)
    points = np.array([[200.0, 100.0], [300.0, 220.0]])
    _, first = crop(frame, points, 0.0)
    assert np.allclose(first, (130.0, 40.0, 240.0))  # twice the face (120 px), centred on it at (250, 160)
    for index in range(1, 60):  # the face walks 200 pixels to the right in two seconds
        _, window = crop(frame, points + [index * 200 / 60, 0], index / 30)
    assert abs(window[0] - (130.0 + 59 * 200 / 60)) < 12  # the window went with it
