import time

import cv2
import numpy as np
import pytest

from neuropresence.capture import FrameSourceError
from neuropresence.capture.feed import LiveFeed

CLIP_FRAMES = 12
CLIP_FPS = 60.0


@pytest.fixture
def clip(tmp_path):
    """A short clip whose frame number is written into its brightness."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), CLIP_FPS, (64, 48))
    for i in range(CLIP_FRAMES):
        writer.write(np.full((48, 64, 3), 20 * i, dtype=np.uint8))
    writer.release()
    return str(path)


def test_frames_arrive_in_order_at_the_clip_rate(clip):
    with LiveFeed(clip) as feed:
        assert not feed.is_camera
        started = time.perf_counter()
        last, seen = -1, []
        while len(seen) < 6:
            item = feed.read(after_index=last, timeout=1.0)
            assert item is not None
            assert item.index > last
            last = item.index
            seen.append(item)
        elapsed = time.perf_counter() - started
    assert seen[0].image.shape == (48, 64, 3)
    assert elapsed > 4 / CLIP_FPS  # paced like a camera, not read as fast as the disk allows
    assert all(a.captured_at < b.captured_at for a, b in zip(seen, seen[1:]))


def test_a_slow_reader_gets_the_newest_frame_and_the_rest_are_dropped(clip):
    with LiveFeed(clip) as feed:
        first = feed.read(timeout=1.0)
        time.sleep(5 / CLIP_FPS)  # the reader is busy for about five frames
        newest = feed.read(after_index=first.index, timeout=1.0)
    assert newest.index - first.index >= 3  # the frames in between were skipped, not queued


def test_a_clip_starts_again_from_the_top(clip):
    with LiveFeed(clip) as feed:
        item = feed.read(timeout=1.0)
        while item.index < CLIP_FRAMES + 2:
            item = feed.read(after_index=item.index, timeout=1.0)
            assert item is not None
        assert not feed.ended


def test_without_looping_the_feed_ends(clip):
    with LiveFeed(clip, loop=False) as feed:
        last = -1
        while (item := feed.read(after_index=last, timeout=1.0)) is not None:
            last = item.index
        assert feed.ended
        assert last == CLIP_FRAMES - 1


def test_read_times_out_when_nothing_new_arrives(clip):
    with LiveFeed(clip, loop=False) as feed:
        while feed.read(after_index=-1, timeout=1.0) is None:
            pass
        started = time.perf_counter()
        assert feed.read(after_index=10_000, timeout=0.2) is None
        assert time.perf_counter() - started < 1.0


def test_missing_clip_is_reported():
    with pytest.raises(FrameSourceError):
        LiveFeed("does_not_exist.mp4")
