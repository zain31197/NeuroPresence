"""Live demo of Stage 1-2: webcam -> face tracking -> driving signal.

Usage:
    python scripts/demo_capture.py              # default webcam
    python scripts/demo_capture.py --source 1   # another camera
    python scripts/demo_capture.py --source clip.mp4

Press q to quit.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuropresence.capture import FaceTracker, FrameSource, FrameSourceError, TrackStatus

GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)
BANNERS = {
    TrackStatus.NO_FACE: "NO FACE - reenactment paused",
    TrackStatus.MULTIPLE_FACES: "MULTIPLE FACES - reenactment paused",
}


def draw(frame, result, fps):
    lines = [f"fps {fps:5.1f}   tracker {result.latency_ms:5.1f} ms"]
    if result.ok:
        for x, y in result.landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, GREEN, -1)
        x, y, w, h = result.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 1)
        yaw, pitch, roll = result.pose_deg
        lines.append(f"yaw {yaw:+6.1f}  pitch {pitch:+6.1f}  roll {roll:+6.1f}")
        lines.append(f"jaw open {result.blendshapes.get('jawOpen', 0.0):.2f}")
    else:
        cv2.putText(frame, BANNERS[result.status], (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="0", help="camera index or video file path")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    try:
        frames = FrameSource(source)
    except FrameSourceError as err:
        sys.exit(f"{err}\nCheck that a webcam is connected and not in use by another app.")

    frame_times = deque(maxlen=30)
    with frames, FaceTracker() as tracker:
        while True:
            frame = frames.read()
            if frame is None:
                print("Stream ended or camera read failed.")
                break
            result = tracker.process(frame)
            frame_times.append(time.perf_counter())
            span = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / span if span > 0 else 0.0
            draw(frame, result, fps)
            cv2.imshow("NeuroPresence - capture and tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
