"""Live reenactment: webcam -> tracker -> LivePortrait -> preview window.

Usage:
    python scripts/live.py                          # source captured from the camera
    python scripts/live.py --source-image me.jpg
    python scripts/live.py --source-image me.jpg --camera 1
    python scripts/live.py --source-image me.jpg --camera clip.mp4 --record out.mp4

Keys: q quits, r re-captures the neutral pose (look straight, relaxed face, press r).
The left panel is the camera with metrics; the right panel is the output.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuropresence.capture import FaceTracker, FrameSource, FrameSourceError, TrackStatus
from neuropresence.pipeline import Pipeline
from neuropresence.reenactment import ReenactmentEngine, SourceError

PANEL = 512
WHITE, RED = (255, 255, 255), (0, 0, 255)
BANNERS = {
    TrackStatus.NO_FACE: "NO FACE - showing enrolled frame",
    TrackStatus.MULTIPLE_FACES: "MULTIPLE FACES - showing enrolled frame",
}


def letterbox(frame, size=PANEL):
    """Fit the frame inside a size x size panel without distorting it."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    panel[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return panel


def draw_metrics(panel, result, fps, vram_gb):
    t = result.timing_ms
    lines = [f"fps {fps:5.1f}   total {t['total']:6.1f} ms", f"tracker {t['tracker']:5.1f} ms"]
    if result.live:
        lines.append(f"motion {t['motion']:5.1f} ms   render {t['render']:5.1f} ms")
    lines.append(f"GPU memory {vram_gb:.2f} GB")
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
    if not result.live:
        cv2.putText(panel, BANNERS[result.status], (8, PANEL - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)


def capture_source(frames, tracker, max_frames=150, max_angle_deg=20):
    """Take the source from the camera: the first frame with one roughly frontal face."""
    for _ in range(max_frames):
        frame = frames.read()
        if frame is None:
            break
        track = tracker.process(frame)
        if track.ok and all(abs(a) <= max_angle_deg for a in track.pose_deg):
            return frame
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-image", help="photo of the user to animate; "
                        "if omitted, the source is captured from the camera at start")
    parser.add_argument("--camera", default="0", help="camera index or a video file path")
    parser.add_argument("--record", help="also save the preview to this video file")
    parser.add_argument("--no-window", action="store_true", help="run without a preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames")
    args = parser.parse_args()
    camera = int(args.camera) if args.camera.isdigit() else args.camera

    try:
        frames = FrameSource(camera, width=1280, height=720)
    except FrameSourceError as err:
        sys.exit(f"{err}\nCheck that a webcam is connected and not in use by another app.")
    if args.source_image:
        source = cv2.imread(args.source_image)
        if source is None:
            sys.exit(f"Could not read source image: {args.source_image}")
    else:
        print("Capturing the source from the camera: look straight at it with a relaxed face.")
        with FaceTracker() as enrol_tracker:
            source = capture_source(frames, enrol_tracker)
        if source is None:
            sys.exit("No single front-facing face seen by the camera. Try again or pass --source-image.")
    engine = ReenactmentEngine()
    try:
        engine.set_source(source)
    except SourceError as err:
        sys.exit(f"{err} Use a clear, front-facing photo.")

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (PANEL * 2, PANEL))
    frame_times = deque(maxlen=30)
    count = live_count = 0
    with frames, FaceTracker() as tracker:
        pipeline = Pipeline(tracker, engine)
        while True:
            frame = frames.read()
            if frame is None:
                print("Stream ended or camera read failed.")
                break
            result = pipeline.step(frame)
            count += 1
            live_count += result.live
            frame_times.append(time.perf_counter())
            span = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / span if span > 0 else 0.0

            left = letterbox(frame)
            draw_metrics(left, result, fps, engine.peak_vram_gb())
            preview = np.hstack([left, result.output])
            if writer is not None:
                writer.write(preview)
            if not args.no_window:
                cv2.imshow("NeuroPresence - live", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    engine.reset_reference()
            if args.max_frames and count >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Frames: {count}, reenacted: {live_count}, fallback: {count - live_count}, last fps: {fps:.1f}")


if __name__ == "__main__":
    main()
