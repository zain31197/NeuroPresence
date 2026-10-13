"""Record a minute of your own camera, to tune and measure the system on a real webcam.

Every limit and filter setting so far was set on studio sample clips. A webcam in a room has
dimmer light, more grain and a softer picture, so the settings have to be checked on that too.
This script records you while a window tells you what to do, and saves the video on this
computer only, in data/recordings/, which git ignores. Nothing is uploaded.

    python scripts/record_camera.py                  # camera 0, saved as data/recordings/camera.avi
    python scripts/record_camera.py --name zain      # a name of your own
    python scripts/record_camera.py --camera 1       # another camera

Stop any running session first: a camera can be open in one program at a time.
Press Q in the window to stop early.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FrameSource, FrameSourceError  # noqa: E402

OUT = ROOT / "data" / "recordings"

# What to do, and for how long. Together about a minute: rest first, because everything is measured from it.
STEPS = [
    (4, "Get ready: sit as you would in a meeting"),
    (10, "Sit still and look at the camera, mouth closed"),
    (14, "Talk normally (count aloud, or say anything)"),
    (8, "Nod a few times, then shake your head slowly"),
    (8, "Turn your head slowly left, then right, then back"),
    (8, "Look down at the screen, then back at the camera"),
    (8, "Sit still again and look at the camera"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--name", default="camera", help="file name, without extension")
    args = parser.parse_args()

    try:
        source = FrameSource(args.camera, 1280, 720)
    except FrameSourceError as error:
        raise SystemExit(f"{error}\nStop any running session, and close other programs that use the camera.")
    OUT.mkdir(parents=True, exist_ok=True)
    video_path, times_path = OUT / f"{args.name}.avi", OUT / f"{args.name}.json"

    first = source.read()
    if first is None:
        raise SystemExit("The camera opened but delivered no picture.")
    height, width = first.shape[:2]
    # Motion JPEG keeps each frame as it came: the measurements should see the camera's grain, not a codec's.
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not write {video_path}.")

    total = sum(seconds for seconds, _ in STEPS)
    began = time.perf_counter()
    times, frame = [], first
    while frame is not None:
        at = time.perf_counter() - began
        if at >= total:
            break
        elapsed, step_text, left = 0.0, STEPS[-1][1], 0.0
        for seconds, text in STEPS:
            if at < elapsed + seconds:
                step_text, left = text, elapsed + seconds - at
                break
            elapsed += seconds
        recording = at >= STEPS[0][0]
        if recording:
            writer.write(frame)
            times.append(round(at - STEPS[0][0], 4))

        shown = cv2.flip(frame, 1)  # mirrored, as in the app
        cv2.rectangle(shown, (0, height - 92), (width, height), (20, 16, 12), -1)
        cv2.putText(shown, step_text, (24, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(shown, f"{left:4.1f} s   {'REC' if recording else 'starting'}   total {max(0.0, total - at):4.1f} s left   Q stops",
                    (24, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 235, 90) if recording else (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("NeuroPresence: recording your camera", shown)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
        frame = source.read()

    writer.release()
    source.release()
    cv2.destroyAllWindows()
    if len(times) < 60:
        raise SystemExit("Too little was recorded to be of use. Run it again and let it finish.")
    rate = (len(times) - 1) / (times[-1] - times[0])
    times_path.write_text(json.dumps({"camera": args.camera, "size": [width, height], "frames": len(times),
                                      "camera_fps": round(rate, 2), "times_s": times,
                                      "steps": [{"seconds": seconds, "asked": text} for seconds, text in STEPS[1:]]}))
    print(f"Saved {len(times)} frames, {times[-1]:.0f} s, at {rate:.1f} frames a second from the camera.")
    print(f"  {video_path.relative_to(ROOT)}")
    print(f"  {times_path.relative_to(ROOT)}")
    print("These stay on this computer: data/ is not part of the repository.")


if __name__ == "__main__":
    main()
