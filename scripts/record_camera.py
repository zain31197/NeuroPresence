"""Record a minute of your own camera, to tune and measure the system on a real webcam.

Every limit and filter setting so far was set on studio sample clips. A webcam in a room has
dimmer light, more grain and a softer picture, so the settings have to be checked on that too.
This script records you while a window tells you what to do, and saves the video on this
computer only, in data/recordings/, which git ignores. Nothing is uploaded.

    python scripts/record_camera.py                  # camera 0, saved as data/recordings/camera.avi
    python scripts/record_camera.py --name zain      # a name of your own
    python scripts/record_camera.py --camera 1       # another camera
    python scripts/record_camera.py --speech --name zain_speech
                                                     # speaking, with the microphone recorded too (to <name>.wav),
                                                     # to measure the mouth driven by the voice against the real one
    python scripts/record_camera.py --speech --microphone HyperX   # a microphone of your choice, by part of its name

Stop any running session first: a camera can be open in one program at a time.
Press Q in the window to stop early.
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np

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

# With --speech: speaking instead of moving. The sentences are full of lips that meet (b, m, p), which
# is what loudness alone cannot see, and the quiet stretches show whether the mouth stays shut.
SPEECH_STEPS = [
    (4, "Get ready: sit as you would in a meeting"),
    (6, "Stay quiet, mouth closed"),
    (10, "Say slowly: a, e, i, o, u ... then again: a, e, i, o, u"),
    (12, "Say: Bob bought a big blue map for my mum"),
    (12, "Say: Peter met Pam by the main pool on Monday"),
    (20, "Talk freely, as you would in a meeting"),
    (6, "Stay quiet again, mouth closed"),
]
SOUND_RATE = 16000


class Microphone:
    """The microphone, recorded alongside the camera. Each block is kept with the moment it arrived."""

    def __init__(self, part_of_name=None):
        import sounddevice as sd

        from neuropresence.server.virtual_microphone import list_devices

        found = list_devices()
        names = found["microphones"]
        if part_of_name:
            matches = [name for name in names if part_of_name.lower() in name.lower()]
            if not matches:
                raise SystemExit(f"No microphone matches \"{part_of_name}\". There are: {', '.join(names)}")
            self.name = matches[0]
        else:
            self.name = found["default"] or (names[0] if names else None)
        if self.name is None:
            raise SystemExit("No microphone was found.")
        index = next(i for i, d in enumerate(sd.query_devices()) if d["name"] == self.name and d["max_input_channels"] > 0)
        extra = sd.WasapiSettings(auto_convert=True) if sys.platform == "win32" else None
        self.blocks = []
        self._stream = sd.InputStream(device=index, channels=1, samplerate=SOUND_RATE, dtype="float32",
                                      callback=self._heard, extra_settings=extra)

    def _heard(self, data, frames, moment, status):
        # When the first sample of the block was heard: now, less the block's length.
        self.blocks.append((time.perf_counter() - frames / SOUND_RATE, data[:, 0].copy()))

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def save(self, path, start, seconds):
        """The sound from `start` (a perf_counter moment) for `seconds`, as 16 bit mono WAV."""
        if not self.blocks:
            raise SystemExit("The microphone delivered no sound.")
        first = self.blocks[0][0]
        sound = np.concatenate([block for _, block in self.blocks])
        skip = int(round((start - first) * SOUND_RATE))
        if skip < 0:  # the sound began after the recording: pad its start with silence
            sound, skip = np.concatenate([np.zeros(-skip, np.float32), sound]), 0
        sound = sound[skip:skip + int(seconds * SOUND_RATE)]
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SOUND_RATE)
            out.writeframes((np.clip(sound, -1, 1) * 32767).astype(np.int16).tobytes())
        return len(sound) / SOUND_RATE


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--name", default="camera", help="file name, without extension")
    parser.add_argument("--speech", action="store_true", help="speaking instead of moving, with the microphone recorded")
    parser.add_argument("--microphone", default=None, help="with --speech: part of the microphone's name")
    args = parser.parse_args()
    steps = SPEECH_STEPS if args.speech else STEPS
    microphone = Microphone(args.microphone) if args.speech else None

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

    total = sum(seconds for seconds, _ in steps)
    if microphone is not None:
        microphone.start()
        print(f"Recording the microphone \"{microphone.name}\" too.")
    began = time.perf_counter()
    times, frame = [], first
    while frame is not None:
        at = time.perf_counter() - began
        if at >= total:
            break
        elapsed, step_text, left = 0.0, steps[-1][1], 0.0
        for seconds, text in steps:
            if at < elapsed + seconds:
                step_text, left = text, elapsed + seconds - at
                break
            elapsed += seconds
        recording = at >= steps[0][0]
        if recording:
            writer.write(frame)
            times.append(round(at - steps[0][0], 4))

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
    if microphone is not None:
        microphone.stop()
    cv2.destroyAllWindows()
    if len(times) < 60:
        raise SystemExit("Too little was recorded to be of use. Run it again and let it finish.")
    rate = (len(times) - 1) / (times[-1] - times[0])
    sound = None
    if microphone is not None:
        sound_path = OUT / f"{args.name}.wav"
        # The sound starts where the first recorded frame does: both are measured on the same clock.
        seconds = microphone.save(sound_path, began + steps[0][0] + times[0], times[-1] - times[0] + 1 / rate)
        sound = {"file": sound_path.name, "rate": SOUND_RATE, "seconds": round(seconds, 2), "microphone": microphone.name,
                 "starts_at_frame": 0}
    times_path.write_text(json.dumps({"camera": args.camera, "size": [width, height], "frames": len(times),
                                      "camera_fps": round(rate, 2), "times_s": times,
                                      "steps": [{"seconds": seconds, "asked": text} for seconds, text in steps[1:]],
                                      **({"sound": sound} if microphone is not None else {})}))
    print(f"Saved {len(times)} frames, {times[-1]:.0f} s, at {rate:.1f} frames a second from the camera.")
    print(f"  {video_path.relative_to(ROOT)}")
    print(f"  {times_path.relative_to(ROOT)}")
    if sound is not None:
        print(f"  {(OUT / sound['file']).relative_to(ROOT)}  ({sound['seconds']:.0f} s of sound from {sound['microphone']})")
    print("These stay on this computer: data/ is not part of the repository.")


if __name__ == "__main__":
    main()
