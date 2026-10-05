"""Download the MediaPipe face landmark model into models/."""

import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
TARGET = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"

if __name__ == "__main__":
    if TARGET.exists():
        print(f"Already present: {TARGET}")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, TARGET)
        print(f"Downloaded: {TARGET}")
