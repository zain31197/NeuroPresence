"""Download the small models the project needs into models/.

- MediaPipe face landmark model (4 MB), used by the face tracker.
- ArcFace w600k_r50 (166 MB), used for identity similarity. It is taken from
  InsightFace's official buffalo_l pack (a 275 MB download) and is licensed
  for non-commercial research use only.
"""

import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER = MODELS / "face_landmarker.task"

ARCFACE_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE_MEMBER = "w600k_r50.onnx"
ARCFACE = MODELS / "arcface" / ARCFACE_MEMBER
ARCFACE_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_landmarker():
    if LANDMARKER.exists():
        print(f"Already present: {LANDMARKER}")
        return
    LANDMARKER.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER)
    print(f"Downloaded: {LANDMARKER}")


def fetch_arcface():
    if ARCFACE.exists() and sha256(ARCFACE) == ARCFACE_SHA256:
        print(f"Already present: {ARCFACE}")
        return
    ARCFACE.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the InsightFace model pack (275 MB) ...")
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "buffalo_l.zip"
        urllib.request.urlretrieve(ARCFACE_PACK_URL, pack)
        with zipfile.ZipFile(pack) as archive, open(ARCFACE, "wb") as target:
            target.write(archive.read(ARCFACE_MEMBER))
    if sha256(ARCFACE) != ARCFACE_SHA256:
        ARCFACE.unlink()
        raise SystemExit("The ArcFace model does not match its expected checksum; download it again.")
    print(f"Downloaded: {ARCFACE}")


if __name__ == "__main__":
    fetch_landmarker()
    fetch_arcface()
