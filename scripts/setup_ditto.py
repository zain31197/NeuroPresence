"""Fetch Ditto's code (pinned commit) and its PyTorch weights into third_party/, for the voice driven picture.

Ditto (Ant Group, ACM MM 2025, Apache-2.0) turns speech into LivePortrait's own motion: a HuBERT
speech encoder and a diffusion model that writes expression keypoints, head rotation and shift.
Only the PyTorch weights and their configuration are fetched. Its ready-made TensorRT engines are
built for older graphics cards with TensorRT 8.6, and its ONNX files are only the source for those.
"""

import subprocess
from pathlib import Path

REPO = "https://github.com/antgroup/ditto-talkinghead.git"
COMMIT = "c3e47eee2e626500017a0556b470d6d4182f85e8"
WEIGHTS_REPO = "digital-avatar/ditto-talkinghead"
TARGET = Path(__file__).resolve().parents[1] / "third_party" / "ditto"

if __name__ == "__main__":
    if not (TARGET / "core").exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO, str(TARGET)], check=True)
        subprocess.run(["git", "-C", str(TARGET), "checkout", COMMIT], check=True)
    else:
        print(f"Code already present: {TARGET}")

    from huggingface_hub import snapshot_download

    snapshot_download(
        WEIGHTS_REPO,
        local_dir=TARGET / "checkpoints",
        allow_patterns=["ditto_pytorch/*", "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"],
    )
    print("Ditto is ready.")
