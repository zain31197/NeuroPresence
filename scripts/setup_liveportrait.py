"""Fetch LivePortrait's code (pinned commit) and human-face weights into third_party/."""

import subprocess
from pathlib import Path

REPO = "https://github.com/KlingAIResearch/LivePortrait.git"
COMMIT = "9b294b3d0536135442ea73cb01e6cb3ca7029dd3"
WEIGHTS_REPO = "KlingTeam/LivePortrait"
TARGET = Path(__file__).resolve().parents[1] / "third_party" / "LivePortrait"

if __name__ == "__main__":
    if not (TARGET / "src").exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO, str(TARGET)], check=True)
        subprocess.run(["git", "-C", str(TARGET), "checkout", COMMIT], check=True)
    else:
        print(f"Code already present: {TARGET}")

    from huggingface_hub import snapshot_download

    # About 660 MB. The animal models in the same repository are not needed.
    snapshot_download(
        WEIGHTS_REPO,
        local_dir=TARGET / "pretrained_weights",
        allow_patterns=["liveportrait/*", "insightface/*"],
    )
    print("LivePortrait is ready.")
