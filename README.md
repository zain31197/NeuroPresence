# NeuroPresence

Identity-preserving, latency-bounded neural face reenactment for consent-gated professional telepresence.

Final Year Project, BS Data Science, FAST School of Computing (NUCES Islamabad), Fall 2026 – Spring 2027.

## What it does

The user records one presentable source clip of themselves. During a meeting, their live webcam (head pose, lip motion, expression) drives that clip in near real time, and the reenacted video is delivered to any meeting application through a virtual camera. The system animates only the user's own enrolled likeness and marks its output as synthetic.

## Pipeline

1. **Capture** – webcam frames, face detection and landmarks (MediaPipe FaceMesh)
2. **Motion encoding** – head pose, lip motion, and expression as a compact driving signal
3. **Reenactment** – the driving signal animates the source clip (LivePortrait core)
4. **Identity preservation** – output anchored to the enrolled face embedding (ArcFace)
5. **Consent and disclosure** – self-likeness gate and synthetic-media watermark
6. **Streaming** – composited frames written to a virtual camera (v4l2loopback)

## Targets

| Attribute | Target |
|---|---|
| Per-frame reenactment compute | ≤ 42 ms |
| Throughput | ≥ 24 fps |
| End-to-end latency | ≤ 150 ms |
| Peak GPU memory | ≤ 8 GB |
| Identity similarity (CSIM) | ≥ 0.80 |
| Consent gate true-accept | ≥ 95% |

Target hardware: NVIDIA RTX 5050 (8 GB). The code is developed and run on both Windows 11 and Kubuntu Linux; the virtual camera uses OBS Virtual Camera on Windows and v4l2loopback on Linux.

## Baseline (measured, unoptimized LivePortrait on the target GPU)

| Measurement | Result |
|---|---|
| Mean per-frame latency | 130.1 ms (p95 133.9, p99 138.7) |
| Throughput | 7.7 fps |
| Peak VRAM | 1.57 GB |

Throughput, not memory, is the binding constraint; closing the roughly 3x gap is the main optimization work.

## Team

| Member | Ownership |
|---|---|
| Muhammad Talha Arshad | Reenactment core, real-time optimization, identity-preservation module |
| Zain Shahid | Capture, tracking, motion encoding, temporal stability |
| Sana Ullah Farooqi | Virtual-camera integration, consent and disclosure safeguards, evaluation harness |

Supervisor: Muhammad Aamir Gulzar

## Status

| Stage | State |
|---|---|
| 1–2 Capture, tracking, driving signal | Working (`neuropresence/capture`) |
| 3 Reenactment | Working offline from a video file (`neuropresence/reenactment`); live loop not started |
| 4 Identity preservation | Not started |
| 5 Consent and disclosure | Not started |
| 6 Virtual camera | Not started |

## Setup

Requires Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/download_models.py        # face landmark model, 4 MB
python scripts/setup_liveportrait.py     # LivePortrait code and weights, about 700 MB
```

## Run

```
python scripts/demo_capture.py                 # live webcam, press q to quit
python scripts/demo_capture.py --source clip.mp4
python scripts/reenact_video.py --source me.jpg --driving clip.mp4 --out out.mp4
python -m pytest
```

`reenact_video.py` animates the source image with the motion in the driving video, writes a side-by-side result, and prints per-stage latency, frame rate, and peak GPU memory. The current measurement on Windows is in `results/baseline_windows_rtx5050.json`: 7.9 fps and 126 ms per frame, of which 93 ms is the warp-and-decode step.

The demo overlays face landmarks, head pose (yaw, pitch, roll), jaw opening, frame rate, and tracker latency. With no face or more than one face in frame it shows a banner and reports the frame as not usable for reenactment.

## Third-party components and licensing

This is a research prototype. It builds on LivePortrait, InsightFace/ArcFace, MediaPipe, Wav2Lip/SyncNet, and F5-TTS, several of which are released for research or non-commercial use only. It is not licensed for commercial deployment.

## Responsible use

NeuroPresence is intended for consented self-presentation only. It must not be used to animate another person's likeness or to represent a user as present when they are not.
