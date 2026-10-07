# NeuroPresence

Identity-preserving, latency-bounded neural face reenactment for consent-gated professional telepresence.

Final Year Project, BS Data Science, FAST School of Computing (NUCES Islamabad), Fall 2026 – Spring 2027.

## What it does

The user enrols one presentable picture of themselves. During a meeting, their live webcam (head pose, lip motion, expression) drives that picture in near real time, and the reenacted video is delivered to any meeting application through a virtual camera. The system animates only the user's own enrolled likeness and marks its output as synthetic.

## Pipeline

Before any session, once: **Enrolment** – one picture of the user, checked for sharpness, light, framing and a relaxed, front-facing face. Then, for every camera frame:

1. **Capture** – webcam frames, face detection and landmarks (MediaPipe FaceMesh)
2. **Motion encoding** – head pose, lip motion, and expression as a compact driving signal
3. **Reenactment** – the driving signal animates the enrolled picture (LivePortrait core)
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

## Baseline (measured, unoptimized, RTX 5050 on Windows 11)

Mean over the eight clips in `benchmarks/samples.json` (LivePortrait's sample videos, up to 150 frames each; seven are self-reenactment, where the source is a frame of the driving video). Full results: `results/benchmark_baseline_windows_rtx5050.json`.

| Measurement | Baseline | Target |
|---|---|---|
| Frame rate | 7.5 fps | ≥ 24 fps |
| Render (warp and decode) per frame | 94 ms | ≤ 42 ms |
| Whole pipeline per frame | 134 ms | ≤ 150 ms end to end |
| Peak GPU memory | 1.17 GB | ≤ 8 GB |
| Identity similarity (CSIM), self-reenactment | 0.896 | ≥ 0.80 |
| CSIM of the real driving video against the same source frame | 0.850 | reference |
| Head-pose error | 0.5° while the driver moves 1.4° on average | lower is better |
| Expression error (blendshape units) | 0.026 while the driver moves 0.072 | lower is better |
| Mouth-opening correlation with the driver | 0.96 | higher is better |
| Lag behind the driver | 0.0 frames | lower is better |
| Flicker (warping error) relative to real video | 0.89x | at most 1x |
| Head jitter relative to real video | 1.49x | at most 1x |

What this says: speed is about three times short of the target and is the main optimization work; identity is already above its target and has to stay there as the pipeline is optimized; the output follows the driver with no measurable lag; it flickers no more than real video but its head position trembles about 1.5 times as much, which is the first quality defect to fix.

These clips are studio recordings, not webcam footage of the team, so the numbers will be repeated on the team's own recordings. The proposal-stage measurement of the same unoptimized model was 7.7 fps and 130 ms per frame.

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
| Enrolment | Working: the face is verified live with the camera and only its signature is kept; the meeting picture is uploaded, checked, and accepted only if it shows the same face (`neuropresence/enrolment`) |
| 1–2 Capture, tracking, driving signal | Working (`neuropresence/capture`) |
| 3 Reenactment | Working live in a preview window (`neuropresence/reenactment`, `neuropresence/pipeline.py`) |
| 4 Identity preservation | Identity similarity (CSIM) is measured (`neuropresence/identity`); the live monitor and fallback are not started |
| 5 Consent and disclosure | Partly built: a meeting picture must match the face verified live. Checking the live face before each session, and the mark on the output, are not started |
| 6 Virtual camera | Not started |
| Evaluation | Benchmark of speed, identity and temporal stability (`scripts/benchmark.py`, `neuropresence/evaluation`) |
| Web app | Landing page, Enrolment and Live Studio (`web/`, `neuropresence/server`); Test Lab, Benchmarks and System screens come with the features they belong to |

## Setup

Requires Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/download_models.py        # face landmark model (4 MB) and ArcFace (275 MB download)
python scripts/setup_liveportrait.py     # LivePortrait code and weights, about 700 MB
```

The first benchmark run also downloads the RAFT optical-flow weights (21 MB) through torchvision.

## Web app

The web app is the place to run the system and to test each feature. It has a landing page and two screens, in the order they are used:

- **Enrolment**: verify your face with the camera, then upload the picture people will see. The camera is shown with a face outline and seven checks that update as you move; the picture tells you the one thing to fix next. An uploaded picture is refused if it is not of the same person.
- **Live Studio**: camera and output side by side, the live figures against their targets, where each frame's time goes, an event list, and a switch for every feature.

Build it once (this needs Node.js; it was built and tested with version 24), then start the engine:

```
cd web
npm install
npm run build
cd ..
python -m neuropresence.server --open
```

It opens at http://127.0.0.1:8000. The engine listens on this computer only, and the enrolled picture is kept in `data/enrolment/`, which git ignores. A camera session animates the enrolled picture, so enrol one first. For repeatable tests a session can also be driven from one of the sample clips; a clip animates one of its own frames and enrols nothing.

Two options help when testing: `--camera-file clip.mp4` plays a video file in place of every camera, so the Enrolment screen and camera sessions can be tried without a webcam, and `--data-dir folder` keeps the enrolled picture somewhere other than `data/`.

While working on the front end, run the engine as above and, in a second terminal, `npm run dev` inside `web/`; the page is then at http://localhost:5173 and reloads as you edit.

Every figure in the app comes from the running pipeline or from a saved result file. A new feature gets its own switch in the Live Studio by adding one entry to `neuropresence/server/features.py`.

## Enrolment

Enrolment has two steps, and they answer two different questions.

1. **Who are you? Verify your face with the camera.** The camera runs with seven checks on every frame. The picture is taken after a count of three, as the sharpest open-eyed frame of a half-second burst. It is used only to make a face signature (an ArcFace embedding, 512 numbers) and is not stored.
2. **What should people see? Upload your meeting picture.** A picture you choose, with the light and background you want. It goes through the same seven checks and one more: its face is compared with the signature from step 1, and it is accepted only if they are the same person. This is the picture a camera session animates.

When a session starts, the still picture is shown until you are seen at rest (facing the camera, mouth closed, eyes open; three seconds at most). That frame is the neutral pose, and all movement is measured from it, so a photograph from another camera keeps its proportions. "Reset neutral pose" in the Live Studio takes a new one.

So nobody's picture can be animated except that of the person who sat in front of the camera. Every output frame is made from the meeting picture, so a fault in it shows in all of them, which is why it is checked before it is kept.

What is stored, as plain files in `data/enrolment/`: the face signature and its record (`identity.npy`, `identity.json`), and the meeting picture with its face crop (which sets the neutral pose), its own signature, and a record of what the checks found and how well it matched the face. "Remove picture" deletes the picture and leaves the face; "Forget my face" deletes everything.

**How alike is the same person?** `python scripts/study_same_person.py` measured the similarity of face signatures on the sample material (`results/same_person_study.json`). Of 300 pairs of different people none scored above 0.27; of 550 pairs of the same person none scored below 0.42. The limit is 0.35. Two limits of this measurement: the same-person pairs are frames of one clip, so a picture from another day and camera will score lower than they do, and 25 people is a small set. It will be repeated on the team's own pictures when the consent gate is built.

| Check | Passes when |
|---|---|
| One face in view | Exactly one face is found |
| Facing the camera | The head is turned at most 12° to a side, 15° up or down, and tilted at most 10° |
| Face large enough | The face is at least 200 px tall, and at most 62% of the picture's height |
| Whole head in frame | There is room above the forehead, below the chin and beside the face, and the face is near the middle |
| Enough light | The brightest parts of the face reach 95 of 255, without burning out to white |
| Sharp | Fine detail on the face scores 18 or more (variance of the Laplacian with the face at a standard size) |
| Relaxed face, eyes open | The mouth is closed, the eyes are open, and there is no broad smile |

The outline drawn on the camera picture is worked out from the same limits: it asks for a face 54% of the picture's height, as large as the checks allow with room to spare, because a larger face gives a sharper output. A face that fills the outline passes the size and framing checks.

Light is read from the brightest parts of the face and from blown-out pixels, never from the average brightness, so skin tone does not decide it. A picture that passes can still be told how to be better: below 300 px of face height the screen suggests coming closer while there is room, and below a sharpness of 45 it suggests more light.

The limits come from a measurement, `python scripts/study_enrolment.py`, which takes about 15 minutes and writes `results/enrolment_study.json`. The Enrolment screen draws its two charts from that file.

**What a fault costs.** Six sample clips were each animated from a good frame of themselves and from faulty ones, and every output frame was compared with the real frame of the same instant.

| Source picture | Identity match to the real frame (CSIM) | Share of fine detail kept |
|---|---|---|
| Good | 0.847 | 0.41 |
| Blurred | 0.068 lower | 0.13 |
| Dark | 0.040 lower | 0.14 |
| Mouth open | 0.023 lower | 0.40 |
| Low resolution | 0.021 lower | 0.19 |
| Head turned (found in 1 clip) | 0.015 lower | 0.44 |
| Eyes closed (found in 2 clips) | 0.003 lower | 0.36 |

**Face size.** Three sharp pictures were each enrolled at seven sizes and reproduced without movement. Detail was measured on the generator's own 512 px output, where nothing but the source picture differs between sizes.

| Face height in the enrolled picture | Detail in the output face, against a 260 px face |
|---|---|
| 110 px | 0.17 to 0.38 |
| 150 px | 0.33 to 0.58 |
| 200 px | 0.61 to 0.79 |
| 260 px | 1.00 |
| 300 px | 1.10 to 1.21 |
| 380 px | 1.23 to 1.46 |
| 460 px | 1.70 to 1.71 |

The generator draws the face about 260 px tall, and the first plan was to store larger pictures shrunk to that size. This measurement ruled it out: detail keeps rising with face size, and shrinking a 460 px face to 260 px would have cost about 40% of it. The picture is stored at full size (at most 1280 px on its longer side), the minimum face height was raised from 150 to 200 px, and the screen asks for a larger face while there is room.

Read these numbers with their limits: six clips and three pictures, all sample material and none from the team's own cameras; a turned head was found in one clip and closed eyes in two. The expression checks rest less on these numbers than on what the picture is used for: it is shown as it is whenever tracking is lost.

The proposal speaks of a source clip. The pipeline animates a single picture, so that is what is enrolled; enrolling a clip or several pictures is not built.

## Run from the command line

```
python scripts/demo_capture.py                 # live webcam, press q to quit
python scripts/demo_capture.py --source clip.mp4
python scripts/reenact_video.py --source me.jpg --driving clip.mp4 --out out.mp4
python scripts/live.py --source-image me.jpg   # live reenactment; q quits, r resets the neutral pose
python scripts/benchmark.py                    # speed, identity, motion and stability on the sample clips
python scripts/study_enrolment.py              # what the enrolled picture's faults and face size cost (about 15 minutes)
python scripts/study_same_person.py            # how alike two faces must be to count as the same person
python -m pytest                               # engine and server tests
cd web && npm test                             # front-end tests
```

`live.py` shows the camera with metrics on the left and the reenacted output on the right. With no face or more than one face in view, the output switches to the static enrolled frame. Add `--camera clip.mp4` to drive it from a file, and `--record out.mp4` to save the preview.

`reenact_video.py` animates the source image with the motion in the driving video, writes a side-by-side result, and prints per-stage latency, frame rate, and peak GPU memory.

## Measurements

`benchmark.py` runs the pipeline on every (source, driving video) pair in a manifest and writes one JSON file. Use `--manifest` for your own list of clips, `--out` for the result file and `--save-videos` to keep side-by-side videos. The measurements run outside the timed pipeline.

| Measurement | Question it answers | How it is measured |
|---|---|---|
| Speed | Is it fast enough? | Timers around each stage; the first 10 frames are left out as warm-up |
| Identity (CSIM) | Does the output still look like the source person? | Cosine similarity of ArcFace embeddings of the source image and each output frame |
| Pose and expression error | Does the output move the way the driver moves? | MediaPipe head pose and blendshapes on the driving and the output frame, compared as change since the first frame |
| Lag | Does the output trail the driver? | The time shift that best lines up output motion with driving motion |
| Flicker (warping error) | Does the picture change in ways movement does not explain? | Previous frame moved onto the current one along RAFT optical flow, then compared (Lai et al., 2018) |
| Head jitter | Does the head tremble? | Frame-to-frame acceleration of eye-corner and nose-bridge landmarks, as a share of the eye span |

Three points about reading them:

- CSIM compares with the source image, so a face that never moved would score 1.0. It is always read together with the motion errors.
- Flicker and jitter are reported as a ratio to the same measurement on the real driving video, so 1.0 means "as steady as real video".
- The warping error cannot see a trembling head, because trembling is real movement that the optical flow explains away. That is why head jitter is measured separately; `tests/test_evaluation.py` demonstrates both facts on synthetic clips.

The identity score aligns faces with five MediaPipe landmarks instead of InsightFace's own detector. `scripts/validate_alignment.py` compares the two on 36 sample faces: the similarity scores differ by 0.015 on average (0.077 at most) and correlate at 0.993 (`results/alignment_validation.json`).

The demo overlays face landmarks, head pose (yaw, pitch, roll), jaw opening, frame rate, and tracker latency. With no face or more than one face in frame it shows a banner and reports the frame as not usable for reenactment.

## Project history

`PROJECT_HISTORY.md` records what was built, decided and measured, in order, and is updated with every completed feature.

## Third-party components and licensing

This is a research prototype. It builds on LivePortrait, InsightFace/ArcFace, MediaPipe, RAFT (through torchvision, for evaluation), Wav2Lip/SyncNet, and F5-TTS, several of which are released for research or non-commercial use only. It is not licensed for commercial deployment.

## Responsible use

NeuroPresence is intended for consented self-presentation only. It must not be used to animate another person's likeness or to represent a user as present when they are not.
