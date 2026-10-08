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

**After the driving signal was steadied (8 October 2026).** The face crop now holds still while the head does, the keypoints given to the generator are filtered so a still head is drawn still, and a lost face fades to the still picture instead of snapping. Each has its own switch in the Live Studio. Full results: `results/benchmark_steady_windows_rtx5050.json`.

| Measurement | Baseline | Now | Target |
|---|---|---|---|
| Head jitter relative to real video | 1.49x | 0.98x | at most 1x |
| Flicker relative to real video | 0.89x | 0.73x | at most 1x |
| Lag behind the driver | 0.0 frames | 0.07 frames | lower is better |
| Identity, pose error, expression error, mouth correlation | 0.896, 0.54°, 0.026, 0.96 | 0.896, 0.53°, 0.026, 0.96 | unchanged |
| Frame rate | 7.5 fps | 7.3 fps | ≥ 24 fps |

The head also keeps to the range of movement that looks right on a body that stays still: small movements are followed exactly, and a head thrown far back or turned far aside eases to a stop, and a posture held for a few seconds becomes the new rest position, so leaning back in the chair does not leave the head tilted on a still body ("Natural head range" in the Live Studio).

**After the first round of speed work (8 October 2026).** The two networks that draw the picture run with half-precision weights and are compiled for the GPU (about a minute when the models load), and the face is pasted back into the frame on the GPU. Full results: `results/benchmark_fast_windows_rtx5050.json`.

| Measurement | Baseline | Now | Target |
|---|---|---|---|
| Frame rate | 7.5 fps | 9.9 fps | ≥ 24 fps |
| Render per frame | 94 ms | 67 ms | ≤ 42 ms |
| Whole pipeline per frame | 134 ms | 101 ms | ≤ 150 ms end to end |
| Peak GPU memory | 1.17 GB | 0.67 GB | ≤ 8 GB |
| Identity, pose error, expression error, mouth correlation, jitter, flicker | | unchanged | |

**With TensorRT (8 October 2026).** The three networks run through TensorRT: the two that draw the picture in half precision, the one that reads the movement in full precision. The face is tracked on its own thread while the GPU draws. The first start on a machine converts the networks, which takes about a minute and a half; after that they load from `models/tensorrt` in a second. TensorRT is optional: without it the app uses PyTorch's compiler, and without that plain PyTorch. Full results: `results/benchmark_tensorrt_windows_rtx5050.json`.

| Measurement | Baseline | Now | Target |
|---|---|---|---|
| Render per frame | 94 ms | 37 ms | ≤ 42 ms |
| Frame rate, live session | 7.5 fps | 18.7 fps | ≥ 24 fps |
| Camera to output, live session | about 160 ms | 89 ms | ≤ 150 ms |
| Identity, pose error, expression error, mouth correlation, jitter, flicker | | unchanged | |

The render and latency targets are met. The frame-rate target is not reached yet.

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
| Enrolment | Working: the face is verified live with the camera and only its signature is kept; five poses are registered (facing the camera, turned to each side, tilted up and down) for a richer signature and a measured range of motion; the meeting picture is uploaded, checked, and accepted only if it shows the same face (`neuropresence/enrolment`) |
| 1–2 Capture, tracking, driving signal | Working, with the crop and keypoints steadied (`neuropresence/capture`, `neuropresence/capture/steady.py`) |
| 3 Reenactment | Working live in a preview window (`neuropresence/reenactment`, `neuropresence/pipeline.py`) |
| 4 Identity preservation | Working: the output is scored against the enrolled picture twice a second; a drop that lasts takes a fresh neutral pose, and if that does not help the still picture is shown until you resume (`neuropresence/identity/guard.py`). A delay watchdog shows the still picture while the output arrives too late (`neuropresence/server/watchdog.py`) |
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
- **Live Studio**: camera and output side by side, the live figures against their targets, where each frame's time goes, an event list, and a switch for every feature. When the output stops matching your picture, or arrives too late, the still picture is shown and the screen says why; after an identity fallback one button resumes.

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

Enrolment has three steps, and they answer three different questions.

1. **Who are you? Verify your face with the camera.** The camera runs with seven checks on every frame. The picture is taken after a count of three, as the sharpest open-eyed frame of a half-second burst. It is used only to make a face signature (an ArcFace embedding, 512 numbers) and is not stored.
2. **How far does your head actually move? Register five poses.** Facing the camera again, turned to each side, tilted up and down. Each gets four checks (one face, turned the way asked, light, sharp) instead of the seven above. Two things come of it: the face signature grows richer, since an uploaded picture or a live face is then compared against whichever of up to six registered angles looks most like it, not only the step-one capture; and the reenactment engine's own clamp on how far the head is allowed to move is fitted to this person's own registered left/right/up/down extremes, instead of a generic default set by looking at one person turned in steps. Required once per verified face before step 3 will accept a new upload; a meeting picture enrolled before this step existed is not retroactively blocked.
3. **What should people see? Upload your meeting picture.** A picture you choose, with the light and background you want. It goes through the same seven checks and one more: its face is compared with every signature from steps 1 and 2, and it is accepted only if the best match is the same person. This is the picture a camera session animates.

When a session starts, the still picture is shown until you are seen at rest (facing the camera, mouth closed, eyes open; three seconds at most). That frame is the neutral pose, and all movement is measured from it, so a photograph from another camera keeps its proportions. "Reset neutral pose" in the Live Studio takes a new one.

So nobody's picture can be animated except that of the person who sat in front of the camera. Every output frame is made from the meeting picture, so a fault in it shows in all of them, which is why it is checked before it is kept.

What is stored, as plain files in `data/enrolment/`: the face signature and its record (`identity.npy`, `identity.json`), each registered pose's own signature and the angle measured (`pose_<name>.npy`, `poses.json`), and the meeting picture with its face crop (which sets the neutral pose), its own signature, and a record of what the checks found and how well it matched the face. "Remove picture" deletes the picture and leaves the face and its poses; "Forget my face" deletes everything, poses included. Verifying the face again clears any poses registered for the face before it, since they are specific to that signature.

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
python scripts/study_identity_guard.py         # the identity score in normal use and under faults (about 20 minutes)
python scripts/study_delay.py                  # delay and frame rate of a live session, alone and beside GPU load
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

## Identity fallback and delay watchdog

Two things can make the live output worse than the still picture: it stops looking like the person, or it arrives late. Both are watched during a session, and both have a switch in the Live Studio.

**Identity.** The output is scored against the enrolled picture twice a second (CSIM, on its own thread; one score takes the CPU about 48 ms). `python scripts/study_identity_guard.py` measured what that score looks like (`results/identity_guard_study.json`):

| Case | Mean | Lowest frame | What the guard does with the recorded scores |
|---|---|---|---|
| Normal use, 8 pairs of the benchmark | 0.89 | 0.72 | Nothing, on any pair |
| Head pushed 30° / 45° past its range, head range off | 0.73 / 0.55 | 0.59 / 0.44 | Ends on the still picture in 75% / 100% of runs, after about 4.5 s |
| Head turned 45° further, head range off | 0.65 | 0.47 | Ends on the still picture in 90% of runs |
| The same 45° with the head range on | 0.85 and 0.83 | 0.70 | Nothing for the nod; for the turn, the still picture in 5% of runs |
| Mouth covered in the camera picture | 0.72 | 0.59 | Acts in about half the runs |
| Expression three times too strong | 0.79 | 0.43 | Ends on the still picture in 39% of runs |

Single frames cannot be judged: 9% of normal frames are below the 0.80 the project aims for on average. So the guard (`neuropresence/identity/guard.py`) reads the mean of the last three seconds. Below 0.75, which no three seconds of normal use reached (the lowest was 0.82), it takes a fresh neutral pose and starts the filters again. If the next seconds are still below, or two readings in a row fall under 0.50, the still picture is shown and stays until you press "Resume reenactment". A dip of a second or so does nothing.

The same table shows what the natural head range is worth: with it off, a head pushed 45° brings the match down to 0.55; with it on, it stays at 0.85.

Run in a real session, with the mouth covered in a sample clip, the guard took a fresh neutral pose 3.5 s after the cover appeared, and the match returned to 0.91. That showed a fault in the first version: the cover had become part of the neutral pose, so the output was off once the cover was gone (0.80 where it had been 0.86). A fresh neutral pose that helped is now watched, and when the match steps down again the neutral pose is taken once more; in the same run it then returned to 0.86.

**Delay.** `python scripts/study_delay.py` runs a live session on a sample clip and keeps the GPU busy from a second process (`results/delay_study.json`). The load is artificial; it stands in for a meeting application and is not one.

| GPU busy elsewhere | Frame rate | Delay, mean | Frames over 150 ms |
|---|---|---|---|
| Not at all | 20.4 fps | 82 ms | 0% |
| 25% of the time | 16.2 fps | 93 ms | 0% |
| 50% of the time | 13.0 fps | 110 ms | under 1% |
| 75% of the time | 9.9 fps | 134 ms | 13% |
| All the time | 7.3 fps | 171 ms | 96% |

Late frames are dropped and never queued, so the frame rate falls first and the delay follows slowly. The watchdog (`neuropresence/server/watchdog.py`) reads the mean delay of the last two seconds. Above 150 ms the still picture is shown; the frames are still drawn, hidden, so that the delay stays measured, and the live picture returns when the mean has been under 135 ms for a second. In a real session beside a fully loaded GPU it held the picture back 3.5 s after the load began and released it within two seconds of the load ending.

These figures are for a 480 px sample clip with no browser attached; a camera session with a larger picture runs slower (18.7 fps and 89 ms on 8 October). What a real meeting application takes from the GPU has not been measured: that needs the virtual camera.

## Project history

`PROJECT_HISTORY.md` records what was built, decided and measured, in order, and is updated with every completed feature.

## Third-party components and licensing

This is a research prototype. It builds on LivePortrait, InsightFace/ArcFace, MediaPipe, RAFT (through torchvision, for evaluation), Wav2Lip/SyncNet, and F5-TTS, several of which are released for research or non-commercial use only. It is not licensed for commercial deployment.

## Responsible use

NeuroPresence is intended for consented self-presentation only. It must not be used to animate another person's likeness or to represent a user as present when they are not.
