# NeuroPresence

[![Tests](https://github.com/zain31197/NeuroPresence/actions/workflows/tests.yml/badge.svg)](https://github.com/zain31197/NeuroPresence/actions/workflows/tests.yml)

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

The render and latency targets are met. The frame-rate target is not reached yet; on 11 October the live session went from 18.7 to 22 frames a second (see Speed and even motion).

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
| 3 Reenactment | Working at about 22 frames a second live, with the movement shown in even steps (`neuropresence/reenactment`, `neuropresence/pipeline.py`, `neuropresence/server/pacing.py`) |
| 4 Identity preservation | Working: the output is scored against the enrolled picture twice a second; a drop that lasts takes a fresh neutral pose, and if that does not help the still picture is shown until you sit at rest again or resume (`neuropresence/identity/guard.py`). A delay watchdog shows the still picture while the output arrives too late (`neuropresence/server/watchdog.py`) |
| 5 Consent and disclosure | Working: before a face is verified and before every camera session, two actions chosen at random are asked for and the face is matched all the way through (`neuropresence/consent/liveness.py`, `check.py`); during a session the camera face is compared twice a second; every output frame carries the label "AI reenacted" (`neuropresence/consent/disclosure.py`) |
| 6 Virtual camera | Working on Windows with OBS Virtual Camera, and the voice held back to match it through VB-Audio Virtual Cable; used in Google Meet, Zoom and Microsoft Teams; not yet tried on Kubuntu |
| Evaluation | Benchmark of speed, identity and temporal stability (`scripts/benchmark.py`, `neuropresence/evaluation`) |
| Web app | Landing page, sign-in, and the studio: Overview, Enrolment, Live Studio, Sessions, Security, Your data, Account, Benchmarks and System (`web/`, `neuropresence/server`) |
| Accounts and database | Working: accounts on this computer, eleven tables in one SQLite file, every row a person's own (`neuropresence/accounts.py`, `neuropresence/database.py`, [docs/DATABASE.md](docs/DATABASE.md)) |

## Setup

Requires Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/download_models.py        # face landmark model (4 MB) and ArcFace (275 MB download)
python scripts/setup_liveportrait.py     # LivePortrait code and weights, about 700 MB
python scripts/setup_ditto.py            # the camera-off voice mode: Ditto's code and weights, about 2.2 GB
```

The first benchmark run also downloads the RAFT optical-flow weights (21 MB) through torchvision.

For meetings, install the two devices a meeting application picks up: **OBS Studio** (its virtual camera; on Linux, v4l2loopback) and **VB-Audio Virtual Cable** (the voice held back to match the picture; on Linux, a PipeWire null sink named NeuroPresence). Then build the web app (below).

### Check a computer before a session

```
python scripts/preflight.py --warm
python scripts/preflight.py --demo    # the morning of the demonstration: --warm, then the rehearsal of its failure paths
```

It checks everything a session needs and says how to fix what is missing: Python and its packages, the graphics card, power (a laptop on battery runs its graphics card much slower), the model files, the web app, the account signed in last and its enrolment, the virtual camera and microphone, camera 0 (how many frames a second it really delivers), the internet for Meet, the port and free memory. With `--warm` it also converts the drawing networks for this graphics card (about a minute and a half, once per card), warms the voice model, and draws for a minute as a session does, to say how many frames a second this computer reaches and whether it slows down as it heats. `--camera 1` checks another camera, `--no-camera` none. The report is kept in `logs/preflight.log`. With `--demo` it then rehearses the demonstration's eleven failure paths on that computer (`scripts/rehearse_demo.py`, a stand-in at the camera, no camera opened), adds the result to the report, and ends with "Ready for the demonstration" or what to fix; the cases are kept in `logs/demo_rehearsal.json`. About six minutes on the PC. Run it with the virtual camera switched off in the Live Studio, since it opens that device for a moment.

## Web app

The web app is the place to run the system and to test each feature. It has a landing page and the studio, whose screens are, in the order they are used:

- **Sign in**: an account on this computer, made in half a minute. It keeps the person's face, pictures, sessions, notes and settings apart from anyone else's. The first account made takes over what was on the computer before accounts. A recovery code, shown once, sets a new password when the old one is forgotten; no email is ever sent. To use the same account on another computer, copy the `data` folder to it with the engine closed on both.
- **Overview**: whether everything is ready for a call (enrolment, this computer), a button to go live with the input used last, the week's figures, and the latest sessions and security entries.
- **Enrolment**: verify your face with the camera, then upload the picture people will see. The camera is shown with a face outline and seven checks that update as you move; the picture tells you the one thing to fix next. Before the picture is taken you are asked for two quick actions, so that a photograph cannot be verified. An uploaded picture is refused if it is not of the same person. Every picture that was enrolled stays in **Your pictures**, up to twelve, and any of them is brought back with one press, after the same checks.
- **Live Studio**: camera and output side by side, the live figures against their targets, where each frame's time goes, an event list, and a switch for every feature. A camera session starts with the liveness prompt, shown on the camera picture; if it is not passed, the screen says why and nothing is animated. When the output stops matching your picture, arrives too late, or someone else sits down at the camera, the still picture is shown and the screen says why. After an identity fallback the picture resumes by itself, with a fresh neutral pose, once you sit facing the camera with the mouth closed, since in a meeting you are looking at the meeting and not at this screen; if it still does not match it falls back again, and each automatic try waits twice as long. One button resumes at once, another keeps the still picture until you resume.
- **Sessions**: every past session, grouped by day, with how it ran and how it ended; one opens with its figures drawn over time, its consent check and what happened in it. A session can be given a name, a note and a star, searched for, and deleted; a starred one is kept when the history is cleared.
- **Security**: the retry limit on the liveness check, and a timeline of every check and every change to the enrolment.
- **Your data**: the database itself: its tables and how they refer to one another, drawn from the file, how many rows are yours, whether it is intact, and a copy of everything of yours to take away (JSON).
- **Account**: name, email, password, recovery code, where the account is signed in, the latest tries at its password, and deleting it with everything in it.
- **Benchmarks** and **System**: every saved benchmark run against the targets, and the preflight checks run from the app.

Everything is kept in a database on this computer: `data/neuropresence.db`, a single SQLite file that git ignores (`neuropresence/database.py`). It holds the accounts, the meeting pictures, the sessions with a reading every two seconds, the events, the consent checks, the changes to the enrolment, the notes, and the switches and input chosen, so these survive a restart, the retry count included. Every row belongs to the person who made it and goes with their account when it is deleted. Passwords are kept only as scrypt hashes and sign-ins only as hashes of their tokens. No camera frame and no sound is ever kept, and the face signature stays a file of its own in the person's folder, so that "Forget my face" removes it outright. SQLite comes with Python, so there is nothing to install. The diagram, the data dictionary and what is done to keep it right are in [docs/DATABASE.md](docs/DATABASE.md), and a test holds that document to the schema.

Build it once (this needs Node.js; it was built and tested with version 24), then start the engine:

```
cd web
npm install
npm run build
cd ..
python -m neuropresence.server --open
```

Warnings and errors, with their full traces, are written to `logs/server.log`; if the engine process itself dies, the stacks of all its threads go to `logs/crash.log` (both kept out of git). If the camera stops sending pictures for a moment, as a phone used as a webcam may, the session reconnects to it and goes on; it ends only after 15 seconds without a picture. Every 30 seconds a running session also writes its figures there: frames drawn a second, the delay, and how many new pictures the virtual camera sent, so a picture that looked jerky in a call can be traced afterwards.

It opens at http://127.0.0.1:8000. The engine listens on this computer only and, apart from its health and its measured figures, answers only someone who is signed in. The face signature and the picture in use are kept in the first account's folder `data/enrolment/` and in `data/users/` for the accounts after it, which git ignores. A camera session animates the enrolled picture, so enrol one first. For repeatable tests a session can also be driven from one of the sample clips; a clip animates one of its own frames and enrols nothing.

Two options help when testing: `--camera-file clip.mp4` plays a video file in place of every camera, so the camera view of the Enrolment screen and its checks can be tried without a webcam, and `--data-dir folder` keeps the enrolled picture somewhere other than `data/`. A video file cannot verify a face or start a camera session: it does not answer the liveness prompt, which is what the prompt is for. Sessions without a webcam run on the sample clips.

While working on the front end, run the engine as above and, in a second terminal, `npm run dev` inside `web/`; the page is then at http://localhost:5173 and reloads as you edit.

Every figure in the app comes from the running pipeline or from a saved result file. A new feature gets its own switch in the Live Studio by adding one entry to `neuropresence/server/features.py`.

## Enrolment

Enrolment has three steps, and they answer three different questions.

1. **Who are you? Verify your face with the camera.** The camera runs with seven checks on every frame. Once they pass you are asked for two quick actions (the liveness prompt, see Consent and disclosure), and right after it the picture is taken, as the sharpest open-eyed frame of a half-second burst. It has to show the face that answered the prompt. It is used only to make a face signature (an ArcFace embedding, 512 numbers) and is not stored.
2. **How far does your head actually move? Register five poses.** Facing the camera again, turned to each side, tilted up and down. Each gets four checks (one face, turned the way asked, light, sharp) instead of the seven above. Two things come of it: the face signature grows richer, since an uploaded picture or a live face is then compared against whichever of up to six registered angles looks most like it, not only the step-one capture; and the reenactment engine's own clamp on how far the head is allowed to move is fitted to this person's own registered left/right/up/down extremes, instead of a generic default set by looking at one person turned in steps. A pose is registered only if it matches the verified face, because every signature kept for you is one a session can later be started with. Required once per verified face before step 3 will accept a new upload; a meeting picture enrolled before this step existed is not retroactively blocked.
3. **What should people see? Upload your meeting picture.** A picture you choose, with the light and background you want. It goes through the same seven checks and one more: its face is compared with every signature from steps 1 and 2, and it is accepted only if the best match is the same person. This is the picture a camera session animates.

When a session starts, you answer the liveness prompt; then the still picture is shown until you are seen at rest (facing the camera, mouth closed, eyes open; three seconds at most). That frame is the neutral pose, and all movement is measured from it, so a photograph from another camera keeps its proportions. "Reset neutral pose" in the Live Studio takes a new one.

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
python scripts/study_liveness.py --stand-in    # what can pass the liveness prompt and the face match (about 15 minutes)
python scripts/study_pacing.py                 # how evenly the movement is shown, with and without even motion (about 6 minutes)
python scripts/study_consent_watch.py         # how soon someone else at the camera brings the still picture (about 15 minutes)
python scripts/study_mouth_cover.py           # whether a covered mouth is told from an uncovered one (about 5 minutes; --live: a session too)
python scripts/rehearse_demo.py               # the demo's failure paths on the real engine, with a stand-in (about 3 minutes)
python scripts/rehearse_demo.py --soak 30     # one camera session of half an hour, watched for anything that creeps
python -m pytest                               # engine and server tests
cd web && npm test                             # front-end tests
```

The demonstration itself, what to show in what order and what to do when something goes wrong, is in [docs/DEMO_RUN_SHEET.md](docs/DEMO_RUN_SHEET.md); `rehearse_demo.py` runs each of its failure paths before the day.

The same tests run on GitHub on every push and pull request (`.github/workflows/tests.yml`): the engine and server tests on a clean Linux machine and a clean Windows machine, and the web app's type check, tests and build. Those machines have no graphics card and no downloaded models, so the tests that need either skip themselves and say why; the rest must pass. That also checks that a fresh copy of the repository installs from `requirements.txt` alone.

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

## Speed and even motion

**The heaviest layers in 8 bit (22 November 2026).** The networks that draw are 84% of a frame's 44 ms on the RTX 5050, in 16 bit through TensorRT. The SPADE layers of the generator (45 of its 74 convolutions) and the hourglass of the warping network (11 of 16) now run in 8 bit (`neuropresence/reenactment/quantize.py`); the layers that decide where things move and the generator's own residual blocks stay in 16 bit, because in 8 bit they broke the picture. Drawing the two networks for a frame went from 35.4 to 30.9 ms in the benchmark (`results/benchmark_int8_windows_rtx5050.json` against `results/benchmark_half_windows_rtx5050.json`): identity 0.896 to 0.894, mouth agreement 0.948 to 0.945, pose and expression unchanged; flicker 0.81 to 0.87 of real video and tremble 1.03 to 1.06 (both target at most 1; tremble was missed before too). That small rise is the price, and it was chosen layer by layer: with every SPADE layer in 8 bit the flicker went to 1.02, and leaving one block (`up_0`) in 16 bit brought it back. The ranges each layer needs were measured once on LivePortrait's public samples (`scripts/calibrate_int8.py`, `int8_scales.json`), so no one's face is in them and every machine builds the same networks. Converting them takes a few minutes the first time on a machine; whatever fails falls back to 16 bit, and `NEUROPRESENCE_INT8=0` in the environment turns them off.

**Speed (11 October 2026).** A live session ran at 19.4 frames a second, and at 18.1 with the Live Studio open in a browser. Taking one drawn frame apart showed almost 9 ms of it that was not the networks' work: the arithmetic on the 21 keypoints was done as dozens of tiny GPU operations, each costing more to hand over than to do, and the finished face was copied back into the picture through a slow path. The arithmetic now runs on the CPU and the copy is one solid block, which took 6 of those 9 ms away.

| Measurement | Before | Now | Target |
|---|---|---|---|
| One drawn frame, engine alone, 960 x 1280 picture | 49.0 ms | 43.4 ms | |
| Frame rate, live session | 19.4 fps | 22.0 fps | ≥ 24 fps |
| Frame rate, live session with the Live Studio open | 18.1 fps | 20.3 to 20.8 fps | ≥ 24 fps |
| Frame rate, benchmark | 16.2 fps | 17.6 fps | |
| Identity, pose error, expression error, mouth correlation, jitter, flicker | | unchanged to three decimals | |

The picture drawn is the same: against the output saved before the change, it differs by 0.002 of 255 on average (`results/benchmark_smooth_windows_rtx5050.json` for the quality figures). The live figures are from a session with a 1280 x 720 picture at 30 frames a second standing in for the camera. Over four minutes without a break the rate moved between 20.1 and 21.6 as the graphics card, which runs at its 130 W limit here, changed its clock. The 24 frames a second target is still not met: 36 of the 45 ms of a frame are the two networks that draw the picture, and they have to become cheaper for it.

**Even motion.** The camera delivers 30 frames a second and the output is drawn at about 22. Each output frame was made from the newest camera frame there was, and that frame is sometimes just in and sometimes 33 ms old. The output frames came out evenly, 45 ms apart, but the camera frames they showed were 33 and 67 ms apart by turns, so a head turning steadily was drawn in short and long steps. Now the movement is read between the last two camera frames, for a moment that lies a fixed time back (`neuropresence/server/pacing.py`, `Pipeline.step`). It has a switch in the Live Studio, "Even motion", and is on.

`python scripts/study_pacing.py` runs the pipeline over six sample clips with a live session's timing, once each way (`results/pacing_study.json`):

| Measurement | Newest frame | Even motion |
|---|---|---|
| Moments shown by one output frame and the next, apart | 33 to 67 ms (30 fps clips), 40 to 80 ms (25 fps clips) | 45.5 ms, every time |
| How much the head in the output changes speed from frame to frame, in % of the eye span | 3.77 | 2.71 |
| Head turn against the driver's at the moment shown | 2.59° off | 2.65° off |
| Mouth opening against the driver's (correlation) | 0.898 | 0.899 |

The head moves 28% more evenly and follows the driver as closely as before. The cost is delay: in a live session the movement is 98 ms old when it is shown, steadily (it varied by 1.4 ms), where it was 62 to 95 ms old by turns before, 80 on average. The limit is 150.

The Live Studio now shows what the camera itself delivers, beside its name, because the output cannot be faster than the camera: a webcam in poor light often drops to 15 frames a second by itself, and the event list says so when that is what holds the rate down.

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

**A covered mouth** (`neuropresence/capture/cover.py`, the "Covered mouth" switch). A hand over the mouth used to be drawn as a smeared, lipless mouth until the guard noticed, which it did in about half the runs. Now the camera's mouth is checked on every frame: lips are redder than the cheeks beside them, by an amount learned for the person at the start of the session and followed as the light changes, and a hand or a cup takes that difference away or changes it far. While the mouth is covered the picture's mouth eases to the neutral pose's, closed, the head and eyes go on moving, and the cover is kept out of the neutral pose, the lip anchoring and the mouth watch. The Live Studio says "Covered" beside the mouth signal. It costs 0.3 ms a frame on the CPU.

`python scripts/study_mouth_cover.py` measured it (`results/mouth_cover_study.json`) on Zain's two camera recordings and the twelve sample clips, with covers drawn over the lower face: a hand, a flat skin coloured patch, a white cup, a red cup, fingers over half the mouth, and a hand on the chin below the lips, which is not a cover:

| | Result |
|---|---|
| Uncovered, 168 s of 14 people | Called covered for 0.33 s in all, on one clip where the lips are pressed together |
| People whose lips cannot be told from their skin in their light | 2 of 14: for them it stays off, and nothing changes from before |
| A hand, a flat patch, a white cup | 24 of 24 each, after 0.1 s (Zain's recordings: 0.2 s) |
| Fingers over half the mouth | 22 of 24 |
| A red cup | 8 of 24: on red lips it is the same colour as the lips |
| A hand on the chin, lips showing | Never called a cover, on Zain's recordings or any clip but the one above |

In a session on the real engine (`--live`), with a hand drawn over a talking face, the picture's mouth was held closed for 93% of the covers and its likeness to the picture stayed at 0.90 (0.75 at worst), against a smeared, lipless mouth and 0.80 (0.70) with the switch off; the still picture was not needed either way. The covers are drawn, not filmed; a real hand at a real camera is on the demo run sheet to check by hand.

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

## Consent and disclosure

A picture is animated only for the person it belongs to, and every output frame says what it is. This has three parts, and none of them has a switch.

**The liveness prompt** (`neuropresence/consent/liveness.py`). A photograph of you matches your face signature as well as you do, and so does a recording. So before a face is verified, and before every camera session, two actions are asked for, chosen at random out of four: blink, turn your head to your left, turn it to your right, open your mouth. They come one after the other and each has five seconds. An action that is not seen in time is asked for once more, twice at most, with a line on what was missing ("open your mouth wider"). One of the two is always a turn of the head. Your face has to be at rest before each is asked for, and the wait before it is of a random length; doing something other than what was asked ends the check, except a blink, because people blink all the time. The camera is shown mirrored with the instruction on it, the time left and, for a turn or an open mouth, how far it has got. Meanwhile the output picture says that it starts moving once the two actions are done.

**The face match** (`neuropresence/consent/check.py`). All the way through the prompt the face at the camera is compared, about four times a second, with every signature registered for you (the verified face and the five poses). The best match decides, and 0.35 or more is the same person. A face that does not match is looked at again in the very next frame, and two in a row refuse. So the face that answers is the face that matches: your photograph cannot be held up for the match while someone else does the moving. After the session has started the camera face is still compared once a second. Three readings in a row that do not match put the still picture up; two that match bring the live picture back, with a fresh neutral pose.

**The disclosure mark** (`neuropresence/consent/disclosure.py`). The label "AI reenacted" in the middle of the lower edge of every output frame: live frames, the still picture, and the frames that fade between them. The pipeline draws it as its last step. It is an eighteenth of the frame high and never under 22 pixels, and drawing it takes 0.09 ms a frame.

`python scripts/study_liveness.py --stand-in` measured what the limits rest on, with no camera (`results/liveness_study.json`):

| What is held to the camera | What happened |
|---|---|
| A photograph, held still | Refused in all 1,296 attempts |
| A photograph tilted by up to 65° or swung from side to side | Reads as a head turn of 32.5° at most, and a turn counts from 35°. Refused in all 12,000 attempts |
| A photograph with the eyes covered, or with a dark shape over the mouth | Eye score 0.24 at most (a blink needs 0.50); mouth 0.42 at most (an open mouth needs 0.60) |
| The twelve sample clips played as recordings, from every half second and for every order of actions | Refused in all 3,110 attempts. If no turn were asked for, 4 of 622 would pass |
| A recording made on purpose, with every action in it one after another (written out as readings, not filmed) | About 1 attempt in 7 passes (15% with two or three seconds of rest between the actions). It was 1 in 9 when an action had three seconds and was asked for once |
| A stand-in that does what is asked (two sample faces: blink and open mouth are frames of the clip, the turns are drawn by the engine) | Passes 20 of 20, in 4.5 to 4.8 s on average. Turning the other way: 0 of 20. Against another person's signature: 0 of 20 |
| The same person through a whole clip, against its own first frame (1,474 faces) | Lowest match 0.49, against a limit of 0.35 |
| Another person (132 pairs of clips) | Highest match 0.23 |

A turn of the head is what a flat picture does not show, which is why one is always asked for: with blink and open mouth alone, a recording of a person talking passed four times in 622. How large a turn a tilted photograph reads as depends on how near the camera is. With the camera 3.0 widths of the photograph away the largest reading was 12.7°, at 1.4 widths 18.3°, at 0.8 widths 27.8° and at 0.6 widths 32.5°. That nearest case is a camera with a view 80° wide and a photograph that fills its whole picture.

A session alone ran at 20.8 to 21.0 frames a second and 80 to 81 ms with the watch during the session off and at 20.6 to 21.1 and 81 to 82 ms with it on. The full benchmark is unchanged with the mark in place (`results/benchmark_consent_windows_rtx5050.json`): 16.2 frames a second (16.3 before), render 36.7 ms (36.8), identity 0.898 (0.898), tremble 1.00 (0.99), flicker 0.75 (0.75).

**What this does not stop, and what has not been measured.**

- One person has passed the prompt at a real camera: Zain passed it at his own camera twice on 22 November 2026, the face matched all the way through at 0.57 and 0.80 at the lowest: the only two checks recorded since the database began on 21 November. How often a real user is refused is not measured yet: that needs many tries by more than one person. The first to try, on 12 October 2026, was asked to open the mouth, was not seen doing it within the three seconds an action then had, and the session ended. That is why an action now has five seconds and is asked for again after a miss; whether 0.60 for an open mouth and 35° for a turn suit real faces is still to be measured. The stand-in is not a person, and its head turns are drawn by the engine. On the one face registered so far, poses turned 40° matched the capture facing the camera at 0.67 and 0.79.
- A recording made for the purpose, holding every action, passes about one attempt in seven. Repeated attempts are limited (`neuropresence/consent/attempts.py`): three checks that do not pass cost nothing, then each further one makes the next check wait, one minute, then two, four and so on up to a quarter of an hour, and a check that passes forgets them. Against such a recording the expected time goes from under two minutes to about a quarter of an hour. That slows it; it does not stop it. The counts are kept in the database, so a restart of the engine does not clear them.
- A face that is itself animated live and fed in as the camera answers the prompt as a person does. The stand-in used here is exactly that. The prompt stops photographs and recordings, not that.
- The turn limit of 35° has 2.5° to spare in the hardest case measured. A second measure was looked at, how much a face narrows as it turns, which separates a tilted photograph from a head in the sample material (short of what its turn says by 0.14 or more, against 0.10 at most); it is not used, because the sample clips hold no head turned further than 18° to set it on.
- Someone else who sits down during a session drives the picture for about one and a half seconds (1.9 at most in six tries), until three readings half a second apart have not matched (`scripts/study_consent_watch.py`).
- The app runs on the user's own computer. These checks stand in the way of misuse through the app; they do not bind someone who changes the program.

## Project history

`PROJECT_HISTORY.md` records what was built, decided and measured, in order, and is updated with every completed feature.

## Third-party components and licensing

This is a research prototype. It builds on LivePortrait, InsightFace/ArcFace, MediaPipe, RAFT (through torchvision, for evaluation), Wav2Lip/SyncNet, and F5-TTS, several of which are released for research or non-commercial use only. It is not licensed for commercial deployment.

## Responsible use

NeuroPresence is intended for consented self-presentation only. It must not be used to animate another person's likeness or to represent a user as present when they are not.

## Virtual camera

The output can be offered to meeting applications as an ordinary camera (`neuropresence/server/virtual_camera.py`).

**What to install once.** On Windows, OBS Studio (obsproject.com): it brings the "OBS Virtual Camera", and OBS itself does not have to be running. On Kubuntu, the `v4l2loopback` module: `sudo modprobe v4l2loopback exclusive_caps=1 card_label=NeuroPresence`. The Python side is `pyvirtualcam`, which `requirements.txt` installs.

**How to use it.** In the Live Studio, switch on "Virtual camera". In the meeting application choose "OBS Virtual Camera" as the camera. Start a camera session and answer the prompt.

**What a meeting sees.**

| When | The camera shows |
|---|---|
| No session, or the liveness prompt is still being answered | A plain card: "NeuroPresence is not running". Nothing of the user |
| A camera session is running | The output, with the "AI reenacted" mark |
| The face is lost, the output stops looking like the user, the picture is late, or someone else sits down | The still enrolled picture, with the mark, as in the Live Studio |
| A session from a sample clip | The card: a sample clip animates someone else's face |

The frame sent is the one the Live Studio shows, so there is no path to a meeting that goes around the mark or the fallbacks. Nothing of the user is sent before the consent check has passed: if the enrolled picture were shown while idle, anyone at the computer could put it into a meeting without answering the prompt.

The device is 1280 x 720 at 22 frames a second (20 until 22 November 2026, which is what the measurements below were made at). An upright picture is set in the middle with bars beside it. Each frame is sent once, as soon as it is finished, and while the camera sends a camera session the drawing is held to 22 frames a second ("Even delivery" in the Live Studio).

Why 22 now: a Meet call on 22 November drew only 17 to 18 frames a second, not the 20.0 to 20.2 measured on 23 October. The graphics card is at its 130 W power limit throughout, so whatever else it draws (the meeting, the Live Studio page, a screen recorder) comes out of the engine's share; raising the process's priority or its GPU scheduling class changed nothing. What did: the heaviest layers of the two networks that draw now run in 8 bit (see Speed), the Live Studio holds its endless decoration still while a session is in a meeting and gets no frames while its tab is hidden, and a test call (the real OBS Virtual Camera read and encoded by a browser, the Live Studio open) went from 19.5 to 23.6 frames a second, and to 22.0 steady, with no gap over 57 ms, with the drawing held to 22. Read back from the device at 22, none of the frames drawn at 22 are held long, 2.2% of those drawn at 21.5 are, 7.7% of those drawn at 20.

Why (21 November 2026, `python scripts/study_meeting_cadence.py`, `results/meeting_cadence_study.json`): the engine finishes its frames evenly, 45 ms apart to within 1 ms, but the camera was opened at 30 and fed on a clock of its own, and OBS Virtual Camera's filter picks frames up on yet another clock at the camera's rate. A meeting then held 37% of the frames half again as long as their neighbours, so a steady turn of the head arrived in short and long steps. Measured with the real engine on a sample clip and replayed through each way of sending:

| What a meeting application gets | New frames a second | Frames held half again as long | Step error | Added delay |
|---|---|---|---|---|
| Before: drawn freely, sent on a 30 a second clock, camera at 30 | 22.4 | 34% | 15.8 ms | 31.6 ms |
| Now: drawn at 20, each sent as finished, camera at 20 | 19.5 | 3.0% | 6.7 ms | 24.6 ms |
| Linux (v4l2loopback hands on every frame), before | 22.5 | 33% | 15.5 ms | 16.2 ms |
| Linux, now | 20.0 | 0.1% | 0.0 ms | 0.0 ms |

The step error is, from one new frame shown to the next, the time it stayed up less the time between their being finished: the unevenness a viewer sees as judder. The figures are from the run kept in the results file; a first run the same evening gave 16.3 ms before and 2.0 ms now (0.8% held long), and the difference is one frame of the second run that came 84 ms after the one before. Held to 20 the picture is no later: 86.7 ms from camera to output, against 84.6 drawn freely. What the meeting application then does with the frames (its own encoding rate) is not modelled; a call is needed to see the whole of it.

The Live Studio's own stream to the browser had the same kind of fault: it looked for a new frame every 10 ms, and on Windows such a wait lasts a tick of the system timer, so it looked every 16.7 ms and a frame left for the browser 0 to 17 ms after it was made. It is now woken by each frame as it is made.

**Checked on the device itself** (`python scripts/study_virtual_camera.py`, `results/virtual_camera_study.json`; Windows, OBS Virtual Camera). Numbered test frames, nothing of anyone, were written into the device the way the engine writes and read back by ffmpeg the way a meeting application reads a camera:

| What a reader got, two runs | Frames held half again as long | Step error |
|---|---|---|
| Before: camera at 30, drawn at 22, sent on a 30 a second clock | 41% and 38% | 17.5 and 18.9 ms |
| Now: camera at 20, drawn at 20, sent as made | 0% and 2.1% | 8.8 and 11.1 ms |
| Drawn at 22 on the camera at 20 ("Even delivery" off) | 0%, but 9% of frames never arrived | 16.6 ms |

Reading through ffmpeg adds a few ms to every arrival, so about 8 ms of step error is the floor of this measurement. The 2.1% of the second run is the reader's pick ups falling close to the moments frames are written, so that one now and then is missed or shown twice.

**Switch the virtual camera on before the meeting opens it.** The same check found that a reader keeps the rate it was offered when it opened the camera (read at 20 a second, it went on reading at 20 after the writer was opened again at 30), and that with no writer OBS offers 30. A meeting that opened the camera before NeuroPresence's was on therefore reads it at 30, and the even 20 is lost. So the engine switches the camera back on when it starts if it was on when it stopped (showing only the plain card until a camera session runs), and the Live Studio says so; if a meeting had it open first, turn the meeting's camera off and on once.

**The voice mode** draws at about 16.5 frames a second and cannot be held to 20. On the camera at 20 it gets 21% of its frames held half again as long (step error 21.8 ms, both runs), against 0 and 12% before (15.1 and 24.5 ms): about as before, a little worse on average. It would be even on a camera at 30 with the drawing held to 15 (7.7 ms), but a meeting keeps one rate, and the camera mode is the one the camera is set for.

Meeting applications crop the camera picture to fill their tiles: a wide tile takes off the top and bottom, a narrow one (a phone held upright) the sides. In a Google Meet call on 18 October 2026 the main tile was 1.97 to 1 and cut the mark, then in the lower left corner, in half. So the mark now sits in the middle of the lower edge, which a crop from the sides never reaches, and the picture is kept within the middle 84% of the device's height, which a tile up to 2.1 to 1 leaves whole.

Checked on 14 October 2026 (Windows 11, OBS Studio 32.2.2) by reading the device back by name while a stand-in session ran: the card before and during the check, the marked output once it ran, the marked still picture when another face sat down, the card again after the session stopped. The session ran at 21.1 frames a second and 90 ms with the camera on, as without it. Not yet done: a call in Google Meet and in Zoom, and the same on Kubuntu.

## Voice in step

The picture a meeting sees is late by the time the system takes to make it, about 90 to 110 ms; the voice, sent straight from the microphone, is not. A virtual microphone holds the voice back by the same amount (`neuropresence/server/virtual_microphone.py`), so lips and words arrive together.

**What to install once.** On Windows, VB-Audio Virtual Cable (vb-audio.com/Cable, the free "VBCABLE_Driver_Pack"; run `VBCABLE_Setup_x64.exe` as administrator, then restart). It adds "CABLE Input" to the speakers and "CABLE Output" to the microphones; neither should be made the Windows default. On Kubuntu nothing is installed: `pactl load-module module-null-sink sink_name=NeuroPresence sink_properties=device.description=NeuroPresence` creates the cable. The Python side is `sounddevice`, which `requirements.txt` installs.

**How to use it.** In the Live Studio, switch on "Voice in step" and choose the microphone you speak into. In the meeting application choose "CABLE Output (VB-Audio Virtual Cable)" as the microphone, and the virtual camera as the camera.

**How long the voice is held back.**

| When | The voice is held back |
|---|---|
| No camera session, the prompt is being answered, or a sample clip runs (the meeting sees the card) | 40 ms, the least the two sound devices need |
| A camera session is running | The session's delay from camera to output, plus half a frame of the virtual camera when it is on, plus the adjustment |

The delay follows the session's figure only when it moves by more than 8 ms, and gets there by playing the voice one per cent faster or slower, which no one hears. A change of more than 40 ms (a session starting or stopping) is made at once with a 2.5 ms fade. The two devices keep their own clocks; the same catching up keeps their drift from adding up.

**The adjustment.** The meeting application adds delays of its own that cannot be measured from here, and so do the camera itself and the cable. The adjustment, 10 ms a step and up to 200 ms either way, is for those: words before lips, raise it; lips before words, lower it. The chosen microphone and the adjustment are kept between runs; the switch always starts off.

Nothing is recorded. The voice is held in memory for the length of the delay; the buffer holds two seconds and is written over all the time.

Checked on 15 October 2026 (Windows 11, VB-Audio Virtual Cable, WASAPI at 48 kHz) without opening a microphone: the cable's far end was fed back in as the microphone, so one click went round and round, and the time for each trip was measured. At 40, 100 and 150 ms the trip took the set delay plus 100 to 112 ms, the same to a tenth of a millisecond on every trip, with no catching up once settled. Those 100 to 112 ms are the cable and the Windows sound buffers, and they come on top of the delay. Not yet done: a call in Google Meet and in Zoom with a clap seen and heard, which settles the adjustment, and Kubuntu.

## Measuring during a call

The committee asked what the system does when a meeting application runs beside it on the same computer. `scripts/measure_call.py` measures it in three phases of two minutes: no meeting, a Google Meet call, a Zoom call.

1. Start the engine and a camera session in the Live Studio, with "Virtual camera" and "Voice in step" switched on.
2. Run `python scripts/measure_call.py` and follow it. For each call, join from a second device (a phone) so that video really flows, and keep talking and moving.

Once a second it reads the engine's own figures from its status (frame rate, delay, render time, dropped frames, and whether the still picture was up and why), the graphics card from `nvidia-smi` (how busy, memory, power, clock), and the CPU time of the engine, the browser and Zoom (`psutil`). After each call it asks whether the phone showed the picture, whether the mark could be read, whether the voice was heard and whether lips and voice looked in step. Everything goes to `results/call_measurements_windows_rtx5050.json`, kept after every phase, and a table of the phases side by side is printed at the end. `--phases zoom` measures one phase again and keeps the others; `--seconds 60` makes the phases shorter. Nothing of the picture or the sound is recorded.

Measured on 23 October 2026 (Windows 11, RTX 5050, two minutes a phase, a phone as the second person): with no meeting, in Google Meet and in Zoom the frame rate was 20.0, 20.2 and 20.2 a second, the delay from camera to output 95 ms in all three (never over 150 ms), the render time 39.5, 38.7 and 39.1 ms, and the still picture was never needed. On the phone the picture, the mark and the voice arrived in both, with lips and voice in step. The card was about 86% busy and at its power limit in every phase, so the engine, not the meeting, sets the frame rate. One person at one camera; Zoom's own CPU time was not counted. Microsoft Teams was tried by hand by Zain in November 2026 and worked; its figures have not been measured.

## The voice driving the picture (feasibility)

The proposal's live mode has a second form: with the camera off, the user's voice drives the picture (M10, the audio driven live mode). The first step, a measurement of whether it is feasible on this card, is built in `neuropresence/speech/`; it is not yet a switch in the Live Studio.

- `mouth.py` reads how open the mouth should be from the loudness of the voice, in steps of 10 ms, judged against the room's quiet and the voice's own loudest moments, so the microphone's level does not matter. No network is used.
- `motion.py` makes the movement the engine draws: the mouth along the direction in which LivePortrait's lip keypoints move when a mouth opens, a blink every few seconds along the eye keypoints, a slow sway of the head of about a degree and a slight nod with loud speech. The directions are fitted on LivePortrait's bundled clips (`directions.json`). The engine draws it with `engine.drive(..., motion=...)` from the picture's own pose (`engine.source_reading()`), so the mark, the virtual camera and everything after the engine stay as they are.
- `scripts/study_speech.py` fits the directions, times the engine driven both ways, and measures the voice's mouth against the real one on recordings with sound. `python scripts/record_camera.py --speech --name <you>_speech` makes such a recording: about a minute of vowels, sentences full of lips that meet, and free talk, with the microphone recorded in step with the camera, kept in `data/recordings/`.

Measured on 23 October 2026 (RTX 5050, frames drawn one after another): driven from speech the engine drew 26.1 frames a second, against 23.3 driven from camera frames, because the motion network (5.1 ms a frame) and the face tracker are not needed; the render time is the same. On a recording of Zain speaking at his camera with his headset (66 s, `record_camera.py --speech`), the voice's mouth was open in 45% of the frames against 40% for his real one, within a frame of it in time, and shut in quiet stretches; its agreement with his real lip gap was r = 0.44, best on read sentences (about 0.5) and weakest on free talk (0.14). At his widest openings the picture's mouth opens less than his. That is about as far as loudness alone goes: it cannot tell lips that meet on an "m" from a vowel at the same volume.

## Camera off: the voice drives the picture

In the Live Studio, choose the input **Voice, camera off** and start the session. The camera opens for the liveness check and the face match, as for every camera session, and closes once they are passed. From then on your voice drives the picture: Ditto (`scripts/setup_ditto.py`) turns the speech from the microphone into the motion of your face, which our engine draws with the mark, to the Live Studio and the virtual camera. The microphone is the one chosen under "Voice in step".

The picture follows the voice about 0.9 s behind it. Switch on **Voice in step** for a meeting: it holds the voice back by the same measured delay, so that the meeting gets lips and words together. The delay watchdog does not act in this mode, since the picture is late on purpose; the identity guard still scores every frame.

The first session after the engine starts takes about half a minute to get ready (loading the voice model and compiling it for the card; a minute the first time on a computer).

Measured on 25 October 2026 (RTX 5050), with a recording of Zain fed in place of the microphone in real time: 16.5 frames a second drawn, the frame drawn a median 0.87 s after its speech was heard (95th percentile 0.91 s), 30 of 1,649 frames not ready at their moment and skipped, and the lips following his real lips at r = 0.69 to 0.74.

Measured on 27 October 2026 against Zain's own face driving the picture, over the same minute of speech: the voice driven picture blinks 16 times a minute (his face 12), its head turns and nods with a spread of 0.8 and 1.0 degrees (his 0.8 and 0.8), and its movement is as even as his (0.92 px of change in movement from frame to frame, his 0.98). For the most natural result, enrol a picture with the eyes open and looking into the camera: the voice mode keeps the picture's own eyes between blinks.

The lips open and shut as his did: measured by the gap between the lips, his real face had a median of 0.015 and was open 35% of the time, the picture driven by his face in camera mode 0.020 and 35%, and the voice mode 0.011 and 38%; the drawn lips follow his real ones within one frame (40 ms).

