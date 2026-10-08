# NeuroPresence: project history

A running record of what was built, what was decided and why, what was measured, and what went wrong. New entries go at the end of the log. Whenever an entry is added, the section "Where the project stands" is brought up to date.

## Where the project stands (9 October 2026)

| Stage | State |
|---|---|
| Enrolment (once, before any session) | Working, in three steps: the face is verified live with the camera and only its signature is kept; five poses are registered (facing the camera, turned to each side, tilted up and down), for a richer signature and a measured range of motion; the meeting picture is uploaded, checked, and accepted only if it shows the same face |
| 1. Capture | Working: camera or recorded clip, 478 face landmarks per frame, newest-frame pacing, a face crop that holds still while the head does |
| 2. Motion encoding | Working: head pose, expression signals; the keypoints are steadied, and a lost face fades to the still picture |
| 3. Reenactment | Working, 18 frames per second live against a target of 24; the render target is met |
| 4. Identity | Working: the output is scored against the enrolled picture twice a second; a drop that lasts takes a fresh neutral pose, and if that does not help the still picture is shown until the person resumes. A delay watchdog shows the still picture while the output arrives too late |
| 5. Consent and disclosure | Partly built: a meeting picture must match the face verified live. The check of the live face before each session, and the mark on the output, are not built |
| 6. Virtual camera (meeting apps) | Not built |
| Evaluation | Benchmark of speed, identity, motion and stability, with a saved baseline |
| Web app | Landing page, Enrolment and Live Studio; further screens are added with the features they belong to |

Measured against the targets (benchmark of 8 October 2026, RTX 5050, eight sample clips; the first benchmark, of 6 October, is in brackets where the figure changed):

| Measurement | Measured | Target |
|---|---|---|
| Frame rate | 18.7 fps in a live session, 16.4 in the benchmark (7.5) | at least 24 fps |
| Render time per frame | 37 ms (94) | at most 42 ms |
| Whole pipeline per frame | 61 ms in the benchmark; 89 ms from camera to output in a live session (134) | at most 150 ms end to end |
| Peak GPU memory | 1.28 GB, read from the driver so that it includes TensorRT (1.17) | at most 8 GB |
| Identity match (CSIM), self-reenactment | 0.90 | at least 0.80 |
| Lag behind the driving face | 0.07 frames | none |
| Flicker, against real video | 0.74 times (0.89) | at most 1 time |
| Head jitter, against real video | 0.98 times (1.49) | at most 1 time |

Tests: 248 for the engine and server, 37 for the web app, all passing.

Dates: the mid evaluation is planned for 28 October 2026 (the team's working assumption). The FYP-I final date has not been announced; the progress plan uses 7 December 2026 as a placeholder.

## How the work is done

Agreed on 6 and 7 October 2026.

1. One feature at a time, in the order the product flows. Each is finished before the next one starts.
2. The web app grows with the features. The screen or control for a feature is built together with that feature, so each one can be tested in the app as soon as it exists.
3. Quality comes before pace. A feature is tested, measured and checked by eye before the next one starts.
4. Each step is described first and started only after it is approved.
5. Commits and pushes are made by Zain after each completed feature.
6. Every figure the project shows is measured. Nothing is simulated, and what is not built is labelled as not built.
7. This file gets an entry for every completed feature.

### Order of work

Proposed and approved on 7 October 2026:

1. Enrolment (built on 7 October 2026)
2. Capture and tracking (built on 8 October 2026)
3. Reenactment (done on 8 October 2026: render and delay targets met, frame rate at 19 of 24; a sharper face crop is measured and available, off by default)
4. Identity (built on 9 October 2026, with the delay watchdog)
5. Consent and disclosure (next)
6. Virtual camera and meeting apps

## Before this repository

- **25 August 2026.** Date of proposal version 5.4: real-time face reenactment for meetings on an 8 GB consumer GPU, with a consent gate and a disclosure watermark. Three contributions are claimed: the optimized engine, a meeting-condition benchmark, and the safeguard subsystem.
- **Proposal-stage measurements** (from the proposal): unoptimized LivePortrait on the RTX 5050 ran at 7.7 fps, 130.1 ms per frame and 1.57 GB. Keeping weights in half precision left latency unchanged and cut memory by 26%, which pointed at many small unfused GPU operations as the bottleneck. F5-TTS, OpenVoice and Wav2Lip were shown to run on the same GPU for the later offline mode.
- **A user-interface prototype** with simulated data was built in a separate repository (`zain31197/NeuroPresence_Prototype`). It is not used by this project's web app.

## Log

### 5 October 2026: proposal and rubric reviewed

The proposal and the FYP-I mid-evaluation rubric were read in full. The rubric gives 25 of 100 demo marks to a live core module with failure cases, and asks for commits from all members, tests, design diagrams that match the code, and disclosure of external and AI tools.

Inconsistencies found in the proposal, to be settled in the mid report:

- The bandwidth argument says the system sends compact motion parameters, but the design renders locally and the meeting app still sends full video.
- The watermark is called optional in some sections and always-on in others.
- Section 4.1 describes two modes in English only; other sections describe three modes and Urdu.
- The fallback on failure is a pass-through camera in one section and a static enrolled frame in another.
- "No live face, no output" is given as a safeguard, which the planned phone-relay audio mode would not satisfy.

### 6 October 2026, early hours: repository, capture, reenactment, live loop

- **Repository.** `github.com/zain31197/NeuroPresence` created, public. The proposal and rubric PDFs are kept out of it because they contain phone numbers and grades. The first eight commits were pushed the same day.
- **Progress plan.** `docs/NeuroPresence_Progress_Plan.pdf`: a daily checklist to the FYP-I final with the mid-evaluation deliverables.
- **Two platforms.** Decided to develop on both Windows 11 (the RTX 5050 desktop) and Kubuntu (a teammate's machine), so all code must run on both.
- **Capture and tracking** (`neuropresence/capture`). Frames from a camera or a video file; MediaPipe FaceLandmarker gives 478 landmarks, head pose and expression coefficients; frames with no face or with more than one face are flagged, not guessed.
- **Reenactment engine** (`neuropresence/reenactment`). GPU PyTorch 2.11 (CUDA 12.8) and LivePortrait (pinned commit `9b294b3`) installed. The engine prepares the source once and then drives it one frame at a time, timing each part. First run on a sample face: 7.9 fps and 126 ms per frame, of which 93 ms is the warp-and-decode step. This matches the proposal's baseline.
- **Live loop** (`scripts/live.py`). Camera to tracker to engine to a preview window, falling back to the still enrolled picture when no single face is in view. The source picture can be taken from the camera at start. Run on a real face through a phone camera app at about 7 fps.
- **Black band fixed.** With the head near the top of the camera frame, the face crop reached past the frame edge, was filled with black, and the band was dragged into view when the head moved. The crop now repeats the edge pixels, and the output is the whole source frame with only the face region replaced, so the background cannot move. On a test image the black area in the crop fell from 25.7% to 0.04%.

### 6 October 2026, evening: measurement and benchmark

- **Identity scoring** (`neuropresence/identity`). ArcFace embeddings compared by cosine similarity (CSIM). Faces are aligned with five MediaPipe landmarks, so the identity module does not depend on the reenactment core. Checked against InsightFace's standard alignment on 36 sample faces: scores differ by 0.015 on average, 0.077 at most, correlation 0.993 (`results/alignment_validation.json`).
- **Stability** (`neuropresence/evaluation`). Two measurements, because they catch different faults: warping error (RAFT optical flow) for flicker, and landmark jitter for a trembling head. Tests on synthetic clips show that the warping error cannot see a trembling picture, which is why both are needed.
- **Motion fidelity.** Head-pose error, expression error, mouth-opening correlation and lag against the driving face. These were added because CSIM alone rewards a face that does not move: a frozen copy of the source would score 1.0.
- **Benchmark** (`scripts/benchmark.py`, `benchmarks/samples.json`). Eight sample clips, seven of them self-reenactment. The baseline is in `results/benchmark_baseline_windows_rtx5050.json` and is summarised at the top of this file.
- **Findings.** Identity is above its target at baseline (0.90; real video of the same person scores 0.85 against the same picture). The output follows the driver with no measurable lag and a mouth-opening correlation of 0.96. It flickers no more than real video, but the head trembles about 1.5 times as much. Strong expressions come out weaker: expression error is about a third of the driver's expression movement.

### 6 to 7 October 2026, night: web app, part one

- **Server** (`neuropresence/server`, started with `python -m neuropresence.server`). A FastAPI application that owns the camera and the pipeline on a background thread, streams each camera frame together with the output made from it over a WebSocket, and reports live figures four times a second. It listens on this computer only.
- **Frame pacing** (`neuropresence/capture/feed.py`). A capture thread keeps only the newest frame, so the output never falls behind the camera. At the current speed about three quarters of camera frames are skipped. A recorded clip can stand in for the camera and is played at its own frame rate, which makes live tests repeatable.
- **Neutral pose.** When the picture is taken from the camera, that same frame becomes the neutral pose, so the output starts exactly aligned with the person.
- **Live identity check.** The output is scored against the enrolled picture about once a second on its own thread.
- **Feature switches** (`neuropresence/server/features.py`). Each feature registers one entry and gets its on/off switch in the app. The first two are reenactment and the tracking overlay.
- **Front end** (`web/`: React, TypeScript, Tailwind). A new design, not based on the earlier prototype: light theme, landing page, and the Live Studio with two monitors, five live figures against their targets, the time spent per stage, an event list and the enrolled picture. The landing page reads its results from the latest benchmark file and its stage statuses from the engine, so it cannot claim more than the code does.
- **Checked** in a browser at four screen widths against the real engine with sample clips. With the page open the studio measured 6.5 to 7.3 fps and 154 to 170 ms end to end, a little worse than the benchmark.
- **Enrolled picture** is stored in `data/enrolment/`, which git ignores.

### 7 October 2026: order of work agreed, history started

- The rules under "How the work is done" were set: features in the order of the product flow, the web app growing with them, quality before pace.
- Building the remaining web screens all at once was dropped in favour of adding each screen with its feature.
- Storage was discussed. No database is used: pictures and results are files in the project's own folders. The current position is to add a small local database only when there is a library of sources, consent records, or history to compare.
- This file was started.

### 7 October 2026: enrolment, the first feature in the agreed order

The order of work was approved, and enrolment was built first: taking the one picture that every output frame is made from.

**What was built**

- **Checks** (`neuropresence/enrolment/checks.py`). Seven checks on a picture and its tracked face: one face in view; facing the camera (yaw within 12°, pitch within 15°, roll within 10°); face large enough (at least 200 px tall, at most 62% of the picture's height); whole head in frame; enough light; sharp; relaxed face with the eyes open. Each check that fails says what is wrong and what to do, in two sentences. A check that passes can add advice: come closer below 300 px of face height, more light below a sharpness of 45.
- **Two checks built not to judge looks.** Light is read from the brightest parts of the face (the 95th percentile of its brightness) and from blown-out pixels, never from the average, so a dark-skinned face in good light passes (there is a test for exactly that). Eyes count as closed only above a score that no open eye reached in the sample faces, because the tracker scores open eyes differently from person to person.
- **Storage** (`neuropresence/enrolment/store.py`). Four plain files in `data/enrolment/`: the picture, its face crop (the neutral pose), its face signature (ArcFace embedding) and a record of when it was enrolled and what the checks found. The record is written last, so an interrupted save reads as nothing enrolled. No database.
- **Server.** A camera preview that runs the tracker and the checks on every frame without loading the animation model. Taking a picture keeps the best frame of a half-second burst, scored by sharpness and how open the eyes are. A picture, taken or uploaded, is a candidate until it is confirmed; one that fails a check can be looked at but not enrolled. A camera session now needs an enrolled picture; a sample clip still animates one of its own frames and enrols nothing. The preview closes the camera by itself when no page has asked for a frame for 15 seconds.
- **Enrolment screen** (`web/src/screens/enrolment`). The camera, mirrored, with a face outline and the one thing to fix next written on the picture; a list of the seven checks with the readings that depend on the camera (face height, light, sharpness); a count of three before the picture is taken; a review of the picture as others will see it, with keep or retake; upload through the same checks; the enrolled picture with its details and a way to remove it. Under it, two charts drawn from the study below, with the enrolled picture's own face height marked on the first.
- **Live Studio.** It no longer takes pictures. It shows the enrolled picture with a link to Enrolment, and says so when a camera session cannot start because nothing is enrolled.

**What was measured** (`scripts/study_enrolment.py`, `results/enrolment_study.json`)

Faults in the picture. Six sample clips, each animated from a good frame of itself and from faulty ones, every output frame compared with the real frame of the same instant:

| Source picture | Identity match (CSIM) | Share of fine detail kept |
|---|---|---|
| Good | 0.847 | 0.41 |
| Blurred | 0.068 lower | 0.13 |
| Dark | 0.040 lower | 0.14 |
| Mouth open | 0.023 lower | 0.40 |
| Low resolution | 0.021 lower | 0.19 |
| Head turned (found in 1 clip) | 0.015 lower | 0.44 |
| Eyes closed (found in 2 clips) | 0.003 lower | 0.36 |

Face size. Three sharp pictures, each enrolled at seven sizes. Detail in the generator's own output, against the same picture with a 260 px face: 0.33 to 0.58 at 150 px, 0.61 to 0.79 at 200 px, 1.10 to 1.21 at 300 px, 1.70 at 460 px.

**A plan that the measurement overturned.** The generator draws the face about 260 px tall, whatever the picture (measured: 255 to 261 px over 20 cases). The plan was to store larger pictures shrunk to that size, on the reasoning that anything larger is only stretched on the way back and makes the face softer than its surroundings. A first measurement seemed to agree: with a 460 px face the output kept 0.21 of the picture's detail at its own resolution, against 0.31 at 260 px. Measured on the generator's own output instead, the detail in the face itself was 1.7 times higher from the 460 px picture, and side-by-side pictures showed it. Shrinking would have made the face less sharp in order to make the frame match it. The plan was dropped before it was built, the minimum face height was raised from 150 to 200 px, and the screen now asks for a larger face.

**Checked** in a browser at widths from 390 to 1920 px against the real engine, with a sample clip standing in for the camera (`--camera-file`): empty, camera with a failing check, ready, countdown, review, keeping the picture, the enrolled state, a failed and a passed upload, a camera that cannot be opened, a session running, and the camera closing when the screen is left. A camera session then animated the enrolled picture at 6.6 fps with an identity match of 0.88; the picture and the driving video came from the same sample clip.

**First run on a real camera, the same day.** Zain opened the Enrolment screen on his phone camera (1280 x 720). Sitting at an ordinary distance his face was 396 px tall, 55% of the picture's height, with light 193 and sharpness 31. Six of the seven checks passed; the hair touched the top edge, so "Whole head in frame" failed. The run showed a fault in the screen, not in the checks: the face outline was drawn for a face 47% of the picture's height and 0.72 as wide as it is tall, so a face at a good distance overflowed it, and the hint said "move down or back" when moving back was the wrong thing to do. Three things were changed:

- The outline is now worked out in `checks.py`, beside the limits, and sent to the app with the preview. It asks for a face 54% of the picture's height (389 px at 720p), with room above it of 0.42 of the face's height, and is 0.85 as wide as tall. That width was measured: the tracked face of 41 front-facing sample faces is 0.75 to 0.97 as wide as tall, median 0.85. In a picture taller than wide the outline is limited by the width instead.
- Tests assert, for five shapes of camera picture, that a face filling the outline passes the size and framing checks, and still passes when it is a tenth larger or smaller or 4% of the picture off in any direction.
- The framing hint now depends on the size of the face. A face about the outline's size that sits too high is told "Tilt the camera up a little, or sit lower"; only a face clearly larger than the outline is told to move back.

Sharpness 31 on the real camera is above the limit of 18 but below the 45 at which the output face itself still counts as sharp. Four of the sample clips score in the same range (22 to 35), so the limit does not need changing for it, and on this camera it is the camera, not the checks, that limits how sharp the output can be.

### 7 October 2026, evening: the face and the meeting picture separated

Zain's instruction, after trying enrolment on his own camera: the picture taken with the camera should be used for the face embedding only, the picture for meetings should be one the person uploads (a good one, with the background they want), and it should be used only if its embedding matches the face from the camera.

This also closed a gap in the first version, where any picture could be uploaded and animated, of anyone.

- **Step 1, the face.** The camera, the outline and the seven checks are as before, but confirming the picture now stores only its face signature (`identity.npy`, with a record of the checks). The picture is dropped. The animation model is not needed for this step.
- **Step 2, the meeting picture.** Upload only, and only once a face has been verified. It gets the seven checks and an eighth, "Same person as your face": the similarity of its signature to the verified one has to reach the limit. A picture that fails is shown with the reason and cannot be used.
- **The limit, measured** (`scripts/study_same_person.py`, `results/same_person_study.json`). On 25 people from the sample clips and photographs: 300 pairs of different people, highest similarity 0.27; 550 pairs of the same person, lowest 0.42. The limit is 0.35. A first run put dogs and a monkey from the sample set among the "people" and scored them as alike (up to 0.74), and counted a child's clip and a photograph of the same child as two people (0.90); both were found by looking at the most alike pairs and are now handled in the script.
- **Kept consistent.** Verifying a different face removes a meeting picture that no longer matches it. Removing the picture leaves the face; "Forget my face" removes both. A picture enrolled from the camera under the first version is kept, and its signature, which was taken live, becomes the verified face.
- **Checked** on the real engine and identity model with sample material: a face verified from one clip, a picture of another person refused at a similarity of 0.02, and a picture of the same person accepted at 0.98 and enrolled as the meeting picture. It has not been tried yet with Zain's own camera and photograph.

**8 October 2026: an uploaded picture came out with the head too large.** Zain enrolled an upright photograph and ran a session from his camera: the head was enlarged and the body looked thin beside it. Cause: movement was still measured from the picture's own pose, which was right while the picture came from the same camera and seat, and wrong for a photograph from another camera. The difference between the two was applied as if it were movement. Reproduced on sample material: a photograph driven from a differently framed camera came out with the head 0.87 times its size and turned 14 degrees while the person sat still. The neutral pose is now the person's own resting face at their camera: when a session starts the still picture is shown until a frame is seen with the face to the camera, mouth closed and eyes open (three seconds at most), and all movement is measured from that frame. With that, the same test gives the picture unchanged at rest, and the head stays within 1% of its size while the person moves. Checked on the real engine with an upright picture and a camera of another framing: normal proportions, identity match 0.95.

What this leaves for the consent stage: the same comparison on the live face before each session, and the measurement repeated on the team's own pictures, where a picture from another day and camera will score lower than frames of one clip do.

### 8 October 2026: capture and tracking, the second feature

Approved by Zain with the instruction to keep the focus on quality. The aim was a steady signal driving the picture: the output's head trembled 1.49 times as much as real video.

**Where the tremble came from, measured** on four sample clips before anything was changed:

| Face crop made this way | Tremble against real video |
|---|---|
| As it was: cut fresh around the tracked box of every frame | 1.50 |
| The same, placed in fractions of a pixel | 1.47 |
| One fixed window for the whole clip | 1.41 |
| A window built from the eye corners | 1.93 |
| The tracked box, filtered | 1.08 to 1.14 |

So rounding was not the cause, and a crop that never moves still trembled: part of it is the model's own reading. Two things were then tried on top of the filtered crop. Filtering the pose angles the model reads made it worse (1.19 to 1.24, with half a frame of delay). Filtering the keypoints given to the generator worked: 0.76 with settings that cost three quarters of a frame, and 0.96 with settings that let go as soon as anything moves.

**What was built**

- **Steady crop** (`neuropresence/capture/steady.py`). The crop window's position and size go through a One Euro filter, which holds hard when the head is still and follows when it moves. Readings carry their own time, so it behaves the same at the live loop's 7 frames a second and a clip's 30.
- **Steady head.** The same filter on the keypoints the engine gives the generator.
- **Hold and fade.** When the face is lost, the last live frame is held for a quarter of a second, because the tracker often misses a single frame, and then fades to the still picture over a third of a second. When the face returns the output fades in. Before, it snapped.
- **In the web app.** A switch for each of the two filters, so each can be seen on and off; a Tracking panel in the Live Studio with head turn, head nod and mouth opening as live traces; and the tracking overlay now draws the window the face crop is cut from.

**Result, full benchmark on eight clips** (`results/benchmark_steady_windows_rtx5050.json`): head tremble 1.49 to 0.99 of real video, flicker 0.89 to 0.75. Nothing else moved: identity 0.896, pose error 0.53 degrees, expression error 0.026, mouth correlation 0.96, delay 0.07 of a frame, 7.4 frames a second. Five of the eight clips are now steadier than their real video; three are not (1.36, 1.15 and 1.20).

**The head thrown far back, the same day.** The rule for a head turned too far had been left out of this step, because where the output breaks had not been looked at. Zain then leaned his head back 46 degrees on his camera: the output followed all the way, with the head enlarged and distorted on a body that stayed still. The same picture was rendered with the head turned in steps: natural up to about 12 degrees of nod and 18 of turn, the face stretching from 20 degrees of nod, distorted at 30 to 45. A real head goes further only because the neck and shoulders go with it. The engine now keeps the head to that range (`POSE_RANGE_DEG` in `reenactment/engine.py`): movement is followed exactly up to 8 degrees of nod, 12 of turn and 8 of tilt, then eases toward 15, 22 and 15, which it never passes, and the head keeps its size against the body within 7%. It has its own switch, "Natural head range". Rendered again with the head back 36 degrees: a gentle tilt at normal size. The benchmark is unchanged by it (pose error 0.53 degrees, tremble 0.98, flicker 0.73), because the sample clips stay inside the free range.

### 8 October 2026: reenactment, first round of speed work

Approved by Zain with the instruction to aim for the targets and for something usable in real life. Speed was the largest gap: 7.3 frames a second against 24, with the render step at 93 ms against 42.

**Where the time went, measured** (RTX 5050, each network timed alone): generator 62 ms, warping network 31 ms, the network that reads the movement 16 ms, pasting the face back into a 960 x 1280 picture 25 ms on the CPU, bringing the output to the CPU 2 ms.

**What was tried**

| Change | Result | Kept |
|---|---|---|
| A different convolution setting (cudnn.benchmark) | No gain | No |
| Half-precision weights for the two networks that draw the picture | Render 93 to 86 ms; picture changed by 0.15 of 255 | Yes |
| The same for the network that reads the movement | 5 ms faster, but the benchmark showed tremble back at 1.46, flicker at 1.12 and mouth correlation down to 0.93 | No |
| Channels-last memory layout | Slower (96 ms) | No |
| Pasting the face back on the GPU | 25 to 6 ms on a large picture; the same picture to within 2 of 255 | Yes |
| Compiling the two drawing networks for the GPU | Render 85 to 66 ms; picture changed by 0.19 of 255; about a minute at start-up | Yes |
| The compiler in its most thorough mode | 63 ms for a start-up of nearly three minutes | No |

Compiling needs Triton, which PyTorch does not ship for Windows; the `triton-windows` package provides it and is now in `requirements.txt` for Windows only. If compiling fails on a machine the engine runs uncompiled.

**Result, full benchmark on eight clips** (`results/benchmark_fast_windows_rtx5050.json`): 7.3 to 9.9 frames a second, render 93 to 67 ms, the whole pipeline 138 to 101 ms, GPU memory 1.17 to 0.67 GB. Quality unchanged: identity 0.897, pose error 0.53 degrees, expression error 0.026, mouth correlation 0.96, tremble 0.98, flicker 0.74.

**The lesson of the rejected change.** Putting the movement-reading network into half precision looked harmless on one picture and was only 5 ms faster. The full benchmark showed what it cost. Every speed change is judged by the whole benchmark, not by a timing.

**A held posture looked glued on, the same day.** With the head range limited, Zain leaned back in his chair and stayed there: the output held the head tilted at its limit for as long as he sat like that, on a body that never moves. A posture is not a gesture. The engine now treats a pose held beyond the free range as the new rest position: over about three seconds the head eases back to how it sits in the picture, and movement inside the free range is never touched. Rendered on the real engine with the head back 36 degrees and held: 20 degrees from rest after 2 seconds, 10 after 4, under 3 after 8. It is part of the "Natural head range" switch. In the benchmark the pose error rose from 0.53 to 0.63 degrees, which is the cost of deliberately not following a held pose; identity, expression error, mouth correlation, tremble and flicker are unchanged.

**Second round, approved by Zain the same day.**

- **Tracking beside the GPU.** The face is tracked on its own thread while the GPU draws the frame before. Live: 9.9 to 10.5 frames a second. Less than estimated, because tracking was only 12 ms of each frame; the GPU sets the pace.
- **TensorRT** (`neuropresence/reenactment/accelerate.py`). The generator went from 55 to 25 ms and the warping network from 29 to 14 ms, both in half precision as before. The network that reads the movement went from about 15 to 5 ms in full precision; its readings match the PyTorch version to a thousandth of a degree. Converting takes about a minute and a half the first time on a machine; the engines are then kept in `models/tensorrt` and load in a second. Without TensorRT the engine falls back to PyTorch's compiler, and without that to plain PyTorch.

**Result** (`results/benchmark_tensorrt_windows_rtx5050.json`, and a live session measured with a clip as the camera): render 66 to 37 ms, which meets the 42 ms target; 18.2 frames a second live, with 86 ms from camera to output; 16.4 in the benchmark, which tracks before it draws. Quality unchanged on every measure: identity 0.898, pose error 0.64 degrees, expression error 0.026, mouth correlation 0.955, tremble 0.99, flicker 0.75.

**Third round, the three smaller steps.** The estimate for them had been about 22 frames a second. Measured, they gave half a frame a second:

| Step | Result | Kept |
|---|---|---|
| TensorRT built at its highest optimisation level | Render 37.5 ms against 37.8; a slower first build | No |
| Face crop and movement reading on the tracking thread | 19.8 frames a second, but 121 ms from camera to output, up from 86: the reading needs the GPU, so it queued behind the frame being drawn and held the next one back | No |
| Only the face crop on the tracking thread, started so it is ready as the GPU comes free | 18.7 frames a second, 89 ms from camera to output | Yes |

**Where this leaves the target.** The render target (37 ms against 42) and the delay target (89 ms against 150) are met. 24 frames a second is not: the live session runs at 18.7. The GPU is now busy for about 46 ms of every frame, 38 of them drawing, so the frame rate is set by the generator and the warping network themselves, and the steps around them are used up. Getting from 19 to 24 on this GPU needs the drawing itself to be about a fifth cheaper: a smaller generator, lower-precision arithmetic (which has to be judged on quality), or a faster GPU. The sharper face crop found during enrolment has not been tried yet.

**8-bit arithmetic tried and rejected.** Zain asked for it to be tried, and dropped if quality fell by much. The generator's convolutions were calibrated on 46 of its real inputs from four sample clips and built in 8-bit; 12 other inputs were kept apart to judge by. It was faster, 14.8 ms against 24.8, which would have brought the live session to about 23 frames a second. But its picture differed from the half-precision one by 28 of 255 on average, a signal-to-noise ratio of 17.5 dB: not the same picture. It was not put into the engine. This was the plain method, every convolution quantised from its smallest and largest values. Quantising only the layers that tolerate it, or training the network for 8-bit, might do better and is a piece of work of its own.

**The two things left over in this step.**

- **The GPU memory figure.** PyTorch's count leaves out what TensorRT holds, so the app and the benchmark showed too little. The engine now reads the drop in the card's free memory since it began loading, from the driver: 1.28 GB. Other programs starting or stopping on the GPU can shift that reading.
- **The sharper face crop: measured, and left off.** The model draws about as sharply as the crop it is given. On five pictures, the output's fine detail against the picture's own was 0.73; averaging the picture down more carefully made it softer (0.76 of that); a Lanczos filter raised it to 0.99 on average but unevenly (0.64 to 1.80); strengthening the crop's edges before the model sees it raised it steadily: 0.86 at a strength of 0.05, 0.93 at 0.1, 1.00 at 0.15, 1.08 at 0.2, with identity unchanged at 0.99. The full benchmark then showed the cost: at 0.1 and at 0.2 alike the head tremble went from 0.99 to 1.035, past its target of 1.0, and flicker from 0.75 to 0.81 and 0.86. A run with the sharpening off confirmed it was the cause (tremble 0.994). It is in the engine as `SOURCE_SHARPEN` and set to 0, so every target met stays met. Turning it to 0.1 is a choice between a visibly sharper face and 4% more tremble, and is left to Zain.

**Decision, 8 October 2026.** The speed work stops here for FYP-I at about 19 frames a second live, with the render and delay targets met and quality intact. The 24 frames a second target stays open and is recorded as not met.

### 8 October 2026: enrolment, uploaded-picture checks and pose registration

Approved by Talha (reenactment and identity owner) after testing enrolment on his own camera and a real uploaded photo, using Claude Code to write and test the change, as the rest of this log discloses.

**What was found.** Uploading a casual, real workplace photo (sitting back from the camera, more of the room in view) turned up two faults. First, the size check's hint said "Move closer to the camera," which is an instruction for someone live at a camera, not for a photo already taken; every check an upload fails reused the camera's own wording regardless. Second, the photo was refused on size alone (103 px of face height, under the 200 px floor) even though its face signature matched the verified face at 0.74, comfortably over the 0.35 limit: the size check protects how sharp the output looks, which is a different question from whose face it is, and the two had not been told apart.

**What was built**

- **Upload-specific wording** (`enrolment/checks.py`). Every check's hint now reads differently by origin: the camera keeps its live instructions ("Move closer," "Tilt the camera up"), an upload asks for a different photo instead ("Choose a photo where your face is larger"). `judge`/`evaluate` take an `origin` argument; the default is `"camera"`, so nothing already enrolled changes.
- **Reframing before the stored-size downscale** (`server/runtime.py: _upload_candidate`, `_reframe_on_face`). A wide upload with a small face is first cropped tightly around the face in the file as given, before it is brought down to the 1280 px stored size; downscaling the whole photo first, as the straightforward path did, throws away resolution on background the output never uses. If the file genuinely lacks the detail, the crop cannot invent it, and the picture is still refused, now for a reason about the photo rather than a camera instruction that does not apply to it.
- **Size becomes a warning once identity is confirmed** (`server/runtime.py: _accept_small_face_once_identity_is_confirmed`). Identity is who the picture is of; size is only how sharp the animation looks. Once ArcFace confirms an upload is the enrolled person, a face still too small after reframing no longer blocks it, only warns in the tip a passing check already carries. Left blocking when anything else also fails: a small face on a photo that is also blurred or badly lit is still refused, on that other ground.
- **Pose registration**, a new step between verifying the face and uploading the picture (`enrolment/checks.py`, `store.py`, `candidate.py`; `server/preview.py`, `runtime.py`, `app.py`; `web/src/screens/enrolment/PoseCapture.tsx`). Five more pictures are registered: facing the camera again, turned to each side, tilted up, tilted down. Each gets four checks instead of the frontal seven (one face, turned the right way, light, sharp); "facing the camera" reuses the step-one facing limits, the turned/tilted ones ask for 35° to 85° of yaw or 12° to 40° of pitch. Two things come of it:
  - A richer face signature: an uploaded picture or a live face is compared against the best match of up to six registered signatures (the step-one capture plus up to five poses), not only the first.
  - A measured range of motion (`reenactment/engine.py: calibrate_pose_range`). The generic clamp (free 12°/8°, most 22°/15°, yaw/pitch, set by looking at one person turned in steps) is replaced per person by their own registered left/right/up/down extremes, at the same free:most ratio the generic pair used, so the clamp fits how far this person actually turns instead of one earlier guess.
  - Uploading a new meeting picture is refused until all five are registered (`check_upload`); a meeting picture enrolled before this step existed is not retroactively blocked from being viewed or used, only a new upload is gated.

**Checked.** 217 engine/server tests (up from 182), covering the reframe and warning logic, each pose's angle window and its hints, store persistence and cleanup of the registered poses, the calibration arithmetic, and the API wiring, all passing; `tsc` and the web build clean; the 29 existing web tests unaffected. Walked through by hand on the real engine and a real camera: verify face, register all five poses with live hints responding to real head angles, upload blocked until they are done, then accepted.

**Open items added**

- The calibrated range of motion has been exercised through the server's own tests, not yet watched on a live session actually using one person's own registered poses while they move.
- Whether a second frontal reference sample ("facing the camera" registered again, beside the step-one capture) measurably improves the identity match over the single step-one sample alone has not been measured.

### 8 October 2026: mid report

Zain asked for the FYP-I mid report before step 4: LaTeX source and PDF in the university's template, following the mid-evaluation guidelines and the report rubric (Form 3), with a dummy picture wherever a diagram or screenshot belongs, because Sana Ullah draws the diagrams by hand, and with no hyphens in the text. A first draft came to 175 pages. Zain then asked for 115 to 120 pages, written as the state of the work at the mid evaluation, and gave the proposal committee's decision of 14 September 2026 (approved with minor modifications) with its comments.

- **Where.** `F26_FYP1_Mid_Report_LaTeX_Complete/`, which git ignores: `main.tex`, the files in `chapters/`, `references/references.bib` (23 sources, all cited), 13 dummy pictures in `ThesisFigs/` (nine diagrams, four screenshots), `DIAGRAMS_TO_DRAW.txt` (what each diagram has to show), `make_generated_tables.py` (writes the commit table and the list of tests from the repository; run before each build) and `NeuroPresence_FYP1_Mid_Report.pdf` (118 pages, 56 tables).
- **Content.** Every chapter and appendix of the template is filled from the result files, the README and the code: numbered requirements by module with acceptance checks and a traceability table; a design chapter that describes each diagram box by box; the benchmark per clip. Both test suites were run for it: 182 passed in 75 s, 29 passed.
- **The committee's comments.** Ten points, each with an action, a status and evidence in the action register (chapter 3 and appendix A): presentation, latency and its mitigation, resource use, holding the targets beside meeting software, the original contribution against DeepFaceLive and LivePortrait, the fallback when lips lose synchronization, priorities, what is cut first, liveness, and Urdu.
- **Changes to the template, in `main.tex` only.** `longtable` and `booktabs` are loaded and `\projecttitle` and the student names defined (the template uses them without defining them); citations are set to numbered (the class asks for author and year, which stops the build with the IEEE bibliography style); the second copy of the contents lists is removed; chapters start on the next page, so there are no empty pages. Built with Tectonic, because the Windows machine has no LaTeX. It has not been compiled on Overleaf or with pdfLaTeX.
- **Written into the report as plans, for the team to confirm.** New planned requirements that answer the committee: a liveness prompt at verification and before each session (two random actions out of blink, turn left, turn right, open mouth, each within three seconds); a delay watchdog that shows the still picture while the delay is above 150 ms; the microphone delayed to match the video; the targets measured during a call in Meet and Zoom. The order of the second iteration (virtual camera first, the frame-rate attempt last and limited to one week). The order of cuts if time runs short (phone relay, then Urdu, then the audio-driven mode, then the second baseline). The five proposal inconsistencies of 5 October as settled before; owners for the two modules the proposal does not name (Enrolment: Zain; web app: shared).
- **Left for the team, four places marked "TO FILL".** What Talha and Sana Ullah have done so far, the tag for the mid evaluation, and the supervisor meeting log.
- **Seen in the benchmark file while writing.** Per clip, three clips tremble more than their real video (1.32, 1.08, 1.22) and one flickers more (d18, 1.38). The mean identity match is above 0.80 on every clip, but 22% of the frames of d6 and 59% of the cross-identity clip are below it.

### 9 October 2026: identity, the fourth feature, with the delay watchdog

Approved by Zain on 9 October. The score of the output against the enrolled picture was already measured live; this step makes the system act on it. The delay watchdog was built in the same step, because it ends in the same place: the still picture.

**What the score looks like, measured before any limit was set** (`scripts/study_identity_guard.py`, `results/identity_guard_study.json`). Normal use, the eight pairs of the benchmark, 1,128 frames: mean 0.89, the lowest frame 0.72, and 9% of frames below the 0.80 that is the target on average; the lowest mean of any three seconds was 0.82. Then four clips with one thing wrong at a time:

| Fault | Mean | Lowest frame |
|---|---|---|
| Head pushed 30 / 45 degrees past its range, head range off | 0.73 / 0.55 | 0.59 / 0.44 |
| Head turned 45 degrees further, head range off | 0.65 | 0.47 |
| The same two at 45 degrees with the head range on | 0.85 / 0.83 | 0.70 |
| Mouth covered in the camera picture | 0.72 | 0.59 |
| Expression three times too strong | 0.79 | 0.43 |
| Head 0.75 times its size | 0.79 | 0.70 |
| Neutral pose taken at the least restful frame | 0.85 | 0.66 |
| Another person driving the picture | 0.77 | 0.60 |

Three things follow. Single frames cannot be judged. The natural head range is what keeps the face recognisable at large angles (0.55 without it, 0.85 with it), which had not been measured before. And the score does not see everything: a neutral pose taken at a bad moment, or another person at the camera, stay near the normal range. The second is the consent check's job, not this stage's.

**What was built**

- **The guard** (`neuropresence/identity/guard.py`). It reads the mean of the last three seconds. Below 0.75 it asks for a fresh neutral pose and the filters start again. If the next seconds are still below, or two readings in a row fall under 0.50, the still picture is shown and stays. A dip of a second or so does nothing. A second drop within 30 seconds of a neutral pose that helped goes straight to the still picture. It keeps no clock of its own, so it can be run over a recording.
- **The monitor** scores twice a second in place of once. One score takes the CPU 48 ms; at four a second the frame rate and delay of a session did not move (`results/delay_study.json`).
- **The delay watchdog** (`neuropresence/server/watchdog.py`). It reads the mean delay of the last two seconds. Above 150 ms the still picture is shown; the frames are still drawn, hidden, so the delay stays measured, and the live picture returns when the mean has been under 135 ms for a second.
- **In the session.** The pipeline can draw a frame without showing it (`step(show=False)`). Both holds use the hold and fade that a lost face already had.
- **In the web app.** Two new switches, "Identity fallback" and "Delay watchdog". The identity figure says what the guard is doing. When the still picture goes up for identity, a notice asks for five seconds whether to resume or stay, then stays, with one button to resume (`POST /api/session/resume`, which takes a fresh neutral pose and gives the output a few seconds to prove itself). A late picture gets its own notice and returns by itself.

**The guard run over the recorded scores**, two readings a second, each clip repeated as if the fault went on: nothing on any of the eight normal pairs; the still picture in every run with the head pushed 45 degrees and in 90% with it turned 45 degrees, about 4.5 s after the fault starts; in 75% at 30 degrees; in about half with the mouth covered, whose scores sit on the level; in 5% with the head range on.

**In a real session**, on the real engine and scorer, with the lower face covered in a sample clip from one frame to the next: the guard took a fresh neutral pose 3.5 s after the cover appeared, and the match returned to 0.91 while the cover lasted. **This showed a fault the recordings could not.** The cover had become part of the neutral pose, so once it was taken away the output was off (0.80, where it had been 0.86). A fresh neutral pose that helped is now watched: when the match steps down by 0.08 or more from what it reached, the neutral pose is taken once more. In the same run that came 2 s after the cover was removed, and the match returned to 0.86.

**How the delay behaves when the GPU is shared** (`scripts/study_delay.py`, `results/delay_study.json`; a second process keeps the GPU busy for a set share of the time, on a 480 px sample clip with no browser attached):

| GPU busy elsewhere | Frame rate | Delay, mean | Frames over 150 ms |
|---|---|---|---|
| Not at all | 20.4 | 82 ms | 0% |
| 25% of the time | 16.2 | 93 ms | 0% |
| 50% of the time | 13.0 | 110 ms | under 1% |
| 75% of the time | 9.9 | 134 ms | 13% |
| All the time | 7.3 | 171 ms | 96% |

The frame rate falls first and the delay follows slowly, because late frames are dropped and not queued. The limit is passed only when something else holds the GPU three quarters of the time or more. In a real session beside a fully loaded GPU the watchdog held the picture back 3.5 s after the load began and released it within two seconds of the load ending. This is the first measurement of the system beside other GPU work, which the proposal committee asked about; it is an artificial load, not a meeting application.

**Checked.** 248 engine and server tests (31 new: each level of the guard, the watchdog, the hidden frame, and both in a session through the server) and 37 web tests (8 new), all passing. The full benchmark is unchanged (`results/benchmark_identity_windows_rtx5050.json`): 16.3 frames a second, render 36.8 ms, identity 0.898, tremble 0.99, flicker 0.75. A session alone ran at 20.9 frames a second and 80 ms before the change and 20.4 to 20.9 and 81 to 83 ms after it. Looked at in a browser at 1440 px with a scripted session: the prompt, the notice after five seconds, the resume, the late picture and its return, with no errors in the console.

**Not the same figures as on 8 October.** The 18.7 frames a second and 89 ms of that day were a camera-sized picture; the delay study uses a 480 px sample clip, where a session runs at about 21 and 82 ms. The two are not a before and after.

## Decisions and their reasons

| Decision | Reason |
|---|---|
| Public repository; proposal and rubric PDFs kept out | The PDFs contain personal details |
| Windows and Kubuntu both supported | Team members own different machines; the proposal names Kubuntu |
| LivePortrait fetched by a setup script at a pinned commit; weights not in the repository | Reproducible, and the weights carry their own licence |
| Output is the full source frame with the face blended in | Stable background, no black borders, same shape as a webcam picture |
| Identity alignment from MediaPipe landmarks | Keeps the identity module independent of the reenactment core; validated against the standard method |
| Flicker and jitter measured separately | The flicker measure is blind to a trembling head |
| Motion fidelity measured beside CSIM | CSIM alone can be raised by damping motion |
| ArcFace and the face cropper run on the CPU | The installed ONNX Runtime GPU build needs CUDA 13 libraries that PyTorch does not ship; the CPU works on every machine |
| Newest-frame pacing, late frames dropped | Staying current matters more than showing every frame |
| Web app is local, with a new design | The proposal requires local computation; the earlier prototype was simulated data |
| No database so far (current position, open to change) | One user, one picture, results as files |
| The enrolled picture is checked before it is kept, and the limits are set by measurement | Every output frame is made from it; a blurred picture cost 0.07 of identity match and two thirds of the fine detail |
| The picture is stored at full size, never shrunk to the size the generator draws | Detail in the output face keeps rising with the size of the face in the picture: 1.7 times as much at 460 px as at 260 px |
| Minimum face height 200 px, and advice to come closer below 300 px | At 150 px the output has a third to six tenths of the detail it has at 260 px; 200 px is still within reach of a 640 x 480 camera |
| Light judged from the brightest parts of the face, not the average | The average would fail darker skin in good light |
| A camera session needs an enrolled picture; a sample clip animates its own frame and is never enrolled | The user's picture and the test material stay apart |
| The picture is the best frame of a short burst, taken after a count of three | A single frame can catch a blink or movement; the count gives time to look from the button to the camera |
| The enrolment preview closes the camera when no page is showing it | A camera left on with nobody watching is filming for nothing |
| The camera picture is used for the face signature only; the meeting picture is uploaded and must match it (Zain, 7 October 2026) | The person in front of the camera proves who they are, and chooses separately what people see; nobody else's picture can be animated |
| Two faces count as the same person from a similarity of 0.35 | Measured: different people scored at most 0.27, the same person at least 0.42 |
| The neutral pose is the person's resting face at their camera, not the picture's pose | A picture from another camera came out with the head resized and turned while the person sat still |
| Tremble is removed at its two sources, the crop and the keypoints, with a filter that lets movement through | Each was measured separately; filtering the pose angles instead added delay and made it worse |
| The frame rate is left at about 19 of the 24 targeted (Zain, 8 October 2026) | The remaining route, 8-bit arithmetic, reached about 23 but changed the picture badly; quality comes first |
| A speed change is kept only if the full benchmark shows quality unchanged | Half precision for the movement-reading network was faster and brought the tremble back |
| The identity score is judged as a mean over three seconds, with 0.75 as the level | 9% of normal frames are below 0.80; no three seconds of normal use averaged below 0.82, and outputs that look wrong average 0.73 or less |
| A fresh neutral pose is tried before the still picture | It costs the viewer a moment and cures a neutral pose gone wrong; the still picture stays until the person acts |
| A fresh neutral pose that helped is watched, and taken once more if the match steps down | Seen live: taken while the mouth was covered, it made the cover part of the neutral pose |
| The delay is judged as a mean over two seconds, and late frames are still drawn while hidden | Single frames pass 150 ms long before the session does; drawing hidden keeps the delay measured, so the picture can return by itself |
| A posture held for a few seconds becomes the rest position | A head held at an angle on a body that never moves looks glued on; a nod or a turn still shows |
| The head keeps to a range of movement and to its size | The body in the picture stays still; beyond about 15 degrees of nod or 22 of turn the head looks wrong on it |
| An upload gets its own wording for every check (Talha, 8 October 2026) | A picture already taken cannot be told to move closer or tilt up; the hint has to ask for a different photo instead |
| A small face is a warning, not a block, once identity is confirmed | Size protects sharpness, identity protects whose face it is; they are different questions, and only one of them is who it is of |
| Pose registration is required before a *new* upload, but not retroactive on an existing meeting picture | Forcing it on someone who already finished enrolling under the old flow would be a regression for no benefit to them |
| "Facing the camera" is registered again as one of the five poses, not reused from step one | It is the pose a person is actually in for most of a meeting, so it is the one most worth a second, fresh reference sample |

## Problems found and how they were fixed

| Problem | Cause | Fix |
|---|---|---|
| Black band at the top of the output | Source crop reached past the frame edge and was padded with black | Edge pixels repeated; face blended back into the full frame; regression test |
| Live stream dropped while the models loaded | The status call touched PyTorch while it was still being imported on another thread | GPU figures are read only after the engine has loaded; the stream survives a failed reading; regression tests |
| A server test failed about one run in five | It checked averages before enough frames had arrived | The test waits for 30 frames |
| Stage timings went stale with reenactment off | Stage averages were taken over reenacted frames only | Render time per reenacted frame is reported separately from the average-frame breakdown |
| LivePortrait clone stalled | The repository had moved to a new address | Setup script uses the new address |
| The plan to shrink large enrolled pictures would have softened the output | It assumed the generator's 512 px output caps the detail that is of any use; the first measure compared sizes after resizing, which hid the difference | Measured on the generator's own output before any resizing, and by eye; the plan was dropped and the minimum face size raised |
| On a narrow screen the guidance line covered the lower half of the face | The line is drawn on the picture, which is small at phone width | Below 640 px the same line is shown under the picture |
| On a real camera the face outline was smaller and narrower than a well-placed face, and the hint said to move back | The outline's size was chosen by eye in the web app, apart from the checks, for a face 47% of the picture's height; the measurements say a larger face is better | The outline is worked out beside the check limits and tested against them; the hint tells a well-sized face to shift, not to move back |
| With an uploaded photograph the head came out too large for the body | Movement was measured from the photograph's own pose, so the difference between the photograph and how the person sits at the camera was applied as movement | The neutral pose is taken from the camera, from the first frame at rest when a session starts; tests for the waiting logic |
| A fresh copy of the repository could not have built the web app | The ignore rule `data/`, meant for the enrolled picture, also hid `web/src/data/faceMesh.json`, which the landing page needs. Found on 7 October 2026, before the web app was first committed | The rule now names only the project's own data folder (`/data/`) |

## Open items

- The identity limits were set on short studio clips. A session of many minutes on a real camera has not been scored, and the level of 0.75 has to be checked on the team's own recordings.
- A covered mouth is caught in only about half the runs, because its scores sit on the level, and until the guard acts the mouth is drawn smeared.
- A fresh neutral pose taken by the guard is taken from whatever the camera shows at that moment. The step-down rule repairs the one case seen (a covered mouth); others may exist.
- After an identity fallback the still picture stays until the person resumes, and the prompt is in the Live Studio only. Once the output goes to a meeting through the virtual camera, the person may not be looking at the Live Studio.
- The delay watchdog has been tried against an artificial GPU load, not against a meeting application.
- The pipeline has been run on one team member's face only, and never on Kubuntu.
- The benchmark uses studio sample clips. It has to be repeated on the team's own webcam recordings.
- The phone camera app used so far draws a name banner into the picture, which ends up in the output.
- The proposal says "source clip"; the engine animates a single picture. Whether to support a clip as the source is undecided.
- The enrolment limits were set on sample pictures and clips. One real camera has been tried so far (Zain's phone camera: face 396 px, light 193, sharpness 31), and the checks have been run on it but no picture has been recorded as enrolled from it yet. Other cameras, the teammates' in particular, still have to be tried; the screen shows the readings so the limits can be adjusted from what real cameras give.
- The enrolment study is small: six clips and three pictures. A turned head was found in one clip and closed eyes in two.
- Even from a good picture the output keeps about 0.4 of the fine detail, and the generator's output is sharper from a larger picture although its network reads the face at a fixed 256 px input. Both point at how the face crop is resampled on the way in. To be looked at under Reenactment.
- The studio measures a little slower than the benchmark while the page is open.
- TensorRT and its engines have only been built and run on the Windows machine. On Kubuntu the same code should convert on first start, but that has not been tried.
- Three of the eight benchmark clips still tremble more than their real video (1.36, 1.15, 1.20). The filters were tuned on four clips and at 25 to 30 frames a second; the live loop runs at about 7, where they have not been measured.
- The natural head range's generic default was set by looking at one picture turned in steps, not by a measurement over many faces; it is now replaced per person by their own registered poses (8 October 2026), but that calibration itself has not been watched on a real camera yet, and no one person's registered range has been checked against how far they can actually, comfortably turn.
- The hold and fade have been tested with stand-ins, not watched on a real camera.
- The progress plan does not yet list the web app.
- The proposal inconsistencies listed under 5 October are settled in the draft of the mid report (8 October 2026); the team has not confirmed the settlements yet.
- The mid report is in a local folder that git ignores, by Zain's decision on 7 October 2026. Still missing from it: the design diagrams and screenshots (13 dummy pictures stand in), the two other members' contributions, the meeting log, and the tag for the mid evaluation.
- Nothing has been measured with a meeting application running beside the system. The committee asked about exactly this; it waits for the virtual camera.
- The repository shows commits from one account only. The rubric asks for commits from all members.

## External components and tools

| Component | Used for | Terms |
|---|---|---|
| LivePortrait (Guo et al., 2024) | Reenactment networks | MIT licence for the code |
| MediaPipe FaceLandmarker (Google) | Face tracking | Apache 2.0 |
| ArcFace `w600k_r50` (InsightFace) | Identity embedding | Non-commercial research use |
| RAFT through torchvision (Teed and Deng, 2020) | Optical flow for the flicker measurement | BSD-3 code |
| PyTorch, OpenCV, FastAPI, React, Tailwind CSS | Framework and libraries | Open source |

An AI coding assistant (Claude Code) was used to write and test the code and documents recorded in this log from 5 October 2026, directed by the project team. The mid-evaluation rubric asks for this to be disclosed in the report (Form 3, criterion 8) and for each member to be able to explain the code they own.
