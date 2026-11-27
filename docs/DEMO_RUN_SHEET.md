# Demo run sheet: FYP-1 final, 9 and 10 December 2026

For the team on the day: what to set up, what to show in what order, what to say, and what to do when something goes wrong in the room. The panel scores the demo beyond the happy path, so the failure paths are part of the script, not an accident to hide.

Every failure path below was run on the real engine before the demo (`python scripts/rehearse_demo.py`, results in `results/demo_rehearsal.json`), and a session of half an hour was measured for anything that creeps (`--soak 30`, `results/demo_soak.json`). What those runs could not stand in for is marked **check by hand**.

## The day before

| | What | How | Done |
|---|---|---|---|
| 1 | The laptop is ready | `python scripts/preflight.py --warm` on the demo laptop, plugged in. Read its frame rate in `logs/preflight.log`: 20 or more is fine | |
| 2 | The account | Make it on the laptop and enrol under it, or copy the `data` folder from the PC with the engine closed on both. Write the recovery code on paper and keep it with the laptop | |
| 3 | The rehearsal | `python scripts/rehearse_demo.py` on the laptop: all cases as expected | |
| 4 | Two meetings | A Google Meet link and a Microsoft Teams (or Zoom) link, each with a second device signed in as the other person (a phone is enough) | |
| 5 | The backup | A screen recording of the whole script below on the laptop, played from the laptop's own disk, not the internet | |
| 6 | Kit | Laptop charger, the webcam or phone camera and its cable, a printed photograph of the presenter, the recovery code | |

## In the room, before the panel sits down (10 minutes)

1. Laptop on its charger. Close everything else that uses the graphics card: games, a second browser full of tabs, screen recorders.
2. Start the engine: `python -m neuropresence.server --open`. Sign in.
3. **Live Studio: switch the Virtual camera on now, before any meeting opens it.** A meeting keeps the frame rate the camera offered when it first opened it; opened before, it gets 30, and the picture steps.
4. Open the meeting and choose **OBS Virtual Camera** as the camera. For the voice to stay in step with the lips, switch on **Voice in step** in the Live Studio and choose **CABLE Output (VB-Audio Virtual Cable)** as the meeting's microphone. The meeting shows the plain card: nothing of anyone until a session runs.
5. Overview: everything ticked under "Next session" and "This computer".
6. Light: face lit from the front, no window behind the presenter.

## The script (about 12 minutes)

| Min | Show | Say | What the panel sees |
|---|---|---|---|
| 0 | Landing page | The problem; one picture of you driven live by your own face; runs on this GPU, nothing leaves the computer | |
| 1 | Sign in, then **Enrolment** | Verified face, five poses, the meeting picture, accepted only if it is the same face. **Your pictures**: the library | The picture library in the database |
| 3 | **Live Studio**: Start session on the camera | Two actions at random so that a photo or a recording does not pass; the face is matched all the way through | The prompt, then the live picture, marked "AI reenacted", 20 to 22 frames a second, about 80 to 100 ms |
| 5 | The meeting on the second device | The meeting sees it as an ordinary camera; the mark travels with every frame | The animated picture in Meet or Teams |
| 6 | **Someone else sits down** (a teammate) | The camera face is compared twice a second | The still picture within about a second and a half; live again about a second after the presenter is back |
| 7 | **Cover the mouth** with a hand | The camera's lips are checked on every frame; a hand takes their colour away | The picture keeps a closed mouth while the hand is there and the head goes on moving; the Live Studio says "Covered" |
| 8 | **Pull the camera's cable** for five seconds, then put it back | A phone camera drops frames; the session waits instead of ending | "Reconnecting", then "The camera is back" |
| 9 | Stop. Start again and **hold up the printed photograph** | | Refused, with the reason in words; the meeting never got a picture of anyone |
| 10 | **Sessions**, **Security**, **Your data** | Every session with its figures; every check; the database drawn from the file itself, eleven tables, every row the person's own | The ER diagram on screen |
| 11 | **Benchmarks** | Measured against the targets, with what is not met said plainly | |
| 12 | Questions | | |

The voice mode (camera off) is shown only if asked: it needs Ditto installed on the laptop and runs at about 16 frames a second, 0.8 s behind the speaker.

## When something goes wrong

| What happens | What to do | Why it happens |
|---|---|---|
| The check refuses the presenter | Read the reason aloud, sit facing the camera, try again. Each action has 5 seconds and one more go | The light, or an action done too small. Three refused checks are free; after that each next one waits a minute, then two, and so on. A passed check clears the count |
| The picture stays on the still picture | Live Studio: **Resume now**. It also resumes by itself once the presenter sits facing the camera, mouth closed | The identity guard or the consent watch saw someone else, or a poor match |
| "The picture is arriving late" | Close what else is using the graphics card | The delay watchdog: above 150 ms on average the still picture is shown, and it resumes by itself (rehearsed under full load: it did) |
| The meeting picture steps or stutters | Stop the session, switch the Virtual camera off and on, leave and rejoin the meeting with the camera chosen again | The meeting opened the camera before it was on at 22 |
| The meeting shows only the plain card | Is a session running on the camera? A recorded clip is never sent to a meeting, on purpose | |
| The camera is gone for more than 15 seconds | Plug it back in and start a new session | The session ends then, and says so |
| The engine stopped or the laptop restarted | Start it again: the presenter is still signed in, the virtual camera comes back on, the cut-off session is kept as "interrupted" | Rehearsed by killing the engine mid-session |
| Signed out by mistake, or the password is not accepted | Sign in. Forgotten: **Forgot it?** with the recovery code on paper | Five wrong tries lock the email for 15 minutes: type slowly |
| Nothing works | Play the backup recording, and say that it is a recording | |

## What not to claim

- 24 frames a second: it is 20 to 22 live (target 24), with the card at its power limit.
- That nothing can fool the check: a recording made on purpose with every action passes about one try in seven; the limit on retries slows that to about a quarter of an hour.
- Microsoft Teams figures: it was tried by hand and works, but only Meet and Zoom were measured.
- Kubuntu: say whether it has been run by then.

## Rehearsal results

On the PC (RTX 5050 desktop), the latest run on 27 November 2026, with the consent watch at twice a second and a covered mouth held closed: **11 of 11 as expected**, in 3 minutes (also 11 of 11 on 24 November and earlier on 27 November, before those changes). The stand-in is the sample clip d6; someone else is another sample clip; the hand is drawn over the stand-in's mouth on every frame.

| Case | What the app did |
|---|---|
| A photograph held up | Refused at the liveness check ("Your head did not turn to your right within 5 seconds."); the meeting got no frame of the person |
| The person starts a session | Live 11 s after Start, 22.0 frames a second, 87 ms, face match 1.0; the output marked |
| Someone else sits down | Still picture after 2.0 s (1.5 s typically in `study_consent_watch.py`; 3.3 s on 24 November, with the watch at once a second); live again 1.0 s after the person was back |
| The mouth is covered | Found covered 0.25 s after the hand came, the picture's mouth held closed for 97% of the time and the picture live throughout; the guard steady, nothing repaired (on 24 November, before this was built, the guard took a fresh neutral pose) |
| The camera drops out for 5 s | "Reconnecting", then "The camera is back"; the session went on |
| Another program takes the whole card | Frame rate down to 10 or 11. On 24 November the delay reached 157 ms, the still picture was shown, and the picture resumed by itself; in the runs of 27 November it peaked at 143 to 149 ms, just under the limit, and none was needed |
| Signed out during a session | The session stopped, the meeting got the plain card, the API refused everything |
| The camera is pulled out for good | The session ended after 15 s, saying to check the cable and start again |
| Five wrong passwords | The right password refused too: "Try again in 15 minutes." |
| Another account on the computer | None of the first person's sessions; opening one by its number: not found |
| The engine killed mid-session | After the restart: still signed in, the session kept as "interrupted", the virtual camera on, a new session ran |

**Check by hand**, since a stand-in cannot: a real person passing the prompt in the room's light; a real hand over the mouth (the Live Studio should say "Covered" beside the mouth within a fraction of a second, and the picture keep a closed mouth; if it does not in the room's light, the "Covered mouth" switch can be left as it is, since it then does nothing); a real cable pulled; the meeting application's own picture.

## Half an hour without a break

24 November 2026, on the PC: one camera session of **30 minutes**, the stand-in at the camera, the virtual camera on, the Live Studio open in a browser throughout.

| | First 5 minutes | Last 5 minutes | Worst minute |
|---|---|---|---|
| Frame rate | 22.0 | 22.0 | 22.0 |
| Delay, camera to output | 83 ms | 83 ms | 83 ms |
| Frames a second reaching the meeting | 21.9 to 22.0 | 21.7 to 22.1 | 21.7 |
| Longest gap between two frames | 54 to 131 ms | 53 to 76 ms | 131 ms, in minute 2, while a test run shared the computer; 118 ms in minute 21 without one |
| Engine memory | 5.53 GB | 5.53 GB | no growth |
| Engine threads | 147 | 145 | no growth |
| Graphics card memory in use (the browser's included) | 2.91 to 2.94 GB | 2.94 GB | 27 MB more after half an hour |
| Seconds on the still picture | 0 | 0 | 0 |

Nothing creeps that a 12 minute demo would feel. The browser page's memory stayed between 11 and 26 MB.
