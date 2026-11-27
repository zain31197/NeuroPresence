"""Rehearse the demonstration, the paths beyond the happy one included, with the real engine and no camera.

The panel will not only watch a session run. It will hold up a photograph, sit someone else down, cover the mouth,
pull out the camera, load the graphics card, try passwords and sign out in the middle. This runs each of those on
the real engine, tracker, face match, consent check, identity guard, delay watchdog, accounts and database, and
says for each one what the app did and whether that is what it should do.

The person at the camera is a stand-in: a sample clip that answers the liveness prompt (scripts/study_liveness.py:
StandIn), staged on a 1280 x 720 picture as a webcam would show it, and fed in place of the camera. Someone else is
another sample clip. A covered mouth is a hand-coloured shape drawn over the stand-in's mouth. The virtual camera is
a stand-in that keeps what it was sent; the microphone and the cable are not opened. Everything is kept in a
folder of its own under the system's temporary folder, never in data/.

Usage:
    python scripts/rehearse_demo.py            # about five minutes; writes results/demo_rehearsal.json
    python scripts/rehearse_demo.py --soak 30  # one session of half an hour, served on port 8011 so a browser can
                                               # keep the Live Studio open beside it; writes results/demo_soak.json

What it cannot stand in for: a real person in front of a real camera, a real meeting application, the light of the
room. The run sheet (docs/DEMO_RUN_SHEET.md) says what to check by hand.
"""

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi.testclient import TestClient  # noqa: E402

from neuropresence.capture import FaceTracker  # noqa: E402
from neuropresence.capture.feed import GIVE_UP_SECONDS  # noqa: E402
from neuropresence.consent.disclosure import is_marked  # noqa: E402
from neuropresence.enrolment import POSE_KEYS  # noqa: E402
from neuropresence.identity import SAME_PERSON_CSIM  # noqa: E402
from neuropresence.server import create_app  # noqa: E402
from neuropresence.server.runtime import Runtime  # noqa: E402
from neuropresence.server.virtual_camera import card  # noqa: E402

DRIVING = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving"
CANVAS = (1280, 720)
STAND_IN, OTHERS = "d6", ("d0", "d3", "d9", "d10", "d11", "d12")
PERSON = {"email": "rehearsal@example.com", "name": "Rehearsal", "password": "a long rehearsal password"}
SECOND = {"email": "second@example.com", "name": "Second", "password": "another long password"}
CAMERA = "camera:0"


# ---------------------------------------------------------------- what the camera shows


def stage(frame):
    """A clip frame as a webcam shows a person: the face in the middle of a 1280 x 720 picture."""
    height, width = frame.shape[:2]
    scale = min(CANVAS[1] / height, 1.4)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((CANVAS[1], CANVAS[0], 3), 52, dtype=np.uint8)
    top, left = (CANVAS[1] - small.shape[0]) // 2, (CANVAS[0] - small.shape[1]) // 2
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
    return canvas


def frames_of(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


HAND_FRAMES = 150  # of the talking clip, with a hand drawn over the mouth: five seconds, played round


def covered(frames):
    """The talking frames with a hand over the mouth, as scripts/study_mouth_cover.py draws one: the person's own
    skin with the creases between the fingers, following the face from frame to frame."""
    from study_mouth_cover import cover

    hands = []
    with FaceTracker(video=False) as tracker:
        for frame in frames[:HAND_FRAMES]:
            track = tracker.process(frame)
            if track.ok:
                hands.append(cover(frame, np.asarray(track.landmarks, dtype=np.float32)[:, :2], "hand"))
    if not hands:
        raise ValueError("no face to cover")
    return hands


class Director:
    """Decides what the camera shows: the stand-in (answering the prompt while it is asked, then talking), a
    photograph of them, someone else, the stand-in with a covered mouth, or nothing at all."""

    def __init__(self, runtime, stand_in, talking, other, photo, cover):
        self.rt, self.stand_in, self.talking, self.other, self.photo, self.cover = runtime, stand_in, talking, other, photo, cover
        self.scene = "me"
        self._asked, self._asked_at, self._i = None, 0.0, 0

    def frame(self, now):
        scene = self.scene
        if isinstance(scene, np.ndarray):  # one picture held, such as a head turned and kept there
            return scene
        if scene == "silent":
            return None
        if scene == "photo":
            return self.photo
        self._i += 1
        if scene == "cover":
            return self.cover[self._i % len(self.cover)]
        if scene == "other":
            return self.other[self._i % len(self.other)]
        consent = self.rt.session.snapshot().get("consent") or {}
        if consent.get("stage") not in (None, "passed", "refused"):  # the prompt is running: answer it
            action = consent.get("action")
            if action != self._asked:
                self._asked, self._asked_at = action, now
            return stage(self.stand_in.frame(action, now - self._asked_at))
        self._asked = None
        return self.talking[self._i % len(self.talking)]


class DirectedFeed:
    """What the session reads in place of a camera (capture/feed.py: LiveFeed), at 30 frames a second."""

    is_camera = True

    def __init__(self, director):
        self._director = director
        self._newest, self._index = None, 0
        self._arrived = threading.Condition()
        self._stop = threading.Event()
        self.ended, self.reopened = False, 0
        self._last = time.perf_counter()
        threading.Thread(target=self._run, name="rehearsal-feed", daemon=True).start()

    @property
    def silent_seconds(self):
        return time.perf_counter() - self._last

    @property
    def rate(self):
        return None if self.silent_seconds > 0.5 else 30.0

    def _run(self):
        from neuropresence.capture.feed import FeedFrame

        due = time.perf_counter()
        while not self._stop.is_set():
            due += 1 / 30
            time.sleep(max(0.0, due - time.perf_counter()))
            image = self._director.frame(time.perf_counter())
            if image is None:  # the camera gives nothing: as a real one, it is opened again, then given up
                if self.silent_seconds >= GIVE_UP_SECONDS:
                    break
                continue
            self._last = time.perf_counter()
            with self._arrived:
                self._newest = FeedFrame(image, self._index, self._last)
                self._index += 1
                self._arrived.notify_all()
        with self._arrived:
            self.ended = True
            self._arrived.notify_all()

    def read(self, after_index=-1, timeout=1.0):
        deadline = time.perf_counter() + timeout
        with self._arrived:
            while True:
                if self._newest is not None and self._newest.index > after_index:
                    return self._newest
                left = deadline - time.perf_counter()
                if self.ended or left <= 0:
                    return None
                self._arrived.wait(left)

    def close(self):
        self._stop.set()


class MeetingCamera:
    """Stands in for the virtual camera device: keeps what a meeting would have been sent, and how many of those
    frames were something other than the plain card."""

    device = "Rehearsal Camera"

    def __init__(self, width, height):
        self.card = card((width, height))
        self.sent, self.newest, self.person = 0, None, 0

    def send(self, frame):
        self.sent, self.newest = self.sent + 1, frame
        self.person += not self.shows_card

    @property
    def shows_card(self):
        return self.newest is not None and np.array_equal(self.newest, self.card)

    def close(self):
        pass


# ---------------------------------------------------------------- the runtime


def make_runtime(data_dir, director_box, cameras):
    def feed(source):
        if isinstance(source, int):
            return DirectedFeed(director_box[0])
        from neuropresence.capture.feed import LiveFeed

        return LiveFeed(source)

    def open_camera(width, height, fps):
        cameras.append(MeetingCamera(width, height))
        return cameras[-1]

    return Runtime(data_dir=data_dir, feed_factory=feed, camera_opener=open_camera,
                   sound_devices=lambda: {"microphones": [], "default": None, "cable": None},
                   sound_opener=lambda *args: None)


def turned_to(stand_in, degrees):
    """The stand-in's head turned as near to `degrees` as the engine drew it (left is positive), staged."""
    return stage(min(stand_in.turned, key=lambda pair: abs(pair[0] - degrees))[1])


def enrol(rt, rest, stand_in=None):
    """The stand-in's face verified, its five poses registered and its picture enrolled, as the Enrolment screen
    leaves them: the signatures through the store, the picture through the upload's own checks. The turns to each
    side are registered from the stand-in turned by 30 degrees, as a person turns for them; up and down, which the
    stand-in cannot do, from the face at rest."""
    scorer = rt.scorer()
    with FaceTracker(video=False) as tracker:
        track = tracker.process(rest)
        signature = scorer.embed(rest, track.landmarks)
        rt.store.save_identity(signature, [])
        for pose, angle in zip(POSE_KEYS, (0.0, 30.0, -30.0, 15.0, -15.0)):
            pose_signature = signature
            if stand_in is not None and pose in ("left", "right"):
                frame = turned_to(stand_in, angle)
                seen = tracker.process(frame)
                if seen.ok:
                    pose_signature = scorer.embed(frame, seen.landmarks)
            rt.store.save_pose(pose, pose_signature, angle, [])
    rt.face = rt.store.load_identity()
    rt.check_upload(rest, "rehearsal")
    candidate = rt.candidate
    if not candidate.passed:
        raise RuntimeError("the stand-in's picture did not pass: " + "; ".join(c.hint for c in candidate.checks if not c.passed))
    rt.confirm_candidate()
    return signature


def wait(condition, seconds, every=0.1):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        value = condition()
        if value:
            return value
        time.sleep(every)
    return None


def watch(rt, seconds, every=0.25):
    """What the session showed, every quarter of a second: the holds, the guard, the delay, the state."""
    seen, start = [], time.perf_counter()
    while time.perf_counter() - start < seconds:
        s = rt.session.snapshot()
        guard = rt.identity.snapshot().get("guard", {})
        metrics = s.get("metrics") or {}
        seen.append({"t": round(time.perf_counter() - start, 2), "state": s["state"], "holds": s["holds"],
                     "covered": bool((s.get("tracking") or {}).get("mouth_covered")),
                     "guard": guard.get("state"), "delay_ms": metrics.get("end_to_end_ms"), "fps": metrics.get("fps"),
                     "showing": rt.virtual_camera.snapshot()["showing"]})
        time.sleep(every)
    return seen


def first(seen, test):
    return next((entry["t"] for entry in seen if test(entry)), None)


def events_since(rt, last):
    return [e["message"] for e in rt.events.since(last)]


def keep_gpu_busy(share, seconds):  # in a process of its own
    sys.path.insert(0, str(ROOT / "scripts"))
    from study_delay import keep_gpu_busy as busy

    busy(share, seconds)


# ---------------------------------------------------------------- the rehearsal


def set_up(data_dir):
    """A runtime with the stand-in signed up, verified and enrolled, and the camera handed to the director."""
    from study_liveness import StandIn

    cameras, box = [], [None]
    rt = make_runtime(data_dir, box, cameras)
    client = TestClient(create_app(rt))
    reply = client.post("/api/account/sign-up", json={**PERSON, "remember": True})
    assert reply.status_code == 200, reply.text

    print("Making the stand-ins (about a minute)...", flush=True)
    engine = rt.engine()
    with rt.engine_lock:
        stand_in = StandIn(engine, DRIVING / f"{STAND_IN}.mp4")
        rt.engine_holds = None  # the stand-in used the engine; the enrolled picture goes back in at the session's start
    talking = [stage(f) for f in frames_of(DRIVING / f"{STAND_IN}.mp4")]
    rest = stage(stand_in.rest)
    signature = enrol(rt, rest, stand_in)
    scorer = rt.scorer()
    other = None
    for name in OTHERS:  # someone who is clearly not the stand-in
        frames = [stage(f) for f in frames_of(DRIVING / f"{name}.mp4")]
        with FaceTracker(video=False) as tracker:
            track = tracker.process(frames[0])
        if track.ok and scorer.similarity(signature, scorer.embed(frames[0], track.landmarks)) < SAME_PERSON_CSIM / 2:
            other = frames
            break
    director = Director(rt, stand_in, talking, other, rest, covered(talking))
    box[0] = director
    return rt, client, director, cameras


def rehearse(data_dir):
    results = []
    rt, client, director, cameras = set_up(data_dir)

    def record(name, expected, ok, **found):
        results.append({"case": name, "expected": expected, "as_expected": bool(ok), **found})
        print(f"{'ok  ' if ok else 'FAIL'} {name}: " + ", ".join(f"{k} {v}" for k, v in found.items() if k != "seen"), flush=True)

    client.post("/api/virtual-camera", json={"on": True})  # on before the meeting opens it, as the run sheet says

    # 1. A photograph held up at the start.
    director.scene = "photo"
    last = rt.events.last_id
    client.post("/api/session/start", json={"input": CAMERA})
    ended = wait(lambda: rt.session.snapshot()["state"] in ("error", "idle"), 60)
    consent = rt.session.snapshot().get("consent") or {}
    record("A photograph held up to the camera", "refused at the liveness check, with the reason; nothing sent to the meeting",
           ended and consent.get("stage") == "refused" and cameras[-1].person == 0,
           reason=consent.get("reason"), message=rt.session.snapshot()["message"], meeting_frames_of_the_person=cameras[-1].person)
    client.post("/api/session/stop")

    # 2. The stand-in starts a session: the prompt, then the live picture, marked, in the meeting.
    director.scene = "me"
    started = time.perf_counter()
    reply = client.post("/api/session/start", json={"input": CAMERA})
    running = wait(lambda: rt.session.snapshot()["state"] == "running", 90)
    # The meeting gets the output fitted into its own size; the mark is checked on the output it is made from.
    live = wait(lambda: running and rt.virtual_camera.snapshot()["showing"] == "session" and not cameras[-1].shows_card
                and rt.session.latest_pair() is not None and rt.session.latest_pair().live
                and is_marked(rt.session.latest_pair().output), 20)
    time.sleep(6)
    metrics = rt.session.snapshot().get("metrics") or {}
    record("The person starts a session", "the prompt is answered, the picture goes live, the meeting gets it marked",
           reply.status_code == 200 and running and live, seconds_to_live=round(time.perf_counter() - started, 1),
           fps=metrics.get("fps"), delay_ms=metrics.get("end_to_end_ms"),
           match=(rt.session.snapshot().get("consent") or {}).get("match"))

    # 3. Someone else sits down, then the person comes back.
    director.scene = "other"
    seen = watch(rt, 9)
    held_at = first(seen, lambda e: e["holds"]["consent"] or e["holds"]["identity"])
    director.scene = "me"
    back = watch(rt, 15)
    released_at = first(back, lambda e: not e["holds"]["consent"] and not e["holds"]["identity"] and e["guard"] != "fallback")
    record("Someone else sits down, then the person returns", "the still picture within about 3 s; live again once the person is back",
           held_at is not None and held_at <= 4.0 and released_at is not None, still_after_s=held_at, live_again_after_s=released_at,
           seen=seen + back)

    # 4. The mouth is covered for ten seconds.
    last = rt.events.last_id
    director.scene = "cover"
    seen = watch(rt, 10)
    director.scene = "me"
    back = watch(rt, 12)
    states = sorted({e["guard"] for e in seen if e["guard"]})
    said = events_since(rt, last)
    record("The mouth is covered with a hand", "found covered, the picture's mouth held closed and live; nothing repaired or held back; "
           "live after",
           first(seen, lambda e: e["covered"]) is not None and not any(e["holds"]["identity"] for e in seen)
           and not any("brought back" in m or "neutral pose" in m for m in said)
           and back[-1]["state"] == "running" and not back[-1]["holds"]["identity"],
           covered_after_s=first(seen, lambda e: e["covered"]), covered_share=round(sum(e["covered"] for e in seen) / len(seen), 2),
           guard_states=states, events=said, seen=seen + back)

    # 5. The camera drops out for five seconds.
    last = rt.events.last_id
    director.scene = "silent"
    seen = watch(rt, 5)
    director.scene = "me"
    back = watch(rt, 6)
    said = events_since(rt, last)
    record("The camera drops out for five seconds", "the session waits and says so, and goes on when the camera is back",
           all(e["state"] == "running" for e in seen + back) and any("back" in m for m in said),
           events=said)

    # 6. The graphics card is loaded by something else.
    last = rt.events.last_id
    load = multiprocessing.get_context("spawn").Process(target=keep_gpu_busy, args=(1.0, 20.0))
    load.start()
    seen = watch(rt, 22, every=0.5)
    load.join()
    after = watch(rt, 10, every=0.5)
    delays = [e["delay_ms"] for e in seen if e["delay_ms"]]
    record("Another program takes the graphics card", "if the delay passes 150 ms the still picture is shown, and it resumes after",
           (not any(e["holds"]["delay"] for e in seen) or not after[-1]["holds"]["delay"]) and after[-1]["state"] == "running",
           worst_delay_ms=round(max(delays), 1) if delays else None, delay_hold=any(e["holds"]["delay"] for e in seen),
           lowest_fps=min((e["fps"] for e in seen if e["fps"]), default=None), events=events_since(rt, last), seen=seen + after)

    # 7. Signed out while live.
    meeting = cameras[-1]
    client.post("/api/account/sign-out")
    stopped = wait(lambda: rt.session.snapshot()["state"] == "idle", 15)
    time.sleep(1.0)
    record("Signed out in the middle of a session", "the session stops and the meeting gets no picture of the person",
           stopped and meeting.shows_card and client.get("/api/status").status_code == 401,
           meeting_shows=rt.virtual_camera.snapshot()["showing"], api_after=client.get("/api/status").status_code)

    # 8. The camera is pulled out for good.
    client.post("/api/account/sign-in", json={"email": PERSON["email"], "password": PERSON["password"], "remember": True})
    director.scene = "me"
    client.post("/api/session/start", json={"input": CAMERA})
    wait(lambda: rt.session.snapshot()["state"] == "running", 90)
    director.scene = "silent"
    pulled = time.perf_counter()
    ended = wait(lambda: rt.session.snapshot()["state"] != "running", GIVE_UP_SECONDS + 15)
    record("The camera is pulled out", f"the session ends after {GIVE_UP_SECONDS:.0f} s without a picture, and says why",
           ended and rt.session.snapshot()["message"], after_s=round(time.perf_counter() - pulled, 1),
           message=rt.session.snapshot()["message"])
    client.post("/api/session/stop")

    # 9. Five wrong passwords, on another account.
    client.post("/api/account/sign-up", json={**SECOND, "remember": False})
    client.post("/api/account/sign-out")
    tries = [client.post("/api/account/sign-in", json={"email": SECOND["email"], "password": "not the password", "remember": False}).status_code
             for _ in range(5)]
    right = client.post("/api/account/sign-in", json={"email": SECOND["email"], "password": SECOND["password"], "remember": False})
    record("Five wrong passwords in a row", "the right password is refused too, for a quarter of an hour, with the wait in words",
           tries == [401] * 5 and right.status_code == 429, tries=tries, then=right.status_code, said=right.json().get("detail"))

    # 10. Two people: the second sees nothing of the first.
    client.post("/api/account/sign-in", json={"email": PERSON["email"], "password": PERSON["password"], "remember": True})
    mine = client.get("/api/history").json()["sessions"]
    stranger = TestClient(create_app(rt))
    stranger.post("/api/account/sign-up", json={"email": "third@example.com", "name": "Third", "password": "yet another password", "remember": False})
    theirs = stranger.get("/api/history").json()["sessions"]
    peek = stranger.get(f"/api/history/{mine[0]['id']}").status_code
    record("Another account on the same computer", "none of the first person's sessions, pictures or face",
           theirs == [] and peek == 404 and rt.face is None, first_has=len(mine), second_has=len(theirs), opening_one=peek)
    stranger.post("/api/account/delete", json={"password": "yet another password"})
    client.post("/api/account/sign-in", json={"email": PERSON["email"], "password": PERSON["password"], "remember": True})

    # Leave a session running for the engine's restart.
    director.scene = "me"
    cookie = dict(client.cookies)  # what the person's browser keeps across the engine's restart
    rt.shutdown()
    return results, cookie


def crash_child(data_dir):
    """Run as a process of its own: the engine with a session running, then killed as a crash would."""
    rt = make_runtime(data_dir, [None], [])
    rt.start_session(f"sample:{STAND_IN}")
    wait(lambda: rt.session.snapshot()["state"] == "running", 120)
    time.sleep(4)
    print("crashing", rt.session.snapshot()["state"], flush=True)
    os._exit(3)


def after_restart(data_dir, cookie):
    box, cameras = [None], []
    started = subprocess.run([sys.executable, __file__, "--crash", str(data_dir)], capture_output=True, text=True, timeout=600)
    rt = make_runtime(data_dir, box, cameras)
    try:
        client = TestClient(create_app(rt), cookies=cookie)
        newest = rt.db.sessions(1, rt.user_id)[0] if rt.user_id is not None else None
        reply = client.post("/api/session/start", json={"input": f"sample:{STAND_IN}"})
        running = wait(lambda: rt.session.snapshot()["state"] == "running", 180)
        ok = (started.returncode == 3 and rt.user is not None and newest is not None and newest["outcome"] == "interrupted"
              and rt.virtual_camera.on and reply.status_code == 200 and running)
        return {"case": "The engine is restarted in the middle of a session",
                "expected": "the session is kept as interrupted, the person is still signed in, the virtual camera is on, a new session starts",
                "as_expected": bool(ok), "child_exit": started.returncode, "signed_in": rt.user is not None,
                "last_session": newest and {"outcome": newest["outcome"], "message": newest["message"]},
                "virtual_camera_on": rt.virtual_camera.on, "new_session": reply.status_code,
                "new_session_state": rt.session.snapshot()["state"], "new_session_message": rt.session.snapshot()["message"],
                # What the killed engine printed, only when it did not get as far as it should: it can hold local paths.
                "child_error": started.stderr[-1500:] if started.returncode != 3 else None}
    finally:
        rt.shutdown()


# ---------------------------------------------------------------- a long session


def soak(data_dir, minutes, port):
    """A camera session of `minutes` sent to the meeting, the app served on `port` so that a browser can keep the
    Live Studio open beside it; one line a minute of what could creep: frame rate, delay, the longest gap, the memory
    and threads of the engine, the graphics card's memory, and how many frames the meeting got."""
    import psutil
    import uvicorn

    rt, client, director, cameras = set_up(data_dir)
    server = uvicorn.Server(uvicorn.Config(create_app(rt), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    client.post("/api/virtual-camera", json={"on": True})
    client.post("/api/session/start", json={"input": CAMERA})
    if not wait(lambda: rt.session.snapshot()["state"] == "running", 120):
        raise RuntimeError("the session did not start: " + rt.session.snapshot()["message"])
    print(f"READY: http://127.0.0.1:{port} (sign in as {PERSON['email']}, {PERSON['password']})", flush=True)

    process, rows, started = psutil.Process(), [], time.perf_counter()
    sent, holds = cameras[-1].sent, 0
    print("minute   fps  delay ms  longest gap ms  meeting fps  memory MB  threads  card MB  holds", flush=True)
    for minute in range(1, minutes + 1):
        end = started + 60 * minute
        while time.perf_counter() < end:
            s = rt.session.snapshot()
            if s["state"] != "running":
                raise RuntimeError(f"the session ended in minute {minute}: {s['message']}")
            holds += any(s["holds"].values())
            time.sleep(1.0)
        figures = [f for f in rt.db.session(rt.session.record_id, rt.user_id)["figures"] if f["at"] >= time.time() - 60]
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        row = {
            "minute": minute,
            "fps": float(np.median([f["fps"] for f in figures if f["fps"]])) if figures else None,
            "delay_ms": float(np.median([f["delay_ms"] for f in figures if f["delay_ms"]])) if figures else None,
            "longest_gap_ms": max((f["longest_gap_ms"] for f in figures if f["longest_gap_ms"]), default=None),
            "meeting_fps": round((cameras[-1].sent - sent) / 60, 1),
            "memory_mb": round(process.memory_info().rss / 2**20),
            "threads": process.num_threads(),
            "card_mb": int(gpu[0]) if gpu and gpu[0].isdigit() else None,
            "seconds_held": holds,
        }
        sent, holds = cameras[-1].sent, 0
        rows.append(row)
        print(f"{minute:6d} {row['fps'] or 0:5.1f} {row['delay_ms'] or 0:9.0f} {row['longest_gap_ms'] or 0:15.0f} {row['meeting_fps']:12.1f}"
              f" {row['memory_mb']:10d} {row['threads']:8d} {row['card_mb'] or 0:8d} {row['seconds_held']:6d}", flush=True)
    client.post("/api/session/stop")
    server.should_exit = True
    rt.shutdown()
    return rows


def creep(rows, key):
    """How much a figure moved from the first five minutes to the last five (medians)."""
    values = [r[key] for r in rows if r[key] is not None]
    if len(values) < 10:
        return None
    return round(float(np.median(values[-5:]) - np.median(values[:5])), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--soak", type=int, metavar="MINUTES", help="instead, one long session; writes results/demo_soak.json")
    parser.add_argument("--port", type=int, default=8011, help="where the soak serves the app (default 8011; never the engine's 8000)")
    parser.add_argument("--crash", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.crash:
        crash_child(args.crash)
        return
    if args.soak:
        import torch

        folder = Path(tempfile.mkdtemp(prefix="neuropresence_soak_"))
        try:
            rows = soak(folder, args.soak, args.port)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        report = {"date": datetime.now().isoformat(timespec="seconds"),
                  "machine": {"system": platform.platform(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                  "minutes": args.soak, "stand_in": STAND_IN,
                  "creep_first_to_last_five_minutes": {key: creep(rows, key) for key in ("fps", "delay_ms", "memory_mb", "threads", "card_mb")},
                  "worst": {"lowest_fps": min(r["fps"] for r in rows if r["fps"]),
                            "longest_gap_ms": max(r["longest_gap_ms"] for r in rows if r["longest_gap_ms"])},
                  "minutes_measured": rows}
        out = ROOT / "results" / "demo_soak.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFirst five minutes to last five: {report['creep_first_to_last_five_minutes']}. Written to {out.relative_to(ROOT)}.")
        return

    folder = Path(tempfile.mkdtemp(prefix="neuropresence_rehearsal_"))
    try:
        started = time.perf_counter()
        results, cookie = rehearse(folder)
        restart = after_restart(folder, cookie)
        results.append(restart)
        print(f"{'ok  ' if restart['as_expected'] else 'FAIL'} {restart['case']}: signed in {restart['signed_in']}, "
              f"last session {restart['last_session']}, virtual camera on {restart['virtual_camera_on']}", flush=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    import torch

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": {"system": platform.platform(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "stand_in": STAND_IN, "minutes": round((time.perf_counter() - started) / 60, 1),
        "passed": sum(r["as_expected"] for r in results), "cases": len(results), "results": results,
    }
    out = ROOT / "results" / "demo_rehearsal.json"
    out.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"\n{report['passed']} of {report['cases']} as expected, in {report['minutes']} minutes. Written to {out.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
