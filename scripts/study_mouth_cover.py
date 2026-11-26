"""Can a covered mouth be told from an uncovered one in the camera picture, quickly and without false alarms?

The detector (neuropresence/capture/cover.py) compares how much redder the lips are than the cheeks with what is
usual for the person. This runs it, frame by frame and at each recording's own rate, over:

  uncovered   Zain's two recordings of his own camera (sitting, talking, turning, looking down; data/recordings,
              kept only on his computer) and the twelve sample clips: every moment called covered is a false alarm
  covered     the same recordings with a cover drawn over the lower face twice for 2 s, of six kinds: a flat skin
              coloured patch, a hand (the person's own forehead skin with the creases between fingers), a red cup,
              a white cup, fingers over half the mouth, and a hand on the chin below the lips, which is not a cover
              and must not be called one

The covers are drawn, not filmed: they say what the detector does with what they hold, and the run sheet asks for
a real hand at a real camera before the demonstration.

With --live it also runs a session on the real engine (scripts/rehearse_demo.py's stand-in, talking, a hand drawn
over its mouth four times for 3 s), once with the "Covered mouth" switch off and once on, and measures what the
picture did: how open its mouth was, how alike it stayed to the enrolled picture, and what the identity guard
and the consent watch did.

Usage:
    python scripts/study_mouth_cover.py          # about five minutes, no GPU; writes results/mouth_cover_study.json
    python scripts/study_mouth_cover.py --live   # the session as well, about four minutes more (GPU)
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker  # noqa: E402
from neuropresence.capture.cover import MouthCover, lip_redness  # noqa: E402

DRIVING = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving"
OWN = {"zain": ROOT / "data" / "recordings" / "zain.avi", "zain_speech": ROOT / "data" / "recordings" / "zain_speech.avi"}
SAMPLES = ("d0", "d3", "d6", "d9", "d10", "d11", "d12", "d13", "d14", "d18", "d19", "d20")
SECONDS = 12.0
WINDOWS = ((4.0, 6.0), (8.0, 10.0))  # when the cover is held, in seconds from the start
KINDS = ("flat", "hand", "red_cup", "white_cup", "half", "chin")
NOT_A_COVER = ("chin",)
GRACE_SECONDS = 0.6  # after a cover is taken away, being called covered this much longer is the release, not a false alarm
OWN_STRIDE = 2  # every second frame of the long own recordings, so each takes 24 s of them


def read_frames(path, stride=1):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)
        index += 1
    capture.release()
    return frames, fps / stride


def order(count, fps):
    """Frame indices for SECONDS of the recording, played forth and back again when it is shorter."""
    if count < 2:
        return [0] * int(SECONDS * fps)
    cycle = list(range(count)) + list(range(count - 2, 0, -1))
    return [cycle[i % len(cycle)] for i in range(int(SECONDS * fps))]


def skin_at(frame, points, index):
    x, y = points[index].astype(int)
    patch = frame[max(0, y - 4):y + 4, max(0, x - 4):x + 4].reshape(-1, 3)
    return np.median(patch, axis=0) if len(patch) else np.array([150, 160, 190])


def cover(frame, points, kind):
    """The frame with something over (or, for "chin", under) the mouth. `points` are its tracked landmarks."""
    out = frame.copy()
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    w, h = x1 - x0, y1 - y0
    top = int((points[2][1] + points[13][1]) / 2)  # halfway between the nose and the lips
    bottom = int(min(frame.shape[0] - 1, y1 + 0.1 * h))
    left, right = int(max(0, x0 - 0.08 * w)), int(min(frame.shape[1] - 1, x1 + 0.08 * w))
    if kind == "flat":
        cv2.rectangle(out, (left, top), (right, bottom), tuple(int(c * 0.92) for c in skin_at(frame, points, 50)), -1)
    elif kind in ("hand", "chin"):
        if kind == "chin":
            top = int(points[17][1] + 0.04 * h)  # from below the lower lip
        fy, fx = int(points[10][1] + 0.05 * h), int(points[10][0])
        texture = frame[max(0, fy):fy + max(4, int(0.12 * h)), max(0, fx - int(0.15 * w)):fx + int(0.15 * w)]
        rw, rh = right - left, bottom - top
        if rw < 4 or rh < 4 or texture.size == 0:
            return out
        skin = cv2.resize(texture, (rw, rh), interpolation=cv2.INTER_CUBIC).astype(np.float32) * 0.9
        for j in range(1, 4):  # the creases between the fingers
            row = int(rh * j / 4)
            skin[max(0, row - 2):row + 2] *= 0.7
        shape = np.zeros((rh, rw), np.float32)
        cv2.ellipse(shape, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 1.0, -1)
        shape = cv2.GaussianBlur(shape, (0, 0), 6)[..., None]
        part = out[top:top + rh, left:left + rw].astype(np.float32)
        out[top:top + rh, left:left + rw] = (part * (1 - shape) + skin * shape).astype(np.uint8)
    elif kind in ("red_cup", "white_cup"):
        colour = (40, 60, 150) if kind == "red_cup" else (225, 228, 230)
        cv2.ellipse(out, (int(points[13][0]), int(points[13][1] + 0.05 * h)), (int(0.32 * w), int(0.28 * h)), 0, 0, 360, colour, -1)
    elif kind == "half":
        cv2.rectangle(out, (left, top), (int(points[13][0]), bottom), tuple(int(c * 0.92) for c in skin_at(frame, points, 50)), -1)
    return out


def run(readings, fps):
    """The detector over one sequence of (frame, landmarks or None); returns what it said each frame and whether it
    could tell at all."""
    detector, said = MouthCover(), []
    for i, (frame, landmarks) in enumerate(readings):
        said.append(False if landmarks is None else bool(detector.update(frame, landmarks, i / fps)))
    return said, detector.usable


def judge(said, fps, windows):
    """False alarms outside the windows (beyond the grace after each), and for each window how long until it was
    called covered and how long it stayed so after."""
    t = np.arange(len(said)) / fps
    outside = np.ones(len(said), bool)
    caught = []
    for start, end in windows:
        inside = (t >= start) & (t < end)
        outside &= ~((t >= start) & (t < end + GRACE_SECONDS))
        hits = np.flatnonzero(inside & np.array(said))
        after = np.flatnonzero((t >= end) & ~np.array(said))
        caught.append({"after_s": None if len(hits) == 0 else round(float(t[hits[0]] - start), 2),
                       "share_of_cover": round(float(np.mean(np.array(said)[inside])), 3) if inside.any() else None,
                       "released_after_s": None if len(after) == 0 else round(float(t[after[0]] - end), 2)})
    false = np.array(said) & outside
    runs, longest = 0, 0
    for value in false:
        runs = runs + 1 if value else 0
        longest = max(longest, runs)
    return {"false_s": round(float(false.sum() / fps), 2), "longest_false_s": round(longest / fps, 2), "windows": caught}


# ---------------------------------------------------------------- on the real engine

COVERS, COVER_SECONDS, BETWEEN_SECONDS = 4, 3.0, 4.0


def live(pictures=None):
    """The session with the switch off, then on. Returns what each did while the mouth was covered."""
    import shutil
    import tempfile
    import time

    sys.path.insert(0, str(ROOT / "scripts"))
    from rehearse_demo import CAMERA, set_up, wait

    from neuropresence.pipeline import gap_of

    folder = Path(tempfile.mkdtemp(prefix="neuropresence_cover_"))
    results = {}
    try:
        rt, client, director, _ = set_up(folder)
        with FaceTracker(video=False) as tracker:
            for switch in (False, True):
                client.patch("/api/features/mouth_cover", json={"enabled": switch})
                director.scene = "me"
                client.post("/api/session/start", json={"input": "camera:0"})
                if not wait(lambda: rt.session.snapshot()["state"] == "running", 90):
                    raise RuntimeError("the session did not start: " + rt.session.snapshot()["message"])
                time.sleep(5.0)
                last = rt.events.last_id
                covered, uncovered = [], []
                for k in range(COVERS):
                    for scene, seconds, into in (("cover", COVER_SECONDS, covered), ("me", BETWEEN_SECONDS, uncovered)):
                        director.scene = scene
                        end = time.perf_counter() + seconds
                        while time.perf_counter() < end:
                            time.sleep(0.2)
                            pair = rt.session.latest_pair()
                            s = rt.session.snapshot()
                            identity = rt.identity.snapshot()
                            reading = {"live": bool(pair and pair.live), "holds": s["holds"],
                                       "covered": bool((s.get("tracking") or {}).get("mouth_covered")),
                                       "identity": identity.get("csim"), "guard": (identity.get("guard") or {}).get("state"),
                                       "lips": None, "fps": (s.get("metrics") or {}).get("fps")}
                            if pair is not None and pair.live:
                                track = tracker.process(pair.output)
                                if track.ok:
                                    reading["lips"] = gap_of(np.asarray(track.landmarks, dtype=np.float32))
                                if pictures is not None and scene == "cover" and k == 1 and abs(end - time.perf_counter() - seconds / 2) < 0.15:
                                    cv2.imwrite(str(Path(pictures) / f"cover_{'on' if switch else 'off'}.jpg"), pair.output)
                            into.append(reading)
                events = [e["message"] for e in rt.events.since(last)]
                client.post("/api/session/stop")
                wait(lambda: rt.session.snapshot()["state"] == "idle", 15)

                def figures(readings):
                    lips = [r["lips"] for r in readings if r["lips"] is not None]
                    scores = [r["identity"] for r in readings if r["identity"] is not None]
                    return {"readings": len(readings),
                            "called_covered": round(float(np.mean([r["covered"] for r in readings])), 3),
                            "lips_median": None if not lips else round(float(np.median(lips)), 4),
                            "lips_highest": None if not lips else round(float(np.max(lips)), 4),
                            "lips_spread": None if not lips else round(float(np.std(lips)), 4),
                            "identity_median": None if not scores else round(float(np.median(scores)), 3),
                            "identity_lowest": None if not scores else round(float(np.min(scores)), 3),
                            "still_picture_share": round(float(np.mean([not r["live"] for r in readings])), 3),
                            "guard_states": sorted({r["guard"] for r in readings if r["guard"]}),
                            "held_for": {why: round(float(np.mean([r["holds"][why] for r in readings])), 3)
                                         for why in ("identity", "delay", "consent")},
                            "fps_median": None if not [r for r in readings if r["fps"]] else
                            round(float(np.median([r["fps"] for r in readings if r["fps"]])), 1),
                            "consent_held": any(r["holds"]["consent"] for r in readings)}

                name = "switch on" if switch else "switch off"
                results[name] = {"covered": figures(covered), "uncovered": figures(uncovered), "events": events}
                c = results[name]["covered"]
                print(f"{name:10s} covered: called covered {c['called_covered']:.0%}, lips {c['lips_median']} (highest {c['lips_highest']}, "
                      f"spread {c['lips_spread']}), identity {c['identity_median']} (lowest {c['identity_lowest']}), still picture "
                      f"{c['still_picture_share']:.0%} (held for {c['held_for']}), guard {c['guard_states']}, fps {c['fps_median']}", flush=True)
        client.patch("/api/features/mouth_cover", json={"enabled": True})
        rt.shutdown()
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live", action="store_true", help="also a session on the real engine, the switch off and on")
    parser.add_argument("--pictures", help="a folder to keep one output frame from the middle of a cover, each way")
    args = parser.parse_args()
    sources = {name: (path, OWN_STRIDE) for name, path in OWN.items() if path.exists()}
    sources.update({name: (DRIVING / f"{name}.mp4", 1) for name in SAMPLES})
    report = {"date": datetime.now().isoformat(timespec="seconds"), "machine": platform.platform(),
              "seconds": SECONDS, "windows_s": WINDOWS, "clips": {}}
    with FaceTracker(video=False) as tracker:
        for name, (path, stride) in sources.items():
            frames, fps = read_frames(path, stride)
            sequence = order(len(frames), fps)
            tracked = {}
            for index in set(sequence):
                track = tracker.process(frames[index])
                tracked[index] = np.asarray(track.landmarks, dtype=np.float32)[:, :2] if track.ok else None
            plain = [(frames[i], tracked[i]) for i in sequence]
            said, usable = run(plain, fps)
            usable = bool(usable)
            redness = [lip_redness(f, p) for f, p in plain if p is not None]
            entry = {"fps": round(fps, 1), "usable": usable,
                     "redness_median": round(float(np.median([r for r in redness if r is not None])), 1),
                     "uncovered": judge(said, fps, ())}
            if usable:
                for kind in KINDS:
                    drawn = []
                    for k, i in enumerate(sequence):
                        frame, landmarks = plain[k]
                        if landmarks is not None and any(a <= k / fps < b for a, b in WINDOWS):
                            frame = cover(frames[i], landmarks, kind)
                            track = tracker.process(frame)
                            landmarks = np.asarray(track.landmarks, dtype=np.float32)[:, :2] if track.ok else None
                        drawn.append((frame, landmarks))
                    said, _ = run(drawn, fps)
                    result = judge(said, fps, () if kind in NOT_A_COVER else WINDOWS)
                    if kind in NOT_A_COVER:
                        t = np.arange(len(said)) / fps
                        result["called_covered_s"] = round(float(sum(s for s, x in zip(said, t) if any(a <= x < b for a, b in WINDOWS)) / fps), 2)
                    entry[kind] = result
            report["clips"][name] = entry
            line = f"{name:12s} redness {entry['redness_median']:5.1f} usable {str(usable):5s} false {entry['uncovered']['false_s']:.2f} s"
            for kind in KINDS:
                if kind in entry:
                    r = entry[kind]
                    if kind in NOT_A_COVER:
                        line += f" | {kind}: called covered {r['called_covered_s']:.2f} s"
                    else:
                        after = [w["after_s"] for w in r["windows"]]
                        line += f" | {kind}: {after} false {r['false_s']:.2f}"
            print(line, flush=True)

    usable = [e for e in report["clips"].values() if e["usable"]]
    report["summary"] = {
        "clips": len(report["clips"]), "usable": len(usable),
        "false_alarm_s_uncovered": round(sum(e["uncovered"]["false_s"] for e in report["clips"].values()), 2),
        "uncovered_seconds": round(SECONDS * len(report["clips"]), 1),
        "covers": {kind: {
            "caught": sum(1 for e in usable for w in e[kind]["windows"] if w["after_s"] is not None),
            "of": sum(len(e[kind]["windows"]) for e in usable),
            "median_after_s": float(np.median([w["after_s"] for e in usable for w in e[kind]["windows"] if w["after_s"] is not None] or [np.nan])),
            "false_s": round(sum(e[kind]["false_s"] for e in usable), 2)} for kind in KINDS if kind not in NOT_A_COVER},
        "chin_called_covered_s": round(sum(e["chin"]["called_covered_s"] for e in usable), 2),
    }
    if args.live:
        report["live"] = live(args.pictures)
    out = ROOT / "results" / "mouth_cover_study.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print(f"Written to {out.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
