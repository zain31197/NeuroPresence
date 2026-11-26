"""How soon the consent watch shows the still picture when someone else sits down, and whether it ever does so
for the right person, at different paces of the watch. Real engine, tracker and face match; no camera.

The consent watch (consent/check.py) compares the face at the camera with the registered signatures every
WATCH_SECONDS during a session, and shows the still picture after MISSES_TO_HOLD comparisons in a row that do not
match. In the demo rehearsal of 24 November 2026 that took 3.3 s: long enough for someone else to drive the
picture in front of the panel. Each pace is tried here on the same stand-in (scripts/rehearse_demo.py):

  someone else   another sample clip sits down, six times, for 6 s each, with the stand-in back for 6 s between:
                 how long until the still picture, and until the picture is live again
  turned         the stand-in's head held turned 40 degrees to each side for 8 s: a still picture here is a false one
  talking        the stand-in talking for two minutes: false still pictures, and the frame rate, since each
                 comparison costs the CPU about 50 ms

Usage:
    python scripts/study_consent_watch.py         # about 15 minutes; writes results/consent_watch_study.json
"""

import json
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rehearse_demo import CAMERA, set_up, turned_to, wait  # noqa: E402

from neuropresence.consent import check  # noqa: E402
from neuropresence.server import runtime as runtime_module  # noqa: E402

# (name, seconds between comparisons, misses in a row to hold, matches in a row to release)
PACES = [
    ("as built", 1.0, 3, 2),
    ("twice a second", 0.5, 3, 2),
    ("twice a second, two misses", 0.5, 2, 2),
]
SWAPS, SWAP_SECONDS, TURN_SECONDS, TALK_SECONDS = 6, 6.0, 8.0, 120.0


def held(rt):
    return rt.session.consent_hold


def timed(rt, test, seconds):
    """How long until test() is true, or None within `seconds`; samples every 50 ms."""
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        if test():
            return round(time.perf_counter() - start, 2)
        time.sleep(0.05)
    return None


def held_for(rt, seconds):
    """How many seconds of `seconds` the consent watch held the still picture, and the frame rates seen."""
    start, holding, rates = time.perf_counter(), 0.0, []
    while time.perf_counter() - start < seconds:
        holding += 0.1 if held(rt) else 0.0
        rates.append((rt.session.snapshot().get("metrics") or {}).get("fps"))
        time.sleep(0.1)
    return round(holding, 1), [r for r in rates if r]


def one_pace(rt, client, director, name, every, misses, matches):
    runtime_module.WATCH_SECONDS = every  # the monitor reads it at every comparison
    check.MISSES_TO_HOLD, check.MATCHES_TO_RELEASE = misses, matches
    director.scene = "me"
    client.post("/api/session/start", json={"input": CAMERA})
    if not wait(lambda: rt.session.snapshot()["state"] == "running", 90):
        raise RuntimeError("the session did not start: " + rt.session.snapshot()["message"])
    time.sleep(4.0)

    still, live = [], []
    for _ in range(SWAPS):
        director.scene = "other"
        still.append(timed(rt, lambda: held(rt), SWAP_SECONDS))
        wait(lambda: held(rt), SWAP_SECONDS)  # the rest of the swap, if it was quicker
        director.scene = "me"
        live.append(timed(rt, lambda: not held(rt), SWAP_SECONDS))
        time.sleep(max(0.0, SWAP_SECONDS - (live[-1] or SWAP_SECONDS)))

    turned = {}
    for side, degrees in (("left", 40.0), ("right", -40.0)):
        director.scene = turned_to(director.stand_in, degrees)
        turned[side], _ = held_for(rt, TURN_SECONDS)
        director.scene = "me"
        time.sleep(3.0)

    talking, rates = held_for(rt, TALK_SECONDS)
    client.post("/api/session/stop")
    wait(lambda: rt.session.snapshot()["state"] == "idle", 15)

    found = [s for s in still if s is not None]
    back = [s for s in live if s is not None]
    result = {
        "pace": name, "watch_seconds": every, "misses_to_hold": misses, "matches_to_release": matches,
        "someone_else": {"times": SWAPS, "held": len(found), "seconds_to_still": still,
                         "median_s": float(np.median(found)) if found else None, "longest_s": max(found) if found else None},
        "back_again": {"seconds_to_live": live, "median_s": float(np.median(back)) if back else None},
        "false_still_seconds": {"turned_left_40": turned["left"], "turned_right_40": turned["right"], "talking_120": talking},
        "fps_while_talking": {"median": float(np.median(rates)) if rates else None, "lowest": min(rates) if rates else None},
    }
    print(f"{name:28s} someone else: still after {result['someone_else']['median_s']} s median, {result['someone_else']['longest_s']} s longest "
          f"({len(found)} of {SWAPS}); live again {result['back_again']['median_s']} s | false still: turned {turned['left']} / "
          f"{turned['right']} s, talking {talking} s | fps {result['fps_while_talking']['median']}", flush=True)
    return result


def main():
    folder = Path(tempfile.mkdtemp(prefix="neuropresence_watch_"))
    try:
        rt, client, director, _ = set_up(folder)
        results = [one_pace(rt, client, director, *pace) for pace in PACES]
        rt.shutdown()
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    import torch

    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "machine": {"system": platform.platform(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "same_person_limit": check.SAME_PERSON_CSIM if hasattr(check, "SAME_PERSON_CSIM") else None,
              "paces": results}
    out = ROOT / "results" / "consent_watch_study.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Written to {out.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
