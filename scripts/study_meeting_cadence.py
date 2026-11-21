"""How evenly the output reaches a meeting application: the frames as the engine finishes them, then as the
virtual camera passes them on, then as the meeting application picks them up.

The engine draws its frames evenly (pacing.py: 45 ms apart to within 1 ms). Two things after it can undo that.
The virtual camera sends on a clock of its own, and the application reading the camera picks up the newest
frame on a third. OBS Virtual Camera works that way on Windows: its camera filter wakes at the rate the camera
was opened with and takes whatever frame was written last. A frame drawn at 22 a second and picked up at 30 is
then held for one pick up or for two, by turns, so a head turning at an even speed is shown in short and long
steps; and two frames finished within one tick of a clock mean the first is never seen at all. On Linux,
v4l2loopback hands each written frame to the reader instead, so there only the sender's own clock counts.

This drives a live session through the server's own session code, with a sample clip in place of the camera
(no camera is opened, nothing is sent anywhere), records when each frame is finished, and replays those times
through each way of sending and picking up. The application's pick up clock starts at a random moment, so each
case is replayed from many starting moments and averaged.

What is measured, as the application sees it:
- new frames a second, and the share of finished frames it never shows;
- how long each new frame stays up (one, two or more pick ups), and how much that varies;
- the step error: from one new frame shown to the next, the time it was shown for less the time between their
  being finished. The engine finishes them evenly, so this is the unevenness a viewer sees as judder;
- how late each frame is shown after it was finished, on average: delay added after the engine.

Writes results/meeting_cadence_study.json.

Usage:
    python scripts/study_meeting_cadence.py               # about two minutes
    python scripts/study_meeting_cadence.py --pace 20      # the drawing paced to 20 frames a second as well
"""

import argparse
import json
import platform
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SETTLE_SECONDS, WATCH_SECONDS = 6.0, 40.0
STARTS = 60  # starting moments of the application's clock, per case
WAKE_SPREAD_S = 0.0008  # how far the sender's 30 Hz clock wakes from its time, measured on Windows 11 with Event.wait


def record(clip, pace=None):
    """Finish times (perf_counter seconds) of every frame of a live session over WATCH_SECONDS."""
    from neuropresence.server import metrics
    from neuropresence.server.runtime import Runtime

    finished, delays = [], []
    plain_add = metrics.MetricsWindow.add

    def add(self, finished_at, timing_ms, end_to_end_ms, live, dropped):
        finished.append(finished_at)
        delays.append(end_to_end_ms)
        return plain_add(self, finished_at, timing_ms, end_to_end_ms, live, dropped)

    metrics.MetricsWindow.add = add
    status_ms = []
    try:
        with tempfile.TemporaryDirectory() as data_dir:  # nothing is enrolled: the clip animates its own frame
            runtime = Runtime(data_dir=data_dir)
            if pace is not None:
                runtime.session.pace_fps = pace
            runtime.start_session(f"sample:{clip}")
            deadline = time.time() + 300
            while runtime.session.latest_pair() is None and time.time() < deadline:
                time.sleep(0.5)
            if runtime.session.latest_pair() is None:
                raise SystemExit(f"The session did not start: {runtime.session.snapshot()['message']}")
            time.sleep(SETTLE_SECONDS)
            finished.clear()
            delays.clear()
            # Meanwhile, what the status costs: the browser's stream builds it four times a second.
            end = time.perf_counter() + WATCH_SECONDS
            while time.perf_counter() < end:
                started = time.perf_counter()
                runtime.status()
                status_ms.append((time.perf_counter() - started) * 1000)
                time.sleep(0.25)
            frames, delay_ms = list(finished), list(delays)
            runtime.shutdown()
    finally:
        metrics.MetricsWindow.add = plain_add
    return frames, status_ms, delay_ms


def poll_sender(frames, fps, rng):
    """Today's sender: wakes `fps` times a second and sends the newest finished frame. Returns (sent at, frame)."""
    sent, tick, i = [], frames[0], -1
    while tick < frames[-1]:
        at = tick + rng.gauss(0.0, WAKE_SPREAD_S)
        while i + 1 < len(frames) and frames[i + 1] <= at:
            i += 1
        if i >= 0:
            sent.append((at, i))
        tick += 1.0 / fps
    return sent


def arrival_sender(frames):
    """Each frame sent once, as it is finished."""
    return [(at, i) for i, at in enumerate(frames)]


def picked_up(sent, fps, start):
    """The application's clock: `fps` times a second from `start`, the newest frame sent by then. None for a
    reader that takes every frame as it is written (v4l2loopback)."""
    if fps is None:
        return list(sent)
    shown, j = [], -1
    tick = start
    while tick < sent[-1][0]:
        while j + 1 < len(sent) and sent[j + 1][0] <= tick:
            j += 1
        if j >= 0:
            shown.append((tick, sent[j][1]))
        tick += 1.0 / fps
    return shown


def judge(frames, shown, seconds):
    """What a viewer gets from the pick ups: new frames, frames never shown, holds and the timing error."""
    firsts, last = [], None  # the first pick up of each new frame
    for at, i in shown:
        if i != last:
            firsts.append((at, i))
            last = i
    holds = [(b[0] - a[0]) * 1000 for a, b in zip(firsts, firsts[1:])]
    late = [(at - frames[i]) * 1000 for at, i in firsts]
    steps = [((b[0] - a[0]) - (frames[b[1]] - frames[a[1]])) * 1000 for a, b in zip(firsts, firsts[1:])]
    seen = {i for _, i in firsts}
    span = [i for i in range(len(frames)) if frames[0] + 1.0 <= frames[i] <= frames[-1] - 1.0]
    lost = sum(1 for i in span if i not in seen) / max(1, len(span))
    typical = statistics.median(holds) if holds else 0.0
    return {
        "new_frames_per_second": len(firsts) / seconds,
        "never_shown": lost,
        "hold_ms_mean": statistics.mean(holds) if holds else None,
        "hold_ms_sd": statistics.pstdev(holds) if holds else None,
        "holds_half_again_longer": sum(1 for h in holds if h > 1.5 * typical) / max(1, len(holds)),
        "step_error_ms_sd": statistics.pstdev(steps) if steps else None,
        "late_ms_mean": statistics.mean(late),
    }


def case(frames, sender, camera_fps, rng):
    seconds = frames[-1] - frames[0]
    runs = []
    for _ in range(STARTS if camera_fps is not None else 1):
        sent = poll_sender(frames, 30, rng) if sender == "30 Hz clock" else arrival_sender(frames)
        start = frames[0] + rng.random() / (camera_fps or 30)
        runs.append(judge(frames, picked_up(sent, camera_fps, start), seconds))
    keys = runs[0].keys()
    return {k: round(statistics.mean(r[k] for r in runs if r[k] is not None), 3) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", default="d6", help="the sample clip that stands in for the camera")
    parser.add_argument("--pace", type=float, nargs="*", default=[], help="also record with the drawing paced to these rates")
    parser.add_argument("--out", default=str(ROOT / "results" / "meeting_cadence_study.json"))
    args = parser.parse_args()

    import torch

    rng = random.Random(7)
    runs = []
    for pace in [None, *args.pace]:
        frames, status_ms, delay_ms = record(args.clip, pace)
        gaps = [(b - a) * 1000 for a, b in zip(frames, frames[1:])]
        drawn = {"paced_to": pace, "frames": len(frames), "fps": round((len(frames) - 1) / (frames[-1] - frames[0]), 2),
                 "gap_ms_mean": round(statistics.mean(gaps), 2), "gap_ms_sd": round(statistics.pstdev(gaps), 2),
                 "gap_ms_max": round(max(gaps), 1),
                 "delay_ms_median": round(statistics.median(delay_ms), 1),
                 "delay_ms_p95": round(sorted(delay_ms)[int(0.95 * len(delay_ms))], 1),
                 "status_ms_median": round(statistics.median(status_ms), 2), "status_ms_max": round(max(status_ms), 1)}
        print(f"\nDrawn{'' if pace is None else f', paced to {pace:g}'}: {drawn['fps']} fps, "
              f"{drawn['gap_ms_mean']} ms apart (sd {drawn['gap_ms_sd']}, longest {drawn['gap_ms_max']}), "
              f"delay {drawn['delay_ms_median']} ms (95% within {drawn['delay_ms_p95']})", flush=True)
        cases = []
        for name, sender, camera_fps in [
            ("Today: 30 Hz clock, camera at 30", "30 Hz clock", 30),
            ("Sent as finished, camera at 30", "as finished", 30),
            ("Sent as finished, camera at 20", "as finished", 20),
            ("Sent as finished, camera at 22", "as finished", 22),
            ("Linux reader: 30 Hz clock", "30 Hz clock", None),
            ("Linux reader: sent as finished", "as finished", None),
        ]:
            result = {"case": name, **case(frames, sender, camera_fps, rng)}
            cases.append(result)
            print(f"  {name:36s} {result['new_frames_per_second']:5.1f} new/s  never shown {result['never_shown']:5.1%}  "
                  f"hold {result['hold_ms_mean']:5.1f} ms (sd {result['hold_ms_sd']:4.1f}, long {result['holds_half_again_longer']:5.1%})  "
                  f"step error {result['step_error_ms_sd']:4.1f} ms  late {result['late_ms_mean']:5.1f} ms", flush=True)
        runs.append({"drawn": drawn, "cases": cases})

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": {"os": platform.platform(), "gpu": torch.cuda.get_device_name(0)},
        "clip": args.clip,
        "seconds_watched": WATCH_SECONDS,
        "starting_moments_per_case": STARTS,
        "runs": runs,
        "notes": [
            "A live session through the server's own session code, with a sample clip in place of the camera; no browser "
            "attached, no virtual camera opened. Finish times are recorded; sending and picking up are replayed from them.",
            "The pick up is modelled on OBS Virtual Camera's camera filter, which wakes at the rate the camera was opened "
            "with and takes the newest frame written. The 'Linux reader' cases take every frame written, as v4l2loopback "
            "hands them on. What the meeting application then does with the frames (its own encoding rate) is not modelled.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
