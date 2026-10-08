"""The delay from camera to output of a live session, frame by frame: alone, and beside another
program that keeps the GPU busy.

The target is at most 150 ms. A meeting is not the only thing the GPU does, so this measures
what happens to the frame rate and the delay when something else takes a share of the card, and
how long the delay stays above the limit. The delay watchdog's settings come from it.

It also measures what the identity monitor costs at different rates, since it scores the output
on the CPU while the session runs.

The load is artificial: a second process that runs convolutions for a set share of every 50 ms.
It stands in for a meeting application, it is not one. Writes results/delay_study.json.

Usage:
    python scripts/study_delay.py                 # about five minutes; no camera is opened
    python scripts/study_delay.py --clip d6
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOAD_PERIOD = 0.05
SETTLE_SECONDS, WATCH_SECONDS = 4.0, 24.0
LOADS = (0.25, 0.5, 0.75, 1.0)
MONITOR_INTERVALS = (0.5, 0.25)  # the session starts with the monitor's own interval, then these


def keep_gpu_busy(share, seconds):
    """Run in a second process: convolutions for `share` of every 50 ms."""
    import torch

    x = torch.randn(1, 64, 768, 768, device="cuda")
    conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
    end = time.perf_counter() + seconds
    with torch.no_grad():
        while time.perf_counter() < end:
            started = time.perf_counter()
            while time.perf_counter() - started < share * LOAD_PERIOD:
                conv(x)
                torch.cuda.synchronize()
            rest = LOAD_PERIOD - (time.perf_counter() - started)
            if rest > 0:
                time.sleep(rest)


def describe(case, frames, limit_ms):
    at = np.array([f[0] for f in frames])
    delay = np.array([f[1] for f in frames])
    render = np.array([f[2] for f in frames if f[2] is not None])
    over = delay > limit_ms
    longest, since, in_a_row, run = 0.0, None, 0, 0
    for moment, flag in zip(at, over):
        run = run + 1 if flag else 0
        in_a_row = max(in_a_row, run)
        if flag and since is None:
            since = moment
        if not flag and since is not None:
            longest, since = max(longest, moment - since), None
    if since is not None:
        longest = max(longest, at[-1] - since)
    # The mean over the last two seconds is what the Live Studio shows and what the watchdog reads.
    two_seconds = [float(delay[(at > moment - 2.0) & (at <= moment)].mean()) for moment in at[at >= at[0] + 2.0]]
    return {
        "case": case,
        "frames": len(frames),
        "fps": round((len(frames) - 1) / (at[-1] - at[0]), 1),
        "delay_ms": {"mean": round(float(delay.mean()), 1), "p95": round(float(np.percentile(delay, 95)), 1),
                     "p99": round(float(np.percentile(delay, 99)), 1), "max": round(float(delay.max()), 1)},
        "delay_ms_mean_of_two_seconds": {"lowest": round(min(two_seconds), 1), "highest": round(max(two_seconds), 1)},
        "render_ms": {"mean": round(float(render.mean()), 1), "p99": round(float(np.percentile(render, 99)), 1)},
        "share_of_frames_over_limit": round(float(over.mean()), 3),
        "longest_seconds_over_limit": round(float(longest), 2),
        "most_frames_over_limit_in_a_row": int(in_a_row),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", default="d6", help="the sample clip that stands in for the camera")
    parser.add_argument("--out", default=str(ROOT / "results" / "delay_study.json"))
    parser.add_argument("--load", nargs=2, type=float, metavar=("SHARE", "SECONDS"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.load:
        keep_gpu_busy(*args.load)
        return

    import torch

    from neuropresence.server import metrics
    from neuropresence.server.runtime import Runtime
    from neuropresence.targets import TARGETS

    limit_ms = TARGETS["end_to_end_ms"]
    seen = []
    plain_add = metrics.MetricsWindow.add

    def add(self, finished_at, timing_ms, end_to_end_ms, live, dropped):
        seen.append((finished_at, end_to_end_ms, timing_ms.get("render")))
        return plain_add(self, finished_at, timing_ms, end_to_end_ms, live, dropped)

    metrics.MetricsWindow.add = add  # every frame of the session passes through here

    def watch(case):
        time.sleep(SETTLE_SECONDS)
        seen.clear()
        time.sleep(WATCH_SECONDS)
        report = describe(case, list(seen), limit_ms)
        print(f"  {case:34s} {report['fps']:5.1f} fps   delay {report['delay_ms']['mean']:6.1f} ms "
              f"(p99 {report['delay_ms']['p99']:.0f}, max {report['delay_ms']['max']:.0f})   "
              f"over the limit {report['share_of_frames_over_limit']:.0%} of frames, "
              f"longest {report['longest_seconds_over_limit']:.2f} s", flush=True)
        return report

    with tempfile.TemporaryDirectory() as data_dir:  # nothing is enrolled: the clip animates its own frame
        runtime = Runtime(data_dir=data_dir)
        runtime.start_session(f"sample:{args.clip}")
        deadline = time.time() + 300
        while runtime.session.latest_pair() is None and time.time() < deadline:
            time.sleep(0.5)
        if runtime.session.latest_pair() is None:
            raise SystemExit(f"The session did not start: {runtime.session.snapshot()['message']}")
        monitor_interval = runtime.identity._interval

        cases = [watch("alone")]
        for share in LOADS:
            load = subprocess.Popen([sys.executable, __file__, "--load", str(share),
                                     str(SETTLE_SECONDS + WATCH_SECONDS + 3)])
            cases.append(watch(f"GPU busy elsewhere {share:.0%} of the time"))
            load.wait()
        cases.append(watch("alone again"))
        monitor = []
        for interval in MONITOR_INTERVALS:
            runtime.identity._interval = interval
            monitor.append(watch(f"identity scored every {interval} s"))
        runtime.identity._interval = monitor_interval
        runtime.shutdown()

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": {"os": platform.platform(), "gpu": torch.cuda.get_device_name(0)},
        "clip": args.clip,
        "limit_ms": limit_ms,
        "identity_scored_every_seconds": monitor_interval,
        "seconds_watched_per_case": WATCH_SECONDS,
        "cases": cases,
        "identity_monitor_rate": monitor,
        "notes": [
            "A live session through the server's own session code, with a sample clip in place of the camera. "
            "No browser is attached and no virtual camera exists yet.",
            "The delay is from a frame arriving from the feed to its output being ready.",
            "The load is a second process running convolutions for the given share of every 50 ms. It is not a "
            "meeting application; it shows what happens when the GPU has to be shared.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
