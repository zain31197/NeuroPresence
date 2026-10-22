"""Measure the engine while a meeting application runs beside it: no meeting, Google Meet, Zoom.

Start the engine and a camera session in the Live Studio first, with "Virtual camera" and
"Voice in step" switched on. Then run this script. It takes you through three phases of two minutes,
and reads once a second the engine's own figures (over its local web address, without changing
anything), the graphics card (nvidia-smi) and the CPU time of the engine, the browser and Zoom.
After each call it asks a few yes or no questions about what the other side saw and heard.

    python scripts/measure_call.py                      # all three phases, two minutes each
    python scripts/measure_call.py --phases zoom        # only one phase again; the others are kept
    python scripts/measure_call.py --seconds 60         # shorter phases

During a call, join it from a second device (a phone) so that video really flows, and keep talking
and moving as in a meeting. The results go to results/call_measurements_<name>.json, and a table
for the report is printed at the end. Nothing of the picture or the sound is recorded.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neuropresence.evaluation.call import (STATUS_URL, CpuReader, gpu_name, measure, read_gpu,  # noqa: E402
                                           summarise_phase, table)
from neuropresence.targets import TARGETS  # noqa: E402

RESULTS = ROOT / "results"

PHASES = {
    "baseline": {
        "title": "No meeting",
        "program": None,
        "steps": "Close Google Meet and Zoom. Keep the session live, and talk and move as you would in a call.",
    },
    "meet": {
        "title": "Google Meet",
        "program": "browser",
        "steps": "Join a Google Meet call in the browser. In its settings choose the camera \"OBS Virtual Camera\" "
                 "and the microphone \"CABLE Output (VB-Audio Virtual Cable)\". Join the same call from your phone, "
                 "with its microphone muted. Keep talking and moving.",
    },
    "zoom": {
        "title": "Zoom",
        "program": "zoom",
        "steps": "Leave Meet and close its tab. Join a Zoom meeting in the Zoom app with the same camera and "
                 "microphone, and join it from your phone too. Keep talking and moving.",
    },
}

QUESTIONS = (
    ("picture_seen", "Did the phone show your NeuroPresence picture (not your real camera)?"),
    ("mark_readable", "Could the \"AI reenacted\" mark be read on the phone?"),
    ("voice_heard", "Did the phone play your voice?"),
    ("in_step", "Did lips and voice look in step on the phone?"),
)


URL = STATUS_URL


def read_status():
    try:
        return requests.get(URL, timeout=2).json()
    except (requests.RequestException, ValueError):
        return None


def git_revision():
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ask(question):
    """y, n, or s to skip. Returns True, False or None."""
    while True:
        answer = input(f"  {question} [y/n/s] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        if answer in ("s", "skip", ""):
            return None


def check_session():
    """Stop with what to do when no camera session is live."""
    status = read_status()
    if status is None:
        raise SystemExit(f"The engine does not answer at {URL}. Start it with: "
                         "python -m neuropresence.server --open")
    session = status.get("session") or {}
    if session.get("state") != "running":
        raise SystemExit("No session is running. In the Live Studio choose Camera 0, press Start session and "
                         "answer the prompt, then run this again.")
    if session.get("source") != "enrolment":
        raise SystemExit("The running session is from a sample clip. Meetings only ever get a camera session: "
                         "start one from Camera 0.")
    camera, voice = status.get("virtual_camera") or {}, status.get("virtual_microphone") or {}
    if not camera.get("on"):
        print("Note: \"Virtual camera\" is off. Switch it on in the Live Studio before the Meet and Zoom phases.")
    if not voice.get("on"):
        print("Note: \"Voice in step\" is off. Switch it on in the Live Studio for the voice questions.")


def progress(sample):
    fps = "-" if sample.get("fps") is None else f"{sample['fps']:.1f}"
    delay = "-" if sample.get("delay_ms") is None else f"{sample['delay_ms']:.0f}"
    busy = "-" if sample.get("gpu_util_pct") is None else f"{sample['gpu_util_pct']:.0f}"
    held = f"  still picture: {sample['held']}" if sample.get("held") else ""
    state = "" if sample.get("state") == "running" else f"  session {sample.get('state') or 'unreachable'}"
    print(f"\r  {sample['t']:5.0f} s   {fps:>5} fps   {delay:>4} ms   GPU {busy:>3}%{held}{state}        ",
          end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phases", default="baseline,meet,zoom", help="which phases, in order, comma separated")
    parser.add_argument("--seconds", type=int, default=120, help="length of each phase")
    parser.add_argument("--name", default="windows_rtx5050", help="results/call_measurements_<name>.json")
    parser.add_argument("--url", default=STATUS_URL, help="the engine's status address")
    args = parser.parse_args()
    global URL
    URL = args.url
    chosen = [key.strip() for key in args.phases.split(",") if key.strip()]
    unknown = [key for key in chosen if key not in PHASES]
    if unknown:
        raise SystemExit(f"Unknown phase {', '.join(unknown)}. Choose from {', '.join(PHASES)}.")

    check_session()
    path = RESULTS / f"call_measurements_{args.name}.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"phases": {}}
    result.update({
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": {"os": platform.platform(), "gpu": gpu_name(), "cpu_cores": os.cpu_count(),
                    "python": platform.python_version()},
        "code": git_revision(),
        "targets": TARGETS,
        "seconds_per_phase": args.seconds,
    })
    cpu = CpuReader()

    for key in chosen:
        phase = PHASES[key]
        print(f"\n=== {phase['title']} ===\n{phase['steps']}")
        input("Press Enter when you are ready; the measurement then runs for "
              f"{args.seconds} seconds. ")
        samples = measure(args.seconds, read_status, read_gpu, cpu.read, time.perf_counter, time.sleep,
                          on_sample=progress)
        print()
        summary = summarise_phase(samples)
        if phase["program"] and summary.get(f"cpu_{phase['program']}_pct") is None:
            print(f"  Note: no {phase['title']} program was found running during this phase.")
        answers = {}
        if phase["program"]:
            print("  What did the phone show?")
            answers = {name: ask(question) for name, question in QUESTIONS}
        notes = input("  Anything to note about this phase (Enter for nothing): ").strip()
        result["phases"][key] = {"title": phase["title"], "measured_at": datetime.now().isoformat(timespec="seconds"),
                                 "summary": summary, "answers": answers, "notes": notes, "samples": samples}
        RESULTS.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")  # kept after every phase

    ordered = {key: result["phases"][key] for key in PHASES if key in result["phases"]}
    print(f"\nSaved {path.relative_to(ROOT)}\n")
    print(table(ordered))
    for key, phase in ordered.items():
        if phase.get("answers"):
            said = ", ".join(f"{name.replace('_', ' ')}: {'-' if value is None else ('yes' if value else 'no')}"
                             for name, value in phase["answers"].items())
            print(f"\n{phase['title']}: {said}")


if __name__ == "__main__":
    main()
