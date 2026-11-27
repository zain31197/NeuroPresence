"""Check that this computer is ready for a live session and for the demonstration.

    python scripts/preflight.py              every check, camera 0 included
    python scripts/preflight.py --camera 1   another camera
    python scripts/preflight.py --no-camera  leave the camera alone
    python scripts/preflight.py --warm       also convert the networks for this graphics card, warm the voice model,
                                             and measure how fast this computer draws (about three minutes)
    python scripts/preflight.py --demo       the morning of the demonstration: all of that, then the rehearsal of
                                             its failure paths on this computer (about eight minutes)

Run it with the engine stopped, or with the virtual camera switched off in the Live Studio: the check opens the
virtual camera for a moment. The report is printed and also kept in logs/preflight.log, to send to the team.
See neuropresence/preflight.py for what each check looks at.
"""

import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.preflight import FAIL, rehearse, report, run_checks, warm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Check that this computer is ready for a NeuroPresence session.")
    parser.add_argument("--camera", type=int, default=0, help="the camera to check (default 0)")
    parser.add_argument("--no-camera", action="store_true", help="do not open the camera")
    parser.add_argument("--warm", action="store_true",
                        help="also make the networks ready for this graphics card and measure the drawing speed")
    parser.add_argument("--seconds", type=float, default=60.0, help="how long --warm draws for (default 60)")
    parser.add_argument("--no-voice", action="store_true", help="with --warm: leave the voice model out")
    parser.add_argument("--data-dir", default=str(ROOT / "data"), help="where the enrolment is kept")
    parser.add_argument("--port", type=int, default=8000, help="the port the engine will use (default 8000)")
    parser.add_argument("--demo", action="store_true",
                        help="for the demonstration: --warm, then the rehearsal of its failure paths (scripts/rehearse_demo.py)")
    args = parser.parse_args()
    args.warm = args.warm or args.demo

    print("Checking this computer ...", flush=True)
    checks = run_checks(camera=None if args.no_camera else args.camera, data_dir=args.data_dir, port=args.port)
    ready = not any(c.level == FAIL for c in checks if c.name in ("Graphics card", "Python packages")
                    or c.name.startswith("File: "))
    say = lambda text: print(text, flush=True)  # noqa: E731
    if args.warm:
        if not ready:
            print("Skipping the warm-up: the graphics card, the packages or the model files are not ready.")
        else:
            checks += warm(data_dir=args.data_dir, seconds=args.seconds, voice=not args.no_voice, say=say)
    if args.demo and ready:
        checks += rehearse(say=say)
    text = report(checks, demo=args.demo)
    header = f"NeuroPresence preflight, {datetime.now():%Y-%m-%d %H:%M}, {platform.node()}, {platform.platform()}"
    print("\n" + header + "\n" + text)
    log = ROOT / "logs" / "preflight.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(header + "\n" + text + "\n\n")
    print(f"\nKept in {log}")
    sys.exit(1 if any(c.level == FAIL for c in checks) else 0)


if __name__ == "__main__":
    main()
