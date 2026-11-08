"""Start the NeuroPresence web app on this machine.

    python -m neuropresence.server            # then open http://127.0.0.1:8000
    python -m neuropresence.server --open     # also opens the browser

What goes wrong is written to logs/server.log (kept to three files of 5 MB), so that a failure in a
long meeting can be read afterwards and not only in the terminal it was printed to. If the process
itself dies, in the graphics driver for instance, where Python cannot catch it, the stack of every
thread at that moment goes to logs/crash.log.
"""

import argparse
import faulthandler
import logging
import logging.handlers
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .app import WEB_DIST, create_app
from .runtime import DATA_DIR, Runtime, _default_feed

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def keep_logs(folder=LOG_DIR):
    """Write warnings and errors, with their traces, to folder/server.log, and a dying process's stacks to
    folder/crash.log. Returns the crash file, which has to stay open for as long as the process runs."""
    folder.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(folder / "server.log", maxBytes=5 << 20, backupCount=2,
                                                   encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger("neuropresence")
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    def unhandled(args):  # an error that ends a thread nobody waits on would otherwise only be printed
        logging.getLogger("neuropresence.server").error(
            "Unhandled error in thread %s", args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = unhandled
    crash = open(folder / "crash.log", "a", encoding="utf-8")
    faulthandler.enable(crash, all_threads=True)
    return crash


def main():
    parser = argparse.ArgumentParser(description="Start the NeuroPresence web app.")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the app in the browser")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="where the enrolled picture is kept (default: the data folder of the project)")
    parser.add_argument("--camera-file",
                        help="for testing without a webcam: play this video file in place of every camera")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if (WEB_DIST / "index.html").exists():
        print(f"NeuroPresence is starting at {url}")
    else:
        print(f"The API is starting at {url}, but the web app has not been built.\n"
              "Build it once with:  cd web && npm install && npm run build")
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    feed = _default_feed
    if args.camera_file:
        print(f"Cameras are replaced by the video file {args.camera_file}")

        def feed(source):
            return _default_feed(args.camera_file if isinstance(source, int) else source)

    crash = keep_logs()
    logging.getLogger("neuropresence.server").info("Engine started at %s", url)
    print(f"Errors are written to {LOG_DIR / 'server.log'}")
    runtime = Runtime(feed_factory=feed, data_dir=args.data_dir)
    try:
        uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="warning")
    finally:
        logging.getLogger("neuropresence.server").info("Engine stopped")
        crash.close()


if __name__ == "__main__":
    main()
