"""Start the NeuroPresence web app on this machine.

    python -m neuropresence.server            # then open http://127.0.0.1:8000
    python -m neuropresence.server --open     # also opens the browser
"""

import argparse
import threading
import webbrowser

import uvicorn

from .app import WEB_DIST, create_app
from .runtime import DATA_DIR, Runtime, _default_feed


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

    runtime = Runtime(feed_factory=feed, data_dir=args.data_dir)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
