"""The local web server: a small API over the pipeline, plus the built web app.

It listens on this machine only. Nothing it handles leaves the computer.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .capabilities import ENROLMENT, STAGES
from .preview import NotGoodEnough
from .runtime import Runtime
from .session import SessionError
from .stream import pack_frame

log = logging.getLogger("neuropresence.server")
WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
STATUS_INTERVAL_S = 0.25


class StartRequest(BaseModel):
    input: str = "camera:0"


class FeaturePatch(BaseModel):
    enabled: bool


def create_app(runtime=None, web_dist=WEB_DIST):
    runtime = runtime or Runtime()

    @asynccontextmanager
    async def lifespan(app):
        yield
        runtime.shutdown()

    app = FastAPI(title="NeuroPresence", lifespan=lifespan, docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.runtime = runtime

    def refuse(err, status=409):
        raise HTTPException(status_code=status, detail=str(err)) from None

    def attempt(action):
        """Run a request and turn the ways it can be refused into clear replies.

        409: it cannot be done right now (wrong state). 422: the picture is not
        good enough. 504: the session did not answer.
        """
        try:
            return action()
        except SessionError as err:
            refuse(err)
        except NotGoodEnough as err:
            refuse(err, 422)
        except TimeoutError:
            refuse("That took too long. Try again.", 504)

    def picture(jpeg, missing):
        if jpeg is None:
            raise HTTPException(status_code=404, detail=missing)
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ----------------------------------------------------------------- status

    @app.get("/api/status")
    def status():
        return runtime.status()

    @app.get("/api/capabilities")
    def capabilities():
        return {"enrolment": ENROLMENT, "stages": STAGES}

    @app.get("/api/benchmarks/latest")
    def benchmark():
        latest = runtime.latest_benchmark()
        if latest is None:
            raise HTTPException(status_code=404, detail="No benchmark has been run yet.")
        return latest

    # -------------------------------------------------------------- enrolment

    @app.get("/api/enrolment/guide")
    def enrolment_guide():
        return runtime.enrolment_guide()

    @app.post("/api/enrolment/preview/start")
    def start_preview(request: StartRequest):
        attempt(lambda: runtime.start_preview(request.input))
        return runtime.status()

    @app.post("/api/enrolment/preview/stop")
    def stop_preview():
        runtime.stop_preview()
        return runtime.status()

    @app.post("/api/enrolment/take")
    def take_picture():
        """Take a picture from the open camera. It becomes the candidate, to be confirmed or discarded."""
        return attempt(runtime.take_picture)

    @app.post("/api/enrolment/upload")
    def upload(file: UploadFile = File(...)):
        """Check an uploaded picture. It becomes the candidate even if a check fails, so the result can be shown."""
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            refuse("That picture is larger than 15 MB. Choose a smaller one.", 413)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            refuse("That file is not a picture this app can read. Use a JPEG or PNG.", 422)
        return attempt(lambda: runtime.check_upload(image))

    @app.post("/api/enrolment/confirm")
    def confirm():
        """Enrol the candidate. This loads the animation model the first time, which takes a few seconds."""
        try:
            return attempt(runtime.confirm_candidate)
        except ValueError as err:  # the animation model found no usable face
            refuse(f"{err} Try a clearer, front-facing picture.", 422)

    @app.delete("/api/enrolment/candidate")
    def discard_candidate():
        runtime.discard_candidate()
        return {"ok": True}

    @app.delete("/api/enrolment")
    def remove_enrolment():
        attempt(runtime.remove_enrolment)
        return {"ok": True}

    @app.get("/api/enrolment/picture")
    def enrolment_picture():
        return picture(runtime.enrolment_jpeg(), "No picture is enrolled yet.")

    @app.get("/api/enrolment/candidate/picture")
    def candidate_picture():
        return picture(runtime.candidate_jpeg(), "There is no picture waiting to be confirmed.")

    # ---------------------------------------------------------------- session

    @app.post("/api/session/start")
    def start(request: StartRequest):
        attempt(lambda: runtime.start_session(request.input))
        return runtime.status()

    @app.post("/api/session/stop")
    def stop():
        runtime.stop_session()
        return runtime.status()

    @app.post("/api/session/neutral")
    def neutral():
        attempt(runtime.reset_neutral)
        return {"ok": True}

    # --------------------------------------------------------------- features

    @app.patch("/api/features/{key}")
    def set_feature(key: str, patch: FeaturePatch):
        try:
            feature = runtime.features.set(key, patch.enabled)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown feature: {key}") from None
        runtime.events.add("info", f"{feature['label']} turned {'on' if feature['enabled'] else 'off'}.")
        return feature

    # ----------------------------------------------------------------- stream

    @app.websocket("/api/stream")
    async def stream(socket: WebSocket):
        """Pushes status four times a second and every new frame as it is made:
        camera and output pairs in a live session, camera frames in the enrolment preview."""
        await socket.accept()
        loop = asyncio.get_running_loop()
        last_frame, last_event, next_status = None, 0, 0.0
        try:
            while True:
                now = loop.time()
                if now >= next_status:
                    next_status = now + STATUS_INTERVAL_S
                    try:
                        status = runtime.status()
                    except Exception:  # one bad reading must not cut the picture off
                        log.exception("Could not read the status; the stream continues.")
                        status = None
                    if status is not None:
                        events = runtime.events.since(last_event)
                        if events:
                            last_event = events[-1]["id"]
                        await socket.send_text(json.dumps({"status": status, "events": events}))
                pair = runtime.latest_pair()
                if pair is not None and (pair.kind, pair.id) != last_frame:
                    last_frame = (pair.kind, pair.id)
                    overlay = pair.kind == "live" and runtime.features.enabled("tracking_overlay")
                    await socket.send_bytes(await asyncio.to_thread(pack_frame, pair, overlay))
                else:
                    await asyncio.sleep(0.01)
        except (WebSocketDisconnect, RuntimeError):
            pass  # the browser tab closed

    # ------------------------------------------------------------ the web app

    web_dist = Path(web_dist).resolve()
    if (web_dist / "index.html").exists():
        if (web_dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def web_app(path: str):
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            file = web_dist / path
            if path and file.is_file() and web_dist in file.resolve().parents:
                return FileResponse(file)
            return FileResponse(web_dist / "index.html")  # the app handles its own routes

    return app
