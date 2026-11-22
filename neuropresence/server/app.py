"""The local web server: a small API over the pipeline, plus the built web app.

It listens on this machine only. Nothing it handles leaves the computer. Everything but the landing page's
figures needs a signed-in account (account_routes.py).
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

from .account_routes import SIGNED_OUT, add_account_routes, signed_in
from .capabilities import ENROLMENT, LIVE, MODES, STAGES
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


class CameraTest(BaseModel):
    input: str = "camera:0"


class FeaturePatch(BaseModel):
    enabled: bool


class SessionChange(BaseModel):
    title: str | None = None
    note: str | None = None
    starred: bool | None = None


class CameraSwitch(BaseModel):
    on: bool


class MicrophoneChange(BaseModel):
    on: bool | None = None
    microphone: str | None = None
    offset_ms: float | None = None


def create_app(runtime=None, web_dist=WEB_DIST):
    runtime = runtime or Runtime()

    @asynccontextmanager
    async def lifespan(app):
        yield
        runtime.shutdown()

    app = FastAPI(title="NeuroPresence", lifespan=lifespan, docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.runtime = runtime
    add_account_routes(app, runtime)

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
        return {"enrolment": ENROLMENT, "stages": STAGES, "modes": MODES, "live": LIVE}

    @app.get("/api/history")
    def history(limit: int = 100, search: str = "", starred: bool = False):
        """The account's past sessions, newest first, with totals (database.py)."""
        return runtime.history(max(1, min(limit, 1000)), search[:100], starred)

    @app.get("/api/history/{session_id}")
    def history_session(session_id: int):
        """One past session: its figures over time, its events and its consent check."""
        try:
            return runtime.session_record(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="No such session.") from None

    @app.patch("/api/history/{session_id}")
    def change_session(session_id: int, change: SessionChange):
        """Give a past session a name or a note, or a star."""
        try:
            return runtime.change_session(session_id, change.model_dump(exclude_unset=True))
        except KeyError:
            raise HTTPException(status_code=404, detail="No such session.") from None

    @app.delete("/api/history/{session_id}")
    def delete_session(session_id: int):
        try:
            runtime.delete_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="No such session, or it is still running.") from None
        return {"ok": True}

    @app.delete("/api/history")
    def clear_history():
        """Forget the session history. A running session is kept, and the starred ones."""
        runtime.clear_history()
        return runtime.history()

    @app.get("/api/security")
    def security(limit: int = 200):
        """Every consent check and change to the enrolment, newest first, and the retry limit."""
        return runtime.security(limit=max(1, min(limit, 1000)))

    @app.delete("/api/security")
    def clear_security():
        runtime.clear_security()
        return runtime.security()

    @app.get("/api/preferences")
    def preferences():
        """The app's own choices it remembers, such as the input last used."""
        return runtime.preferences()

    @app.patch("/api/preferences")
    def set_preferences(changes: dict):
        return attempt(lambda: runtime.set_preferences(changes))

    @app.get("/api/benchmarks")
    def benchmarks():
        """Every saved benchmark and the newest call measurement, for the Benchmarks screen."""
        return runtime.benchmarks()

    @app.get("/api/system")
    def system(internet: bool = True):
        """Is this computer ready for a session: the preflight checks, run by the engine."""
        return runtime.system(internet=internet)

    @app.post("/api/system/camera")
    def system_camera(body: CameraTest):
        """Open a camera for two seconds and count the pictures it sends."""
        return attempt(lambda: runtime.test_camera(body.input))

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

    @app.post("/api/enrolment/pose/{pose}/start")
    def start_pose_preview(pose: str, request: StartRequest):
        """Open the camera to register one pose: 'left', 'right', 'up' or 'down'.

        Taking and confirming the picture reuse the ordinary /take and /confirm endpoints:
        the preview is already in pose mode, so every frame is judged against that pose's
        own checks (see enrolment/checks.py: evaluate_pose) instead of the frontal ones.
        """
        attempt(lambda: runtime.start_pose_preview(pose, request.input))
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
        return attempt(lambda: runtime.check_upload(image, Path(file.filename or "").stem))

    @app.post("/api/enrolment/confirm")
    def confirm():
        """Keep the candidate: a camera picture as the face signature, an uploaded one as the meeting picture.

        An uploaded picture loads the animation model the first time, which takes a few seconds.
        """
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
        """Remove the meeting picture. The verified face stays."""
        attempt(runtime.remove_enrolment)
        return {"ok": True}

    @app.delete("/api/enrolment/face")
    def forget_face():
        """Remove the face signature and the meeting picture with it."""
        attempt(runtime.forget_face)
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

    @app.post("/api/session/resume")
    def resume():
        """Go live again after an identity fallback."""
        attempt(runtime.resume_reenactment)
        return runtime.status()

    @app.post("/api/session/stay")
    def stay():
        """Keep the still picture after an identity fallback, rather than resuming by itself."""
        attempt(runtime.stay_on_still_picture)
        return runtime.status()

    # --------------------------------------------------------------- features

    @app.patch("/api/features/{key}")
    def set_feature(key: str, patch: FeaturePatch):
        try:
            feature = runtime.features.set(key, patch.enabled)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown feature: {key}") from None
        runtime.events.add("info", f"{feature['label']} turned {'on' if feature['enabled'] else 'off'}.")
        return feature

    # --------------------------------------------------------- virtual camera

    @app.post("/api/virtual-camera")
    def virtual_camera(switch: CameraSwitch):
        """Offer the output to meeting applications as a camera, or stop offering it."""
        return attempt(lambda: runtime.switch_virtual_camera(switch.on))

    # ----------------------------------------------------- virtual microphone

    @app.patch("/api/virtual-microphone")
    def virtual_microphone(change: MicrophoneChange):
        """Switch the delayed voice on or off, choose the microphone, or adjust the delay."""
        return attempt(lambda: runtime.change_virtual_microphone(change.on, change.microphone, change.offset_ms))

    # ----------------------------------------------------------------- stream

    @app.websocket("/api/stream")
    async def stream(socket: WebSocket):
        """Pushes status four times a second and every new frame as it is made:
        camera and output pairs in a live session, camera frames in the enrolment preview."""
        await socket.accept()
        user = await asyncio.to_thread(signed_in, runtime, socket.cookies)
        if user is None:
            await socket.close(SIGNED_OUT)  # after accepting: before it, a browser is not told the code
            return
        loop = asyncio.get_running_loop()
        last_frame, last_event, next_status, seen = None, 0, 0.0, None
        wanted = {"frames": True}

        async def listen():
            """What the page asks for. {"frames": false} while its tab is hidden (a meeting in front of it): then no
            pictures are packed and sent for nobody, which costs the computer the drawing needs. Status goes on."""
            try:
                while True:
                    text = await socket.receive_text()
                    try:
                        message = json.loads(text)
                    except ValueError:
                        continue  # not for here
                    if isinstance(message, dict) and isinstance(message.get("frames"), bool):
                        wanted["frames"] = message["frames"]
            except (WebSocketDisconnect, RuntimeError, KeyError):
                pass  # the tab closed

        listener = asyncio.create_task(listen())
        try:
            while True:
                now = loop.time()
                if now >= next_status:
                    next_status = now + STATUS_INTERVAL_S
                    if await asyncio.to_thread(signed_in, runtime, socket.cookies) is None:
                        await socket.close(SIGNED_OUT)  # signed out meanwhile, here or by another account
                        return
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
                if not wanted["frames"]:
                    await asyncio.sleep(max(0.02, next_status - loop.time()))
                elif pair is not None and (pair.kind, pair.id) != last_frame:
                    last_frame = (pair.kind, pair.id)
                    overlay = pair.kind == "live" and runtime.features.enabled("tracking_overlay")
                    await socket.send_bytes(await asyncio.to_thread(pack_frame, pair, overlay))
                else:
                    # Woken by the next frame as soon as it is made, and in time for the next status.
                    wait = min(STATUS_INTERVAL_S, max(0.0, next_status - loop.time()))
                    seen = await asyncio.to_thread(runtime.wait_for_frame, seen, wait)
        except (WebSocketDisconnect, RuntimeError):
            pass  # the browser tab closed
        finally:
            listener.cancel()

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
