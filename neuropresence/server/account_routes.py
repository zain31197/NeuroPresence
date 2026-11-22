"""The accounts over HTTP: signing up and in, the account itself, its data and its pictures.

Everything under /api needs a signed-in account, the one the engine is open for, except what is listed in PUBLIC:
the landing page's figures, and what a person needs before they are signed in. The sign-in is a cookie that scripts
cannot read and that a browser sends only for this site's own pages (HttpOnly, SameSite=Strict).
"""

import json
from datetime import datetime

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from ..accounts import AccountError, device_of
from .session import SessionError

COOKIE = "neuropresence_login"
SIGNED_OUT = 4401  # the code the stream is closed with when no one is signed in (401 does not exist there)
PUBLIC = {
    "/api/health", "/api/engine", "/api/capabilities", "/api/benchmarks", "/api/benchmarks/latest",
    "/api/account", "/api/account/sign-up", "/api/account/sign-in", "/api/account/sign-out", "/api/account/reset",
    "/api/docs", "/api/openapi.json",
}


class SignUp(BaseModel):
    email: str
    name: str
    password: str
    remember: bool = True


class SignIn(BaseModel):
    email: str
    password: str
    remember: bool = True


class Reset(BaseModel):
    email: str
    code: str
    password: str


class AccountChange(BaseModel):
    name: str | None = None
    email: str | None = None
    password: str | None = None


class PasswordChange(BaseModel):
    current: str
    new: str


class WithPassword(BaseModel):
    password: str


class PictureName(BaseModel):
    name: str


def shown(user):
    return None if user is None else {key: value for key, value in user.items() if key != "folder"}


def signed_in(runtime, cookies):
    """The account these cookies sign in, if it is the one the engine is open for; otherwise None."""
    user = runtime.accounts.user_of(cookies.get(COOKIE))
    return user if user is not None and user["id"] == runtime.user_id else None


def add_account_routes(app, runtime):
    accounts = runtime.accounts

    @app.middleware("http")
    async def need_account(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in PUBLIC:
            if await run_in_threadpool(signed_in, runtime, request.cookies) is None:
                return JSONResponse({"detail": "Sign in first."}, status_code=401)
        return await call_next(request)

    def attempt(action):
        try:
            return action()
        except AccountError as err:
            raise HTTPException(status_code=err.status, detail=str(err)) from None
        except SessionError as err:
            raise HTTPException(status_code=409, detail=str(err)) from None
        except KeyError:
            raise HTTPException(status_code=404, detail="There is no such picture.") from None

    def enter(user, request, response, remember):
        """Sign a user in on this browser, and open the engine for them."""
        runtime.may_open(user)  # refused while someone else's session runs, before anyone is signed out
        token, seconds = accounts.new_login(user["id"], device_of(request.headers.get("user-agent")), remember)
        runtime.open_account(user)
        response.set_cookie(COOKIE, token, max_age=int(seconds) if remember else None, httponly=True, samesite="strict", path="/")

    def me():
        return runtime.user

    # ------------------------------------------------------------ public

    @app.get("/api/health")
    def health():
        """Does the engine answer? Nothing else: anyone on this computer may ask."""
        return {"ok": True}

    @app.get("/api/engine")
    def engine():
        """How the engine runs, for a measurement beside a call (scripts/measure_call.py). Nothing of the account's."""
        return runtime.engine_status()

    @app.get("/api/account")
    def account(request: Request):
        """Who is signed in on this browser, and whether this computer has any account yet."""
        user = signed_in(runtime, request.cookies)
        return {"user": shown(user), "first": user is None and not accounts.any(),
                "claims": user is None and not accounts.any() and (runtime.face is not None or runtime.enrolment is not None
                                                                   or bool(runtime.db.sessions(1)))}

    @app.post("/api/account/sign-up")
    def sign_up(body: SignUp, request: Request, response: Response):
        def action():
            if runtime.session.active:
                raise SessionError("A session is running. Stop it before making an account.")
            user, code = accounts.sign_up(body.email, body.name, body.password)
            enter(user, request, response, body.remember)
            return {"user": shown(runtime.user), "recovery_code": code}
        return attempt(action)

    @app.post("/api/account/sign-in")
    def sign_in(body: SignIn, request: Request, response: Response):
        def action():
            user = accounts.check(body.email, body.password)
            enter(user, request, response, body.remember)
            return {"user": shown(runtime.user)}
        return attempt(action)

    @app.post("/api/account/sign-out")
    def sign_out(request: Request, response: Response):
        """Signed out on this browser. If nowhere else is signed in, the engine closes the account too."""
        user = signed_in(runtime, request.cookies)
        accounts.sign_out(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/")
        if user is not None and not accounts.devices(user["id"]):
            runtime.close_account()
        return {"ok": True}

    @app.post("/api/account/reset")
    def reset(body: Reset, request: Request, response: Response):
        """A forgotten password: the recovery code sets a new one and signs in. The next code is in the reply."""
        def action():
            user, code = accounts.reset_password(body.email, body.code, body.password)
            enter(user, request, response, True)
            return {"user": shown(runtime.user), "recovery_code": code}
        return attempt(action)

    # ------------------------------------------------------------ the account

    @app.patch("/api/account")
    def change(body: AccountChange):
        def action():
            runtime.open_account(accounts.update(me()["id"], body.name, body.email, body.password))
            return {"user": shown(runtime.user)}
        return attempt(action)

    @app.post("/api/account/password")
    def password(body: PasswordChange, request: Request):
        """A new password. Every other browser is signed out."""
        attempt(lambda: accounts.change_password(me()["id"], body.current, body.new, request.cookies.get(COOKIE)))
        runtime.events.add("info", "Password changed. Other browsers were signed out.")
        return {"ok": True}

    @app.post("/api/account/recovery-code")
    def recovery_code(body: WithPassword):
        return {"recovery_code": attempt(lambda: accounts.new_recovery_code(me()["id"], body.password))}

    @app.get("/api/account/devices")
    def devices(request: Request):
        return {"devices": accounts.devices(me()["id"], request.cookies.get(COOKIE)), "attempts": accounts.attempts(me()["id"])}

    @app.delete("/api/account/devices")
    def sign_out_others(request: Request):
        accounts.sign_out_others(me()["id"], request.cookies.get(COOKIE))
        return devices(request)

    @app.post("/api/account/delete")
    def delete(body: WithPassword, response: Response):
        """Delete the account with everything of its."""
        attempt(lambda: accounts.confirm(me()["id"], body.password))
        runtime.delete_account()
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    # ------------------------------------------------------------ its data

    @app.get("/api/account/data")
    def data():
        """The database as this account's data: its tables, columns and keys, and how many rows are theirs."""
        return runtime.db.overview(runtime.user_id)

    @app.get("/api/account/export")
    def export():
        """Everything the database holds about this account, as a file."""
        body = {"app": "NeuroPresence", "exported_at": datetime.now().isoformat(timespec="seconds"),
                "account": shown(me()), "tables": runtime.db.export(runtime.user_id)}
        return Response(json.dumps(body, indent=1), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="neuropresence-data.json"', "Cache-Control": "no-store"})

    # ------------------------------------------------------------ its pictures

    @app.get("/api/pictures")
    def pictures():
        return runtime.pictures()

    @app.get("/api/pictures/{picture_id}/image")
    def picture_image(picture_id: int, thumbnail: bool = False):
        found = runtime.picture_image(picture_id, thumbnail)
        if found is None:
            raise HTTPException(status_code=404, detail="There is no such picture.")
        return Response(found, media_type="image/jpeg" if thumbnail else "image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/pictures/{picture_id}/use")
    def use_picture(picture_id: int):
        """Make it the meeting picture. It is checked as an upload is, the face match included."""
        return attempt(lambda: runtime.use_picture(picture_id))

    @app.patch("/api/pictures/{picture_id}")
    def rename_picture(picture_id: int, body: PictureName):
        return attempt(lambda: runtime.rename_picture(picture_id, body.name))

    @app.delete("/api/pictures/{picture_id}")
    def delete_picture(picture_id: int):
        return attempt(lambda: runtime.delete_picture(picture_id))
