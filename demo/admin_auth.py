"""
Optional single-admin login for a self-hosted deployment.

Off unless ADMIN_EMAIL, ADMIN_PASSWORD_HASH and ADMIN_SESSION_SECRET are all set. When on, every page
and API route of the demo needs a signed session cookie. The reverse proxy must also call /auth/check
before it forwards a request to the Realtime server, which has no authentication of its own (see
deploy/gcp/Caddyfile).

Only a scrypt hash of the password is stored. Create one with:

    python demo/admin_auth.py hash-password
"""

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import sys
import time
from urllib.parse import urlsplit

COOKIE_NAME = "s2s_admin"
SESSION_SECONDS = 12 * 60 * 60
MAX_FAILURES = 5
# Failures from all addresses together, so that rotating addresses does not give unlimited guesses.
MAX_GLOBAL_FAILURES = 50
FAILURE_WINDOW_SECONDS = 15 * 60
MAX_BODY_BYTES = 4096

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**15, 8, 1
_SCRYPT_MAXMEM = 128 * 1024 * 1024

PUBLIC_PATHS = {"/login", "/auth/check", "/healthz"}


class ConfigError(RuntimeError):
    """The login is partly configured. Starting anyway could leave the app open."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, maxmem=_SCRYPT_MAXMEM
    )
    # Colons, not dollar signs, so the value survives shell-style environment files.
    return f"scrypt:{_SCRYPT_N}:{_SCRYPT_R}:{_SCRYPT_P}:{salt.hex()}:{digest.hex()}"


def _parse_hash(encoded: str) -> tuple[int, int, int, bytes, bytes]:
    scheme, n, r, p, salt, digest = encoded.split(":")
    if scheme != "scrypt":
        raise ValueError("unknown scheme")
    return int(n), int(r), int(p), bytes.fromhex(salt), bytes.fromhex(digest)


def verify_password(password: str, encoded: str) -> bool:
    n, r, p, salt, digest = _parse_hash(encoded)
    actual = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=_SCRYPT_MAXMEM)
    return hmac.compare_digest(actual, digest)


class AdminAuth:
    def __init__(self, email: str, password_hash: str, secret: str) -> None:
        if not email or len(secret) < 32:
            raise ConfigError("ADMIN_EMAIL must be set and ADMIN_SESSION_SECRET needs at least 32 characters.")
        try:
            _parse_hash(password_hash)
        except ValueError as exc:
            raise ConfigError("ADMIN_PASSWORD_HASH is not a hash made by admin_auth.py.") from exc
        self.email = email.strip().lower()
        self.password_hash = password_hash
        # Including the hash in the key signs everyone out when the password changes.
        self._key = hashlib.sha256(f"{secret}|{self.email}|{password_hash}".encode()).digest()
        self._failures: list[tuple[float, str]] = []

    @classmethod
    def from_env(cls, env=os.environ) -> "AdminAuth | None":
        names = ("ADMIN_EMAIL", "ADMIN_PASSWORD_HASH", "ADMIN_SESSION_SECRET")
        values = [env.get(name, "").strip() for name in names]
        if not any(values):
            return None
        missing = [name for name, value in zip(names, values) if not value]
        if missing:
            raise ConfigError(f"Admin login is only partly configured; missing {', '.join(missing)}.")
        return cls(*values)

    # ---- sessions ----

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def issue(self, now: float | None = None) -> str:
        expires = int((time.time() if now is None else now) + SESSION_SECONDS)
        body = json.dumps({"exp": expires, "id": secrets.token_hex(8)}).encode()
        payload = base64.urlsafe_b64encode(body).decode()
        return f"{payload}.{self._sign(payload)}"

    def valid(self, token: str | None, now: float | None = None) -> bool:
        payload, _, signature = (token or "").partition(".")
        # Compare bytes: compare_digest rejects non-ASCII strings with an exception.
        if not payload or not hmac.compare_digest(self._sign(payload).encode(), signature.encode()):
            return False
        try:
            expires = json.loads(base64.urlsafe_b64decode(payload))["exp"]
        except (ValueError, KeyError, TypeError):
            return False
        return (time.time() if now is None else now) < expires

    # ---- rate limit ----

    def locked(self, client: str, now: float | None = None) -> bool:
        cutoff = (time.time() if now is None else now) - FAILURE_WINDOW_SECONDS
        self._failures = [entry for entry in self._failures if entry[0] > cutoff]
        own = sum(1 for _, address in self._failures if address == client)
        return own >= MAX_FAILURES or len(self._failures) >= MAX_GLOBAL_FAILURES

    def record_failure(self, client: str, now: float | None = None) -> None:
        self._failures.append((time.time() if now is None else now, client))

    def clear_failures(self, client: str) -> None:
        self._failures = [entry for entry in self._failures if entry[1] != client]

    def check(self, email: str, password: str) -> bool:
        # Always hash, so that a wrong email takes as long as a wrong password.
        password_ok = verify_password(password, self.password_hash)
        email_ok = hmac.compare_digest(email.strip().lower().encode(), self.email.encode())
        return password_ok and email_ok


def same_origin(request) -> bool:
    """Reject requests another site made the browser send. Non-browser clients send no Origin."""
    origin = request.headers.get("origin")
    if not origin:
        return True
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return urlsplit(origin).netloc.lower() == host.lower()


LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex"><title>Sign in</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:#0a0b10;color:#f5f6fa;
font:16px/1.5 Inter,system-ui,sans-serif}
form{width:100%;max-width:360px;padding:32px;background:#13151c;border:1px solid rgba(255,255,255,.08);
border-radius:22px}
h1{margin:0 0 24px;font-size:1.35rem;font-weight:600}
label{display:block;margin-bottom:16px;font-size:.875rem;color:rgba(245,246,250,.75)}
input{display:block;width:100%;margin-top:6px;padding:11px 12px;font:inherit;color:inherit;background:#1b1e29;
border:1px solid rgba(255,255,255,.16);border-radius:8px}
input:focus-visible,button:focus-visible{outline:2px solid #8b7dff;outline-offset:2px}
button{width:100%;margin-top:8px;padding:12px;font:inherit;font-weight:600;color:#0a0b10;background:#f5f6fa;border:0;
border-radius:8px;cursor:pointer}
.error{margin:0 0 16px;padding:10px 12px;font-size:.875rem;color:#ff9aa2;background:rgba(255,106,117,.12);
border-radius:8px}
</style></head><body><form method="post" action="/login"><h1>Admin sign in</h1>__ERROR__
<label>Email<input name="email" type="email" autocomplete="username" required autofocus></label>
<label>Password<input name="password" type="password" autocomplete="current-password" required></label>
<button type="submit">Sign in</button></form></body></html>"""

_LOGIN_HEADERS = {
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
    "frame-ancestors 'none'; base-uri 'none'",
}


def attach(app, admin: AdminAuth | None) -> None:
    """Protect every route of ``app`` with the admin login. Does nothing without a configured admin."""
    from urllib.parse import parse_qs

    from fastapi import Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from starlette.concurrency import run_in_threadpool

    if admin is None:
        return

    def login_page(error: str = "", status: int = 200) -> HTMLResponse:
        block = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
        return HTMLResponse(LOGIN_PAGE.replace("__ERROR__", block), status, headers=_LOGIN_HEADERS)

    def signed_in(request: Request) -> bool:
        return admin.valid(request.cookies.get(COOKIE_NAME))

    def is_https(request: Request) -> bool:
        return request.url.scheme == "https"

    @app.middleware("http")
    async def require_admin(request: Request, call_next):
        path = request.url.path
        if path not in PUBLIC_PATHS and not signed_in(request):
            if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
                return RedirectResponse("/login", 303)
            return JSONResponse({"detail": "Sign in first."}, 401)
        if request.method not in ("GET", "HEAD") and not same_origin(request):
            return JSONResponse({"detail": "Cross-site request refused."}, 403)
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/auth/check")
    async def auth_check(request: Request):
        """Called by the reverse proxy before it forwards a request to the Realtime server."""
        return Response(status_code=204 if signed_in(request) and same_origin(request) else 401)

    @app.get("/login")
    async def login_form(request: Request):
        return RedirectResponse("/", 303) if signed_in(request) else login_page()

    @app.post("/login")
    async def login(request: Request):
        client = request.client.host if request.client else "unknown"
        if admin.locked(client):
            return login_page("Too many attempts. Try again in 15 minutes.", 429)
        raw = b""
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > MAX_BODY_BYTES:
                return login_page("The request is too large.", 413)
        form = parse_qs(raw.decode("utf-8", "replace"))
        # Count the attempt before the slow hash, so parallel requests cannot slip past the limit.
        admin.record_failure(client)
        if not await run_in_threadpool(admin.check, form.get("email", [""])[0], form.get("password", [""])[0]):
            return login_page("The email or password is wrong.", 401)
        admin.clear_failures(client)
        response = RedirectResponse("/", 303)
        response.set_cookie(
            COOKIE_NAME,
            admin.issue(),
            max_age=SESSION_SECONDS,
            httponly=True,
            secure=is_https(request),
            samesite="lax",
            path="/",
        )
        return response

    @app.post("/logout")
    async def logout():
        response = RedirectResponse("/login", 303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response


if __name__ == "__main__":
    if sys.argv[1:] != ["hash-password"]:
        sys.exit("usage: python admin_auth.py hash-password   (reads the password from the terminal or stdin)")
    if sys.stdin.isatty():
        import getpass

        entered = getpass.getpass("New admin password: ")
        if entered != getpass.getpass("Repeat it: "):
            sys.exit("The passwords differ.")
    else:
        entered = sys.stdin.readline().rstrip("\n")
    if not entered:
        sys.exit("The password is empty.")
    print(hash_password(entered))
