import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
admin_auth = importlib.import_module("admin_auth")

EMAIL, PASSWORD, SECRET = "admin@example.com", "correct horse", "s" * 32
HASH = admin_auth.hash_password(PASSWORD)


@pytest.fixture
def admin():
    return admin_auth.AdminAuth(EMAIL, HASH, SECRET)


@pytest.fixture
def client(admin):
    app = FastAPI()
    admin_auth.attach(app, admin)

    @app.get("/")
    def index():
        return {"page": "index"}

    @app.post("/api/search")
    def search():
        return {"ok": True}

    return TestClient(app, base_url="https://voice.example.com", follow_redirects=False)


def _login(client, email=EMAIL, password=PASSWORD):
    return client.post("/login", data={"email": email, "password": password})


def test_password_hash_round_trip():
    assert admin_auth.verify_password(PASSWORD, HASH)
    assert not admin_auth.verify_password("wrong", HASH)
    assert PASSWORD not in HASH and "$" not in HASH


def test_everything_needs_a_session_except_login_and_health(client):
    assert client.get("/").status_code == 401
    assert client.post("/api/search").status_code == 401
    assert client.get("/auth/check").status_code == 401
    assert client.get("/docs").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/login").status_code == 200
    assert client.get("/healthz").status_code == 200
    redirect = client.get("/", headers={"accept": "text/html"})
    assert (redirect.status_code, redirect.headers["location"]) == (303, "/login")


def test_login_gives_a_protected_cookie_and_access(client):
    response = _login(client, email=" Admin@Example.com ")
    assert response.status_code == 303
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie
    assert client.get("/").status_code == 200
    assert client.get("/auth/check").status_code == 204


@pytest.mark.parametrize(("email", "password"), [(EMAIL, "wrong"), ("other@example.com", PASSWORD), ("", "")])
def test_wrong_credentials_are_refused(client, email, password):
    response = _login(client, email, password)
    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_login_is_rate_limited_even_for_the_right_password(client):
    for _ in range(admin_auth.MAX_FAILURES):
        assert _login(client, password="wrong").status_code == 401
    assert _login(client).status_code == 429


def test_rate_limit_is_per_client_expires_and_has_a_global_cap(admin):
    for _ in range(admin_auth.MAX_FAILURES):
        admin.record_failure("a", now=0)
    assert admin.locked("a", now=1) and not admin.locked("b", now=1)
    assert not admin.locked("a", now=admin_auth.FAILURE_WINDOW_SECONDS + 1)
    for n in range(admin_auth.MAX_GLOBAL_FAILURES):
        admin.record_failure(f"ip{n}", now=2000)
    assert admin.locked("fresh", now=2001)


def test_tokens_expire_and_cannot_be_forged(admin):
    token = admin.issue(now=0)
    assert admin.valid(token, now=1)
    assert not admin.valid(token, now=admin_auth.SESSION_SECONDS + 1)
    payload, signature = token.split(".")
    for bad in (None, "", ".", payload, f"{payload}.", f"{payload}.0{signature[1:]}x", f"x{token}", "é.é"):
        assert not admin.valid(bad, now=1)
    other = admin_auth.AdminAuth(EMAIL, admin_auth.hash_password("new password"), SECRET)
    assert not other.valid(token, now=1)  # Changing the password signs everyone out.


def test_cross_site_requests_are_refused(client):
    _login(client)
    evil = {"origin": "https://evil.example"}
    assert client.get("/auth/check", headers=evil).status_code == 401
    assert client.post("/api/search", headers=evil).status_code == 403
    assert client.post("/api/search", headers={"origin": "https://voice.example.com"}).status_code == 200


def test_logout_removes_the_cookie(client):
    _login(client)
    assert client.post("/logout").status_code == 303
    assert client.get("/auth/check").status_code == 401


def test_large_login_bodies_are_refused(client):
    assert _login(client, password="x" * 10000).status_code == 413


def test_partial_or_invalid_configuration_fails_closed():
    assert admin_auth.AdminAuth.from_env({}) is None
    with pytest.raises(admin_auth.ConfigError):
        admin_auth.AdminAuth.from_env({"ADMIN_EMAIL": EMAIL})
    with pytest.raises(admin_auth.ConfigError):
        admin_auth.AdminAuth(EMAIL, "plain password", SECRET)
    with pytest.raises(admin_auth.ConfigError):
        admin_auth.AdminAuth(EMAIL, HASH, "short")
