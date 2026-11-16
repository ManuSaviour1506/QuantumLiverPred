"""Authentication: registration rules, login, lock-out, refresh rotation and reuse detection, logout, password
change and reset by e-mail (file mail backend)."""
import re
from pathlib import Path

from conftest import PASSWORD, Client


def test_register_validation(live):
    c = Client(live["url"])
    r = c.post("/api/v1/auth/register", json={"email": "not-an-email", "password": PASSWORD, "full_name": "X"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = c.post("/api/v1/auth/register", json={"email": "weak@example.org", "password": "short", "full_name": "X"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "weak_password"
    r = c.post("/api/v1/auth/register", json={"email": "Dup@Example.org", "password": PASSWORD, "full_name": "D"})
    assert r.status_code == 201 and r.json()["email"] == "dup@example.org" and r.json()["role"] == "researcher"
    assert "password" not in r.text.lower().replace("password_reset", "")
    r = c.post("/api/v1/auth/register", json={"email": "dup@example.org", "password": PASSWORD, "full_name": "D"})
    assert r.status_code == 409


def test_login_me_refresh_logout(live):
    c = Client(live["url"])
    c.post("/api/v1/auth/register", json={"email": "flow@example.org", "password": PASSWORD, "full_name": "F"})
    assert c.post("/api/v1/auth/login", json={"email": "flow@example.org", "password": "Wrong-pass-123"}).status_code == 401
    assert c.post("/api/v1/auth/login", json={"email": "nobody@example.org", "password": PASSWORD}).status_code == 401
    r = c.post("/api/v1/auth/login", json={"email": "flow@example.org", "password": PASSWORD})
    assert r.status_code == 200
    tok = r.json()
    assert tok["token_type"] == "bearer" and tok["expires_in"] > 0 and tok["user"]["email"] == "flow@example.org"
    c.token = tok["access_token"]
    assert c.get("/api/v1/auth/me").json()["email"] == "flow@example.org"
    # refresh rotates; re-using the old refresh token revokes the family
    r2 = c.post("/api/v1/auth/refresh", json={"refresh_token": tok["refresh_token"]})
    assert r2.status_code == 200
    new = r2.json()
    assert new["refresh_token"] != tok["refresh_token"]
    reuse = c.post("/api/v1/auth/refresh", json={"refresh_token": tok["refresh_token"]})
    assert reuse.status_code == 401
    assert c.post("/api/v1/auth/refresh", json={"refresh_token": new["refresh_token"]}).status_code == 401
    # logout revokes the access token immediately
    r3 = c.post("/api/v1/auth/login", json={"email": "flow@example.org", "password": PASSWORD}).json()
    c.token = r3["access_token"]
    assert c.post("/api/v1/auth/logout", json={"refresh_token": r3["refresh_token"]}).status_code == 200
    assert c.get("/api/v1/auth/me").status_code == 401
    assert c.post("/api/v1/auth/refresh", json={"refresh_token": r3["refresh_token"]}).status_code == 401


def test_invalid_tokens(live):
    c = Client(live["url"])
    c.token = "not.a.jwt"
    assert c.get("/api/v1/auth/me").status_code == 401
    import jwt
    c.token = jwt.encode({"sub": "x", "exp": 1, "iat": 0, "jti": "j", "iss": "lti-v10", "type": "access"},
                         live["settings"].jwt_secret, algorithm="HS256")
    r = c.get("/api/v1/auth/me")
    assert r.status_code == 401 and "expired" in r.json()["error"]["message"].lower()
    c.token = jwt.encode({"sub": "x", "exp": 9999999999, "iat": 0, "jti": "j", "iss": "lti-v10", "type": "access"},
                         "another-secret-that-is-long-enough-123456", algorithm="HS256")
    assert c.get("/api/v1/auth/me").status_code == 401


def test_lockout(live):
    c = Client(live["url"])
    c.post("/api/v1/auth/register", json={"email": "lock@example.org", "password": PASSWORD, "full_name": "L"})
    for _ in range(live["settings"].max_failed_logins):
        assert c.post("/api/v1/auth/login", json={"email": "lock@example.org", "password": "Wrong-123456"}).status_code == 401
    r = c.post("/api/v1/auth/login", json={"email": "lock@example.org", "password": PASSWORD})
    assert r.status_code == 423 and r.json()["error"]["code"] == "account_locked"


def test_change_and_reset_password(live):
    c = Client(live["url"])
    c.post("/api/v1/auth/register", json={"email": "reset@example.org", "password": PASSWORD, "full_name": "R"})
    c.token = c.post("/api/v1/auth/login", json={"email": "reset@example.org", "password": PASSWORD}).json()["access_token"]
    assert c.post("/api/v1/users/me/password", json={"current_password": "wrong", "new_password": "New-Pass-4567"}).status_code == 400
    assert c.post("/api/v1/users/me/password", json={"current_password": PASSWORD, "new_password": "New-Pass-4567"}).status_code == 200
    c.token = None
    assert c.post("/api/v1/auth/login", json={"email": "reset@example.org", "password": "New-Pass-4567"}).status_code == 200
    outbox = Path(live["settings"].mail_file_dir)
    before = set(outbox.glob("*.eml")) if outbox.exists() else set()
    r = c.post("/api/v1/auth/forgot-password", json={"email": "reset@example.org"})
    assert r.status_code == 200
    assert c.post("/api/v1/auth/forgot-password", json={"email": "unknown@example.org"}).json() == r.json()
    new = [p for p in outbox.glob("*.eml") if p not in before]
    assert len(new) == 1
    import email
    from email import policy
    body = email.message_from_bytes(new[0].read_bytes(), policy=policy.default).get_content()
    token = re.search(r"token=([A-Za-z0-9_\-]+)", body).group(1)
    assert c.post("/api/v1/auth/reset-password", json={"token": token, "new_password": "weak"}).status_code == 422
    assert c.post("/api/v1/auth/reset-password", json={"token": token, "new_password": "Reset-Pass-789"}).status_code == 200
    assert c.post("/api/v1/auth/reset-password", json={"token": token, "new_password": "Reset-Pass-790"}).status_code == 400
    assert c.post("/api/v1/auth/login", json={"email": "reset@example.org", "password": "Reset-Pass-789"}).status_code == 200


def test_profile_and_admin(live, researcher, admin):
    r = researcher.patch("/api/v1/users/me", json={"full_name": "Dr Test", "preferences": {"theme": "dark"}})
    assert r.status_code == 200 and r.json()["preferences"]["theme"] == "dark"
    assert researcher.patch("/api/v1/users/me", json={"preferences": {"evil": 1}}).status_code == 422
    assert researcher.get("/api/v1/users").status_code == 403
    page = admin.get("/api/v1/users?page_size=100").json()
    assert page["meta"]["total"] >= 2
    me = admin.get("/api/v1/auth/me").json()
    assert admin.patch(f"/api/v1/users/{me['id']}", json={"role": "viewer"}).status_code == 400
    other = next(u for u in page["items"] if u["email"] == "dup@example.org")
    r = admin.patch(f"/api/v1/users/{other['id']}", json={"is_active": False})
    assert r.status_code == 200 and r.json()["is_active"] is False
    c = Client(live["url"])
    assert c.post("/api/v1/auth/login", json={"email": "dup@example.org", "password": PASSWORD}).status_code == 403
