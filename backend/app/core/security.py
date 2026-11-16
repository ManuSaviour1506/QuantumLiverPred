"""Password hashing (Argon2id), JWT access tokens and opaque refresh / reset tokens.

Passwords: Argon2id via `cryptography` (OWASP minimum parameters by default: m = 19 MiB, t = 2, p = 1), stored as a
PHC string `$argon2id$v=19$m=...,t=...,p=...$<salt>$<hash>`. Verification is constant time.

Access tokens: short-lived HS256 JWTs (`type=access`, `sub` = user id, `role`, `jti`). Logout puts the `jti` on a deny
list until it expires.

Refresh tokens and password-reset tokens: random 384-bit opaque strings. Only their SHA-256 is stored, so a database
leak does not reveal usable tokens. Refresh tokens rotate on every use; presenting a revoked one revokes the family.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
import secrets
import uuid

import jwt
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

from app.core.errors import unauthorized


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s + "=" * (-len(s) % 4))


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime | None) -> str | None:
    return t.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z") if t else None


def parse_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex


class PasswordHasher:
    def __init__(self, memory_kib: int = 19456, iterations: int = 2, lanes: int = 1):
        self.m, self.t, self.p = memory_kib, iterations, lanes

    def _derive(self, password: str, salt: bytes, m: int, t: int, p: int) -> bytes:
        return Argon2id(salt=salt, length=32, iterations=t, lanes=p, memory_cost=m).derive(password.encode("utf-8"))

    def hash(self, password: str) -> str:
        salt = os.urandom(16)
        digest = self._derive(password, salt, self.m, self.t, self.p)
        return f"$argon2id$v=19$m={self.m},t={self.t},p={self.p}${_b64(salt)}${_b64(digest)}"

    def verify(self, password: str, encoded: str) -> bool:
        try:
            _, algo, version, params, salt, digest = encoded.split("$")
            if algo != "argon2id" or version != "v=19":
                return False
            kv = dict(item.split("=") for item in params.split(","))
            got = self._derive(password, _unb64(salt), int(kv["m"]), int(kv["t"]), int(kv["p"]))
            return hmac.compare_digest(got, _unb64(digest))
        except (ValueError, KeyError):
            return False

    def needs_rehash(self, encoded: str) -> bool:
        return f"m={self.m},t={self.t},p={self.p}" not in encoded


# A fixed hash used to spend the same time when an e-mail address does not exist (prevents user enumeration by timing).
_DUMMY_PASSWORD = "not-a-real-password-" + secrets.token_hex(8)


def dummy_verify(hasher: PasswordHasher) -> None:
    hasher.verify(_DUMMY_PASSWORD, hasher.hash(_DUMMY_PASSWORD))


def password_problems(password: str, min_length: int) -> list[str]:
    problems = []
    if len(password) < min_length:
        problems.append(f"must be at least {min_length} characters")
    if len(password) > 256:
        problems.append("must be at most 256 characters")
    classes = sum(bool(any(f(c) for c in password)) for f in (str.islower, str.isupper, str.isdigit))
    if classes < 2:
        problems.append("must mix at least two of: lower-case letters, upper-case letters, digits")
    if password.lower() in {"password1234", "1234567890", "qwertyuiop", "liverliver1"}:
        problems.append("is too common")
    return problems


def create_access_token(user_id: str, role: str, secret: str, minutes: int, issuer: str,
                        algorithm: str = "HS256") -> tuple[str, dt.datetime, str]:
    now = utcnow()
    exp = now + dt.timedelta(minutes=minutes)
    jti = uuid.uuid4().hex
    claims = {"sub": user_id, "role": role, "type": "access", "iat": int(now.timestamp()),
              "exp": int(exp.timestamp()), "jti": jti, "iss": issuer}
    return jwt.encode(claims, secret, algorithm=algorithm), exp, jti


def decode_access_token(token: str, secret: str, issuer: str, algorithm: str = "HS256") -> dict:
    try:
        claims = jwt.decode(token, secret, algorithms=[algorithm], issuer=issuer,
                            options={"require": ["exp", "iat", "sub", "jti", "iss"]})
    except jwt.ExpiredSignatureError:
        raise unauthorized("Access token expired")
    except jwt.InvalidTokenError:
        raise unauthorized("Invalid access token")
    if claims.get("type") != "access":
        raise unauthorized("Invalid token type")
    return claims


def new_opaque_token() -> tuple[str, str]:
    """Return (token for the client, SHA-256 hex to store)."""
    token = secrets.token_urlsafe(48)
    return token, token_hash(token)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_ip(ip: str | None, secret: str) -> str | None:
    if not ip:
        return None
    return hmac.new(secret.encode(), ip.encode(), hashlib.sha256).hexdigest()[:16]
