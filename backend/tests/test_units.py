"""Unit tests: migrations, password hashing, tokens, storage keys, archive and manifest safety, rate limiter."""
import io
import zipfile
from pathlib import Path

import pytest

from app.core import security
from app.core.config import Settings
from app.core.middleware import RateLimiter
from app.db import migrate
from app.db.database import Database, split_statements
from app.services.storage import LocalStorage, StorageError

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v10.data import ingest  # noqa: E402


def test_migrations_up_down_status(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'm.db'}")
    assert migrate.upgrade(db) == ["0001_initial", "0002_comparison_runs", "0003_benchmark_runs"]
    assert migrate.upgrade(db) == []
    st = migrate.status(db)
    assert all(m["applied"] and m["checksum_ok"] for m in st)
    assert migrate.downgrade(db) == ["0003_benchmark_runs"]       # latest migration only
    with db.session() as c:
        assert c.scalar("SELECT COUNT(*) FROM schema_migrations") == 2
    assert migrate.downgrade(db) == ["0002_comparison_runs"]
    assert migrate.downgrade(db) == ["0001_initial"]
    with db.session() as c:
        assert c.scalar("SELECT COUNT(*) FROM schema_migrations") == 0
    assert migrate.upgrade(db) == ["0001_initial", "0002_comparison_runs", "0003_benchmark_runs"]


def test_migration_checksum_protection(tmp_path, monkeypatch):
    db = Database(f"sqlite:///{tmp_path / 'c.db'}")
    migrate.upgrade(db)
    with db.session() as c:
        c.run("UPDATE schema_migrations SET checksum = 'tampered'")
    with pytest.raises(migrate.MigrationError):
        migrate.upgrade(db)


def test_split_statements():
    assert split_statements("-- c\nCREATE TABLE a (x INT);\n\nINSERT INTO a VALUES (1);\n") == \
        ["CREATE TABLE a (x INT)", "INSERT INTO a VALUES (1)"]


def test_password_hashing():
    h = security.PasswordHasher(1024, 1)
    enc = h.hash("Secret-Pass-1")
    assert enc.startswith("$argon2id$v=19$m=1024,t=1,p=1$")
    assert h.verify("Secret-Pass-1", enc) and not h.verify("secret-pass-1", enc)
    assert not h.verify("x", "garbage")
    assert security.PasswordHasher(2048, 1).needs_rehash(enc)
    assert security.password_problems("short", 10)
    assert not security.password_problems("Long-Enough-123", 10)


def test_access_tokens():
    tok, exp, jti = security.create_access_token("u1", "admin", "s" * 40, 5, "iss")
    claims = security.decode_access_token(tok, "s" * 40, "iss")
    assert claims["sub"] == "u1" and claims["role"] == "admin" and claims["jti"] == jti
    with pytest.raises(Exception):
        security.decode_access_token(tok, "t" * 40, "iss")
    with pytest.raises(Exception):
        security.decode_access_token(tok, "s" * 40, "other-issuer")


def test_storage_keys(tmp_path):
    s = LocalStorage(tmp_path)
    s.put_bytes("uploads/u1/a.png", b"x")
    assert s.get_bytes("uploads/u1/a.png") == b"x"
    for bad in ("../x", "/etc/passwd", "a/../../b", "a//b", "", "uploads/"):
        with pytest.raises(StorageError):
            s.path(bad)


def test_ingest_safety(tmp_path):
    for bad in ("../a.png", "/abs.png", "a/../../b.png", "~/x.png"):
        with pytest.raises(ingest.IngestError):
            ingest.safe_rel(bad)
    zp = tmp_path / "z.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("ok/a.txt", "1")
        z.writestr("../escape.txt", "2")
    with pytest.raises(ingest.IngestError):
        ingest.safe_extract(zp, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_rate_limiter():
    rl = RateLimiter()
    s = Settings.from_env({"ENVIRONMENT": "test", "RATE_LIMIT_AUTH": "3/minute"})
    for _ in range(3):
        rl.enforce("auth", "1.2.3.4", s)
    with pytest.raises(Exception) as e:
        rl.enforce("auth", "1.2.3.4", s)
    assert e.value.status == 429
    rl.enforce("auth", "5.6.7.8", s)


def test_token_refresh_has_its_own_rate_bucket():
    """V11: a page load refreshes the access token; refresh must not share the strict login budget (10/min per IP),
    otherwise a few reloads log the user out. Login stays strict, refresh gets its own (higher, configurable) limit."""
    from app.api.v1 import auth as auth_api
    rl = RateLimiter()
    s = Settings.from_env({"ENVIRONMENT": "test", "RATE_LIMIT_AUTH": "3/minute", "RATE_LIMIT_REFRESH": "5/minute"})
    for _ in range(3):
        rl.enforce("auth", "1.2.3.4", s)
    for _ in range(5):                      # login budget exhausted, refresh still allowed
        rl.enforce("refresh", "1.2.3.4", s)
    with pytest.raises(Exception) as e:
        rl.enforce("refresh", "1.2.3.4", s)
    assert e.value.status == 429
    eps = {(ep.method, ep.path): ep.rate for ep in auth_api.router.endpoints}
    assert eps[("POST", "/api/v1/auth/refresh")] == "refresh"
    assert eps[("POST", "/api/v1/auth/login")] == "auth"


def test_production_settings_require_secrets():
    with pytest.raises(RuntimeError):
        Settings.from_env({"ENVIRONMENT": "production", "DATABASE_URL": "postgresql://x@y/z"})
    with pytest.raises(RuntimeError):
        Settings.from_env({"ENVIRONMENT": "production", "JWT_SECRET": "x" * 40})
    with pytest.raises(RuntimeError):
        Settings.from_env({"ENVIRONMENT": "production", "JWT_SECRET": "x" * 40, "DATABASE_URL": "postgresql://a@b/c",
                           "ALLOW_SYNTHETIC_MODELS": "true"})
    s = Settings.from_env({"ENVIRONMENT": "production", "JWT_SECRET": "x" * 40, "DATABASE_URL": "postgresql://a@b/c"})
    assert s.is_production and not s.auto_migrate and s.hsts


def test_log_redaction():
    import json
    import logging
    from app.core.logging import JsonFormatter, TextFormatter
    rec = logging.LogRecord("lti.test", logging.INFO, __file__, 1, "event", None, None)
    rec.fields = {"password": "hunter2-secret", "Authorization": "Bearer abc.def", "email": "a@b.org",
                  "filename": "patient_name.png", "refresh_token": "tok", "model_id": "v10_vqc_radiomics"}
    out = json.loads(JsonFormatter().format(rec))
    for k in ("password", "Authorization", "email", "filename", "refresh_token"):
        assert out[k] == "[redacted]"
    assert out["model_id"] == "v10_vqc_radiomics"
    text = TextFormatter().format(rec)
    assert "hunter2" not in text and "abc.def" not in text and "a@b.org" not in text and "patient_name" not in text
