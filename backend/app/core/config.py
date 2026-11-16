"""Application settings, read from environment variables (and an optional `.env` file for local development).

Every setting has a safe development default except the ones that must be secret in production: in production the
application refuses to start without a strong JWT secret and an explicit database URL.
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
VERSION = "11.0.0"

DISCLAIMER = ("This platform is intended for academic research and software experimentation only. It has not been "
              "clinically validated and must not be used for diagnosis, treatment decisions or emergency medical "
              "decisions. Model outputs are experimental predictions and require independent clinical validation.")

log = logging.getLogger("lti.config")


def load_dotenv(path: Path) -> None:
    """Minimal `.env` reader: KEY=VALUE lines, `#` comments, optional quotes. Never overrides real environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _bool(env: dict, key: str, default: bool) -> bool:
    v = env.get(key)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(env: dict, key: str, default: int) -> int:
    v = env.get(key)
    return int(v) if v not in (None, "") else default


def _list(env: dict, key: str, default: list[str]) -> list[str]:
    v = env.get(key)
    if v is None or v.strip() == "":
        return list(default)
    return [p.strip() for p in v.split(",") if p.strip()]


def parse_rate(spec: str) -> tuple[int, int]:
    """'10/minute' -> (10, 60). Units: second, minute, hour, day."""
    n, unit = spec.split("/", 1)
    seconds = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit.strip().rstrip("s")]
    return int(n), seconds


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    app_name: str = "Liver Tumour Intelligence Platform"
    version: str = VERSION
    project_root: Path = PROJECT_ROOT
    database_url: str = ""
    auto_migrate: bool = True
    jwt_secret: str = ""
    jwt_secret_generated: bool = False
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "lti-v10"
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    password_min_length: int = 10
    max_failed_logins: int = 5
    lockout_minutes: int = 15
    allow_registration: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    storage_backend: str = "local"
    storage_dir: Path = BACKEND_DIR / "data" / "storage"
    s3_bucket: str = ""
    s3_endpoint_url: str = ""
    s3_region: str = ""
    s3_prefix: str = "lti/"
    max_upload_mb: int = 20
    max_dataset_upload_mb: int = 500
    retention_days: int = 30
    store_original_filenames: bool = True
    worker_mode: str = "inline"          # inline: jobs run in the API process; external: separate worker process
    worker_threads: int = 2
    worker_poll_seconds: float = 1.0
    rate_limit_auth: str = "10/minute"
    rate_limit_refresh: str = "60/minute"   # V11: token refresh has its own bucket (every page load refreshes)
    rate_limit_predict: str = "30/minute"
    rate_limit_default: str = "300/minute"
    allow_synthetic_models: bool = False
    preload_models: bool = True
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"
    frontend_base_url: str = "http://localhost:5173"
    mail_backend: str = "disabled"       # disabled | smtp | file
    mail_from: str = "no-reply@localhost"
    mail_file_dir: Path = BACKEND_DIR / "data" / "outbox"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    log_level: str = "INFO"
    log_json: bool = True
    hsts: bool = False
    argon2_memory_kib: int = 19456
    argon2_iterations: int = 2
    sentry_dsn: str = ""
    dataset_roots: list[Path] = field(default_factory=list)
    compare_max_concurrent_runs: int = 2      # comparison runs executing at once (each starts 4 family workers)
    compare_model_workers_per_family: int = 1  # models at once inside one family pipeline
    compare_run_timeout_s: int = 300          # models not started by then are reported as timeout

    # ------------------------------------------------------------------ derived values
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_dataset_upload_bytes(self) -> int:
        return self.max_dataset_upload_mb * 1024 * 1024

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        if env is None:
            # backend/.env = local development settings (written by setup.sh); the root .env belongs to
            # docker-compose and is read only when there is no backend/.env (never both: their paths differ)
            local_env = BACKEND_DIR / ".env"
            load_dotenv(local_env if local_env.is_file() else PROJECT_ROOT / ".env")
            env = dict(os.environ)
        environment = env.get("ENVIRONMENT", "development").strip().lower()
        if environment not in ("development", "test", "production"):
            raise RuntimeError(f"ENVIRONMENT must be development, test or production, not {environment!r}")
        secret = env.get("JWT_SECRET", "")
        generated = False
        if not secret:
            if environment == "production":
                raise RuntimeError("JWT_SECRET must be set in production (at least 32 random characters)")
            secret, generated = secrets.token_urlsafe(48), True
        if environment == "production" and len(secret) < 32:
            raise RuntimeError("JWT_SECRET is too short for production (minimum 32 characters)")
        db = env.get("DATABASE_URL", "")
        if not db:
            if environment == "production":
                raise RuntimeError("DATABASE_URL must be set in production (PostgreSQL recommended)")
            db = f"sqlite:///{(BACKEND_DIR / 'data' / 'lti.db').as_posix()}"
        storage_dir = Path(env.get("STORAGE_DIR") or (BACKEND_DIR / "data" / "storage"))
        s = cls(
            environment=environment,
            database_url=db,
            auto_migrate=_bool(env, "AUTO_MIGRATE", environment != "production"),
            jwt_secret=secret,
            jwt_secret_generated=generated,
            access_token_minutes=_int(env, "ACCESS_TOKEN_MINUTES", 15),
            refresh_token_days=_int(env, "REFRESH_TOKEN_DAYS", 7),
            password_min_length=_int(env, "PASSWORD_MIN_LENGTH", 10),
            max_failed_logins=_int(env, "MAX_FAILED_LOGINS", 5),
            lockout_minutes=_int(env, "LOCKOUT_MINUTES", 15),
            allow_registration=_bool(env, "ALLOW_REGISTRATION", True),
            cors_origins=_list(env, "CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"]),
            storage_backend=env.get("STORAGE_BACKEND", "local"),
            storage_dir=storage_dir,
            s3_bucket=env.get("S3_BUCKET", ""),
            s3_endpoint_url=env.get("S3_ENDPOINT_URL", ""),
            s3_region=env.get("S3_REGION", ""),
            s3_prefix=env.get("S3_PREFIX", "lti/"),
            max_upload_mb=_int(env, "MAX_UPLOAD_MB", 20),
            max_dataset_upload_mb=_int(env, "MAX_DATASET_UPLOAD_MB", 500),
            retention_days=_int(env, "RETENTION_DAYS", 30),
            store_original_filenames=_bool(env, "STORE_ORIGINAL_FILENAMES", True),
            worker_mode=env.get("WORKER_MODE", "inline"),
            worker_threads=_int(env, "WORKER_THREADS", 2),
            worker_poll_seconds=float(env.get("WORKER_POLL_SECONDS", "1.0")),
            rate_limit_auth=env.get("RATE_LIMIT_AUTH", "10/minute"),
            rate_limit_refresh=env.get("RATE_LIMIT_REFRESH", "60/minute"),
            rate_limit_predict=env.get("RATE_LIMIT_PREDICT", "30/minute"),
            rate_limit_default=env.get("RATE_LIMIT_DEFAULT", "300/minute"),
            allow_synthetic_models=_bool(env, "ALLOW_SYNTHETIC_MODELS", False),
            preload_models=_bool(env, "PRELOAD_MODELS", True),
            frontend_dist=Path(env.get("FRONTEND_DIST") or (PROJECT_ROOT / "frontend" / "dist")),
            frontend_base_url=env.get("FRONTEND_BASE_URL", "http://localhost:5173"),
            mail_backend=env.get("MAIL_BACKEND", "disabled"),
            mail_from=env.get("MAIL_FROM", "no-reply@localhost"),
            mail_file_dir=Path(env.get("MAIL_FILE_DIR") or (BACKEND_DIR / "data" / "outbox")),
            smtp_host=env.get("SMTP_HOST", ""),
            smtp_port=_int(env, "SMTP_PORT", 587),
            smtp_user=env.get("SMTP_USER", ""),
            smtp_password=env.get("SMTP_PASSWORD", ""),
            smtp_starttls=_bool(env, "SMTP_STARTTLS", True),
            log_level=env.get("LOG_LEVEL", "INFO"),
            log_json=_bool(env, "LOG_JSON", True),
            hsts=_bool(env, "HSTS", environment == "production"),
            argon2_memory_kib=_int(env, "ARGON2_MEMORY_KIB", 19456),
            argon2_iterations=_int(env, "ARGON2_ITERATIONS", 2),
            sentry_dsn=env.get("SENTRY_DSN", ""),
            dataset_roots=[Path(p).expanduser() for p in _list(env, "DATASET_ROOTS", [str(PROJECT_ROOT / "dataset")])],
            compare_max_concurrent_runs=_int(env, "COMPARE_MAX_CONCURRENT_RUNS", 2),
            compare_model_workers_per_family=_int(env, "COMPARE_MODEL_WORKERS_PER_FAMILY", 1),
            compare_run_timeout_s=_int(env, "COMPARE_RUN_TIMEOUT_S", 300),
        )
        if s.worker_mode not in ("inline", "external"):
            raise RuntimeError("WORKER_MODE must be inline or external")
        if s.storage_backend not in ("local", "s3"):
            raise RuntimeError("STORAGE_BACKEND must be local or s3")
        if s.mail_backend not in ("disabled", "smtp", "file"):
            raise RuntimeError("MAIL_BACKEND must be disabled, smtp or file")
        if s.is_production and s.allow_synthetic_models:
            raise RuntimeError("ALLOW_SYNTHETIC_MODELS must not be enabled in production")
        return s
