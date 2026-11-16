"""Backend test fixtures: a real uvicorn server on a free port with a fresh database and storage.

These are integration tests over real HTTP (no mocked responses). Environment variables:
  LTI_TEST_DATABASE_URL   database to test against (default: a fresh SQLite file). For PostgreSQL use e.g.
                          postgresql://postgres@localhost:5432/lti_test (psycopg installed, or LTI_TEST_PGWIRE=1)
  LTI_TEST_DATASET_DIR    the verified LiTS dataset folder (enables tests with real CT slices); default: the
                          V10 sample folder dataset/sample_data/lits_sample if present
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

sys.dont_write_bytecode = True
BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def dataset_dir() -> Path | None:
    env = os.environ.get("LTI_TEST_DATASET_DIR")
    for cand in ([Path(env)] if env else []) + [ROOT / "dataset" / "sample_data" / "lits_sample"]:
        if cand and (cand / "labels" / "slices.csv").exists():
            return cand
    return None


def _reset_db(url: str) -> None:
    if not url.startswith("postgres"):
        return
    from app.db.database import Database
    db = Database(url)
    with db.session() as c:
        for t in ("benchmark_runs", "comparison_model_results", "comparison_runs", "comparison_uploads", "audit_logs", "jobs", "prediction_artifacts", "predictions", "experiments", "training_runs",
                  "model_versions", "models", "dataset_files", "dataset_versions", "datasets", "password_reset_tokens",
                  "token_denylist", "refresh_tokens", "users", "roles", "schema_migrations"):
            c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")


@pytest.fixture(scope="session")
def live(tmp_path_factory):
    import uvicorn
    from app.core.config import Settings
    from app.main import create_app
    from src.v10.data.synthetic import train_synthetic_model
    tmp = tmp_path_factory.mktemp("lti")
    # Test project root: real V9 artifacts and results (read-only, via symlinks) + a private artifacts/v10 holding
    # the real V10 models (symlinked) and the synthetic software-test model.
    proj = tmp / "project"
    (proj / "artifacts" / "v10").mkdir(parents=True)
    for name in ("classical", "quantum", "hybrid", "ood", "segmentation", "weights", "manifests"):
        if (ROOT / "artifacts" / name).exists():
            (proj / "artifacts" / name).symlink_to(ROOT / "artifacts" / name)
    (proj / "results").symlink_to(ROOT / "results")
    real_v10 = ROOT / "artifacts" / "v10"
    if real_v10.exists():
        for d in real_v10.iterdir():
            if d.is_dir():
                (proj / "artifacts" / "v10" / d.name).symlink_to(d)
    train_synthetic_model(proj / "artifacts" / "v10")
    url = os.environ.get("LTI_TEST_DATABASE_URL") or f"sqlite:///{tmp / 'test.db'}"
    _reset_db(url)
    roots = [str(tmp)]
    ds = dataset_dir()
    if ds:
        roots.append(str(ds.parent))
    env = {"ENVIRONMENT": "test", "DATABASE_URL": url, "JWT_SECRET": "test-secret-" + "x" * 40,
           "STORAGE_DIR": str(tmp / "storage"), "ALLOW_SYNTHETIC_MODELS": "true", "PRELOAD_MODELS": "true",
           "ARGON2_MEMORY_KIB": "1024", "ARGON2_ITERATIONS": "1", "RATE_LIMIT_AUTH": "1000/minute", "RATE_LIMIT_REFRESH": "1000/minute",
           "RATE_LIMIT_DEFAULT": "5000/minute", "RATE_LIMIT_PREDICT": "1000/minute", "MAIL_BACKEND": "file",
           "MAIL_FILE_DIR": str(tmp / "outbox"), "WORKER_MODE": "inline", "WORKER_POLL_SECONDS": "0.2",
           "WORKER_THREADS": "2", "MAX_UPLOAD_MB": "5", "MAX_DATASET_UPLOAD_MB": "50", "LOG_JSON": "false",
           "LOG_LEVEL": "WARNING", "DATASET_ROOTS": ",".join(roots), "FRONTEND_DIST": str(tmp / "no_frontend"),
           "RETENTION_DAYS": "30"}
    settings = Settings.from_env(env).with_overrides(project_root=proj)
    app = create_app(settings)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if requests.get(base + "/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    # wait for the model engine (V9 registry + V10 artifacts) to finish loading
    for _ in range(600):
        st = app.state.engine.status()
        if st["state"] in ("ready", "failed"):
            break
        time.sleep(0.1)
    yield {"url": base, "app": app, "settings": settings, "tmp": tmp, "dataset": ds}
    server.should_exit = True
    th.join(10)


class Client:
    def __init__(self, base: str):
        self.base, self.token, self.refresh = base, None, None
        self.s = requests.Session()

    def req(self, method: str, path: str, **kw):
        h = kw.pop("headers", {})
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return self.s.request(method, self.base + path, headers=h, timeout=120, **kw)

    def get(self, path, **kw):
        return self.req("GET", path, **kw)

    def post(self, path, **kw):
        return self.req("POST", path, **kw)

    def patch(self, path, **kw):
        return self.req("PATCH", path, **kw)

    def delete(self, path, **kw):
        return self.req("DELETE", path, **kw)


PASSWORD = "Correct-Horse-42"


def make_user(live, email: str, role: str = "researcher") -> Client:
    c = Client(live["url"])
    r = c.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD, "full_name": "Test User"})
    assert r.status_code in (201, 409), r.text
    if role != "researcher":
        st = live["app"].state
        with st.db.session() as conn:
            conn.run("UPDATE users SET role_id = ? WHERE email = ?", (f"role_{role}", email))
    r = c.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    c.token, c.refresh = r.json()["access_token"], r.json()["refresh_token"]
    return c


@pytest.fixture(scope="session")
def researcher(live):
    return make_user(live, "researcher@example.org")


@pytest.fixture(scope="session")
def admin(live):
    return make_user(live, "admin@example.org", "admin")


@pytest.fixture(scope="session")
def viewer(live):
    return make_user(live, "viewer@example.org", "viewer")


def wait_for(fn, timeout: float = 120, every: float = 0.3):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = fn()
        if last is not None:
            return last
        time.sleep(every)
    raise AssertionError(f"timed out; last value {last!r}")


def poll_prediction(client: Client, pid: str, timeout: float = 180) -> dict:
    def check():
        r = client.get(f"/api/v1/predictions/{pid}")
        assert r.status_code == 200, r.text
        j = r.json()
        return j if j["status"] in ("succeeded", "failed") else None
    return wait_for(check, timeout)
