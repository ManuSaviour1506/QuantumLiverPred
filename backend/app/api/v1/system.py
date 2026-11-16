"""Health checks, API status and application information."""
from __future__ import annotations

import importlib.util

from app.core.apikit import APIRouter, Ctx
from app.core.config import DISCLAIMER
from app.schemas.models import SystemInfoOut
from app.services import mail

router = APIRouter("", tags=["system"])

OPTIONAL = ["torch", "torchvision", "monai", "pennylane", "qiskit", "xgboost", "pydicom", "nibabel", "boto3",
            "psycopg", "sklearn", "scipy", "skimage"]


@router.get("/health", auth=False)
def health(ctx: Ctx):
    """Liveness probe: the process is up. Does not touch the database."""
    return {"status": "ok", "version": ctx.settings.version}


@router.get("/health/ready", auth=False)
def ready(ctx: Ctx):
    """Readiness probe: database reachable and migrations applied; model engine state reported."""
    from starlette.responses import Response
    from app.core.apikit import JSONResponse
    checks = {}
    try:
        with ctx.db.session() as c:
            checks["database"] = "ok"
            checks["migrations"] = c.scalar("SELECT COUNT(*) FROM schema_migrations")
    except Exception as e:
        checks["database"] = f"error: {type(e).__name__}"
    eng = ctx.state.engine.status()
    checks["engine"] = eng["state"]
    ok = checks["database"] == "ok" and eng["state"] != "failed"
    body = {"status": "ready" if ok else "not_ready", "checks": checks}
    return JSONResponse(body, status_code=200 if ok else 503)


def _optional_packages() -> dict:
    return {name: importlib.util.find_spec(name) is not None for name in OPTIONAL}


def execution_modes(pk: dict) -> list[dict]:
    return [
        {"id": "custom_numpy_simulator", "label": "Custom NumPy state-vector simulator", "available": True,
         "hardware": False},
        {"id": "pennylane_simulator", "label": "PennyLane simulator", "available": pk.get("pennylane", False),
         "hardware": False},
        {"id": "qiskit_simulator", "label": "Qiskit simulator", "available": pk.get("qiskit", False),
         "hardware": False},
        {"id": "quantum_hardware", "label": "Quantum hardware", "available": False, "hardware": True,
         "note": "no hardware provider is configured; no result in this platform comes from quantum hardware"},
    ]


@router.get("/api/v1/system/info", auth=False, response=SystemInfoOut)
def info(ctx: Ctx):
    """Version, configuration summary (no secrets), optional packages and quantum execution modes."""
    s = ctx.settings
    pk = _optional_packages()
    return {"app": s.app_name, "version": s.version, "environment": s.environment, "disclaimer": DISCLAIMER,
            "database": {"dialect": ctx.db.dialect},
            "storage": ctx.state.storage.describe(), "worker_mode": s.worker_mode,
            "engine": ctx.state.engine.status(), "optional_packages": pk, "execution_modes": execution_modes(pk),
            "registration_open": s.allow_registration, "password_reset_available": mail.available(s),
            "limits": {"max_upload_mb": s.max_upload_mb, "max_dataset_upload_mb": s.max_dataset_upload_mb,
                       "retention_days": s.retention_days,
                       "image_formats": ["PNG", "JPEG", "DICOM (uncompressed)", "NIfTI (.nii, .nii.gz)"]}}
