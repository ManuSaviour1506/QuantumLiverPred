"""Model registry service: exposes the engine's model descriptors and mirrors them into the `models` and
`model_versions` tables (so the registry, deployment status and metrics are queryable and auditable)."""
from __future__ import annotations

import logging

from app.core.errors import ApiError, not_found
from app.core.security import iso, new_id, utcnow
from app.db.database import loads

log = logging.getLogger("lti.models")


class ModelService:
    def __init__(self, settings, db, engine_holder):
        self.settings, self.db, self.holder = settings, db, engine_holder
        self._synced = False

    def engine(self):
        try:
            eng = self.holder.ensure()
        except Exception:
            raise ApiError(503, "engine_unavailable", "The model engine failed to load; see the API status page",
                           {"error": self.holder.status().get("error")})
        if not self._synced:
            try:
                self.sync()
            except Exception:
                log.error("registry sync failed", exc_info=True)
        return eng

    @staticmethod
    def _out(d: dict) -> dict:
        return {"id": d["id"], "name": d.get("name") or d["id"], "family": d.get("family") or "unknown",
                "architecture": d.get("architecture") or "", "task": d.get("task") or "",
                "description": d.get("architecture_detail") if isinstance(d.get("architecture_detail"), str) else "",
                "version": d.get("version") or "", "source": d.get("source") or "", "status": d.get("status"),
                "deployment_status": d.get("deployment_status") or "not_deployed", "servable": bool(d.get("servable")),
                "reason": d.get("reason"), "training_status": d.get("training_status"),
                "training_date": d.get("training_date"), "dataset_version": d.get("dataset_version"),
                "execution_mode": d.get("execution_mode"), "artifact_path": d.get("artifact_path"),
                "artifact_sha256": d.get("artifact_sha256"), "size_bytes": d.get("size_bytes"), "role": d.get("role"),
                "input_contract": d.get("input_contract") or {}, "metrics": d.get("metrics") or {},
                "availability": d.get("availability"), "compare_family": d.get("compare_family"),
                "dependencies": d.get("dependencies") or [],
                "metadata": {k: d.get(k) for k in ("architecture_detail", "threshold", "calibrated", "parameters",
                                                   "training_data", "circuit", "configuration", "preprocessing",
                                                   "quantum_backend", "seeds") if d.get(k) is not None}}

    def list(self, family: str | None = None, servable_only: bool = False) -> list[dict]:
        out = [self._out(d) for d in self.engine().list()]
        if family:
            out = [m for m in out if m["family"] == family]
        if servable_only:
            out = [m for m in out if m["servable"]]
        return out

    def availability(self) -> dict:
        eng = self.engine()
        try:
            last = self.comparison_service.last_success()
        except Exception:
            last = {}
        out = []
        for d in eng.list(include_legacy=True):
            m = self._out(d)
            ic = m["input_contract"]
            out.append({"id": m["id"], "name": m["name"], "family": m["compare_family"] or m["family"],
                        "registry_family": m["family"], "architecture": m["architecture"], "version": m["version"],
                        "source": m["source"], "role": m["role"], "availability": m["availability"],
                        "status": m["status"], "servable": m["servable"], "reason": m["reason"],
                        "training_dataset": m["dataset_version"], "training_status": m["training_status"],
                        "weights_location": m["artifact_path"], "artifact_sha256": m["artifact_sha256"],
                        "dependencies": m["dependencies"], "execution_mode": m["execution_mode"],
                        "input_requirements": {"formats": ic.get("formats"), "patch_px": ic.get("patch_px"),
                                               "context_slices_each_side": ic.get("context_slices_each_side", 0),
                                               "requires_context": bool(ic.get("requires_context")),
                                               "features": ic.get("features")},
                        "threshold": m["metadata"].get("threshold"), "last_successful_inference": last.get(m["id"])})
        from collections import Counter
        return {"models": out, "counts": dict(Counter(o["availability"] for o in out)),
                "weights": eng.weights, "categories": {
                    "available": "loads, passes integrity + self-test, all dependencies present",
                    "requires_context": "available, but needs neighbouring slices for every prediction",
                    "missing_dependency": "a runtime dependency (backbone weights / upstream model) is missing",
                    "missing_weights": "the model's own weight file is missing",
                    "legacy": "V8-era model kept for reproducibility; not offered",
                    "not_trained": "never trained in this environment",
                    "failed_integrity": "artifact present but failed integrity or self-test",
                    "non_deployable": "loads, but cannot make meaningful predictions"}}

    def get(self, model_id: str) -> dict:
        try:
            return self._out(self.engine().describe(model_id))
        except KeyError:
            raise not_found("Model")

    def metrics(self, model_id: str) -> dict:
        m = self.get(model_id)
        notes = []
        if m["source"] == "v9":
            notes.append("Recorded V9 results on V9's fixed split (patch level); see caveats.")
        if m["source"] == "v10":
            notes.append("V10 estimates come from nested leave-one-unit-out evaluation on reconstructed scan groups.")
        return {"id": m["id"], "version": m["version"], "evaluations": (m["metrics"] or {}).get("evaluations", []),
                "notes": notes}

    def status(self, model_id: str) -> dict:
        m = self.get(model_id)
        return {"id": m["id"], "status": m["status"], "deployment_status": m["deployment_status"],
                "servable": m["servable"], "reason": m["reason"], "engine_loaded": self.holder.engine is not None,
                "checks": {"artifact_sha256": m["artifact_sha256"], "training_status": m["training_status"],
                           "execution_mode": m["execution_mode"]}}

    def sync(self) -> dict:
        eng = self.holder.ensure()
        now = iso(utcnow())
        n = 0
        with self.db.session() as c:
            with c.transaction():
                for d in eng.list(include_legacy=True):
                    m = self._out(d)
                    c.run("INSERT INTO models (id, name, family, architecture, task, description, created_at, updated_at) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET name = excluded.name, "
                          "family = excluded.family, architecture = excluded.architecture, task = excluded.task, "
                          "updated_at = excluded.updated_at",
                          (m["id"], m["name"], m["family"], m["architecture"], m["task"], m["description"] or "",
                           now, now))
                    c.run("INSERT INTO model_versions (id, model_id, version, source, status, deployment_status, "
                          "training_status, training_date, dataset_version, execution_mode, artifact_path, "
                          "artifact_sha256, size_bytes, metrics, metadata, registered_at, updated_at) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                          "ON CONFLICT (model_id, version) DO UPDATE SET status = excluded.status, "
                          "deployment_status = excluded.deployment_status, training_status = excluded.training_status, "
                          "artifact_sha256 = excluded.artifact_sha256, size_bytes = excluded.size_bytes, "
                          "metrics = excluded.metrics, metadata = excluded.metadata, updated_at = excluded.updated_at",
                          (new_id("mv_"), m["id"], m["version"] or "unversioned", m["source"], m["status"],
                           m["deployment_status"], m["training_status"], m["training_date"], m["dataset_version"],
                           m["execution_mode"], m["artifact_path"], m["artifact_sha256"], m["size_bytes"],
                           m["metrics"], m["metadata"], now, now))
                    n += 1
        self._synced = True
        return {"synced": n}

    def db_versions(self) -> list[dict]:
        with self.db.session() as c:
            rows = c.all("SELECT model_id, version, source, status, deployment_status, registered_at, updated_at "
                         "FROM model_versions ORDER BY model_id")
        return rows

    def name_of(self, model_id: str) -> str | None:
        eng = self.holder.engine
        if eng is None:
            return None
        try:
            return eng.describe(model_id).get("name")
        except KeyError:
            return None

    @staticmethod
    def metric_summary(m: dict) -> dict:
        ev = (m.get("metrics") or {}).get("evaluations") or []
        return ev[0] if ev else {}

    @staticmethod
    def loads(v):
        return loads(v, {})
