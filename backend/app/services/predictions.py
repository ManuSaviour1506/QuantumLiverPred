"""Prediction workflow: validate upload -> store input privately -> queue job -> real inference in a worker ->
store result, preview and explanation images -> history, report, export, deletion and retention.

No prediction is ever produced without a servable, integrity-checked model; failures are stored as failures with a
user-safe message (never a stack trace or a server path).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import logging

from app.core.apikit import UploadedFile
from app.core.config import DISCLAIMER
from app.core.errors import ApiError, not_found
from app.core.logging import log_event
from app.core.security import iso, new_id, parse_iso, utcnow
from app.db.database import loads
from app.services import audit

log = logging.getLogger("lti.predictions")

EXT = {"png": "png", "jpeg": "jpg", "dicom": "dcm", "nifti": "nii"}
SORTS = {"created_at": "p.created_at ASC", "-created_at": "p.created_at DESC", "probability": "p.probability ASC",
         "-probability": "p.probability DESC", "model_id": "p.model_id ASC", "-model_id": "p.model_id DESC"}


def _detect(data: bytes, name: str | None) -> str:
    from src.preprocessing import formats
    try:
        fmt = formats.detect_format(data, name)
    except formats.UnsupportedFormat as e:
        raise ApiError(415, "unsupported_media_type", str(e))
    if fmt not in EXT:
        raise ApiError(415, "unsupported_media_type", f"unsupported image format {fmt!r}")
    try:
        formats.decode(data, name)
    except (formats.UnsupportedFormat, formats.CorruptImage) as e:
        raise ApiError(422, "corrupt_image", f"The file could not be decoded as {fmt.upper()}: {e}")
    return fmt


class PredictionService:
    def __init__(self, settings, db, storage, engine_holder, jobs, model_service):
        self.settings, self.db, self.storage = settings, db, storage
        self.holder, self.jobs, self.models = engine_holder, jobs, model_service

    # ------------------------------------------------------------------ create
    def create(self, user: dict, model_id: str, file: UploadedFile, prev: UploadedFile | None,
               nxt: UploadedFile | None, explain: bool, ip: str | None) -> dict:
        if user["role"] == "viewer":
            raise ApiError(403, "forbidden", "Viewers cannot run predictions")
        m = self.models.get(model_id)
        if not m["servable"]:
            raise ApiError(409, "model_unavailable", f"Model '{model_id}' is not available for prediction",
                           {"status": m["status"], "reason": m["reason"]})
        contract = m.get("input_contract") or {}
        if contract.get("requires_context") and contract.get("context_slices_each_side", 0) > 1:
            raise ApiError(422, "unsupported_model_input",
                           "This model reads five adjacent slices, which this upload form does not support")
        if contract.get("requires_context") and (prev is None or nxt is None):
            raise ApiError(422, "context_required",
                           "This model reads adjacent slices: upload the previous and next slices as well")
        data = file.read_bytes()
        fmt = _detect(data, file.filename)
        ctx = {}
        for key, f in (("prev", prev), ("next", nxt)):
            if f is not None:
                b = f.read_bytes()
                ctx[key] = (b, _detect(b, f.filename))
        pid = new_id("pred_")
        now = utcnow()
        base = f"uploads/{user['id']}/{pid}"
        self.storage.put_bytes(f"{base}/input.{EXT[fmt]}", data)
        for key, (b, f_fmt) in ctx.items():
            self.storage.put_bytes(f"{base}/{key}.{EXT[f_fmt]}", b)
        expires = iso(now + dt.timedelta(days=self.settings.retention_days)) if self.settings.retention_days > 0 else None
        name = file.filename if self.settings.store_original_filenames else None
        with self.db.session() as c:
            with c.transaction():
                c.run("INSERT INTO predictions (id, user_id, model_id, model_version, dataset_version, execution_mode, "
                      "status, result, input_name, input_sha256, input_format, input_bytes, created_at, expires_at) "
                      "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
                      (pid, user["id"], model_id, m["version"], m["dataset_version"], m["execution_mode"],
                       {"explain_requested": bool(explain), "context": sorted(ctx)}, name, file.sha256, fmt,
                       file.size, iso(now), expires))
                for kind, key in [("input", f"{base}/input.{EXT[fmt]}")] + \
                                 [(k, f"{base}/{k}.{EXT[v[1]]}") for k, v in ctx.items()]:
                    blob = data if kind == "input" else ctx[kind][0]
                    c.run("INSERT INTO prediction_artifacts (id, prediction_id, kind, storage_key, mime, size_bytes, "
                          "sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (new_id("pa_"), pid, kind, key, "application/octet-stream", len(blob),
                           hashlib.sha256(blob).hexdigest(), iso(now)))
                self.jobs.enqueue(c, "prediction", pid)
                audit.log(c, "prediction.created", user["id"], "prediction", pid, ip, {"model_id": model_id},
                          secret=self.settings.jwt_secret)
        self.jobs.wake()
        log_event(log, "prediction queued", prediction_id=pid, model_id=model_id, input_sha256=file.sha256[:12])
        return {"id": pid, "status": "queued", "poll_url": f"/api/v1/predictions/{pid}"}

    # ------------------------------------------------------------------ execute (background job)
    def execute(self, pid: str) -> None:
        from src.v10.engine.interface import InputRejected, ModelInput, ModelUnavailable
        with self.db.session() as c:
            row = c.one("SELECT * FROM predictions WHERE id = ?", (pid,))
            if row is None or row["status"] not in ("queued", "running"):
                return
            arts = {a["kind"]: a["storage_key"] for a in
                    c.all("SELECT kind, storage_key FROM prediction_artifacts WHERE prediction_id = ?", (pid,))}
            c.run("UPDATE predictions SET status = 'running', started_at = ? WHERE id = ?", (iso(utcnow()), pid))
        req = loads(row["result"], {})
        status, fields = "failed", {}
        try:
            eng = self.holder.ensure()
            inp = ModelInput(self.storage.get_bytes(arts["input"]), f"input.{EXT.get(row['input_format'], 'bin')}",
                             self.storage.get_bytes(arts["prev"]) if "prev" in arts else None,
                             self.storage.get_bytes(arts["next"]) if "next" in arts else None)
            po = eng.predict(row["model_id"], inp, explain=bool(req.get("explain_requested")))
            images = dict(po.images)
            expl = dict(po.explanation)
            gc = expl.get("gradcam")
            if isinstance(gc, dict) and gc.get("png_b64"):
                import base64
                images["gradcam"] = base64.b64decode(gc.pop("png_b64"))
            now = iso(utcnow())
            with self.db.session() as c:
                with c.transaction():
                    for kind, png in images.items():
                        key = f"uploads/{row['user_id']}/{pid}/{kind}.png"
                        self.storage.put_bytes(key, png)
                        c.run("INSERT INTO prediction_artifacts (id, prediction_id, kind, storage_key, mime, size_bytes, "
                              "sha256, created_at) VALUES (?, ?, ?, ?, 'image/png', ?, ?, ?)",
                              (new_id("pa_"), pid, kind, key, len(png), hashlib.sha256(png).hexdigest(), now))
            result = {"status": po.status, "label": po.label, "reason": po.reason,
                      "probability_note": po.probability_note, "details": po.details, "explanation": expl,
                      "timings_ms": po.timings_ms, "notes": po.notes, "explain_requested": req.get("explain_requested"),
                      "context": req.get("context")}
            status = "succeeded"
            fields = {"label": po.label, "probability": po.probability, "threshold": po.threshold,
                      "result": result, "processing_ms": po.timings_ms.get("total"), "error": None,
                      "model_version": po.model_version, "execution_mode": po.execution_mode,
                      "dataset_version": po.dataset_version}
        except InputRejected as e:
            fields = {"error": e.message, "result": {**req, "status": "rejected"}}
        except ModelUnavailable as e:
            fields = {"error": f"Model unavailable: {e}", "result": {**req, "status": "model_unavailable"}}
        except Exception as e:
            log.error("prediction failed", extra={"fields": {"prediction_id": pid}}, exc_info=True)
            fields = {"error": f"Inference failed ({type(e).__name__}); the error was logged",
                      "result": {**req, "status": "error"}}
        with self.db.session() as c:
            sets = ", ".join(f"{k} = ?" for k in fields)
            c.run(f"UPDATE predictions SET status = ?, finished_at = ?{', ' + sets if sets else ''} WHERE id = ?",
                  (status, iso(utcnow()), *fields.values(), pid))
            audit.log(c, f"prediction.{status}", row["user_id"], "prediction", pid, None,
                      {"model_id": row["model_id"], "label": fields.get("label")}, secret=self.settings.jwt_secret)

    # ------------------------------------------------------------------ read
    def _owned(self, c, user: dict, pid: str) -> dict:
        row = c.one("SELECT * FROM predictions WHERE id = ?", (pid,))
        if row is None or (row["user_id"] != user["id"] and user["role"] != "admin"):
            raise not_found("Prediction")
        return row

    def _summary(self, r: dict) -> dict:
        return {"id": r["id"], "user_id": r["user_id"], "model_id": r["model_id"],
                "model_name": self.models.name_of(r["model_id"]), "model_version": r["model_version"],
                "status": r["status"], "label": r["label"], "probability": r["probability"], "threshold": r["threshold"],
                "input_name": r["input_name"], "input_format": r["input_format"], "processing_ms": r["processing_ms"],
                "execution_mode": r["execution_mode"], "created_at": r["created_at"], "finished_at": r["finished_at"]}

    def get(self, user: dict, pid: str) -> dict:
        with self.db.session() as c:
            r = self._owned(c, user, pid)
            arts = c.all("SELECT kind, mime, size_bytes FROM prediction_artifacts WHERE prediction_id = ? "
                         "AND mime = 'image/png' ORDER BY kind", (pid,))
        out = self._summary(r)
        out.update({"dataset_version": r["dataset_version"], "input_sha256_prefix": (r["input_sha256"] or "")[:16],
                    "input_bytes": r["input_bytes"], "started_at": r["started_at"], "expires_at": r["expires_at"],
                    "error": r["error"], "result": loads(r["result"], {}), "disclaimer": DISCLAIMER,
                    "artifacts": [{"kind": a["kind"], "mime": a["mime"], "size_bytes": a["size_bytes"],
                                   "url": f"/api/v1/predictions/{pid}/artifacts/{a['kind']}"} for a in arts]})
        return out

    def _filters(self, user: dict, q) -> tuple[str, list]:
        where, params = [], []
        if not (q.all_users and user["role"] == "admin"):
            where.append("p.user_id = ?")
            params.append(user["id"])
        if q.model_id:
            where.append("p.model_id = ?")
            params.append(q.model_id)
        if q.status:
            where.append("p.status = ?")
            params.append(q.status)
        if q.label:
            where.append("p.label = ?")
            params.append(q.label)
        if q.date_from:
            where.append("p.created_at >= ?")
            params.append(q.date_from)
        if q.date_to:
            where.append("p.created_at < ?")
            params.append((dt.date.fromisoformat(q.date_to) + dt.timedelta(days=1)).isoformat())
        if q.q:
            like = f"%{q.q.lower()}%"
            where.append("(LOWER(COALESCE(p.input_name, '')) LIKE ? OR LOWER(p.model_id) LIKE ? "
                         "OR LOWER(COALESCE(p.label, '')) LIKE ?)")
            params += [like, like, like]
        return (" WHERE " + " AND ".join(where)) if where else "", params

    def list(self, user: dict, q) -> tuple[list[dict], int]:
        where, params = self._filters(user, q)
        with self.db.session() as c:
            total = int(c.scalar("SELECT COUNT(*) FROM predictions p" + where, params) or 0)
            rows = c.all(f"SELECT * FROM predictions p{where} ORDER BY {SORTS[q.sort]}, p.id LIMIT ? OFFSET ?",
                         (*params, q.page_size, (q.page - 1) * q.page_size))
        return [self._summary(r) for r in rows], total

    def export_csv(self, user: dict, q) -> str:
        where, params = self._filters(user, q)
        with self.db.session() as c:
            rows = c.all(f"SELECT * FROM predictions p{where} ORDER BY {SORTS[q.sort]} LIMIT 10000", params)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "created_at", "finished_at", "model_id", "model_version", "execution_mode", "status", "label",
                    "probability", "threshold", "input_name", "input_format", "input_sha256_prefix", "processing_ms",
                    "dataset_version", "error"])
        for r in rows:
            w.writerow([r["id"], r["created_at"], r["finished_at"], r["model_id"], r["model_version"],
                        r["execution_mode"], r["status"], r["label"], r["probability"], r["threshold"], r["input_name"],
                        r["input_format"], (r["input_sha256"] or "")[:16], r["processing_ms"], r["dataset_version"],
                        r["error"]])
        buf.write(f"# {DISCLAIMER}\n")
        return buf.getvalue()

    def artifact(self, user: dict, pid: str, kind: str) -> tuple[bytes, str]:
        with self.db.session() as c:
            self._owned(c, user, pid)
            a = c.one("SELECT storage_key, mime FROM prediction_artifacts WHERE prediction_id = ? AND kind = ? "
                      "AND mime = 'image/png'", (pid, kind))
        if a is None:
            raise not_found("Artifact")
        return self.storage.get_bytes(a["storage_key"]), a["mime"]

    def delete(self, user: dict, pid: str, ip: str | None) -> None:
        with self.db.session() as c:
            r = self._owned(c, user, pid)
            with c.transaction():
                c.run("DELETE FROM predictions WHERE id = ?", (pid,))
                audit.log(c, "prediction.deleted", user["id"], "prediction", pid, ip, secret=self.settings.jwt_secret)
        self.storage.delete_prefix(f"uploads/{r['user_id']}/{pid}")

    def purge_expired(self) -> int:
        now = iso(utcnow())
        with self.db.session() as c:
            rows = c.all("SELECT id, user_id FROM predictions WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
            for r in rows:
                c.run("DELETE FROM predictions WHERE id = ?", (r["id"],))
                audit.log(c, "prediction.expired", r["user_id"], "prediction", r["id"], secret=self.settings.jwt_secret)
        for r in rows:
            self.storage.delete_prefix(f"uploads/{r['user_id']}/{r['id']}")
        return len(rows)

    def report_pdf(self, user: dict, pid: str) -> bytes:
        from app.services.reports import prediction_report
        p = self.get(user, pid)
        if p["status"] != "succeeded":
            raise ApiError(409, "not_ready", "A report is available only for completed predictions")
        images = {}
        for a in p["artifacts"]:
            if a["kind"] in ("preview", "patch_map", "gradcam"):
                images[a["kind"]] = self.artifact(user, pid, a["kind"])[0]
        try:
            model = self.models.get(p["model_id"])
        except ApiError:
            model = None
        return prediction_report(p, model, images)

    @staticmethod
    def age_days(row: dict) -> float:
        t = parse_iso(row["created_at"])
        return (utcnow() - t).total_seconds() / 86400
