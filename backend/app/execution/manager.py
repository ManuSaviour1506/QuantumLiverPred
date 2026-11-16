"""Comparison run manager: validated uploads -> concurrent four-family runs -> live events -> persisted history.

Runs execute inside the API process on a bounded thread pool (COMPARE_MAX_CONCURRENT_RUNS, default 2). Each run starts
the four family pipelines of `src.v10.engine.compare.ComparisonRunner` concurrently. Heavy work never runs on the
event loop: the HTTP handlers only enqueue, and the live stream reads an in-memory event log that the run thread
appends to (Server-Sent Events, `GET /api/v1/compare/{run_id}/stream`).

Isolation: every run has its own id, input bytes, feature cache, cancel flag and event log, so results of different
uploads are never mixed. Model objects shared between runs are protected by per-model locks in the engine.
Cancellation and the time limit stop models that have not started yet; a model already executing finishes (Python
threads cannot be interrupted safely) and its result is still recorded.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from app.core.errors import ApiError
from app.core.logging import log_event
from app.core.security import iso, new_id, utcnow

log = logging.getLogger("lti.compare")
TERMINAL = ("completed", "cancelled", "failed")


class RunState:
    def __init__(self, run_id: str, user_id: str):
        self.run_id, self.user_id = run_id, user_id
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.cancel = threading.Event()
        self.status = "queued"
        self.result: dict | None = None
        self.created = time.time()

    def emit(self, event: str, data: dict) -> None:
        with self.cond:
            self.events.append({"id": len(self.events), "event": event, "data": data, "t": round(time.time(), 3)})
            self.cond.notify_all()


def hardware_info() -> dict:
    try:
        import numpy as np
        blas = (np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}) or {}).get("name") \
            if hasattr(np.__config__, "CONFIG") else None
    except Exception:
        blas = None
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "processor": platform.processor() or None, "gpu": None,
            "gpu_note": "all models run on CPU (NumPy / scikit-learn); no GPU is used", "blas": blas,
            "quantum": "exact state-vector simulation on CPU; no quantum hardware"}


class ComparisonManager:
    def __init__(self, settings, db, storage, engine_holder, model_service):
        self.settings, self.db, self.storage = settings, db, storage
        self.holder, self.models = engine_holder, model_service
        self.pool = ThreadPoolExecutor(max_workers=max(1, settings.compare_max_concurrent_runs),
                                       thread_name_prefix="cmp-run")
        self.runs: "OrderedDict[str, RunState]" = OrderedDict()
        self._lock = threading.Lock()
        self.hardware = hardware_info()

    # ------------------------------------------------------------------ lifecycle
    def start(self, user: dict, upload: dict, mode: str, model_ids: list[str] | None) -> dict:
        eng = self.models.engine()
        rid = new_id("run_")
        st = RunState(rid, user["id"])
        with self._lock:
            self.runs[rid] = st
            while len(self.runs) > 200:                       # bounded memory: oldest finished runs leave memory
                k, v = next(iter(self.runs.items()))
                if v.status not in TERMINAL:
                    break
                self.runs.pop(k)
        now = iso(utcnow())
        with self.db.session() as c:
            c.run("INSERT INTO comparison_runs (id, user_id, upload_id, mode, status, requested_models, hardware, "
                  "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (rid, user["id"], upload["id"], mode, "queued", model_ids or [], self.hardware, now))
        st.emit("run_queued", {"run_id": rid, "mode": mode, "upload_id": upload["id"]})
        self.pool.submit(self._execute, st, upload, mode, model_ids, eng)
        log_event(log, "comparison queued", run_id=rid, mode=mode, n_requested=len(model_ids or []))
        return {"run_id": rid, "status": "queued", "mode": mode, "upload_id": upload["id"],
                "stream_url": f"/api/v1/compare/{rid}/stream"}

    def _execute(self, st: RunState, upload: dict, mode: str, model_ids: list[str] | None, eng) -> None:
        from src.v10.engine.compare import ComparisonRunner, FAMILY_LABEL
        rid = st.run_id
        st.status = "running"
        self._update(rid, status="running", started_at=iso(utcnow()))
        try:
            data = self.storage.get_bytes(upload["storage"]["input"])
            ctx = {k: self.storage.get_bytes(v) for k, v in (upload["storage"].get("context") or {}).items()}
            runner = ComparisonRunner(eng, self.settings.compare_model_workers_per_family)
            st.emit("run_started", {"run_id": rid, "families": list(FAMILY_LABEL), "family_labels": FAMILY_LABEL,
                                    "scheduling": {"family_workers": 4,
                                                   "model_workers_per_family": runner.model_workers}})
            t0 = time.perf_counter()
            prepared = runner.prepare(data, upload.get("input_name"), ctx.get("prev"), ctx.get("next"),
                                      ctx.get("prev2"), ctx.get("next2"), run_id=rid)
            st.emit("preprocessing_completed", prepared.summary())
            plan = runner.plan(prepared, model_ids if mode == "run_selected" else None)
            st.emit("plan", {"models": [{k: i.get(k) for k in ("model_id", "name", "family", "architecture",
                                                                "eligible", "skip_status", "skip_reason",
                                                                "context_slices_each_side", "quantum", "version",
                                                                "threshold", "execution_mode")} for i in plan]})
            result = runner.run(prepared, plan, emit=st.emit, cancel=st.cancel,
                                deadline_s=self.settings.compare_run_timeout_s)
            result["end_to_end_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            result["engine"] = {"load_seconds": eng.load_seconds, "note": "models are loaded once per process at "
                                "start-up; per-run timings are warm unless cold_start is true"}
            result["hardware"] = self.hardware
            result["upload"] = {"id": upload["id"], "input_name": upload.get("input_name"),
                                "modality": upload.get("modality"), "sha256_prefix": (upload.get("input_sha256") or "")[:16]}
            status = "cancelled" if st.cancel.is_set() else "completed"
            for m in result["models"]:
                m.pop("trace", None)
            self._persist(rid, status, result)
            st.result = result
            st.status = status
            st.emit("run_cancelled" if status == "cancelled" else "run_completed",
                    {"run_id": rid, "status": status, "result": result})
        except Exception as e:
            log.error("comparison run failed", exc_info=True)
            msg = getattr(e, "message", None) or f"{type(e).__name__}: {e}"
            self._update(rid, status="failed", error=msg[:2000], finished_at=iso(utcnow()))
            st.status = "failed"
            st.emit("run_failed", {"run_id": rid, "status": "failed", "error": msg})

    def _update(self, rid: str, **cols) -> None:
        sets = ", ".join(f"{k} = ?" for k in cols)
        with self.db.session() as c:
            c.run(f"UPDATE comparison_runs SET {sets} WHERE id = ?", (*cols.values(), rid))

    def _persist(self, rid: str, status: str, result: dict) -> None:
        now = iso(utcnow())
        ag, rk = result["agreement"], result["ranking"]
        summary = {"families": result["families"], "agreement": ag, "ranking": {k: rk[k] for k in rk
                                                                                if k != "models_by_inference_ms"},
                   "preprocessing_ms": result["preprocessing"]["timings_ms"], "wall_ms": result["wall_ms"],
                   "stopped": result["preprocessing"].get("stopped"), "scheduling": result["scheduling"]["strategy"]}
        with self.db.session() as c:
            with c.transaction():
                c.run("UPDATE comparison_runs SET status = ?, n_models = ?, n_completed = ?, n_failed = ?, wall_ms = ?, "
                      "fastest_family = ?, slowest_family = ?, majority_label = ?, agreement_pct = ?, summary = ?, "
                      "result = ?, finished_at = ? WHERE id = ?",
                      (status, len(result["models"]), ag["completed"],
                       sum(1 for m in result["models"] if m["status"] == "error"), result["wall_ms"],
                       rk["fastest_family"], rk["slowest_family"], ag["majority_label"], ag["agreement_pct"],
                       summary, result, now, rid))
                for m in result["models"]:
                    t = m.get("timings_ms") or {}
                    c.run("INSERT INTO comparison_model_results (id, run_id, model_id, model_version, family, status, "
                          "label, probability, threshold, confidence, queue_ms, features_ms, inference_ms, total_ms, "
                          "cold_start, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (new_id("cmr_"), rid, m["model_id"], m.get("version"), m["family"], m["status"], m.get("label"),
                           m.get("probability"), m.get("threshold"), m.get("confidence"), t.get("queue"),
                           t.get("features"), t.get("inference"), t.get("total"),
                           None if m.get("cold_start") is None else int(bool(m.get("cold_start"))),
                           (m.get("reason") or None) and str(m.get("reason"))[:1000], now))

    # ------------------------------------------------------------------ control
    def state(self, rid: str) -> RunState | None:
        return self.runs.get(rid)

    def cancel(self, user: dict, rid: str) -> dict:
        st = self.runs.get(rid)
        if st is None or (st.user_id != user["id"] and user.get("role") != "admin"):
            row = self._row(user, rid)
            return {"run_id": rid, "status": row["status"], "cancel_requested": False,
                    "detail": "run is not executing in this process"}
        if st.status in TERMINAL:
            return {"run_id": rid, "status": st.status, "cancel_requested": False, "detail": "run already finished"}
        st.cancel.set()
        st.emit("cancel_requested", {"run_id": rid})
        return {"run_id": rid, "status": st.status, "cancel_requested": True,
                "detail": "models that have not started will be skipped; a model already executing finishes first"}

    def _row(self, user: dict, rid: str) -> dict:
        with self.db.session() as c:
            row = c.one("SELECT * FROM comparison_runs WHERE id = ?", (rid,))
        if row is None or (row["user_id"] != user["id"] and user.get("role") != "admin"):
            raise ApiError(404, "not_found", "Comparison run not found")
        return row

    # ------------------------------------------------------------------ live stream (SSE)
    def stream(self, user: dict, rid: str, last_id: int = -1):
        st = self.runs.get(rid)
        if st is None:
            row = self._row(user, rid)
            res = json.loads(row["result"] or "{}")

            def replay():
                ev = "run_completed" if row["status"] == "completed" else f"run_{row['status']}"
                payload = {"run_id": rid, "status": row["status"], "result": res or None, "error": row["error"],
                           "replayed": True}
                yield _sse(0, ev, payload)
            return replay()
        if st.user_id != user["id"] and user.get("role") != "admin":
            raise ApiError(404, "not_found", "Comparison run not found")

        def gen():
            nxt = last_id + 1
            last_beat = time.time()
            while True:
                with st.cond:
                    if nxt >= len(st.events) and st.status not in TERMINAL:
                        st.cond.wait(timeout=10)
                    batch = st.events[nxt:]
                for e in batch:
                    yield _sse(e["id"], e["event"], e["data"])
                    nxt = e["id"] + 1
                if st.status in TERMINAL and nxt >= len(st.events):
                    return
                if time.time() - last_beat > 9:
                    last_beat = time.time()
                    yield ": keep-alive\n\n"
        return gen()


def _sse(eid: int, event: str, data: dict) -> str:
    from app.core.apikit import to_json_bytes
    return f"id: {eid}\nevent: {event}\ndata: {to_json_bytes(data).decode()}\n\n"
