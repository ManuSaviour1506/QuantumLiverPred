"""New-evaluation benchmark runs (V11) and the live process monitor.

Benchmark runs evaluate registered models on labelled slices with `src.v10.engine.benchmark.run_benchmark` in a
single background worker (one benchmark at a time, so measured latencies are not distorted by each other). They are
stored in `benchmark_runs`, separate from the historical metrics recorded in the model cards.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from app.core.config import PROJECT_ROOT
from app.core.errors import ApiError
from app.core.security import iso, new_id, utcnow
from app.db.database import loads

log = logging.getLogger("lti.benchmark")


class BenchmarkService:
    def __init__(self, settings, db, model_service):
        self.settings, self.db, self.models = settings, db, model_service
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark")
        self.cancels: dict[str, threading.Event] = {}
        self.progress: dict[str, tuple[int, int]] = {}

    def datasets(self) -> list[dict]:
        from src.v10.engine.benchmark import describe_datasets
        return describe_datasets(PROJECT_ROOT)

    def start(self, user: dict, dataset: str, split: str | None, max_slices: int, model_ids: list[str] | None) -> dict:
        if user["role"] == "viewer":
            raise ApiError(403, "forbidden", "Viewers cannot start benchmark runs")
        if dataset not in {d["id"] for d in self.datasets()}:
            raise ApiError(404, "not_found", f"Labelled dataset {dataset!r} is not available in this installation")
        eng = self.models.engine()
        bid = new_id("bench_")
        with self.db.session() as c:
            c.run("INSERT INTO benchmark_runs (id, user_id, dataset, split, max_slices, model_ids, status, created_at) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (bid, user["id"], dataset, split, max_slices, model_ids or [],
                                                      "queued", iso(utcnow())))
        self.cancels[bid] = threading.Event()
        self.pool.submit(self._run, bid, eng, dataset, split, max_slices, model_ids)
        return {"id": bid, "status": "queued"}

    def _update(self, bid: str, **cols):
        sets = ", ".join(f"{k} = ?" for k in cols)
        with self.db.session() as c:
            c.run(f"UPDATE benchmark_runs SET {sets} WHERE id = ?", (*cols.values(), bid))

    def _run(self, bid, eng, dataset, split, max_slices, model_ids):
        from src.v10.engine.benchmark import run_benchmark
        self._update(bid, status="running", started_at=iso(utcnow()))
        last = [0.0]

        def progress(i, n):
            self.progress[bid] = (i, n)
            if time.time() - last[0] > 2 or i == n:
                last[0] = time.time()
                self._update(bid, progress=round(i / max(n, 1), 4), n_slices=n)
        try:
            res = run_benchmark(eng, PROJECT_ROOT, dataset, split or None, max_slices or 0, model_ids or None,
                                progress=progress, cancel=self.cancels[bid])
            self._update(bid, status="cancelled" if res["cancelled"] else "completed", result=res,
                         n_slices=res["n_slices"], progress=1.0 if not res["cancelled"] else
                         round(res["n_processed"] / max(res["n_slices"], 1), 4), finished_at=iso(utcnow()))
        except Exception as e:
            log.error("benchmark run failed", exc_info=True)
            self._update(bid, status="failed", error=f"{type(e).__name__}: {e}"[:2000], finished_at=iso(utcnow()))

    def list(self, user: dict) -> list[dict]:
        with self.db.session() as c:
            rows = c.all("SELECT id, user_id, dataset, split, max_slices, status, progress, n_slices, error, created_at, "
                         "finished_at FROM benchmark_runs ORDER BY created_at DESC LIMIT 50")
        return rows

    def get(self, user: dict, bid: str) -> dict:
        with self.db.session() as c:
            r = c.one("SELECT * FROM benchmark_runs WHERE id = ?", (bid,))
        if r is None:
            raise ApiError(404, "not_found", "Benchmark run not found")
        r["result"] = loads(r["result"], {}) or None
        r["model_ids"] = loads(r["model_ids"], [])
        if bid in self.progress and r["status"] == "running":
            r["processed"], r["total"] = self.progress[bid]
        return r

    def cancel(self, user: dict, bid: str) -> dict:
        ev = self.cancels.get(bid)
        if ev is None:
            raise ApiError(404, "not_found", "Benchmark run is not executing in this process")
        ev.set()
        return {"id": bid, "cancel_requested": True}

    def export_csv(self, user: dict, bid: str) -> str:
        r = self.get(user, bid)
        res = r["result"] or {}
        buf = io.StringIO()
        w = csv.writer(buf)
        keys = ["n_slices", "n_scored", "n_tumour", "n_no_tumour", "n_uncertain", "accuracy", "precision", "recall",
                "specificity", "f1", "roc_auc", "pr_auc", "ece", "inference_ms_median", "inference_ms_p95",
                "thread_cpu_ms_median"]
        w.writerow(["benchmark_id", "dataset", "split", "model_id", "family", "role"] + keys + ["tp", "fn", "fp", "tn"])
        for m in res.get("models", []):
            for role, mt in m["by_role"].items():
                cf = mt["confusion"]
                w.writerow([bid, res.get("dataset"), res.get("split"), m["model_id"], m["family"], role] +
                           [mt.get(k) for k in keys] + [cf["tp"], cf["fn"], cf["fp"], cf["tn"]])
        return buf.getvalue()


class ProcessMonitor:
    """Samples this API process every second (CPU %, resident memory, threads) for the system-monitor page."""

    def __init__(self, keep: int = 300):
        self.samples: deque = deque(maxlen=keep)
        self.started = time.time()
        self._stop = threading.Event()
        self._t = None

    def start(self):
        if self._t is None:
            self._t = threading.Thread(target=self._loop, name="process-monitor", daemon=True)
            self._t.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        from src.v10.engine.compare import rss_mb
        ncpu = os.cpu_count() or 1
        w0, c0 = time.perf_counter(), time.process_time()
        while not self._stop.wait(1.0):
            w, c = time.perf_counter(), time.process_time()
            self.samples.append({"t": round(time.time(), 1), "cpu_percent": round(100 * (c - c0) / max(w - w0, 1e-9) / ncpu, 1),
                                 "rss_mb": rss_mb(), "threads": threading.active_count()})
            w0, c0 = w, c

    def snapshot(self) -> dict:
        from src.v10.engine.compare import gpu_info, peak_rss_mb, rss_mb
        return {"uptime_s": round(time.time() - self.started, 1), "cpu_count": os.cpu_count(), "rss_mb": rss_mb(),
                "peak_rss_mb": peak_rss_mb(), "threads": threading.active_count(), "gpu": gpu_info(),
                "samples": list(self.samples)}
