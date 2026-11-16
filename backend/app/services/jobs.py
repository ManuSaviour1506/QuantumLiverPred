"""Database-backed job queue for long-running work (inference, dataset validation, experiments).

The API only inserts a `jobs` row and returns 202; a runner claims the row atomically
(`UPDATE ... WHERE status = 'queued'`) and executes it. Runners are either threads inside the API process
(WORKER_MODE=inline, default) or a separate process (`python -m app.worker`, WORKER_MODE=external), so the event loop
never waits for a model. Clients poll the resource (e.g. GET /api/v1/predictions/{id}) for the status.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import traceback
from typing import Callable

from app.core.logging import log_event
from app.core.security import iso, new_id, utcnow

log = logging.getLogger("lti.jobs")


class JobQueue:
    def __init__(self, db, settings, handlers: dict[str, Callable[[str], None]] | None = None):
        self.db, self.settings = db, settings
        self.handlers: dict[str, Callable[[str], None]] = dict(handlers or {})
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{new_id()[:6]}"
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._busy = 0
        self._lock = threading.Lock()

    def register(self, kind: str, fn: Callable[[str], None]) -> None:
        self.handlers[kind] = fn

    def enqueue(self, conn, kind: str, ref_id: str) -> str:
        jid = new_id("job_")
        conn.run("INSERT INTO jobs (id, kind, ref_id, status, attempts, created_at) VALUES (?, ?, ?, 'queued', 0, ?)",
                 (jid, kind, ref_id, iso(utcnow())))
        return jid

    def wake(self) -> None:
        self._wake.set()

    def claim(self) -> dict | None:
        with self.db.session() as c:
            for row in c.all("SELECT id, kind, ref_id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 5"):
                n = c.run("UPDATE jobs SET status = 'running', claimed_by = ?, started_at = ?, attempts = attempts + 1 "
                          "WHERE id = ? AND status = 'queued'", (self.worker_id, iso(utcnow()), row["id"]))
                c.commit()
                if n == 1:
                    return row
        return None

    def _finish(self, job_id: str, status: str, error: str | None = None) -> None:
        with self.db.session() as c:
            c.run("UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                  (status, error, iso(utcnow()), job_id))

    def run_once(self) -> bool:
        job = self.claim()
        if job is None:
            return False
        handler = self.handlers.get(job["kind"])
        with self._lock:
            self._busy += 1
        try:
            if handler is None:
                raise RuntimeError(f"no handler for job kind {job['kind']!r}")
            handler(job["ref_id"])
            self._finish(job["id"], "succeeded")
            log_event(log, "job finished", job_id=job["id"], kind=job["kind"])
        except Exception as e:
            log.error("job failed", extra={"fields": {"job_id": job["id"], "kind": job["kind"],
                                                       "error": type(e).__name__}}, exc_info=True)
            self._finish(job["id"], "failed", f"{type(e).__name__}: {str(e)[:500]}")
        finally:
            with self._lock:
                self._busy -= 1
        return True

    def drain(self, max_jobs: int = 100) -> int:
        """Run queued jobs synchronously (tests and the CLI)."""
        n = 0
        while n < max_jobs and self.run_once():
            n += 1
        return n

    def recover(self) -> int:
        """Re-queue jobs left 'running' by a process that stopped (at most 3 attempts), fail the rest."""
        with self.db.session() as c:
            rows = c.all("SELECT id, attempts FROM jobs WHERE status = 'running'")
            for r in rows:
                if r["attempts"] < 3:
                    c.run("UPDATE jobs SET status = 'queued', claimed_by = NULL WHERE id = ?", (r["id"],))
                else:
                    c.run("UPDATE jobs SET status = 'failed', error = 'interrupted three times', finished_at = ? "
                          "WHERE id = ?", (iso(utcnow()), r["id"]))
        return len(rows)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:
                log.error("job runner error", exc_info=True)
                worked = False
            if not worked:
                self._wake.wait(self.settings.worker_poll_seconds)
                self._wake.clear()

    def start(self, threads: int) -> None:
        for i in range(threads):
            t = threading.Thread(target=self._loop, name=f"lti-job-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    def pending(self) -> int:
        with self.db.session() as c:
            return int(c.scalar("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')") or 0)

    def error_text(self) -> str:
        return traceback.format_exc(limit=0)
