"""Background worker process (WORKER_MODE=external): claims queued jobs from the database and runs them.

    python -m app.worker            (from backend/, same environment variables as the API)

Several workers may run at once; each job is claimed atomically. The API process then only enqueues work.
"""
from __future__ import annotations

import logging
import signal
import sys
import time

from app.core.config import Settings


def main() -> int:
    from app.main import create_app
    settings = Settings.from_env()
    app = create_app(settings, start_background=False)
    st = app.state
    log = logging.getLogger("lti.worker")
    stop = {"flag": False}

    def handle(sig, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    st.jobs.recover()
    if settings.preload_models:
        try:
            st.engine.ensure()
        except Exception:
            log.error("model engine failed to load; prediction jobs will fail with a clear error")
    log.info("worker started", extra={"fields": {"worker": st.jobs.worker_id}})
    while not stop["flag"]:
        if not st.jobs.run_once():
            time.sleep(settings.worker_poll_seconds)
    log.info("worker stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
