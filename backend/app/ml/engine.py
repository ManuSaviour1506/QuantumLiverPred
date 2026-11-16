"""Bridge between the API and the Python model engine (`src/v10/engine`).

The engine (V9 models through the vendored V9 code, V10 models from `artifacts/v10`) is loaded once per process, in a
background thread at start-up when PRELOAD_MODELS=true, otherwise on first use. Until it is loaded, model endpoints
report `loading`; if loading fails the error is reported and no prediction is ever produced.
"""
from __future__ import annotations

import logging
import sys
import threading
import time

log = logging.getLogger("lti.engine")


class EngineHolder:
    def __init__(self, settings):
        self.settings = settings
        self._engine = None
        self._error: str | None = None
        self._state = "not_loaded"
        self._lock = threading.Lock()
        self._loaded_at: float | None = None
        self._seconds: float | None = None

    def _load(self):
        root = str(self.settings.project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.v10.engine.registry import ModelEngine
        t0 = time.time()
        eng = ModelEngine(self.settings.project_root, allow_synthetic=self.settings.allow_synthetic_models)
        eng.load()
        self._seconds = round(time.time() - t0, 2)
        return eng

    def ensure(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                self._state = "loading"
                try:
                    self._engine = self._load()
                    self._state, self._loaded_at, self._error = "ready", time.time(), None
                except Exception as e:  # reported, never replaced by a fake model
                    log.error("model engine failed to load", exc_info=True)
                    self._state, self._error = "failed", f"{type(e).__name__}: {e}"
                    raise
        return self._engine

    def preload_async(self):
        def run():
            try:
                self.ensure()
            except Exception:
                pass
        threading.Thread(target=run, name="lti-engine-preload", daemon=True).start()

    @property
    def engine(self):
        return self._engine

    def reload(self):
        with self._lock:
            self._engine = None
            self._state = "not_loaded"
        return self.ensure()

    def status(self) -> dict:
        out = {"state": self._state, "error": self._error, "load_seconds": self._seconds}
        if self._engine is not None:
            out.update(self._engine.summary())
        return out
