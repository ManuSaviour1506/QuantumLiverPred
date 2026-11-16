"""Run records: every training / evaluation run writes a self-contained folder

results/v10/runs/<timestamp>_<name>/
    config.json        full configuration incl. seeds, model architecture, preprocessing
    split.json         which units / rows were used for what
    environment.json   Python, platform, package versions
    log.txt            training log
    metrics.json       evaluation metrics (with evaluation unit)
    predictions.csv    out-of-fold predictions used for calibration and the threshold
    summary.json       timestamps, artifact path, status
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from src.v10.utils import clean, environment


class RunRecorder:
    def __init__(self, root: Path, name: str):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = Path(root) / "results" / "v10" / "runs" / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self._log = open(self.dir / "log.txt", "a")
        (self.dir / "environment.json").write_text(json.dumps(clean(environment()), indent=1))

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def write(self, name: str, obj) -> None:
        (self.dir / name).write_text(json.dumps(clean(obj), indent=1, default=str))

    def finish(self, status: str, **extra) -> dict:
        s = {"status": status, "started_utc": self.started,
             "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "seconds": round(time.time() - self.t0, 1), **extra}
        self.write("summary.json", s)
        self._log.close()
        return s
