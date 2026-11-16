"""Small utilities for V10 scripts: JSON output without NaN, run manifests with environment and input hashes."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False))
    os.replace(tmp, path)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def environment() -> dict:
    mods = {}
    for m in ("numpy", "scipy", "sklearn", "pandas", "pennylane", "qiskit", "torch"):
        try:
            mods[m] = __import__(m).__version__
        except Exception:
            mods[m] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "packages": mods,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "execution": "classical simulation only; no quantum hardware"}


def run_manifest(root, inputs: list, seeds: dict, extra: dict | None = None) -> dict:
    root = Path(root)
    files = {}
    for p in inputs:
        fp = root / p
        files[str(p)] = sha256(fp) if fp.exists() else None
    return {"environment": environment(), "inputs_sha256": files, "seeds": seeds, **(extra or {})}
