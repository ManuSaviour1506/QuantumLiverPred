"""Small, dependency-free helpers: hashing, JSON, pickle-free array bundles."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def pixel_sha(gray_u8: np.ndarray, n: int = 12) -> str:
    """Hash of decoded 8-bit pixel values (row-major); independent of PNG encoder settings."""
    return hashlib.sha256(np.ascontiguousarray(gray_u8, dtype=np.uint8).tobytes()).hexdigest()[:n]


def clean_nan(o):
    """Recursively replace NaN / inf by None so the JSON is standard (browsers and strict encoders reject NaN)."""
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean_nan(o.tolist())
    return o


def write_json(path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(clean_nan(obj), f, indent=2, default=_jsonable, allow_nan=False)
    os.replace(tmp, path)


def read_json(path) -> Any:
    with open(path) as f:
        return json.load(f)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_arrays(path, **arrays) -> str:
    """Save a model as a plain .npz (no pickle) and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return sha256_file(path)


def load_arrays(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def seed_everything(seed: int = 42) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return np.random.default_rng(seed)
