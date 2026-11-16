"""Model artifact format shared by every V8 model (classical, quantum, hybrid).

artifacts/<family>/<model_id>/model.npz   numeric parameters only (np.savez, loaded with allow_pickle=False)
artifacts/<family>/<model_id>/model.json  metadata: architecture, training configuration and log, calibration,
                                          operating threshold, validation / test metrics, SHA-256 of model.npz,
                                          status ("trained" | "not_trained" | "failed") and, for quantum models,
                                          the execution backend (always a simulator in V8).
A model whose model.json says "trained" but whose npz is missing or whose hash differs is refused at load time.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from src.common.io import load_arrays, read_json, save_arrays, sha256_file, write_json
from src.common.paths import ARTIFACTS


class ModelIntegrityError(RuntimeError):
    pass


def model_dir(family: str, model_id: str) -> Path:
    return ARTIFACTS / family / model_id


def save_model(family: str, model_id: str, arrays: dict, meta: dict) -> Path:
    d = model_dir(family, model_id)
    d.mkdir(parents=True, exist_ok=True)
    sha = save_arrays(d / "model.npz", **arrays)
    meta = dict(meta)
    meta.update({"model_id": model_id, "family": family, "npz_sha256": sha, "status": meta.get("status", "trained"),
                 "saved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    write_json(d / "model.json", meta)
    return d


def update_meta(family: str, model_id: str, **fields) -> None:
    p = model_dir(family, model_id) / "model.json"
    meta = read_json(p)
    meta.update(fields)
    write_json(p, meta)


def load_model(family: str, model_id: str, verify: bool = True) -> tuple[dict, dict]:
    d = model_dir(family, model_id)
    meta = read_json(d / "model.json")
    if meta.get("status") != "trained":
        return {}, meta
    npz = d / "model.npz"
    if not npz.exists():
        raise ModelIntegrityError(f"{model_id}: model.npz missing")
    if verify and sha256_file(npz) != meta.get("npz_sha256"):
        raise ModelIntegrityError(f"{model_id}: model.npz hash does not match model.json")
    return load_arrays(npz), meta


def list_models() -> list[tuple[str, str]]:
    out = []
    for fam in ("classical", "quantum", "hybrid"):
        base = ARTIFACTS / fam
        if base.exists():
            out += [(fam, p.name) for p in sorted(base.iterdir()) if (p / "model.json").exists()]
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
