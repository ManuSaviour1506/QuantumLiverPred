"""V10 artifact format and integrity checks.

artifacts/v10/<model_id>/
    model_card.json   identity, architecture, preprocessing, training data (real or synthetic), dataset version,
                      seeds, metrics with split and evaluation unit, environment, SHA-256 of every weight file
    model.npz         numeric parameters (loaded with allow_pickle=False)     — quantum / hybrid / NumPy models
    model.joblib      scikit-learn pipeline (pickle-based: loaded only after its SHA-256 matches the card and only
                      from the trusted artifacts directory; never accepted from users)

A model is refused when: the card or a weight file is missing; a hash differs; the card's preprocessing signature
does not match the code's current signature (e.g. the feature extractor changed); or the card's format version is
unknown. Refused models are reported with the reason and never produce predictions.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


class ArtifactError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocessing_signature(feature_set: str, preproc: dict, patch_px: int = 96, stride: int = 64) -> str:
    """Hash of everything that must be identical between training and inference."""
    from src.v10.features.extract import FEATURE_SETS
    payload = {"feature_set": feature_set, "feature_dim": FEATURE_SETS.get(feature_set, {}).get("dim"),
               "preproc": preproc, "patch_px": patch_px, "stride": stride, "pipeline": "v9_patch_front_end_v1"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def save(root: Path, model_id: str, card: dict, arrays: dict | None = None, sklearn_obj=None) -> Path:
    d = Path(root) / model_id
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    if arrays is not None:
        np.savez(d / "model.npz", **{k: np.asarray(v) for k, v in arrays.items()})
        files["model.npz"] = sha256_file(d / "model.npz")
    if sklearn_obj is not None:
        import joblib
        joblib.dump(sklearn_obj, d / "model.joblib", compress=3)
        files["model.joblib"] = sha256_file(d / "model.joblib")
    card = dict(card)
    card.update({"model_id": model_id, "format_version": FORMAT_VERSION, "files": files,
                 "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    (d / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    return d


def load_card(d: Path) -> dict:
    p = Path(d) / "model_card.json"
    if not p.is_file():
        raise ArtifactError("model_card.json missing")
    card = json.loads(p.read_text())
    if card.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(f"unknown artifact format version {card.get('format_version')!r}")
    return card


def verify(d: Path, card: dict) -> None:
    for name, digest in (card.get("files") or {}).items():
        p = Path(d) / name
        if not p.is_file():
            raise ArtifactError(f"{name} missing")
        if sha256_file(p) != digest:
            raise ArtifactError(f"{name} SHA-256 does not match the model card")


def load_arrays(d: Path) -> dict:
    with np.load(Path(d) / "model.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_sklearn(d: Path):
    import joblib
    return joblib.load(Path(d) / "model.joblib")


def directory_size(d: Path) -> int:
    return int(sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file()))
