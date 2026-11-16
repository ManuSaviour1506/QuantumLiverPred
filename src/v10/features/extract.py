"""Patch-level feature extractors shared by training and inference (identical code paths).

radiomics        40 handcrafted intensity / texture / gradient features (V9 `src/classical/baselines.py`), no training
imagenet_gap     1,280-d global-average-pooled EfficientNet-B0 features with frozen ImageNet weights (V9 NumPy backbone)
cnn_embedding    embedding of the V9 2.5-D CNN. EXPLORATORY ONLY: that CNN was trained once on V9's training groups,
                 so any cross-validation that uses it as a feature extractor leaks information across units.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FEATURE_SETS = {
    "radiomics": {"dim": 40, "leakage": "none (no learned parameters)", "status": "confirmatory"},
    "imagenet_gap": {"dim": 1280, "leakage": "none (externally pretrained, frozen)", "status": "secondary"},
    "cnn_embedding": {"dim": None, "leakage": "V9 CNN trained on V9 training groups: leaks across units",
                      "status": "exploratory"},
}


def radiomics(patches2d: np.ndarray) -> np.ndarray:
    from src.classical import baselines as BL
    return BL.features(np.asarray(patches2d, np.uint8)) if len(patches2d) else np.zeros((0, 40))


def imagenet_gap(patches2d: np.ndarray, bs: int = 64) -> np.ndarray:
    from src.classical import backbones as B
    from src.classical.nn import Tensor
    from src.inference.features import backbone
    from src.preprocessing.ct import to_rgb
    net = backbone("efficientnet_b0")
    out = []
    for s in range(0, len(patches2d), bs):
        h = net.frozen(B.Backbone.preprocess(to_rgb(np.asarray(patches2d[s:s + bs], np.uint8))))
        out.append(net.tail(Tensor(h))["final"].data.mean((1, 2)).astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 1280), np.float32)


def cnn_embedding(stacks25d: np.ndarray, registry) -> np.ndarray:
    from src.inference.registry import _down64
    net = registry.models[registry.selected_cnn].m
    return net.embed(_down64(np.asarray(stacks25d, np.uint8)).astype(np.float32) / 255.0)


def extract(name: str, patches2d: np.ndarray, stacks25d: np.ndarray | None = None, registry=None) -> np.ndarray:
    if name == "radiomics":
        return radiomics(patches2d)
    if name == "imagenet_gap":
        return imagenet_gap(patches2d)
    if name == "cnn_embedding":
        return cnn_embedding(stacks25d, registry)
    raise ValueError(f"unknown feature set {name!r}")


def cached(name: str, patches2d: np.ndarray, cache_dir: Path, stacks25d=None, registry=None, log=print) -> np.ndarray:
    """Extract once per (feature set, exact patch bytes); cache as .npz keyed by a SHA-256 of the inputs."""
    h = hashlib.sha256(np.ascontiguousarray(patches2d).tobytes()).hexdigest()[:16]
    p = Path(cache_dir) / f"{name}_{h}.npz"
    if p.exists():
        return np.load(p)["X"]
    log(f"extracting {name} for {len(patches2d)} patches")
    X = extract(name, patches2d, stacks25d, registry)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=X)
    return X
