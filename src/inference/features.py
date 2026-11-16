"""Shared feature extractors used at inference (identical code path to training)."""
from __future__ import annotations

import functools

import numpy as np

from src.classical import backbones as B
from src.classical.nn import Tensor
from src.preprocessing.ct import to_rgb


@functools.lru_cache(maxsize=4)
def backbone(arch: str):
    return B.build(arch)


def imagenet_features(gray_u8: np.ndarray, arch: str = "efficientnet_b0", bs: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """ImageNet (not fine-tuned) GAP features and intermediate-stage GAP features of grey uint8 images (N, H, W)."""
    net = backbone(arch)
    F, M = [], []
    for s in range(0, len(gray_u8), bs):
        h = net.frozen(B.Backbone.preprocess(to_rgb(gray_u8[s:s + bs])))
        M.append(net._mid.astype(np.float32))
        F.append(net.tail(Tensor(h))["final"].data.mean((1, 2)).astype(np.float32))
    return np.concatenate(F), np.concatenate(M)
