"""Grad-CAM (Selvaraju et al., 2017) for the NumPy CNNs.

The class-activation map for the tumour logit is ReLU(sum_c alpha_c A_c), with A the activations of a chosen
convolutional block and alpha_c the spatial mean of d(logit)/dA_c. Computed in inference mode (batch-norm running
statistics, no dropout); gradients flow only through the layers after the chosen block.

A heat map shows which pixels influenced this model's score. It does not show that the model is medically right,
and at the 8 x 8 resolution of the last block it is coarse.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.classical.numpy_cnn import MaxPool2


def gradcam(net, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X (N, C, 64, 64) in [0, 1] -> (cams (N, 64, 64) in [0, 1], logits (N,)).

    Uses the last convolutional block (before the final max-pool): the layers after it (max-pool, global pooling,
    dropout, dense, ReLU) have exact inference-mode backward rules, so the gradients are exact."""
    pools = [i for i, l in enumerate(net.layers) if isinstance(l, MaxPool2)]
    k = pools[-1]
    h = net._norm(X.astype(np.float32))
    for layer in net.layers[:k]:
        h = layer.forward(h, False)
    A = h
    z = A
    rest = net.layers[k:]
    for layer in rest:
        z = layer.forward(z, isinstance(layer, MaxPool2))  # MaxPool2 caches its routing only in train mode
    g = np.ones_like(z)
    for layer in reversed(rest):
        g = layer.backward(g)
    alpha = g.mean((2, 3), keepdims=True)
    cam = np.maximum((alpha * A).sum(1), 0)
    out = np.zeros((len(X), X.shape[-2], X.shape[-1]), np.float32)
    for i, c in enumerate(cam):
        m = c.max()
        c = c / m if m > 0 else c
        out[i] = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize(out.shape[1:][::-1], Image.BILINEAR)) / 255.0
    return out, z.reshape(-1)


def heat_overlay(gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """gray (H, W) uint8, cam (h, w) in [0, 1] -> RGB uint8 with a red-yellow heat overlay."""
    H, W = gray.shape
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
    base = np.stack([gray] * 3, -1).astype(np.float32)
    heat = np.stack([255 * np.clip(2 * c, 0, 1), 255 * np.clip(2 * c - 1, 0, 1), np.zeros_like(c)], -1)
    a = 0.55 * c[..., None]
    return np.clip(base * (1 - a) + heat * a, 0, 255).astype(np.uint8)
