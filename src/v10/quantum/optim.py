"""Adam optimiser on flat parameter vectors."""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, n: int, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0

    def step(self, params: np.ndarray, grad: np.ndarray, lr: float | None = None) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return params - (lr if lr is not None else self.lr) * mh / (np.sqrt(vh) + self.eps)


def cosine_lr(base: float, epoch: int, epochs: int, floor: float = 0.1) -> float:
    return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * epoch / max(1, epochs))))


def sigmoid(z):
    z = np.asarray(z, float)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n, pos = len(y), max(1, int(y.sum()))
    neg = max(1, n - pos)
    return np.where(y == 1, n / (2 * pos), n / (2 * neg))
