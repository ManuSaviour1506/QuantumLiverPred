"""A small convolutional neural network implemented in NumPy (forward and backward passes).

Why NumPy: PyTorch and TensorFlow could not be installed in the environment where these experiments were run
(no package index access). This network therefore has no ImageNet pre-training; the pretrained ResNet-50 path
(`src/models/cnn.py`, `scripts/train_cnn_mac.py`) is the one to use wherever PyTorch is available.

Architecture (configurable):

    input (C x S x S, standardised with training-set mean/std)
    [Conv3x3 -> BatchNorm -> ReLU] x convs_per_block -> MaxPool2   for each entry of `filters`
    GlobalAveragePool -> Dropout -> Dense(hidden) -> ReLU -> Dropout -> Dense(1) -> sigmoid

Training: class-weighted binary cross-entropy, AdamW (decoupled weight decay on weights only), linear warm-up then
cosine learning-rate decay, on-the-fly augmentation (random shift / small rotation / intensity jitter; flips off by
default because left-right flips swap the liver's side), early stopping on validation loss with the best-validation
parameters restored. Every gradient is checked against finite differences in tests/test_numpy_cnn.py.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DT = np.float32


# ----------------------------------------------------------------------------------------------- layers
class Layer:
    trainable: tuple[str, ...] = ()
    decay: tuple[str, ...] = ()

    def forward(self, x: np.ndarray, train: bool) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, g: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class Conv2D(Layer):
    """3x3 (odd k) convolution, stride 1, 'same' zero padding, via im2col."""
    trainable = ("W", "b")
    decay = ("W",)

    def __init__(self, cin: int, cout: int, k: int, rng: np.random.Generator, dtype=DT):
        if k % 2 != 1:
            raise ValueError("kernel size must be odd")
        self.cin, self.cout, self.k = cin, cout, k
        self.W = (rng.standard_normal((cin * k * k, cout)) * np.sqrt(2.0 / (cin * k * k))).astype(dtype)  # He init
        self.b = np.zeros(cout, dtype)

    def forward(self, x, train):
        n, c, h, w = x.shape
        k, p = self.k, self.k // 2
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        cols = sliding_window_view(xp, (k, k), axis=(2, 3)).transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, c * k * k)
        self._cache = (x.shape, cols if train else None)
        out = cols @ self.W + self.b
        return out.reshape(n, h, w, self.cout).transpose(0, 3, 1, 2)

    def backward(self, g):
        (n, c, h, w), cols = self._cache
        k, p = self.k, self.k // 2
        gf = g.transpose(0, 2, 3, 1).reshape(n * h * w, self.cout)
        self.dW = cols.T @ gf
        self.db = gf.sum(0)
        dcols = (gf @ self.W.T).reshape(n, h, w, c, k, k)
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p), dtype=g.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + h, j:j + w] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + h, p:p + w]


class BatchNorm2D(Layer):
    trainable = ("gamma", "beta")

    def __init__(self, c: int, momentum: float = 0.1, eps: float = 1e-5, dtype=DT):
        self.gamma, self.beta = np.ones(c, dtype), np.zeros(c, dtype)
        self.running_mean, self.running_var = np.zeros(c, dtype), np.ones(c, dtype)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, train):
        if train:
            mu, var = x.mean((0, 2, 3)), x.var((0, 2, 3))
            m = x.shape[0] * x.shape[2] * x.shape[3]
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mu).astype(x.dtype)
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var * m / max(m - 1, 1)).astype(x.dtype)
        else:
            mu, var = self.running_mean, self.running_var
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu[None, :, None, None]) * inv[None, :, None, None]
        self._cache = (xhat, inv)
        return self.gamma[None, :, None, None] * xhat + self.beta[None, :, None, None]

    def backward(self, g):
        xhat, inv = self._cache
        ax = (0, 2, 3)
        m = g.shape[0] * g.shape[2] * g.shape[3]
        self.dgamma = (g * xhat).sum(ax)
        self.dbeta = g.sum(ax)
        dxhat = g * self.gamma[None, :, None, None]
        return (inv[None, :, None, None] / m) * (m * dxhat - dxhat.sum(ax)[None, :, None, None]
                                                  - xhat * (dxhat * xhat).sum(ax)[None, :, None, None])


class ReLU(Layer):
    def forward(self, x, train):
        self._mask = x > 0
        return x * self._mask

    def backward(self, g):
        return g * self._mask


class MaxPool2(Layer):
    def forward(self, x, train):
        n, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(f"MaxPool2 needs even spatial size, got {h}x{w}")
        xr = x.reshape(n, c, h // 2, 2, w // 2, 2)
        out = xr.max((3, 5))
        if train:
            mask = xr == out[:, :, :, None, :, None]
            self._mask = mask / mask.sum((3, 5), keepdims=True)  # ties share the gradient
            self._shape = x.shape
        return out

    def backward(self, g):
        return (g[:, :, :, None, :, None] * self._mask).reshape(self._shape).astype(g.dtype)


class GlobalAvgPool(Layer):
    def forward(self, x, train):
        self._shape = x.shape
        return x.mean((2, 3))

    def backward(self, g):
        n, c, h, w = self._shape
        return np.broadcast_to(g[:, :, None, None] / (h * w), self._shape).astype(g.dtype)


class Dropout(Layer):
    def __init__(self, rate: float, rng: np.random.Generator):
        self.rate, self.rng = float(rate), rng

    def forward(self, x, train):
        if not train or self.rate <= 0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) >= self.rate).astype(x.dtype) / (1.0 - self.rate)
        return x * self._mask

    def backward(self, g):
        return g if self._mask is None else g * self._mask


class Dense(Layer):
    trainable = ("W", "b")
    decay = ("W",)

    def __init__(self, din: int, dout: int, rng: np.random.Generator, dtype=DT, gain: float = 2.0):
        self.W = (rng.standard_normal((din, dout)) * np.sqrt(gain / din)).astype(dtype)
        self.b = np.zeros(dout, dtype)

    def forward(self, x, train):
        self._x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW = self._x.T @ g
        self.db = g.sum(0)
        return g @ self.W.T


class SEBlock(Layer):
    """V9: squeeze-and-excitation (Hu et al., 2018): channel weights from GAP -> FC -> ReLU -> FC -> sigmoid."""

    def __init__(self, c: int, rng: np.random.Generator, reduction: int = 4):
        r = max(c // reduction, 4)
        self.fc1, self.act, self.fc2 = Dense(c, r, rng), ReLU(), Dense(r, c, rng, gain=1.0)

    @property
    def sublayers(self):
        return [self.fc1, self.fc2]

    def forward(self, x, train):
        self._x = x
        z = self.fc2.forward(self.act.forward(self.fc1.forward(x.mean((2, 3)), train), train), train)
        self._s = (1.0 / (1.0 + np.exp(-z))).astype(x.dtype)
        return x * self._s[:, :, None, None]

    def backward(self, g):
        x, s = self._x, self._s
        ds = (g * x).sum((2, 3))
        dz = ds * s * (1 - s)
        dsq = self.fc1.backward(self.act.backward(self.fc2.backward(dz)))
        h, w = x.shape[2], x.shape[3]
        return (g * s[:, :, None, None] + dsq[:, :, None, None] / (h * w)).astype(g.dtype)


class Conv3DStem(Layer):
    """V9: 3x3x3 convolution over a (depth, H, W) slice stack, 'same' padding, then mean over depth.

    Input (N, D, H, W) with the D adjacent slices as depth; output (N, cout, H, W). The depth average after a true
    3-D convolution lets a 2-D network follow; it is the 3-D stem of the '3-D CNN' variant.
    """
    trainable = ("W", "b")
    decay = ("W",)

    def __init__(self, depth: int, cout: int, rng: np.random.Generator, dtype=DT):
        self.depth, self.cout = depth, cout
        self.W = (rng.standard_normal((27, cout)) * np.sqrt(2.0 / 27)).astype(dtype)
        self.b = np.zeros(cout, dtype)

    def forward(self, x, train):
        n, d, h, w = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (1, 1)))
        cols = sliding_window_view(xp, (3, 3, 3), axis=(1, 2, 3)).reshape(n * d * h * w, 27)
        self._cache = (x.shape, cols if train else None)
        out = (cols @ self.W + self.b).reshape(n, d, h, w, self.cout)
        return out.mean(1).transpose(0, 3, 1, 2)

    def backward(self, g):
        (n, d, h, w), cols = self._cache
        gf = np.broadcast_to((g.transpose(0, 2, 3, 1) / d)[:, None], (n, d, h, w, self.cout)).reshape(-1, self.cout)
        self.dW = cols.T @ gf
        self.db = gf.sum(0)
        dcols = (gf @ self.W.T).reshape(n, d, h, w, 3, 3, 3)
        dxp = np.zeros((n, d + 2, h + 2, w + 2), dtype=g.dtype)
        for a in range(3):
            for i in range(3):
                for j in range(3):
                    dxp[:, a:a + d, i:i + h, j:j + w] += dcols[..., a, i, j]
        return dxp[:, 1:1 + d, 1:1 + h, 1:1 + w]


class ResidualBlock(Layer):
    """V6: conv3x3-BN-ReLU-conv3x3-BN + shortcut (identity, or conv1x1-BN when the width changes), then ReLU."""

    def __init__(self, cin: int, cout: int, rng: np.random.Generator, se: bool = False):
        self.sub = [Conv2D(cin, cout, 3, rng), BatchNorm2D(cout), ReLU(), Conv2D(cout, cout, 3, rng), BatchNorm2D(cout)]
        if se:  # V9: squeeze-and-excitation on the residual branch
            self.sub.append(SEBlock(cout, rng))
        self.proj = [Conv2D(cin, cout, 1, rng), BatchNorm2D(cout)] if cin != cout else []
        self.out = ReLU()

    @property
    def sublayers(self):
        out = []
        for layer in self.sub + self.proj:
            out += getattr(layer, "sublayers", [layer])
        return out

    def forward(self, x, train):
        h = x
        for layer in self.sub:
            h = layer.forward(h, train)
        s = x
        for layer in self.proj:
            s = layer.forward(s, train)
        return self.out.forward(h + s, train)

    def backward(self, g):
        g = self.out.backward(g)
        gh = g
        for layer in reversed(self.sub):
            gh = layer.backward(gh)
        gs = g
        for layer in reversed(self.proj):
            gs = layer.backward(gs)
        return gh + gs


# ----------------------------------------------------------------------------------------------- config
@dataclass
class CNNConfig:
    in_channels: int = 1
    image_size: int = 64
    filters: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    convs_per_block: int = 1
    hidden: int = 64
    dropout: float = 0.3
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 30
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.05
    patience: int = 6
    class_weight: str | None = "balanced"
    augment_shift: int = 4          # pixels
    augment_rotate: float = 10.0    # degrees
    augment_intensity: float = 0.1  # +/- relative contrast and brightness
    augment_flip: bool = False
    ema_decay: float = 0.0          # >0: evaluate/keep an exponential moving average of the weights (per step)
    seed: int = 42
    residual: bool = False          # V6: residual blocks (2 convs + shortcut) instead of plain conv blocks
    grad_clip: float = 0.0          # V6: clip the global gradient norm (0 = off)
    schedule_epochs: int | None = None  # V6: LR-schedule horizon (default max_epochs); lets a fixed-epoch refit
                                        # follow exactly the schedule used during early stopping
    se: bool = False                # V9: squeeze-and-excitation inside residual blocks
    stem3d: int = 0                 # V9: >0 -> 3x3x3 convolutional stem with this many filters over the input slices
    focal_gamma: float = 0.0        # V9: focal loss (Lin et al., 2017); 0 = class-weighted cross-entropy
    augment_gamma: float = 0.0      # V9: random gamma in [1-g, 1+g] (CT window / contrast variation)
    augment_noise: float = 0.0      # V9: Gaussian noise std (image scale 0..1), drawn per sample in [0, value]
    augment_blur: float = 0.0       # V9: probability of a Gaussian blur (sigma 0.5-1.2; reconstruction-kernel variation)
    early_stop: str = "loss"        # V9: 'loss' (V8 behaviour) or 'auc' (validation ROC-AUC, ties broken by loss)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------------------------- network
class NumpyCNN:
    def __init__(self, cfg: CNNConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        self.layers: list[Layer] = []
        cin, size = cfg.in_channels, cfg.image_size
        if cfg.stem3d:
            self.layers += [Conv3DStem(cfg.in_channels, cfg.stem3d, rng), BatchNorm2D(cfg.stem3d), ReLU()]
            cin = cfg.stem3d
        for f in cfg.filters:
            if cfg.residual:
                self.layers.append(ResidualBlock(cin, f, rng, se=cfg.se))
                cin = f
            else:
                for _ in range(cfg.convs_per_block):
                    self.layers += [Conv2D(cin, f, 3, rng), BatchNorm2D(f), ReLU()]
                    cin = f
            self.layers.append(MaxPool2())
            size //= 2
        if size < 1:
            raise ValueError("too many pooling stages for the image size")
        self.layers += [GlobalAvgPool(), Dropout(cfg.dropout, rng), Dense(cin, cfg.hidden, rng), ReLU()]
        self.embed_index = len(self.layers)  # activations after this ReLU are the embedding
        self.layers += [Dropout(cfg.dropout, rng), Dense(cfg.hidden, 1, rng, gain=1.0)]
        self.mean = np.zeros(cfg.in_channels, DT)
        self.std = np.ones(cfg.in_channels, DT)
        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.train_seconds: float | None = None
        self._adam_t = 0
        self._m: dict = {}
        self._v: dict = {}
        self._ema: dict = {}

    # -------------------------------------------------------------------------------- core passes
    def _norm(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean[None, :, None, None]) / self.std[None, :, None, None]).astype(DT)

    def forward(self, X: np.ndarray, train: bool, upto: int | None = None) -> np.ndarray:
        h = X
        for layer in self.layers[: upto if upto is not None else len(self.layers)]:
            h = layer.forward(h, train)
        return h

    def backward(self, g: np.ndarray) -> None:
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def _flat(self):
        """(state-key prefix, layer) for every leaf layer, including those inside residual blocks."""
        for li, layer in enumerate(self.layers):
            subs = getattr(layer, "sublayers", None)
            if subs is None:
                yield f"{li}", layer
            else:
                for si, sl in enumerate(subs):
                    yield f"{li}.{si}", sl

    def params(self):
        for pre, layer in self._flat():
            for name in layer.trainable:
                yield (pre, name), layer, name, "d" + name, name in layer.decay

    @staticmethod
    def focal_loss_and_grad(logits: np.ndarray, y: np.ndarray, w: np.ndarray, gamma: float) -> tuple[float, np.ndarray]:
        """Class-weighted focal loss and its gradient with respect to the logits (V9)."""
        z = logits.reshape(-1).astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-z))
        lp, l1p = -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z)  # log p, log(1-p)
        l_pos = -((1 - p) ** gamma) * lp
        l_neg = -(p ** gamma) * l1p
        loss = float(np.sum(w * np.where(y == 1, l_pos, l_neg)) / np.sum(w))
        g_pos = gamma * (1 - p) ** gamma * p * lp - (1 - p) ** (gamma + 1)
        g_neg = p ** (gamma + 1) - gamma * p ** gamma * (1 - p) * l1p
        g = (w * np.where(y == 1, g_pos, g_neg) / np.sum(w)).astype(logits.dtype).reshape(logits.shape)
        return loss, g

    @staticmethod
    def loss_and_grad(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
        z = logits.reshape(-1).astype(np.float64)
        sp = np.logaddexp(0.0, z)  # softplus, stable
        loss = float(np.sum(w * (sp - y * z)) / np.sum(w))
        p = 1.0 / (1.0 + np.exp(-z))
        g = (w * (p - y) / np.sum(w)).astype(logits.dtype).reshape(logits.shape)
        return loss, g

    # -------------------------------------------------------------------------------- optimiser
    def _adam_step(self, lr: float, b1=0.9, b2=0.999, eps=1e-8) -> None:
        self._adam_t += 1
        t = self._adam_t
        scale = 1.0
        if self.cfg.grad_clip:
            gn = float(np.sqrt(sum(float((getattr(l_, g_).astype(np.float64) ** 2).sum()) for _, l_, _, g_, _ in self.params())))
            self.last_grad_norm = gn
            scale = min(1.0, self.cfg.grad_clip / (gn + 1e-12))
        for key, layer, name, gname, decay in self.params():
            p, g = getattr(layer, name), getattr(layer, gname).astype(DT) * DT(scale)
            m = self._m.get(key, np.zeros_like(p))
            v = self._v.get(key, np.zeros_like(p))
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            self._m[key], self._v[key] = m, v
            step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            if decay and self.cfg.weight_decay:
                step = step + lr * self.cfg.weight_decay * p
            setattr(layer, name, (p - step).astype(DT))
        if self.cfg.ema_decay > 0:
            d = self.cfg.ema_decay
            for key, layer, name, _, _ in self.params():
                cur = getattr(layer, name)
                self._ema[key] = cur.copy() if key not in self._ema else (d * self._ema[key] + (1 - d) * cur).astype(DT)

    def _swap_ema(self) -> None:
        """Exchange live and EMA weights (no-op without EMA)."""
        if self.cfg.ema_decay <= 0 or not self._ema:
            return
        for key, layer, name, _, _ in self.params():
            live = getattr(layer, name)
            setattr(layer, name, self._ema[key])
            self._ema[key] = live

    def _lr(self, epoch: int, step: int, steps_per_epoch: int) -> float:
        c = self.cfg
        e = epoch + step / max(steps_per_epoch, 1)
        if e < c.warmup_epochs:
            return c.learning_rate * (e + 1e-3) / c.warmup_epochs
        horizon = c.schedule_epochs or c.max_epochs
        prog = (e - c.warmup_epochs) / max(horizon - c.warmup_epochs, 1)
        return c.learning_rate * (c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0))))

    # -------------------------------------------------------------------------------- augmentation
    def _augment(self, xb: np.ndarray) -> np.ndarray:
        import cv2

        c = self.cfg
        out = np.empty_like(xb)
        s = xb.shape[-1]
        for i in range(len(xb)):
            img = xb[i]
            ang = self.rng.uniform(-c.augment_rotate, c.augment_rotate) if c.augment_rotate else 0.0
            tx, ty = (self.rng.integers(-c.augment_shift, c.augment_shift + 1, 2) if c.augment_shift else (0, 0))
            M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, 1.0)
            M[:, 2] += (tx, ty)
            warped = np.stack([cv2.warpAffine(ch, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=float(ch.min())) for ch in img])
            if c.augment_flip and self.rng.random() < 0.5:
                warped = warped[:, :, ::-1]
            if c.augment_gamma:
                gm = 1 + self.rng.uniform(-c.augment_gamma, c.augment_gamma)
                warped = np.clip(warped, 0, None) ** gm
            if c.augment_blur and self.rng.random() < c.augment_blur:
                sig = self.rng.uniform(0.5, 1.2)
                warped = np.stack([cv2.GaussianBlur(ch, (0, 0), sig) for ch in warped])
            if c.augment_intensity:
                a = 1 + self.rng.uniform(-c.augment_intensity, c.augment_intensity)
                b = self.rng.uniform(-c.augment_intensity, c.augment_intensity) * 0.5
                warped = warped * a + b
            if c.augment_noise:
                warped = warped + self.rng.normal(0, self.rng.uniform(0, c.augment_noise), warped.shape)
            out[i] = warped
        return out

    # -------------------------------------------------------------------------------- training
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            log=None) -> "NumpyCNN":
        """X: (N, C, S, S) float in [0, 1]. Validation data drive early stopping and nothing else."""
        c = self.cfg
        X = X.astype(DT)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 4 or X.shape[1] != c.in_channels or X.shape[2] != c.image_size:
            raise ValueError(f"expected (N, {c.in_channels}, {c.image_size}, {c.image_size}), got {X.shape}")
        self.mean = X.mean((0, 2, 3)).astype(DT)
        self.std = (X.std((0, 2, 3)) + 1e-6).astype(DT)
        if c.class_weight == "balanced":
            pos = y.mean()
            cw = {1: 0.5 / max(pos, 1e-6), 0: 0.5 / max(1 - pos, 1e-6)}
        else:
            cw = {0: 1.0, 1: 1.0}
        n = len(X)
        steps = int(np.ceil(n / c.batch_size))
        best_loss, best_state, wait, best_key = np.inf, None, 0, None
        t0 = time.perf_counter()
        for epoch in range(c.max_epochs):
            order = self.rng.permutation(n)
            tr_loss, seen = 0.0, 0
            for step in range(steps):
                idx = order[step * c.batch_size:(step + 1) * c.batch_size]
                if len(idx) < 2:
                    continue
                xb = self._augment(X[idx]) if (c.augment_shift or c.augment_rotate or c.augment_intensity or c.augment_flip
                                              or c.augment_gamma or c.augment_noise or c.augment_blur) else X[idx]
                yb = y[idx]
                wb = np.where(yb == 1, cw[1], cw[0])
                logits = self.forward(self._norm(xb), train=True)
                if c.focal_gamma:
                    loss, g = self.focal_loss_and_grad(logits, yb, wb, c.focal_gamma)
                else:
                    loss, g = self.loss_and_grad(logits, yb, wb)
                self.backward(g)
                self._adam_step(self._lr(epoch, step, steps))
                tr_loss += loss * len(idx)
                seen += len(idx)
            rec = {"epoch": epoch + 1, "train_loss": tr_loss / max(seen, 1), "lr": self._lr(epoch, steps, steps),
                   "grad_norm": getattr(self, "last_grad_norm", None),
                   "seconds": time.perf_counter() - t0}
            self._swap_ema()  # evaluate (and checkpoint) the averaged weights
            p_tr = self.predict_proba(X)
            rec.update(train_acc=float(((p_tr >= 0.5) == y).mean()))
            if X_val is not None:
                pv = self.predict_proba(X_val)
                yv = np.asarray(y_val, dtype=np.float64)
                wv = np.where(yv == 1, cw[1], cw[0])
                zv = np.log(np.clip(pv, 1e-7, 1 - 1e-7) / np.clip(1 - pv, 1e-7, 1))
                rec["val_loss"] = self.loss_and_grad(zv, yv, wv)[0]
                rec["val_acc"] = float(((pv >= 0.5) == yv).mean())
                try:
                    from sklearn.metrics import roc_auc_score

                    rec["val_roc_auc"] = float(roc_auc_score(yv, pv)) if len(set(yv)) > 1 else None
                except Exception:
                    rec["val_roc_auc"] = None
                if c.early_stop == "auc" and rec["val_roc_auc"] is not None:
                    key = (rec["val_roc_auc"], -rec["val_loss"])
                    better = best_key is None or key > best_key
                else:
                    key, better = None, rec["val_loss"] < best_loss - 1e-4
                if better:
                    best_loss, best_state, wait, best_key = rec["val_loss"], self.get_state(), 0, key
                    self.best_epoch = epoch + 1
                else:
                    wait += 1
            self._swap_ema()  # back to live weights for the next epoch
            self.history.append(rec)
            if log:
                log(rec)
            if X_val is not None and wait >= c.patience:
                break
        if best_state is not None:
            self.set_state(best_state)
        else:
            self._swap_ema()  # no validation: keep the averaged weights of the last epoch
            if X_val is None:
                self.best_epoch = len(self.history)
        self.train_seconds = time.perf_counter() - t0
        return self

    # -------------------------------------------------------------------------------- inference
    def _batched(self, X: np.ndarray, upto: int | None, bs: int = 128) -> np.ndarray:
        outs = [self.forward(self._norm(X[i:i + bs].astype(DT)), train=False, upto=upto) for i in range(0, len(X), bs)]
        return np.concatenate(outs) if outs else np.zeros((0,))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._batched(X, None).reshape(-1).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class, shape (N,)."""
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Hidden-layer activations (N, hidden) used as CNN features for the classical and quantum models."""
        return self._batched(X, self.embed_index).astype(np.float64)

    # -------------------------------------------------------------------------------- persistence
    def get_state(self) -> dict[str, np.ndarray]:
        st = {"mean": self.mean.copy(), "std": self.std.copy()}
        for pre, layer in self._flat():
            for name in layer.trainable:
                st[f"{pre}.{name}"] = getattr(layer, name).copy()
            if isinstance(layer, BatchNorm2D):
                st[f"{pre}.running_mean"] = layer.running_mean.copy()
                st[f"{pre}.running_var"] = layer.running_var.copy()
        return st

    def set_state(self, st: dict[str, np.ndarray]) -> None:
        self.mean, self.std = st["mean"].astype(DT), st["std"].astype(DT)
        for pre, layer in self._flat():
            for name in layer.trainable:
                setattr(layer, name, st[f"{pre}.{name}"].astype(DT))
            if isinstance(layer, BatchNorm2D):
                layer.running_mean = st[f"{pre}.running_mean"].astype(DT)
                layer.running_var = st[f"{pre}.running_var"].astype(DT)

    def n_parameters(self) -> int:
        return int(sum(getattr(layer, name).size for _, layer, name, _, _ in self.params()))

    def describe(self) -> list[str]:
        rows, size, cin = [], self.cfg.image_size, self.cfg.in_channels
        rows.append(f"input {cin}x{size}x{size}")
        if self.cfg.stem3d:
            rows.append(f"3-D conv 3x3x3 over the {cin} slices -> {self.cfg.stem3d} maps, mean over depth, BatchNorm + ReLU")
            cin = self.cfg.stem3d
        for f in self.cfg.filters:
            if self.cfg.residual:
                rows.append(f"residual block {cin}->{f}: 2x(conv3x3+BN){' + squeeze-excitation' if self.cfg.se else ''}, shortcut {'identity' if cin == f else 'conv1x1+BN'}, ReLU ({size}x{size})")
                cin = f
            for _ in range(0 if self.cfg.residual else self.cfg.convs_per_block):
                rows.append(f"conv3x3 {cin}->{f} + BatchNorm + ReLU ({size}x{size})")
                cin = f
            size //= 2
            rows.append(f"maxpool 2x2 -> {f}x{size}x{size}")
        rows += [f"global average pool -> {cin}", f"dropout {self.cfg.dropout}", f"dense {cin}->{self.cfg.hidden} + ReLU (embedding)",
                 f"dropout {self.cfg.dropout}", f"dense {self.cfg.hidden}->1 + sigmoid"]
        return rows

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"config": self.cfg.to_dict(), "history": self.history, "best_epoch": self.best_epoch,
                "train_seconds": self.train_seconds, "n_parameters": self.n_parameters(), "architecture": self.describe(),
                "format": "numpy_cnn_v1"}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **self.get_state())

    @staticmethod
    def load(path: str | Path) -> "NumpyCNN":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != "numpy_cnn_v1":
            raise ValueError(f"{path} is not a numpy_cnn_v1 checkpoint")
        m = NumpyCNN(CNNConfig(**meta["config"]))
        m.set_state({k: z[k] for k in z.files if k != "meta"})
        m.history, m.best_epoch, m.train_seconds = meta["history"], meta["best_epoch"], meta["train_seconds"]
        return m
