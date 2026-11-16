"""Minimal tape-based autodiff for CNN layers in NumPy (NHWC, float32).

Used for (a) inference of ImageNet-pretrained backbones and (b) fine-tuning of their final stages, where only a
few layers need gradients. Operations record a backward closure on the active tape only when gradients are
enabled, so inference has no overhead beyond the forward computation.

Correctness of every backward rule is tested numerically in tests/test_v6.py.
"""
from __future__ import annotations

import contextlib

import numpy as np

_TAPE: list | None = None


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "name")

    def __init__(self, data, requires_grad=False, name=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self.name = name

    @property
    def shape(self):
        return self.data.shape


def param(arr, name="") -> Tensor:
    return Tensor(np.array(arr, dtype=np.float32), requires_grad=True, name=name)


@contextlib.contextmanager
def tape():
    """Record operations; call backward(loss) inside or after the block."""
    global _TAPE
    old, _TAPE = _TAPE, []
    try:
        yield _TAPE
    finally:
        _TAPE = old if old is not None else None


def _needs(*ts):
    return _TAPE is not None and any(isinstance(t, Tensor) and t.requires_grad for t in ts)


def _acc(t: Tensor, g):
    if t.requires_grad:
        t.grad = g if t.grad is None else t.grad + g


def _record(out: Tensor, fn):
    out.requires_grad = True
    _TAPE.append((out, fn))


def backward(loss: Tensor, tp: list):
    loss.grad = np.ones_like(loss.data)
    for out, fn in reversed(tp):
        if out.grad is not None:
            fn(out.grad)


# ----------------------------------------------------------------------------------------------- convolution
def _pad(x, p):
    if p == 0:
        return x
    return np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))


def _im2col(xp, kh, kw, s, Ho, Wo):
    N, _, _, C = xp.shape
    cols = np.empty((N, Ho, Wo, kh, kw, C), dtype=xp.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, i, j, :] = xp[:, i:i + s * Ho:s, j:j + s * Wo:s, :]
    return cols.reshape(N * Ho * Wo, kh * kw * C)


def conv2d(x: Tensor, w: Tensor, b: Tensor | None = None, stride=1, pad=0) -> Tensor:
    """Standard convolution. w is stored PyTorch-style (O, C, kh, kw)."""
    X, W = x.data, w.data
    O, C, kh, kw = W.shape
    N, H, Wd, _ = X.shape
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kw) // stride + 1
    Wm = W.transpose(2, 3, 1, 0).reshape(kh * kw * C, O)
    if kh == 1 and kw == 1 and pad == 0:
        xs = X[:, ::stride, ::stride, :] if stride > 1 else X
        cols = xs.reshape(-1, C)
    else:
        cols = _im2col(_pad(X, pad), kh, kw, stride, Ho, Wo)
    out = cols @ Wm
    if b is not None:
        out += b.data
    y = Tensor(out.reshape(N, Ho, Wo, O))
    if _needs(x, w, b):
        def fn(g):
            gf = g.reshape(-1, O)
            if w.requires_grad:
                _acc(w, (cols.T @ gf).reshape(kh, kw, C, O).transpose(3, 2, 0, 1))
            if b is not None and b.requires_grad:
                _acc(b, gf.sum(0))
            if x.requires_grad:
                dcols = gf @ Wm.T
                if kh == 1 and kw == 1 and pad == 0:
                    dx = np.zeros_like(X)
                    if stride > 1:
                        dx[:, ::stride, ::stride, :] = dcols.reshape(N, Ho, Wo, C)
                    else:
                        dx = dcols.reshape(X.shape)
                else:
                    dcols = dcols.reshape(N, Ho, Wo, kh, kw, C)
                    dxp = np.zeros((N, H + 2 * pad, Wd + 2 * pad, C), np.float32)
                    for i in range(kh):
                        for j in range(kw):
                            dxp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride, :] += dcols[:, :, :, i, j, :]
                    dx = dxp[:, pad:pad + H, pad:pad + Wd, :] if pad else dxp
                _acc(x, dx)
        _record(y, fn)
    return y


def depthwise_conv2d(x: Tensor, w: Tensor, stride=1, pad=0) -> Tensor:
    """Depthwise convolution; w stored PyTorch-style (C, 1, kh, kw)."""
    X = x.data
    C, _, kh, kw = w.data.shape
    N, H, Wd, _ = X.shape
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kw) // stride + 1
    xp = _pad(X, pad)
    wk = w.data[:, 0].transpose(1, 2, 0)  # kh, kw, C
    out = np.zeros((N, Ho, Wo, C), np.float32)
    for i in range(kh):
        for j in range(kw):
            out += xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride, :] * wk[i, j]
    y = Tensor(out)
    if _needs(x, w):
        def fn(g):
            if w.requires_grad:
                dw = np.empty((kh, kw, C), np.float32)
                for i in range(kh):
                    for j in range(kw):
                        dw[i, j] = (xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride, :] * g).sum((0, 1, 2))
                _acc(w, dw.transpose(2, 0, 1)[:, None])
            if x.requires_grad:
                dxp = np.zeros_like(xp)
                for i in range(kh):
                    for j in range(kw):
                        dxp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride, :] += g * wk[i, j]
                _acc(x, dxp[:, pad:pad + H, pad:pad + Wd, :] if pad else dxp)
        _record(y, fn)
    return y


# ----------------------------------------------------------------------------------------------- normalisation
def batchnorm(x: Tensor, gamma: Tensor, beta: Tensor, mean: np.ndarray, var: np.ndarray, eps=1e-5) -> Tensor:
    """Batch norm with FROZEN running statistics (eval-mode BN). gamma/beta may be trainable.

    Fine-tuning with frozen BN statistics is the usual choice for small batches and small datasets."""
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x.data - mean) * inv
    y = Tensor(xh * gamma.data + beta.data)
    if _needs(x, gamma, beta):
        def fn(g):
            if gamma.requires_grad:
                _acc(gamma, (g * xh).reshape(-1, xh.shape[-1]).sum(0))
            if beta.requires_grad:
                _acc(beta, g.reshape(-1, g.shape[-1]).sum(0))
            if x.requires_grad:
                _acc(x, g * (gamma.data * inv))
        _record(y, fn)
    return y


def batchnorm_train(x: Tensor, gamma: Tensor, beta: Tensor, eps=1e-5) -> Tensor:
    """Batch norm using batch statistics (training-mode BN for 2-D inputs (N, F) or 4-D NHWC)."""
    axes = tuple(range(x.data.ndim - 1))
    mu = x.data.mean(axes)
    var = x.data.var(axes)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x.data - mu) * inv
    y = Tensor(xh * gamma.data + beta.data)
    m = int(np.prod([x.data.shape[a] for a in axes]))
    if _needs(x, gamma, beta):
        def fn(g):
            if gamma.requires_grad:
                _acc(gamma, (g * xh).sum(axes))
            if beta.requires_grad:
                _acc(beta, g.sum(axes))
            if x.requires_grad:
                gxh = g * gamma.data
                _acc(x, inv / m * (m * gxh - gxh.sum(axes) - xh * (gxh * xh).sum(axes)))
        _record(y, fn)
    return y, mu, var


# ----------------------------------------------------------------------------------------------- activations
def relu(x: Tensor) -> Tensor:
    y = Tensor(np.maximum(x.data, 0))
    if _needs(x):
        _record(y, lambda g: _acc(x, g * (x.data > 0)))
    return y


def sigmoid_np(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def silu(x: Tensor) -> Tensor:
    s = sigmoid_np(x.data)
    y = Tensor(x.data * s)
    if _needs(x):
        _record(y, lambda g: _acc(x, g * (s * (1 + x.data * (1 - s)))))
    return y


def sigmoid(x: Tensor) -> Tensor:
    s = sigmoid_np(x.data)
    y = Tensor(s)
    if _needs(x):
        _record(y, lambda g: _acc(x, g * s * (1 - s)))
    return y


# ----------------------------------------------------------------------------------------------- structure
def add(a: Tensor, b: Tensor) -> Tensor:
    y = Tensor(a.data + b.data)
    if _needs(a, b):
        def fn(g):
            _acc(a, g)
            _acc(b, g)
        _record(y, fn)
    return y


def mul_channel(x: Tensor, s: Tensor) -> Tensor:
    """x (N,H,W,C) * s (N,1,1,C) (squeeze-excitation scaling)."""
    y = Tensor(x.data * s.data)
    if _needs(x, s):
        def fn(g):
            _acc(x, g * s.data)
            _acc(s, (g * x.data).sum((1, 2), keepdims=True))
        _record(y, fn)
    return y


def concat(ts: list[Tensor]) -> Tensor:
    y = Tensor(np.concatenate([t.data for t in ts], axis=-1))
    if _needs(*ts):
        sizes = np.cumsum([0] + [t.data.shape[-1] for t in ts])

        def fn(g):
            for t, a, b in zip(ts, sizes[:-1], sizes[1:]):
                _acc(t, g[..., a:b])
        _record(y, fn)
    return y


def gap(x: Tensor, keepdims=False) -> Tensor:
    y = Tensor(x.data.mean((1, 2), keepdims=keepdims))
    if _needs(x):
        H, W = x.data.shape[1:3]

        def fn(g):
            gg = g if keepdims else g[:, None, None, :]
            _acc(x, np.broadcast_to(gg, x.data.shape) / (H * W))
        _record(y, fn)
    return y


def linear(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    """x (N, F) @ w.T (PyTorch layout (O, F))."""
    out = x.data @ w.data.T
    if b is not None:
        out = out + b.data
    y = Tensor(out)
    if _needs(x, w, b):
        def fn(g):
            if w.requires_grad:
                _acc(w, g.T @ x.data)
            if b is not None and b.requires_grad:
                _acc(b, g.sum(0))
            if x.requires_grad:
                _acc(x, g @ w.data)
        _record(y, fn)
    return y


def dropout(x: Tensor, p: float, rng: np.random.Generator | None) -> Tensor:
    if rng is None or p <= 0:
        return x
    m = (rng.random(x.data.shape) >= p).astype(np.float32) / (1 - p)
    y = Tensor(x.data * m)
    if _needs(x):
        _record(y, lambda g: _acc(x, g * m))
    return y


def maxpool(x: np.ndarray, k=3, s=2, p=1) -> np.ndarray:
    """Inference-only max pooling (only used inside frozen stems)."""
    N, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), constant_values=-np.inf) if p else x
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    out = np.full((N, Ho, Wo, C), -np.inf, np.float32)
    for i in range(k):
        for j in range(k):
            np.maximum(out, xp[:, i:i + s * Ho:s, j:j + s * Wo:s, :], out=out)
    return out


def avgpool2(x: np.ndarray) -> np.ndarray:
    """Inference-only 2x2 / stride-2 average pooling (DenseNet transitions)."""
    N, H, W, C = x.shape
    H2, W2 = H // 2, W // 2
    return x[:, :2 * H2, :2 * W2].reshape(N, H2, 2, W2, 2, C).mean((2, 4))


# ----------------------------------------------------------------------------------------------- losses
def bce_with_logits(z: Tensor, y: np.ndarray, pos_weight: float = 1.0, focal_gamma: float = 0.0) -> Tensor:
    """Mean (optionally class-weighted / focal) binary cross-entropy on logits z (N,)."""
    zz = z.data.reshape(-1)
    p = sigmoid_np(zz)
    w = np.where(y == 1, pos_weight, 1.0)
    # numerically stable log-sigmoid
    log_p = -np.logaddexp(0, -zz)
    log_1p = -np.logaddexp(0, zz)
    if focal_gamma > 0:
        pt = np.where(y == 1, p, 1 - p)
        fw = (1 - pt) ** focal_gamma
    else:
        fw = np.ones_like(zz)
    loss = -(w * fw * (y * log_p + (1 - y) * log_1p)).mean()
    out = Tensor(np.array(loss, np.float32))
    if _needs(z):
        def fn(g):
            if focal_gamma > 0:  # gradient of focal loss (derived per element)
                pt = np.where(y == 1, p, 1 - p)
                lpt = np.where(y == 1, log_p, log_1p)
                dpt = np.where(y == 1, p * (1 - p), -p * (1 - p))
                d = -(w * (-(focal_gamma) * (1 - pt) ** (focal_gamma - 1) * dpt * lpt
                           + (1 - pt) ** focal_gamma * dpt / np.maximum(pt, 1e-12)))
            else:
                d = w * (p - y)
            _acc(z, (g * d / len(zz)).reshape(z.data.shape).astype(np.float32))
        _record(out, fn)
    return out


# ----------------------------------------------------------------------------------------------- optimiser
class AdamW:
    """AdamW with parameter groups (differential learning rates) and global-norm gradient clipping."""

    def __init__(self, groups: list[dict], betas=(0.9, 0.999), eps=1e-8, clip: float | None = 1.0):
        self.groups = groups  # each: {"params": [Tensor], "lr": float, "wd": float}
        self.b1, self.b2, self.eps, self.clip = betas[0], betas[1], eps, clip
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for g in groups for p in g["params"]}
        self.v = {id(p): np.zeros_like(p.data) for g in groups for p in g["params"]}

    def params(self):
        return [p for g in self.groups for p in g["params"]]

    def grad_norm(self):
        return float(np.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params() if p.grad is not None)))

    def step(self, lr_scale: float = 1.0) -> float:
        self.t += 1
        gn = self.grad_norm()
        k = min(1.0, self.clip / (gn + 1e-12)) if self.clip else 1.0
        for g in self.groups:
            lr = g["lr"] * lr_scale
            for p in g["params"]:
                if p.grad is None:
                    continue
                gr = p.grad * k
                m, v = self.m[id(p)], self.v[id(p)]
                m *= self.b1
                m += (1 - self.b1) * gr
                v *= self.b2
                v += (1 - self.b2) * gr * gr
                mh = m / (1 - self.b1 ** self.t)
                vh = v / (1 - self.b2 ** self.t)
                if g.get("wd", 0):
                    p.data -= lr * g["wd"] * p.data
                p.data -= lr * mh / (np.sqrt(vh) + self.eps)
        return gn

    def zero_grad(self):
        for p in self.params():
            p.grad = None

    def state(self):
        return {"t": self.t, "m": {k: v.copy() for k, v in self.m.items()}, "v": {k: v.copy() for k, v in self.v.items()}}


def lr_schedule(step: int, total: int, warmup: int) -> float:
    """Linear warm-up then cosine annealing to 0 (multiplier of the base learning rate)."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * min(1.0, t)))
