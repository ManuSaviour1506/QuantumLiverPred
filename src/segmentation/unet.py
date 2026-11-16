"""2.5-D U-Net for liver and tumour segmentation, implemented in NumPy (no deep-learning framework is installable here).

Input: 3 adjacent axial slices (previous, current, next) at 128 x 128. Output: 64 x 64 logits for 3 classes
(0 background, 1 liver, 2 tumour) - the resolution of the verified expert maps (each cell = 8 x 8 pixels of 512 x 512).

    enc1  [conv3x3-BN-ReLU] x2   3 -> c1   @128   -> maxpool
    enc2  [conv3x3-BN-ReLU] x2  c1 -> c2   @64    -> maxpool
    enc3  [conv3x3-BN-ReLU] x2  c2 -> c3   @32    -> maxpool
    mid   [conv3x3-BN-ReLU] x2  c3 -> c4   @16
    dec3  upsample x2, concat enc3, [conv3x3-BN-ReLU] x2 -> c3   @32
    dec2  upsample x2, concat enc2, [conv3x3-BN-ReLU] x2 -> c2   @64
    head  conv1x1 c2 -> 3                                         @64

Loss: class-weighted softmax cross-entropy + soft Dice on liver and tumour. Layers come from the gradient-checked
NumPy CNN (src/classical/numpy_cnn.py); the new pieces (upsampling, skip concatenation, softmax/Dice) are checked
in tests/test_v9.py.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.classical.numpy_cnn import DT, BatchNorm2D, Conv2D, MaxPool2, ReLU


class Upsample2:
    def forward(self, x, train):
        return x.repeat(2, axis=2).repeat(2, axis=3)

    def backward(self, g):
        n, c, h, w = g.shape
        return g.reshape(n, c, h // 2, 2, w // 2, 2).sum((3, 5))


def _block(cin, cout, rng):
    return [Conv2D(cin, cout, 3, rng), BatchNorm2D(cout), ReLU(), Conv2D(cout, cout, 3, rng), BatchNorm2D(cout), ReLU()]


@dataclass
class UNetConfig:
    in_channels: int = 3
    widths: list[int] = field(default_factory=lambda: [16, 32, 64, 96])
    class_weights: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    dice_weight: float = 1.0
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 30
    warmup_epochs: int = 1
    patience: int = 6
    neg_per_pos: float = 0.5          # no-tumour (no-liver) slices sampled per tumour slice each epoch
    augment_shift: int = 6            # pixels at 128 x 128 (mask shifted by half)
    augment_rotate: float = 10.0
    augment_gamma: float = 0.3
    augment_noise: float = 0.03
    augment_blur: float = 0.3
    augment_intensity: float = 0.15
    seed: int = 42

    def to_dict(self):
        return asdict(self)


class UNet25D:
    def __init__(self, cfg: UNetConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        c1, c2, c3, c4 = cfg.widths
        r = self.rng
        self.enc1, self.enc2, self.enc3 = _block(cfg.in_channels, c1, r), _block(c1, c2, r), _block(c2, c3, r)
        self.mid = _block(c3, c4, r)
        self.dec3, self.dec2 = _block(c4 + c3, c3, r), _block(c3 + c2, c2, r)
        self.head = Conv2D(c2, 3, 1, r)
        self.pool1, self.pool2, self.pool3 = MaxPool2(), MaxPool2(), MaxPool2()
        self.up3, self.up2 = Upsample2(), Upsample2()
        self.mean = np.zeros(cfg.in_channels, DT)
        self.std = np.ones(cfg.in_channels, DT)
        self.history: list[dict] = []
        self.best_epoch = None
        self.train_seconds = None
        self._t, self._m, self._v = 0, {}, {}

    # ------------------------------------------------------------------ structure
    def _blocks(self):
        return {"enc1": self.enc1, "enc2": self.enc2, "enc3": self.enc3, "mid": self.mid, "dec3": self.dec3,
                "dec2": self.dec2, "head": [self.head]}

    def _leaves(self):
        for bname, layers in self._blocks().items():
            for i, l in enumerate(layers):
                yield f"{bname}.{i}", l

    def params(self):
        for key, l in self._leaves():
            for name in l.trainable:
                yield key, l, name, name in l.decay

    def n_parameters(self):
        return int(sum(getattr(l, n).size for _, l, n, _ in self.params()))

    @staticmethod
    def _run(layers, x, train):
        for l in layers:
            x = l.forward(x, train)
        return x

    @staticmethod
    def _back(layers, g):
        for l in reversed(layers):
            g = l.backward(g)
        return g

    # ------------------------------------------------------------------ passes
    def forward(self, x, train):
        e1 = self._run(self.enc1, x, train)
        e2 = self._run(self.enc2, self.pool1.forward(e1, True), train)
        e3 = self._run(self.enc3, self.pool2.forward(e2, True), train)
        m = self._run(self.mid, self.pool3.forward(e3, True), train)
        self._c3 = e3.shape[1]
        d3 = self._run(self.dec3, np.concatenate([self.up3.forward(m, train), e3], 1), train)
        self._c2 = e2.shape[1]
        d2 = self._run(self.dec2, np.concatenate([self.up2.forward(d3, train), e2], 1), train)
        return self.head.forward(d2, train)

    def backward(self, g):
        g = self.head.backward(g)
        g = self._back(self.dec2, g)
        cu = g.shape[1] - self._c2
        g_up2, g_e2 = g[:, :cu], g[:, cu:]
        g = self._back(self.dec3, self.up2.backward(g_up2))
        cu = g.shape[1] - self._c3
        g_up3, g_e3 = g[:, :cu], g[:, cu:]
        g = self._back(self.mid, self.up3.backward(g_up3))
        g = self.pool3.backward(g) + g_e3
        g = self._back(self.enc3, g)
        g = self.pool2.backward(g) + g_e2
        g = self._back(self.enc2, g)
        g = self.pool1.backward(g)
        self._back(self.enc1, g)

    # ------------------------------------------------------------------ loss
    def loss_and_grad(self, logits, y):
        """Weighted CE + soft Dice (liver, tumour). logits (N,3,64,64), y (N,64,64) int."""
        c = self.cfg
        z = logits.astype(np.float64)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(1, keepdims=True)
        oh = np.stack([(y == k) for k in range(3)], 1).astype(np.float64)
        w = np.asarray(c.class_weights, float)[y]
        ce = -(w * np.log(np.clip((p * oh).sum(1), 1e-12, None))).sum() / w.sum()
        g_ce = (p - oh) * w[:, None] / w.sum()
        dice_l, g_d = 0.0, np.zeros_like(p)
        for k in (1, 2):
            pk, ok = p[:, k], oh[:, k]
            inter, den = (pk * ok).sum(), pk.sum() + ok.sum()
            d = (2 * inter + 1.0) / (den + 1.0)
            dice_l += (1 - d) / 2
            dd_dp = (2 * ok * (den + 1.0) - (2 * inter + 1.0)) / (den + 1.0) ** 2
            gp = np.zeros_like(p)
            gp[:, k] = -dd_dp / 2
            # chain through softmax: dL/dz_j = p_j (gp_j - sum_i gp_i p_i)
            g_d += p * (gp - (gp * p).sum(1, keepdims=True))
        loss = ce + c.dice_weight * dice_l
        g = g_ce + c.dice_weight * g_d
        return float(loss), g.astype(DT)

    # ------------------------------------------------------------------ optimiser
    def _lr(self, epoch, step, steps):
        c = self.cfg
        e = epoch + step / max(steps, 1)
        if e < c.warmup_epochs:
            return c.learning_rate * (e + 1e-3) / c.warmup_epochs
        prog = (e - c.warmup_epochs) / max(c.max_epochs - c.warmup_epochs, 1)
        return c.learning_rate * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0))))

    def _adam(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for key, l, name, decay in self.params():
            p, g = getattr(l, name), getattr(l, "d" + name).astype(DT)
            k = (key, name)
            m = b1 * self._m.get(k, 0) + (1 - b1) * g
            v = b2 * self._v.get(k, 0) + (1 - b2) * g * g
            self._m[k], self._v[k] = m, v
            step = lr * (m / (1 - b1 ** self._t)) / (np.sqrt(v / (1 - b2 ** self._t)) + eps)
            if decay:
                step = step + lr * self.cfg.weight_decay * p
            setattr(l, name, (p - step).astype(DT))

    # ------------------------------------------------------------------ augmentation
    def _augment(self, xb, yb):
        import cv2
        c = self.cfg
        X, Y = np.empty_like(xb), np.empty_like(yb)
        for i in range(len(xb)):
            ang = self.rng.uniform(-c.augment_rotate, c.augment_rotate)
            tx, ty = self.rng.integers(-c.augment_shift, c.augment_shift + 1, 2)
            M = cv2.getRotationMatrix2D((64, 64), ang, 1.0)
            M[:, 2] += (tx, ty)
            Mm = cv2.getRotationMatrix2D((32, 32), ang, 1.0)
            Mm[:, 2] += (tx / 2, ty / 2)
            img = np.stack([cv2.warpAffine(ch, M, (128, 128), flags=cv2.INTER_LINEAR, borderValue=0.0) for ch in xb[i]])
            Y[i] = cv2.warpAffine(yb[i].astype(np.uint8), Mm, (64, 64), flags=cv2.INTER_NEAREST, borderValue=0)
            if c.augment_gamma:
                img = np.clip(img, 0, None) ** (1 + self.rng.uniform(-c.augment_gamma, c.augment_gamma))
            if c.augment_blur and self.rng.random() < c.augment_blur:
                s = self.rng.uniform(0.5, 1.0)
                img = np.stack([cv2.GaussianBlur(ch, (0, 0), s) for ch in img])
            if c.augment_intensity:
                img = img * (1 + self.rng.uniform(-c.augment_intensity, c.augment_intensity)) + \
                    self.rng.uniform(-c.augment_intensity, c.augment_intensity) * 0.5
            if c.augment_noise:
                img = img + self.rng.normal(0, self.rng.uniform(0, c.augment_noise), img.shape)
            X[i] = img
        return X.astype(DT), Y

    def _norm(self, X):
        return ((X - self.mean[None, :, None, None]) / self.std[None, :, None, None]).astype(DT)

    # ------------------------------------------------------------------ training
    def fit(self, X, Y, Xv, Yv, log=None):
        """X (N,3,128,128) float in [0,1], Y (N,64,64) uint8. Early stopping on validation mean Dice (liver, tumour)."""
        c = self.cfg
        X = X.astype(DT)
        self.mean, self.std = X.mean((0, 2, 3)).astype(DT), (X.std((0, 2, 3)) + 1e-6).astype(DT)
        pos = np.where((Y == 2).any((1, 2)) | (Y == 1).any((1, 2)))[0]
        neg = np.setdiff1d(np.arange(len(X)), pos)
        best, best_state, wait = -1.0, None, 0
        t0 = time.perf_counter()
        for epoch in range(c.max_epochs):
            k = min(len(neg), int(round(c.neg_per_pos * len(pos))))
            order = self.rng.permutation(np.concatenate([pos, self.rng.choice(neg, k, replace=False)]))
            steps = int(np.ceil(len(order) / c.batch_size))
            tot = 0.0
            for s in range(steps):
                idx = order[s * c.batch_size:(s + 1) * c.batch_size]
                if len(idx) < 2:
                    continue
                xb, yb = self._augment(X[idx], Y[idx])
                logits = self.forward(self._norm(xb), True)
                loss, g = self.loss_and_grad(logits, yb)
                self.backward(g)
                self._adam(self._lr(epoch, s, steps))
                tot += loss * len(idx)
            pv = self.predict(Xv)
            dl, dt = dice(pv >= 1, Yv >= 1), dice(pv == 2, Yv == 2)
            rec = {"epoch": epoch + 1, "train_loss": tot / len(order), "val_dice_liver": dl, "val_dice_tumour": dt,
                   "lr": self._lr(epoch, steps, steps), "seconds": time.perf_counter() - t0}
            self.history.append(rec)
            if log:
                log(rec)
            score = (dl + dt) / 2
            if score > best + 1e-4:
                best, best_state, wait, self.best_epoch = score, self.get_state(), 0, epoch + 1
            else:
                wait += 1
                if wait >= c.patience:
                    break
        if best_state is not None:
            self.set_state(best_state)
        self.train_seconds = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ inference
    def predict_proba(self, X, bs=16):
        out = []
        for i in range(0, len(X), bs):
            z = self.forward(self._norm(X[i:i + bs].astype(DT)), False).astype(np.float64)
            z -= z.max(1, keepdims=True)
            e = np.exp(z)
            out.append((e / e.sum(1, keepdims=True)).astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 3, 64, 64), np.float32)

    def predict(self, X):
        return self.predict_proba(X).argmax(1).astype(np.uint8)

    # ------------------------------------------------------------------ persistence
    def get_state(self):
        st = {"mean": self.mean.copy(), "std": self.std.copy()}
        for key, l in self._leaves():
            for name in l.trainable:
                st[f"{key}.{name}"] = getattr(l, name).copy()
            if isinstance(l, BatchNorm2D):
                st[f"{key}.running_mean"], st[f"{key}.running_var"] = l.running_mean.copy(), l.running_var.copy()
        return st

    def set_state(self, st):
        self.mean, self.std = st["mean"].astype(DT), st["std"].astype(DT)
        for key, l in self._leaves():
            for name in l.trainable:
                setattr(l, name, st[f"{key}.{name}"].astype(DT))
            if isinstance(l, BatchNorm2D):
                l.running_mean, l.running_var = st[f"{key}.running_mean"].astype(DT), st[f"{key}.running_var"].astype(DT)

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"config": self.cfg.to_dict(), "history": self.history, "best_epoch": self.best_epoch,
                "train_seconds": self.train_seconds, "n_parameters": self.n_parameters(), "format": "unet25d_v1", **(extra or {})}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **self.get_state())

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != "unet25d_v1":
            raise ValueError(f"{path} is not a unet25d_v1 checkpoint")
        m = UNet25D(UNetConfig(**meta["config"]))
        m.set_state({k: z[k] for k in z.files if k != "meta"})
        m.history, m.best_epoch, m.train_seconds = meta["history"], meta["best_epoch"], meta["train_seconds"]
        m.meta = meta
        return m


def dice(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0
