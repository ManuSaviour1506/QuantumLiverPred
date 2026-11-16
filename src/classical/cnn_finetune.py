"""Fine-tuning of ImageNet-pretrained DenseNet-121 / ResNet-50 / EfficientNet-B0 on CT patches (NumPy, CPU).

What is trained: the backbone TAIL (densenet121: dense layers 13-16 of block 4 + norm5; resnet50: the last bottleneck
block; efficientnet_b0: the last MBConv block + conv_head + bn2) and a new linear classification head. Earlier layers
keep their ImageNet weights (their outputs are cached once, including augmented views of every training patch).
BatchNorm layers in the tail use their running statistics (eval mode); their scale/shift are trained.

Training: class-balanced BCE (pos_weight = n_neg / n_pos), AdamW (tail lr 1e-4, head lr 1e-3, weight decay 1e-4),
linear warm-up + cosine decay, dropout 0.2 before the head, gradient clipping, a random augmented view of every
training patch per epoch, early stopping on validation ROC-AUC (patience 4) with the best checkpoint restored and
written to disk. (Validation loss was the first criterion; it rose from epoch 2 on while validation AUC still improved,
because class-weighted training makes the logits over-confident - probabilities are recalibrated on validation data
afterwards, so the ranking criterion is the appropriate one.) Validation data are used for early stopping, calibration
and threshold choice; never test data.
"""
from __future__ import annotations

import time

import numpy as np

from src.classical import backbones as B
from src.classical import nn
from src.classical.nn import Tensor

TAIL_LR, HEAD_LR, WD = 1e-4, 1e-3, 1e-4


class TailClassifier:
    def __init__(self, arch: str, seed: int = 42):
        self.arch = arch
        self.net = B.build(arch)
        self.net.set_trainable(True)
        c_final = {"densenet121": 1024, "resnet50": 2048, "efficientnet_b0": 1280}[arch]
        rng = np.random.default_rng(seed)
        self.w = nn.param(rng.normal(0, 0.01, (1, c_final)), "head.w")
        self.b = nn.param(np.zeros(1), "head.b")
        self.rng = rng

    def logits(self, h: np.ndarray, train: bool = False) -> Tensor:
        f = self.net.tail(Tensor(h.astype(np.float32)))["final"]
        g = nn.gap(f)
        if train:
            g = nn.dropout(g, 0.2, self.rng)
        return nn.linear(g, self.w, self.b)

    def gap_from_frozen(self, h: np.ndarray, bs: int = 128) -> np.ndarray:
        """Fine-tuned penultimate features (GAP after the fine-tuned tail)."""
        out = []
        for s in range(0, len(h), bs):
            out.append(nn.gap(self.net.tail(Tensor(h[s:s + bs].astype(np.float32)))["final"]).data)
        return np.concatenate(out).astype(np.float64)

    def predict_from_frozen(self, h: np.ndarray, bs: int = 128) -> np.ndarray:
        out = []
        for s in range(0, len(h), bs):
            out.append(self.logits(h[s:s + bs]).data.reshape(-1))
        return np.concatenate(out).astype(np.float64)

    def predict_images(self, gray_u8: np.ndarray, bs: int = 32) -> np.ndarray:
        from src.preprocessing.ct import to_rgb
        out = []
        for s in range(0, len(gray_u8), bs):
            h = self.net.frozen(B.Backbone.preprocess(to_rgb(gray_u8[s:s + bs])))
            out.append(self.logits(h).data.reshape(-1))
        return np.concatenate(out).astype(np.float64)

    # ---------------------------------------------------------------- persistence (fine-tuned params only)
    def state(self) -> dict:
        st = {"tail::" + k: self.net.P[k].data.copy() for k in self.net.tail_keys()}
        st["head::w"], st["head::b"] = self.w.data.copy(), self.b.data.copy()
        return st

    def load_state(self, st: dict) -> None:
        for k, v in st.items():
            if k.startswith("tail::"):
                self.net.P[k[6:]].data = np.asarray(v, np.float32)
        self.w.data = np.asarray(st["head::w"], np.float32)
        self.b.data = np.asarray(st["head::b"], np.float32)


def _bce(z, y):
    return float(np.mean(np.logaddexp(0, z) - y * z))


def train_tail(arch: str, views: list[np.ndarray], y_tr: np.ndarray, h_val: np.ndarray, y_val: np.ndarray,
               max_epochs: int = 20, batch: int = 32, patience: int = 4, seed: int = 42, checkpoint=None, log=print,
               select: str = "auc", focal_gamma: float = 0.0):
    """views: list of arrays (N_train, h, w, C) - view 0 = original, others = augmented copies, same row order."""
    from sklearn.metrics import roc_auc_score
    m = TailClassifier(arch, seed)
    opt = nn.AdamW([{"params": [p for p in m.net.tail_params() if p.requires_grad], "lr": TAIL_LR, "wd": WD},
                    {"params": [m.w, m.b], "lr": HEAD_LR, "wd": WD}], clip=1.0)
    n = len(y_tr)
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    steps = int(np.ceil(n / batch))
    total, warm = steps * max_epochs, steps
    rng = np.random.default_rng(seed)
    best, best_state, wait, hist, t0, step = np.inf, None, 0, [], time.time(), 0
    for ep in range(max_epochs):
        order = rng.permutation(n)
        tl, seen = 0.0, 0
        for s in range(steps):
            idx = order[s * batch:(s + 1) * batch]
            if len(idx) < 2:
                continue
            v = rng.integers(0, len(views), len(idx))
            hb = np.stack([views[vi][i] for vi, i in zip(v, idx)]).astype(np.float32)
            with nn.tape() as tp:
                z = m.logits(hb, train=True)
                loss = nn.bce_with_logits(z, y_tr[idx].astype(np.float64), pos_weight=pos_w, focal_gamma=focal_gamma)
                opt.zero_grad()
                nn.backward(loss, tp)
            opt.step(nn.lr_schedule(step, total, warm))
            step += 1
            tl += float(loss.data) * len(idx)
            seen += len(idx)
        zv = m.predict_from_frozen(h_val)
        vl = _bce(zv, y_val)
        auc = float(roc_auc_score(y_val, zv)) if len(np.unique(y_val)) == 2 else float("nan")
        rec = {"epoch": ep + 1, "train_loss": tl / max(seen, 1), "val_loss": vl, "val_auc": auc,
               "seconds": round(time.time() - t0, 1)}
        hist.append(rec)
        log(f"[{arch}] epoch {ep + 1}: train {rec['train_loss']:.4f} val {vl:.4f} auc {auc:.4f} ({rec['seconds']}s)")
        crit = -auc if select == "auc" else vl
        if crit < best - 1e-4:
            best, best_state, wait = crit, m.state(), 0
            if checkpoint:
                checkpoint(best_state, rec)
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state(best_state)
    key = [(-r["val_auc"] if select == "auc" else r["val_loss"]) for r in hist]
    best_epoch = int(np.argmin(key) + 1)
    return m, {"history": hist, "best_epoch": best_epoch, "train_seconds": round(time.time() - t0, 1),
               "pos_weight": pos_w, "max_epochs": max_epochs, "batch": batch, "patience": patience,
               "tail_lr": TAIL_LR, "head_lr": HEAD_LR, "weight_decay": WD, "n_train_views": len(views),
               "early_stopping": f"validation {'ROC-AUC' if select == 'auc' else 'loss'}, patience {patience}",
               "focal_gamma": focal_gamma}
