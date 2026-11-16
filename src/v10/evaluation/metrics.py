"""Classification metrics (no scikit-learn dependency, so results can be cross-checked against scikit-learn in tests).

roc_auc            Mann-Whitney estimate with ties counted as one half
average_precision  step-wise area under the precision-recall curve (scikit-learn's definition)
threshold_metrics  accuracy, balanced accuracy, sensitivity (= recall), specificity, precision, F1, Matthews correlation
                   coefficient and the confusion matrix, at a threshold chosen elsewhere (on validation data)
log_loss, brier, ece (equal-width bins)
youden_threshold   threshold maximising sensitivity + specificity - 1 on the data given (use validation data only)
"""
from __future__ import annotations

import numpy as np


def roc_auc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_pair_counts(y, s) -> dict:
    """Exact numerator of the AUC: correctly ordered, tied and wrongly ordered positive-negative pairs."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    p, q = s[y == 1], s[y == 0]
    gt = int((p[:, None] > q[None, :]).sum())
    eq = int((p[:, None] == q[None, :]).sum())
    return {"correct": gt, "tied": eq, "wrong": int(len(p) * len(q) - gt - eq), "pairs": int(len(p) * len(q))}


def average_precision(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    distinct = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(ys)[distinct]
    fp = (distinct + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    rec_prev = np.r_[0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def threshold_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(p, float) >= thr
    tp = int((pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"threshold": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": float(np.nanmean([sens, spec])), "sensitivity": sens, "recall": sens,
            "specificity": spec, "precision": prec, "f1": f1, "mcc": float(mcc)}


def log_loss(y, p, eps: float = 1e-12) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= a) & ((p < b) if k < bins - 1 else (p <= b))
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def youden_threshold(y, p) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    cand = np.unique(p)
    best, thr = -np.inf, 0.5
    for t in cand:
        m = threshold_metrics(y, p, t)
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best:
            best, thr = j, t
    return float(thr)


def full_metrics(y, score, prob, thr) -> dict:
    out = {"roc_auc": roc_auc(y, score), "pr_auc": average_precision(y, score), "log_loss": log_loss(y, prob),
           "brier": brier(y, prob), "ece": ece(y, prob)}
    out.update(threshold_metrics(y, prob, thr))
    return out
