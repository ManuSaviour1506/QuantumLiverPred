"""V9 evaluation helpers: operating points chosen on validation, threshold metrics, group cross-validation folds."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def high_sensitivity_threshold(y_val, p_val, target: float = 0.98) -> float:
    """Highest validation specificity subject to validation sensitivity >= target (chosen on validation only)."""
    y_val, p_val = np.asarray(y_val), np.asarray(p_val, float)
    if len(np.unique(y_val)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = np.where(tpr >= target)[0]
    k = int(ok[0]) if len(ok) else int(np.argmax(tpr))
    t = float(thr[k])
    return float(min(max(t, 1e-6), 1 - 1e-6)) if np.isfinite(t) else 0.5


def threshold_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(p, float) >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"threshold": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens, "specificity": spec,
            "precision": prec, "recall": sens, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": (sens + spec) / 2}


def logo_folds(groups: np.ndarray, dev_groups: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-group-out folds over the development groups (never the test groups)."""
    groups = np.asarray(groups)
    out = []
    for g in dev_groups:
        held = np.where(groups == g)[0]
        fit = np.where(np.isin(groups, dev_groups) & (groups != g))[0]
        out.append((g, fit, held))
    return out
