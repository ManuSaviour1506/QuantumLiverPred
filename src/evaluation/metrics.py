"""Metrics for the fair comparison (every model is scored by these functions on the same held-out test patches).

Threshold-free:  ROC-AUC, PR-AUC (average precision)
At the threshold chosen on VALIDATION data:  accuracy, precision, recall (= sensitivity), specificity, F1,
                 balanced accuracy, confusion matrix
Calibration:     Brier score, expected calibration error (10 equal-width bins), reliability table
Uncertainty:     95 % CIs from a two-stage cluster bootstrap (resample scans, then slices within each resampled scan;
                 every patch of a resampled slice is kept), 2,000 resamples, seed 42. With few test scans the
                 between-patient part of the uncertainty is poorly estimated; this is reported alongside the CIs.
Paired comparison: bootstrap of the AUC difference between two models on the same resamples.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _safe_ap(y, s):
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def threshold_metrics(y, p, thr) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (tp and (prec + sens) > 0) else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": sens, "sensitivity": sens,
            "specificity": spec, "f1": f1, "balanced_accuracy": np.nanmean([sens, spec]),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def calibration(y, p, bins: int = 10) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.sum():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(conf - acc)
            rows.append({"bin_lo": float(a), "bin_hi": float(b), "n": int(m.sum()), "mean_predicted": conf,
                         "observed_rate": acc})
    return {"brier": float(((p - y) ** 2).mean()), "ece": float(ece), "reliability": rows}


def choose_threshold(y_val, p_val, rule: str = "youden") -> float:
    """Pick the operating threshold on validation data only."""
    y_val, p_val = np.asarray(y_val), np.asarray(p_val)
    if len(np.unique(y_val)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_val, p_val)
    j = tpr - fpr if rule == "youden" else tpr
    k = int(np.argmax(j))
    t = float(thr[k])
    return float(min(max(t, 1e-6), 1 - 1e-6)) if np.isfinite(t) else 0.5


def roc_points(y, s, n: int = 60) -> dict:
    if len(np.unique(y)) < 2:
        return {"fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(y, s)
    grid = np.linspace(0, 1, n)
    return {"fpr": grid.round(4).tolist(), "tpr": np.interp(grid, fpr, tpr).round(4).tolist()}


def _cluster_resample(scans, slices, rng):
    """Indices of a two-stage bootstrap resample (scan -> slice -> all patches of the slice)."""
    us = np.unique(scans)
    pick = rng.choice(us, size=len(us), replace=True)
    out = []
    for s in pick:
        sl = np.unique(slices[scans == s])
        for t in rng.choice(sl, size=len(sl), replace=True):
            out.append(np.where((scans == s) & (slices == t))[0])
    return np.concatenate(out)


def bootstrap_ci(y, p, thr, scans, slices, n_boot: int = 2000, seed: int = 42) -> dict:
    y, p, scans, slices = (np.asarray(v) for v in (y, p, scans, slices))
    rng = np.random.default_rng(seed)
    keys = ("auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy")
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        ix = _cluster_resample(scans, slices, rng)
        yy, pp = y[ix], p[ix]
        if len(np.unique(yy)) < 2:
            continue
        m = threshold_metrics(yy, pp, thr)
        vals["auc"].append(_safe_auc(yy, pp))
        vals["pr_auc"].append(_safe_ap(yy, pp))
        for k in keys[2:]:
            vals[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v else [float("nan")] * 2
            for k, v in vals.items()} | {"n_boot_valid": len(vals["auc"])}


def paired_auc_difference(y, s1, s2, scans, slices, n_boot: int = 2000, seed: int = 42) -> dict:
    y, s1, s2, scans, slices = (np.asarray(v) for v in (y, s1, s2, scans, slices))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = _cluster_resample(scans, slices, rng)
        if len(np.unique(y[ix])) < 2:
            continue
        d.append(_safe_auc(y[ix], s1[ix]) - _safe_auc(y[ix], s2[ix]))
    d = np.asarray(d)
    return {"delta_auc": _safe_auc(y, s1) - _safe_auc(y, s2), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def full_report(y, p_cal, score, thr, scans, slices, n_boot: int = 2000) -> dict:
    """Everything the comparison table needs for one model on one split."""
    rep = {"n": int(len(y)), "n_pos": int(np.sum(y)), "n_neg": int(len(y) - np.sum(y)),
           "auc": _safe_auc(y, score), "pr_auc": _safe_ap(y, score), "threshold": float(thr)}
    rep.update(threshold_metrics(y, p_cal, thr))
    rep["calibration"] = calibration(y, p_cal)
    rep["roc"] = roc_points(y, score)
    rep["ci95"] = bootstrap_ci(y, p_cal, thr, scans, slices, n_boot=n_boot)
    rep["per_scan_auc"] = {str(s): _safe_auc(np.asarray(y)[np.asarray(scans) == s], np.asarray(score)[np.asarray(scans) == s])
                           for s in np.unique(scans)}
    return rep
