"""V7 position-aware and patient-level metrics (V6 audit Problems A and B).

relative position r = (slice_index - min) / (max - min) over a scan's included slices (label-free).
centrality = -|r - 0.5|. It is used ONLY to measure the position shortcut (baselines, matched pairs, strata,
training weights) and never as a model input.

position-matched AUC: concordance over (tumour, non-tumour) pairs from the SAME scan whose relative positions differ by
less than `tol`; pooled over scans (pair-weighted). 0.5 = no tumour signal beyond position.
All confidence intervals resample patients (never slices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from src.v4.metrics import binary, ece


def relative_position(slice_index, scans) -> np.ndarray:
    z, scans = np.asarray(slice_index, float), np.asarray(scans)
    r = np.zeros_like(z)
    for u in np.unique(scans):
        m = scans == u
        lo, hi = z[m].min(), z[m].max()
        r[m] = (z[m] - lo) / (hi - lo) if hi > lo else 0.5
    return r


def centrality(rel) -> np.ndarray:
    return -np.abs(np.asarray(rel, float) - 0.5)


def _conc(p, n):
    return (p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])


def matched_pair_counts(y, s, scans, rel, tol: float = 0.05) -> dict:
    """Per scan: (concordant-pair sum, number of position-matched pairs)."""
    y, s, scans, rel = map(np.asarray, (y, s, scans, rel))
    out = {}
    for u in np.unique(scans):
        m = scans == u
        P, N = m & (y == 1), m & (y == 0)
        if not P.any() or not N.any():
            out[str(u)] = (0.0, 0)
            continue
        near = np.abs(rel[P][:, None] - rel[N][None, :]) < tol
        out[str(u)] = (float(_conc(s[P], s[N])[near].sum()), int(near.sum()))
    return out


def position_matched_auc(y, s, scans, rel, tol: float = 0.05) -> float:
    c = matched_pair_counts(y, s, scans, rel, tol)
    den = sum(v[1] for v in c.values())
    return float(sum(v[0] for v in c.values()) / den) if den else float("nan")


def within_scan_aucs(y, s, scans) -> dict:
    y, s, scans = map(np.asarray, (y, s, scans))
    return {str(u): float(roc_auc_score(y[scans == u], s[scans == u])) for u in np.unique(scans)
            if len(np.unique(y[scans == u])) == 2}


def position_stratified_auc(y, s, scans, rel, bins=(0.0, 1 / 3, 2 / 3, 1.0001)) -> dict:
    """Within-scan AUC inside each relative-position stratum (lower edge, middle, upper edge)."""
    y, s, scans, rel = map(np.asarray, (y, s, scans, rel))
    out = {}
    for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (rel >= lo) & (rel < hi)
        a = within_scan_aucs(y[m], s[m], scans[m])
        out[["lower", "middle", "upper"][k]] = {"mean_auc": float(np.mean(list(a.values()))) if a else float("nan"),
                                                 "n_scans": len(a), "n_slices": int(m.sum())}
    return out


def patient_bootstrap(units, stat, n_boot: int = 2000, seed: int = 42, level: float = 0.95):
    """stat(list_of_units) -> float; resamples patients with replacement. Returns (low, high)."""
    units = np.asarray(sorted(set(units)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = stat(list(rng.choice(units, len(units))))
        if np.isfinite(v):
            vals.append(v)
    a = (1 - level) / 2
    return (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))) if vals else (float("nan"), float("nan"))


def summarize(df: pd.DataFrame, tol: float = 0.05, n_boot: int = 2000, seed: int = 42) -> dict:
    """df columns: unit, site, label, raw, rel, prob, pred_A, pred_B, pred_C (prob/pred may be NaN / -1 for failed folds)."""
    df = df.reset_index(drop=True)
    y, s, u, rel = df.label.to_numpy(int), df.raw.to_numpy(float), df.unit.to_numpy(), df.rel.to_numpy(float)
    by = {k: np.where(u == k)[0] for k in np.unique(u)}
    w = within_scan_aucs(y, s, u)
    pm = matched_pair_counts(y, s, u, rel, tol)
    res = {"n_patients": len(by), "n_slices": int(len(df)), "n_positive": int(y.sum()),
           "majority_class_accuracy": float(max(y.mean(), 1 - y.mean())),
           "within_scan_auc": float(np.mean(list(w.values()))), "per_patient_auc": w,
           "position_matched_auc": position_matched_auc(y, s, u, rel, tol),
           "position_matched_pairs": int(sum(v[1] for v in pm.values())),
           "position_stratified_auc": position_stratified_auc(y, s, u, rel)}
    ok = np.isfinite(df.prob.to_numpy(float)) if "prob" in df else np.zeros(len(df), bool)
    res["failed_slices"] = int((~ok).sum())
    res["failed_patients"] = sorted(set(u[~ok]))
    if ok.any():
        yy, pp = y[ok], df.prob.to_numpy(float)[ok]
        res["pooled_roc_auc"] = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else float("nan")
        res["pr_auc"] = float(average_precision_score(yy, pp))
        res["ece"] = ece(yy, pp)
        res["brier"] = float(np.mean((pp - yy) ** 2))
        for t in ("A", "B", "C"):
            res[f"threshold_{t}"] = binary(yy, df[f"pred_{t}"].to_numpy(int)[ok])
    # patient-bootstrap CIs
    def sub(units):
        return np.concatenate([by[k] for k in units])

    def stat_within(units):
        return float(np.mean([w[k] for k in units if k in w]))

    def stat_pm(units):
        num = sum(pm[str(k)][0] for k in units)
        den = sum(pm[str(k)][1] for k in units)
        return num / den if den else float("nan")

    def stat_acc(t):
        def f(units):
            i = sub(units)
            i = i[ok[i]]
            return float((df[f"pred_{t}"].to_numpy(int)[i] == y[i]).mean()) if len(i) else float("nan")
        return f

    def stat_bal(units):
        i = sub(units)
        i = i[ok[i]]
        yy, pp = y[i], df.pred_B.to_numpy(int)[i]
        return 0.5 * ((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean()) if len(np.unique(yy)) == 2 else float("nan")

    cm = matched_pair_counts(y, s, u, centrality(rel), tol)
    res["centrality_matched_auc"] = float(sum(v[0] for v in cm.values()) / max(1, sum(v[1] for v in cm.values())))
    res["centrality_matched_pairs"] = int(sum(v[1] for v in cm.values()))

    def stat_cm(units):
        den = sum(cm[str(k)][1] for k in units)
        return sum(cm[str(k)][0] for k in units) / den if den else float("nan")

    ci = {"within_scan_auc": patient_bootstrap(list(by), stat_within, n_boot, seed),
          "position_matched_auc": patient_bootstrap(list(by), stat_pm, n_boot, seed),
          "centrality_matched_auc": patient_bootstrap(list(by), stat_cm, n_boot, seed)}
    if ok.any():
        ci["accuracy_A"] = patient_bootstrap(list(by), stat_acc("A"), n_boot, seed)
        ci["balanced_accuracy_B"] = patient_bootstrap(list(by), stat_bal, n_boot, seed)
    res["ci95"] = ci
    res["ci_method"] = f"percentile bootstrap over {len(by)} patients, {n_boot} resamples, seed {seed}"
    return res


def per_patient_table(df: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    rows = []
    for k, d in df.groupby("unit"):
        y, s = d.label.to_numpy(int), d.raw.to_numpy(float)
        pm = matched_pair_counts(y, s, d.unit.to_numpy(), d.rel.to_numpy(), tol)[str(k)]
        ok = np.isfinite(d.prob.to_numpy(float)) if "prob" in d else np.zeros(len(d), bool)
        rows.append({"patient": k, "site": d.site.iloc[0], "slices": len(d), "tumour_fraction": float(y.mean()),
                     "within_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
                     "matched_pairs": pm[1], "matched_auc": pm[0] / pm[1] if pm[1] else np.nan,
                     "accuracy_A": float((d.pred_A.to_numpy(int)[ok] == y[ok]).mean()) if ok.any() else np.nan,
                     "balanced_accuracy_B": float(binary(y[ok], d.pred_B.to_numpy(int)[ok])["balanced_accuracy"])
                     if ok.any() and len(np.unique(y[ok])) == 2 else np.nan,
                     "predicted_positive_B": float((d.pred_B.to_numpy(int)[ok] == 1).mean()) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def per_site_table(df: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    rows = []
    for k, d in df.groupby("site"):
        y, s, u, r = d.label.to_numpy(int), d.raw.to_numpy(float), d.unit.to_numpy(), d.rel.to_numpy()
        ok = np.isfinite(d.prob.to_numpy(float))
        rows.append({"site": k, "patients": d.unit.nunique(), "slices": len(d),
                     "within_auc": float(np.mean(list(within_scan_aucs(y, s, u).values()))),
                     "matched_auc": position_matched_auc(y, s, u, r, tol),
                     "accuracy_A": float((d.pred_A.to_numpy(int)[ok] == y[ok]).mean()) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def paired_patient_test(a: dict, b: dict) -> dict:
    """Wilcoxon signed-rank on per-patient values present in both dicts."""
    keys = sorted(set(a) & set(b))
    x, z = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    d = x - z
    try:
        p = float(wilcoxon(x, z).pvalue) if np.any(d != 0) else float("nan")
    except ValueError:
        p = float("nan")
    return {"n": len(keys), "mean_a": float(x.mean()), "mean_b": float(z.mean()), "mean_diff": float(d.mean()),
            "a_better": int((d > 0).sum()), "p": p}


def reliability_bins(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return pd.DataFrame([{"bin": b, "mean_pred": p[idx == b].mean(), "observed": y[idx == b].mean(), "n": int((idx == b).sum())}
                         for b in range(bins) if (idx == b).any()])


# --- V7 deviation 1 (logged in V7_EXPERIMENT_LOG.md): the pre-registered relative-position matching does not remove
# the position shortcut (tumour runs are contiguous, so in a matched pair straddling a run boundary the tumour slice is
# the more central one; the no-image position rule scores ~0.93 on it). Two stricter analyses are added. ---

def centrality_matched_auc(y, s, scans, rel, tol: float = 0.05) -> float:
    """Pairs matched on centrality (|r - 0.5|), the V6-audit definition: mirror-image positions count as equal."""
    return position_matched_auc(y, s, scans, centrality(rel), tol)


def incremental_over_position(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """For each held-out patient: a logistic model fitted on the OTHER patients' rows (their own out-of-sample scores and
    labels) with (a) position only [one-hot relative-position bins + centrality] and (b) position + within-scan
    z-scored model score; returns each patient's within-scan AUC of (a) and (b). (b) > (a) means the model adds tumour
    information beyond slice position. The held-out patient's labels are used only for its AUC."""
    from sklearn.linear_model import LogisticRegression
    d = df.reset_index(drop=True)
    z = d.groupby("unit").raw.transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)).to_numpy()
    b = np.clip((d.rel.to_numpy() * bins).astype(int), 0, bins - 1)
    P = np.c_[np.eye(bins)[b], -np.abs(d.rel.to_numpy() - 0.5)]
    y, u = d.label.to_numpy(int), d.unit.to_numpy()
    rows = []
    for k in np.unique(u):
        tr, te = u != k, u == k
        if len(np.unique(y[te])) < 2:
            continue
        a = LogisticRegression(max_iter=2000).fit(P[tr], y[tr]).decision_function(P[te])
        Q = np.c_[P, z]
        c = LogisticRegression(max_iter=2000).fit(Q[tr], y[tr])
        rows.append({"patient": k, "auc_position_only": roc_auc_score(y[te], a),
                     "auc_position_plus_model": roc_auc_score(y[te], c.decision_function(Q[te])),
                     "model_coefficient": float(c.coef_[0, -1])})
    out = pd.DataFrame(rows)
    out["gain"] = out.auc_position_plus_model - out.auc_position_only
    return out
