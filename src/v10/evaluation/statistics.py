"""Group-level statistical comparison of two models scored on the same samples.

The independent unit is the group (reconstructed scan group in V10), never the patch. With k groups:
  group_bootstrap_delta   resample groups with replacement (optionally slices within groups) and recompute the pooled
                          AUC difference; a k-group bootstrap has at most k^k distinct resamples, so for small k the
                          interval is coarse and under-covers between-group variability
  group_sign_flip_test    exact paired permutation test on per-group AUC differences d_g (weighted by n_pos * n_neg):
                          under H0 the sign of each d_g is exchangeable; all 2^k sign patterns are enumerated. Groups
                          with d_g = 0 or a single class contribute nothing. The smallest attainable two-sided p-value is
                          2 / 2^k_effective, reported with every result
  holm                    Holm step-down adjustment within a pre-declared family
  tost_from_ci            equivalence is declared only if the (1 - 2 alpha) interval lies inside (-margin, +margin);
                          a non-significant difference is never read as equivalence
  delong                  DeLong test for correlated AUCs; assumes independent samples, which patches are not (reported
                          only as a patch-level reference, never as the confirmatory test)
"""
from __future__ import annotations

import itertools

import numpy as np

from src.v10.evaluation.metrics import roc_auc


def group_auc_differences(y, s1, s2, groups) -> list[dict]:
    y, s1, s2, groups = (np.asarray(v) for v in (y, s1, s2, groups))
    out = []
    for g in np.unique(groups):
        m = groups == g
        yy = y[m]
        n1, n0 = int((yy == 1).sum()), int((yy == 0).sum())
        if n1 == 0 or n0 == 0:
            out.append({"group": str(g), "n": int(m.sum()), "auc1": None, "auc2": None, "delta": None, "weight": 0})
            continue
        a1, a2 = roc_auc(yy, s1[m]), roc_auc(yy, s2[m])
        out.append({"group": str(g), "n": int(m.sum()), "auc1": a1, "auc2": a2, "delta": a1 - a2, "weight": n1 * n0})
    return out


def group_sign_flip_test(y, s1, s2, groups, max_enumerate: int = 20, n_random: int = 100000, seed: int = 0) -> dict:
    rows = group_auc_differences(y, s1, s2, groups)
    use = [r for r in rows if r["delta"] is not None]
    d = np.array([r["delta"] for r in use])
    wts = np.array([r["weight"] for r in use], float)
    if len(d) == 0:
        return {"statistic": None, "p_two_sided": None, "groups": rows}
    wts = wts / wts.sum()
    T = float(np.sum(wts * d))
    k = len(d)
    eff = int((np.abs(d) > 1e-15).sum())
    if k <= max_enumerate:
        signs = np.array(list(itertools.product([1, -1], repeat=k)))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], size=(n_random, k))
    null = signs @ (wts * d)
    p = float(np.mean(np.abs(null) >= abs(T) - 1e-15))
    return {"statistic": T, "p_two_sided": p, "groups_used": k, "groups_with_nonzero_difference": eff,
            "smallest_attainable_p": float(2.0 / 2 ** eff) if eff > 0 else 1.0,
            "enumerated": k <= max_enumerate, "groups": rows}


def group_bootstrap_delta(y, s1, s2, groups, slices=None, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    y, s1, s2, groups = (np.asarray(v) for v in (y, s1, s2, groups))
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_g = {g: np.where(groups == g)[0] for g in ug}
    sl = np.asarray(slices) if slices is not None else None
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        parts = []
        for g in pick:
            ix = idx_g[g]
            if sl is not None:
                us = np.unique(sl[ix])
                for t in rng.choice(us, size=len(us), replace=True):
                    parts.append(ix[sl[ix] == t])
            else:
                parts.append(ix)
        ix = np.concatenate(parts)
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(roc_auc(y[ix], s1[ix]) - roc_auc(y[ix], s2[ix]))
    vals = np.asarray(vals)
    d0 = roc_auc(y, s1) - roc_auc(y, s2)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    lo90, hi90 = np.percentile(vals, [5, 95])
    return {"delta_auc": float(d0), "ci": [float(lo), float(hi)], "ci90": [float(lo90), float(hi90)],
            "n_valid": int(len(vals)), "n_groups": int(len(ug)), "distinct_group_resamples_max": int(len(ug) ** len(ug)),
            "resampling": "groups then slices" if sl is not None else "groups"}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def tost_from_ci(ci90, margin: float) -> dict:
    lo, hi = ci90
    return {"margin": margin, "ci90": [lo, hi], "equivalent": bool(lo > -margin and hi < margin),
            "rule": "equivalence only if the 90% interval lies inside (-margin, +margin)"}


def delong(y, s1, s2) -> dict:
    y = np.asarray(y).astype(int)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(s):
        p, q = s[pos], s[neg]
        psi = (p[:, None] > q[None, :]) + 0.5 * (p[:, None] == q[None, :])
        return psi.mean(1), psi.mean(0), psi.mean()
    v10a, v01a, A = comps(s1)
    v10b, v01b, Bv = comps(s2)
    S10 = np.cov(np.vstack([v10a, v10b]))
    S01 = np.cov(np.vstack([v01a, v01b]))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    if var <= 0:
        return {"delta_auc": float(A - Bv), "z": None, "p_two_sided": None, "note": "zero variance"}
    from math import erf, sqrt
    z = (A - Bv) / np.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return {"delta_auc": float(A - Bv), "z": float(z), "p_two_sided": float(p),
            "assumption": "independent samples (violated for patches from the same scan group)"}
