"""Nested, unit-grouped cross-validation for classical, quantum and hybrid models.

Outer loop: leave one reconstructed scan-group unit out (8 units). Inner loop: grouped k-fold over the remaining units
(no unit is ever split between fit and validation). Everything that is chosen — preprocessing parameters, PCA,
hyperparameters, Platt calibration, the operating threshold — is chosen from training units only; the outer unit is
scored once per seed.

Primary estimand (frozen in configs/v10/analysis_plan.json before any outer prediction exists): within-unit ROC-AUC
(positive-negative pairs inside the held-out unit only), summarised as the unweighted mean over units with at least
10 patches of each class. Pooled out-of-fold AUC, pair-weighted AUC and every unit's own result are secondary.
Seeds are averaged within a unit before inference; seed variability is reported separately from unit variability.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.v10.evaluation.grouped_cv import FoldPreprocessor, assert_disjoint, inner_folds, outer_folds
from src.v10.evaluation.metrics import full_metrics, roc_auc, youden_threshold


@dataclass
class ModelSpec:
    key: str
    name: str
    family: str                       # classical | quantum | hybrid | control
    grid: list[dict]
    build: Callable[[dict, int], object]
    preproc: dict = field(default_factory=dict)
    stochastic: bool = False
    fixed_from: str | None = None     # reuse the configuration selected for another spec (ablations)


class SklearnAdapter:
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def decision_function(self, X):
        from src.v10.classical.models import decision
        return decision(self.pipe, X)


def platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    s = np.asarray(scores, float).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(s, np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(scores, ab) -> np.ndarray:
    z = ab[0] * np.asarray(scores, float) + ab[1]
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def qualifying_units(y, units, min_per_class: int = 10) -> list[str]:
    out = []
    for u in np.unique(units):
        yy = y[units == u]
        if (yy == 1).sum() >= min_per_class and (yy == 0).sum() >= min_per_class:
            out.append(str(u))
    return out


def sign_flip(deltas: dict[str, float]) -> dict:
    """Exact two-sided sign-flip test of the unweighted mean of per-unit differences."""
    keys = sorted(deltas)
    d = np.array([deltas[k] for k in keys], float)
    if len(d) == 0:
        return {"p_two_sided": None}
    T = float(d.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = (signs * d).mean(1)
    p = float(np.mean(np.abs(null) >= abs(T) - 1e-12))
    return {"mean_difference": T, "p_two_sided": p, "units": len(d), "smallest_attainable_p": 2 / 2 ** len(d),
            "units_favouring_first": int((d > 0).sum()), "units_favouring_second": int((d < 0).sum()),
            "per_unit": {k: float(v) for k, v in zip(keys, d)}}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def _fit_predict(spec: ModelSpec, cfg: dict, seed: int, X, y, fit_idx, pred_idx):
    pp = FoldPreprocessor(**spec.preproc).fit(X, fit_idx)
    Xf, Xp = pp.transform(X[fit_idx]), pp.transform(X[pred_idx])
    model = spec.build(cfg, seed)
    t0 = time.perf_counter()
    model.fit(Xf, y[fit_idx])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    s = np.asarray(model.decision_function(Xp), float)
    pred_ms = (time.perf_counter() - t1) * 1000 / max(1, len(pred_idx))
    return s, model, pp, fit_s, pred_ms


def run_nested(X: np.ndarray, y: np.ndarray, units: np.ndarray, specs: list[ModelSpec], seeds: list[int],
               inner_k: int = 3, log=print, min_per_class: int = 10) -> dict:
    X, y, units = np.asarray(X, float), np.asarray(y).astype(int), np.asarray(units).astype(str)
    qual = qualifying_units(y, units, min_per_class)
    folds = outer_folds(units)
    results: dict = {"units": sorted(np.unique(units).tolist()), "qualifying_units": qual, "inner_k": inner_k,
                     "seeds": seeds, "models": {}}
    selected: dict[str, list[dict]] = {}
    for spec in specs:
        t_spec = time.perf_counter()
        per_unit, oof = {}, np.full((len(seeds) if spec.stochastic else 1, len(y)), np.nan)
        oof_prob = np.full_like(oof, np.nan)
        thresholds = np.full(len(y), np.nan)
        chosen = []
        for fi, (u, tr, te) in enumerate(folds):
            assert_disjoint(units, tr, te)
            inner = inner_folds(units, tr, inner_k, seed=fi)
            for _b, fit_idx, va_idx in inner:
                assert_disjoint(units, fit_idx, va_idx, te)
            # ---- inner selection (seed 0 only)
            if spec.fixed_from:
                cands = [selected[spec.fixed_from][fi]["config"]]
            else:
                cands = spec.grid
            best = None
            for cfg in cands:
                scores_va = np.full(len(y), np.nan)
                aucs = []
                for _b, fit_idx, va_idx in inner:
                    s, *_ = _fit_predict(spec, cfg, 0, X, y, fit_idx, va_idx)
                    scores_va[va_idx] = s
                    if len(np.unique(y[va_idx])) == 2:
                        aucs.append(roc_auc(y[va_idx], s))
                score = float(np.mean(aucs)) if aucs else -np.inf
                if best is None or score > best["inner_auc"]:
                    best = {"config": cfg, "inner_auc": score, "oof": scores_va.copy()}
            m = ~np.isnan(best["oof"])
            ab = platt(best["oof"][m], y[m])
            thr = youden_threshold(y[m], apply_platt(best["oof"][m], ab))
            thresholds[te] = thr
            seed_list = seeds if spec.stochastic else [seeds[0]]
            seed_rows = []
            fit_secs, pred_ms, n_params, describe = [], [], None, None
            for si, sd in enumerate(seed_list):
                s, model, pp, fs, pm = _fit_predict(spec, best["config"], sd, X, y, tr, te)
                p = apply_platt(s, ab)
                oof[si, te], oof_prob[si, te] = s, p
                fit_secs.append(fs)
                pred_ms.append(pm)
                if n_params is None:
                    n_params = getattr(model, "n_params", lambda: None)() if hasattr(model, "n_params") else None
                    describe = model.describe() if hasattr(model, "describe") else None
                yy = y[te]
                seed_rows.append({"seed": sd, "within_unit_auc": roc_auc(yy, s) if len(np.unique(yy)) == 2 else None})
            mean_s, mean_p = oof[:, te].mean(0), oof_prob[:, te].mean(0)
            yy = y[te]
            unit_auc = roc_auc(yy, mean_s) if len(np.unique(yy)) == 2 else None
            seed_aucs = [r["within_unit_auc"] for r in seed_rows if r["within_unit_auc"] is not None]
            per_unit[u] = {"n": int(len(te)), "positives": int(yy.sum()), "negatives": int((1 - yy).sum()),
                           "pairs": int(yy.sum() * (1 - yy).sum()), "within_unit_auc": unit_auc,
                           "seed_aucs": seed_aucs, "seed_sd": float(np.std(seed_aucs, ddof=1)) if len(seed_aucs) > 1 else None,
                           "threshold": float(thr), "platt": ab, "selected_config": best["config"],
                           "inner_auc": best["inner_auc"],
                           "threshold_metrics": full_metrics(yy, mean_s, mean_p, thr) if len(np.unique(yy)) == 2 else
                           {"accuracy": float(np.mean((mean_p >= thr) == yy))},
                           "fit_seconds_mean": float(np.mean(fit_secs)), "predict_ms_per_patch": float(np.mean(pred_ms))}
            chosen.append({"unit": u, "config": best["config"], "inner_auc": best["inner_auc"]})
            log(f"  {spec.key:<24} outer unit {u:<8} AUC {unit_auc if unit_auc is None else round(unit_auc, 4)}"
                f"  config {best['config']}")
        selected[spec.key] = chosen
        q_aucs = [per_unit[u]["within_unit_auc"] for u in qual if per_unit[u]["within_unit_auc"] is not None]
        pairs = np.array([per_unit[u]["pairs"] for u in qual], float)
        mean_s_all, mean_p_all = np.nanmean(oof, 0), np.nanmean(oof_prob, 0)
        pooled = full_metrics(y, mean_p_all, mean_p_all, 0.5)
        pooled_thr = full_metrics(y, mean_p_all, mean_p_all, float(np.nanmedian(thresholds)))
        sds = [per_unit[u]["seed_sd"] for u in qual if per_unit[u]["seed_sd"] is not None]
        results["models"][spec.key] = {
            "name": spec.name, "family": spec.family, "preprocessing": spec.preproc,
            "primary_mean_within_unit_auc": float(np.mean(q_aucs)) if q_aucs else None,
            "within_unit_auc_min": float(np.min(q_aucs)) if q_aucs else None,
            "within_unit_auc_max": float(np.max(q_aucs)) if q_aucs else None,
            "pair_weighted_within_unit_auc": float(np.sum(pairs * np.array(q_aucs)) / pairs.sum()) if q_aucs else None,
            "pooled_oof": pooled, "pooled_oof_at_fold_thresholds": {
                k: v for k, v in full_metrics(y, mean_p_all, mean_p_all, 0.5).items() if k in ("roc_auc", "pr_auc")},
            "pooled_threshold_metrics": pooled_thr,
            "mean_seed_sd_within_unit": float(np.mean(sds)) if sds else None,
            "per_unit": per_unit, "selected_configs": chosen,
            "n_parameters": n_params, "architecture": describe,
            "fit_seconds_mean": float(np.mean([per_unit[u]["fit_seconds_mean"] for u in per_unit])),
            "predict_ms_per_patch": float(np.mean([per_unit[u]["predict_ms_per_patch"] for u in per_unit])),
            "total_seconds": round(time.perf_counter() - t_spec, 1),
        }
        results.setdefault("_oof", {})[spec.key] = {"score": mean_s_all, "prob": mean_p_all}
        log(f"{spec.key}: mean within-unit AUC {results['models'][spec.key]['primary_mean_within_unit_auc']}"
            f" ({results['models'][spec.key]['total_seconds']} s)")
    return results


def compare(results: dict, pairs: list[tuple[str, str]], family_name: str) -> dict:
    """Per-unit AUC differences, exact sign-flip tests and Holm correction for a pre-declared family."""
    qual = results["qualifying_units"]
    out, pv = {}, {}
    for a, b in pairs:
        ma, mb = results["models"].get(a), results["models"].get(b)
        if not ma or not mb:
            continue
        d = {u: ma["per_unit"][u]["within_unit_auc"] - mb["per_unit"][u]["within_unit_auc"] for u in qual
             if ma["per_unit"][u]["within_unit_auc"] is not None and mb["per_unit"][u]["within_unit_auc"] is not None}
        t = sign_flip(d)
        out[f"{a} vs {b}"] = t
        if t["p_two_sided"] is not None:
            pv[f"{a} vs {b}"] = t["p_two_sided"]
    adj = holm(pv)
    for k in out:
        out[k]["p_holm"] = adj.get(k)
        out[k]["interpretation"] = ("significant after Holm correction" if adj.get(k) is not None and adj[k] < 0.05 else
                                    "not significant: this is NOT evidence of equivalence")
    return {"family": family_name, "comparisons": out}
