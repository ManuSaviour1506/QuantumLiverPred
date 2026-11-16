"""Train deployable V10 models on all units and save them as integrity-checked artifacts.

Configuration: the one most often selected by the inner loops of the nested evaluation (falls back to the first grid
entry when no nested result exists, which is recorded). Calibration (Platt) and the operating threshold (Youden J)
come from leave-one-unit-out out-of-fold predictions over all training units; the final model is then fitted on all
units. The model card records the nested evaluation's outer-fold results as the model's performance estimate — the
final model itself has no untouched test set, and the card says so.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v10.engine import artifacts as A
from src.v10.engine.v10_models import late_fusion_to_arrays, pp_to_arrays
from src.v10.evaluation.grouped_cv import FoldPreprocessor, assert_disjoint, outer_folds
from src.v10.evaluation.metrics import full_metrics, roc_auc, youden_threshold
from src.v10.evaluation.nested import ModelSpec, apply_platt, platt
from src.v10.training.runs import RunRecorder

KIND = {"classical": "v10_patch_sklearn", "control": None, "quantum": None, "hybrid": "v10_patch_hybrid"}
RADIOMICS_NAMES = (["mean", "std", "skewness", "kurtosis", "p10", "p25", "p50", "p75", "p90", "entropy"]
                   + [f"band_{i}" for i in range(5)]
                   + [f"glcm_d{d}_{p}" for d in (1, 3) for p in ("contrast", "dissimilarity", "homogeneity", "energy",
                                                                 "correlation", "ASM")]
                   + [f"lbp_{i}" for i in range(10)] + ["grad_mean", "grad_std", "grad_p90"])


def kind_of(spec: ModelSpec) -> str:
    k = spec.key
    if k == "vqc":
        return "v10_patch_vqc"
    if k == "qksvm":
        return "v10_patch_qksvm"
    if k == "late_fusion":
        return "v10_patch_late_fusion"
    if k.startswith("hybrid_"):
        return "v10_patch_hybrid"
    return "v10_patch_sklearn"


def chosen_config(spec: ModelSpec, nested: dict | None) -> tuple[dict, str]:
    if nested and spec.key in nested.get("models", {}):
        sel = [json.dumps(s["config"], sort_keys=True) for s in nested["models"][spec.key]["selected_configs"]]
        cfg, n = collections.Counter(sel).most_common(1)[0]
        return json.loads(cfg), f"most frequent inner-loop selection ({n} of {len(sel)} outer folds)"
    return spec.grid[0], "first grid entry (no nested evaluation result available)"


def _arrays_for(spec: ModelSpec, model, pp: FoldPreprocessor, ab) -> tuple[dict, object]:
    arrays = pp_to_arrays(pp)
    arrays["platt"] = np.array(ab, float)
    sk = None
    k = kind_of(spec)
    if k == "v10_patch_sklearn":
        sk = model.pipe
    elif k == "v10_patch_late_fusion":
        arrays.update(late_fusion_to_arrays(model))
    else:
        arrays.update(model.to_arrays())
    return arrays, sk


def nested_evaluation_summary(spec_key: str, nested: dict | None, features: str) -> list[dict]:
    if not nested or spec_key not in nested.get("models", {}):
        return [{"name": "nested evaluation", "status": "not available for this model"}]
    m = nested["models"][spec_key]
    return [{
        "name": f"Nested leave-one-unit-out evaluation ({features})", "status": "completed (real data)",
        "evaluation_unit": nested.get("evaluation_unit"), "units": nested.get("units"),
        "qualifying_units": nested.get("qualifying_units"), "split": "outer: leave one unit out; inner: grouped 3-fold",
        "metrics": {"primary_mean_within_unit_auc": m["primary_mean_within_unit_auc"],
                    "within_unit_auc_min": m["within_unit_auc_min"], "within_unit_auc_max": m["within_unit_auc_max"],
                    "pair_weighted_within_unit_auc": m["pair_weighted_within_unit_auc"],
                    **{k: m["pooled_threshold_metrics"].get(k) for k in
                       ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
                        "precision", "f1", "mcc", "log_loss", "brier", "ece")},
                    "mean_seed_sd_within_unit": m["mean_seed_sd_within_unit"],
                    "fit_seconds_mean": m["fit_seconds_mean"], "predict_ms_per_patch": m["predict_ms_per_patch"]},
        "per_unit_auc": {u: r["within_unit_auc"] for u, r in m["per_unit"].items()},
        "caveats": ["patch-level estimate on reconstructed scan-group units, not patients",
                    "pooled metrics use per-fold thresholds; the primary metric is the mean within-unit AUC"]}]


def train_final(root: Path, spec: ModelSpec, X: np.ndarray, y: np.ndarray, units: np.ndarray, ds: dict,
                features: str, nested: dict | None, seed: int = 0, model_id: str | None = None) -> Path:
    model_id = model_id or f"v10_{spec.key}_{features}"
    rec = RunRecorder(root, model_id)
    cfg, how = chosen_config(spec, nested)
    rec.log(f"{model_id}: config {cfg} ({how}); {len(y)} patches, {len(np.unique(units))} units, seed {seed}")
    oof = np.full(len(y), np.nan)
    for u, tr, te in outer_folds(units):
        assert_disjoint(units, tr, te)
        pp = FoldPreprocessor(**spec.preproc).fit(X, tr)
        m = spec.build(cfg, seed).fit(pp.transform(X[tr]), y[tr])
        oof[te] = m.decision_function(pp.transform(X[te]))
        rec.log(f"  calibration fold {u}: within-unit AUC "
                f"{roc_auc(y[te], oof[te]) if len(np.unique(y[te])) == 2 else 'n/a (one class)'}")
    ab = platt(oof, y)
    prob = apply_platt(oof, ab)
    thr = youden_threshold(y, prob)
    pp = FoldPreprocessor(**spec.preproc).fit(X)
    model = spec.build(cfg, seed).fit(pp.transform(X), y)
    arrays, sk = _arrays_for(spec, model, pp, ab)
    describe = model.describe() if hasattr(model, "describe") else None
    n_params = model.n_params() if hasattr(model, "n_params") else None
    if n_params is None and sk is not None:
        from src.v10.classical.models import n_parameters
        n_params = n_parameters(sk)
    importance = []
    if sk is not None:
        from src.v10.classical.models import feature_importance
        importance = feature_importance(sk, RADIOMICS_NAMES if features == "radiomics" else None)
    fam = {"classical": "classical", "control": "classical", "quantum": "quantum", "hybrid": "hybrid"}[spec.family]
    if spec.key == "hybrid_twin":
        fam = "classical"
    exec_mode = "custom_numpy_simulator" if fam in ("quantum", "hybrid") else "classical_cpu_numpy"
    card = {
        "name": f"{spec.name} — V10, {features}", "family": fam, "kind": kind_of(spec),
        "architecture": spec.name.split(" (")[0].split(" [")[0], "architecture_detail": describe,
        "task": "tumour vs tumour-free liver tissue (96x96 patches) aggregated to a slice-level call",
        "role": {"control": "control"}.get(spec.family, "primary" if spec.key in ("vqc", "qksvm", "hybrid_no_fusion",
                                                                                 "logreg") else "comparison"),
        "version": "v10.0", "feature_set": features, "feature_dim": int(X.shape[1]),
        "feature_names": RADIOMICS_NAMES if features == "radiomics" else None,
        "preprocessing": spec.preproc, "preprocessing_signature": A.preprocessing_signature(features, spec.preproc),
        "selected_config": cfg, "config_selection": how, "seeds": [seed],
        "training_data": "real", "dataset_version": ds["dataset_version"], "dataset_license": ds["license"],
        "data_split": {"final_fit": "all units", "calibration_and_threshold": "leave-one-unit-out out-of-fold scores",
                       "units": sorted(np.unique(units).tolist()), "n_patches": int(len(y)),
                       "positives": int(y.sum()), "negatives": int((1 - y).sum())},
        "threshold": float(thr), "threshold_rule": "Youden J on calibrated leave-one-unit-out predictions",
        "platt": list(ab), "n_parameters": n_params, "execution_mode": exec_mode,
        "circuit": describe if fam in ("quantum", "hybrid") else None, "feature_importance": importance,
        "evaluations": nested_evaluation_summary(spec.key, nested, features)
        + [{"name": "Leave-one-unit-out calibration predictions (final configuration)", "status": "completed",
            "evaluation_unit": "reconstructed scan-group units", "metrics": full_metrics(y, prob, prob, thr),
            "caveats": ["same predictions were used to fit calibration and the threshold: optimistic"]}],
        "training_history": getattr(model, "history", None),
        "limitations": ["research use only; not clinically validated",
                        "liver gate inherited from V9 (known edge-patch label defect)",
                        "slice-level 'No tumour detected' is not validated: the dataset has no tumour-free liver slices"],
    }
    d = A.save(root / "artifacts" / "v10", model_id, card, arrays, sk)
    pd.DataFrame({"unit": units, "label": y, "oof_score": oof, "oof_prob": prob}).to_csv(
        rec.dir / "predictions.csv", index=False)
    rec.write("config.json", {"model_id": model_id, "spec": spec.key, "config": cfg, "selection": how,
                              "preprocessing": spec.preproc, "seed": seed, "features": features,
                              "architecture": describe})
    rec.write("split.json", card["data_split"])
    rec.write("metrics.json", card["evaluations"])
    rec.finish("succeeded", artifact=str(d.relative_to(root)), dataset_version=ds["dataset_version"])
    return d
