"""V11 fix for audit finding 3: the radiomics decision tree could never make a confident call.

Root cause (two interacting rules of the V10 final-training pipeline, src/v10/training/final.py):
  1. Calibration: Platt scaling on the tree's score = logit(clip(leaf probability, 1e-6)). Pure leaves give scores of
     +/-13.8, so the fitted Platt slope is tiny (0.14) and every probability is squeezed into [0.103, 0.855].
  2. Threshold: Youden J is searched over the probabilities the model actually attains. For a model with a handful of
     distinct outputs the optimum is one of those values (here the top one, 0.855), so p == threshold, and the
     inference rule "Tumour only if p >= threshold + 0.10" can never be met.

Fix (the tree itself — joblib pipeline, features, preprocessing — is untouched):
  * calibration: isotonic regression on the SAME leave-one-unit-out out-of-fold scores the original Platt fit used
    (results/v10/runs/<run>_v10_decision_tree_radiomics/predictions.csv); no new data, no test data;
  * threshold: Youden J on those calibrated predictions, placed midway between the selected attained probability and
    the next lower attained probability (identical training-set decisions, but no output sits on the threshold).
  * validation: the calibration method is re-fitted leave-one-unit-out (each unit calibrated by the other units) to
    report an honest ECE / sensitivity / specificity; the in-sample numbers are reported as optimistic.
The original v10.0 artifact is archived unchanged in artifacts/archive/v10_decision_tree_radiomics_v10.0/.

    python scripts/v11_recalibrate_decision_tree.py            # writes artifacts/v10/v10_decision_tree_radiomics (v10.1)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MID = "v10_decision_tree_radiomics"
BAND = 0.10


def ece(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == k).mean() * abs(y[i == k].mean() - p[i == k].mean()) for k in range(bins) if (i == k).any()))


def calls(p, thr):
    return {"tumour": round(float((p >= thr + BAND).mean()), 4), "no_tumour": round(float((p < thr - BAND).mean()), 4),
            "uncertain": round(float(((p >= thr - BAND) & (p < thr + BAND)).mean()), 4)}


def main() -> int:
    from sklearn.isotonic import IsotonicRegression
    from src.v10.engine import artifacts as A
    from src.v10.evaluation.metrics import threshold_metrics, youden_threshold, roc_auc
    d = ROOT / "artifacts" / "v10" / MID
    card = A.load_card(d)
    if card.get("version") == "v10.1":
        print("already recalibrated (v10.1); nothing to do")
        return 0
    runs = sorted((ROOT / "results" / "v10" / "runs").glob(f"*_{MID}/predictions.csv"))
    if not runs:
        raise SystemExit("leave-one-unit-out predictions of the final training run not found")
    rows = list(csv.DictReader(open(runs[-1])))
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["oof_score"]) for r in rows])
    units = np.array([r["unit"] for r in rows])
    old_p = np.array([float(r["oof_prob"]) for r in rows])

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    p = iso.predict(s)
    vals = np.unique(p)
    t_att = youden_threshold(y, p)
    lower = vals[vals < t_att]
    thr = float((t_att + lower.max()) / 2) if len(lower) else float(t_att)
    p_louo = np.zeros_like(p)
    for u in np.unique(units):
        m = units == u
        p_louo[m] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s[~m], y[~m]).predict(s[m])
    m_in, m_louo = threshold_metrics(y, p, thr), threshold_metrics(y, p_louo, thr)
    m_old = threshold_metrics(y, old_p, float(card["threshold"]))
    pick = lambda m: {k: round(float(m[k]), 4) for k in ("sensitivity", "specificity", "precision", "f1")}
    validation = {
        "data": f"{runs[-1].relative_to(ROOT).as_posix()} (leave-one-unit-out out-of-fold scores of the final "
                f"training run; {len(y)} patches, {int(y.sum())} tumour)",
        "v10.0_platt": {"threshold": float(card["threshold"]), "attainable_range": [0.103, 0.855],
                        "calls_with_band": calls(old_p, float(card["threshold"])), "ece": round(ece(y, old_p), 4),
                        "auc": round(float(roc_auc(y, old_p)), 4), "at_threshold": pick(m_old)},
        "v10.1_isotonic_in_sample": {"threshold": thr, "attained_probabilities": [round(float(v), 4) for v in vals],
                                     "calls_with_band": calls(p, thr), "ece": round(ece(y, p), 4),
                                     "auc": round(float(roc_auc(y, p)), 4), "at_threshold": pick(m_in),
                                     "note": "optimistic: calibration, threshold and evaluation use the same predictions"},
        "v10.1_isotonic_unit_validated": {"calls_with_band": calls(p_louo, thr), "ece": round(ece(y, p_louo), 4),
                                          "auc": round(float(roc_auc(y, p_louo)), 4), "at_threshold": pick(m_louo),
                                          "note": "each unit calibrated by an isotonic fit on the other units"},
    }
    arch = ROOT / "artifacts" / "archive" / f"{MID}_v10.0"
    if not arch.exists():
        shutil.copytree(d, arch)
    arr = A.load_arrays(d)
    arr["iso_x"] = np.asarray(iso.X_thresholds_, float)
    arr["iso_y"] = np.asarray(iso.y_thresholds_, float)
    np.savez(d / "model.npz", **arr)
    new = dict(card)
    new.update(
        version="v10.1", threshold=thr,
        threshold_rule="Youden J on isotonic-calibrated leave-one-unit-out predictions, placed midway between the "
                       "selected attained probability and the next lower one (V11 fix; v10.0 rule placed it on an "
                       "attained value)",
        calibration={"method": "isotonic", "fitted_on": "leave-one-unit-out out-of-fold scores of the final training run",
                     "replaces": "Platt (kept in model.npz as 'platt' for reference)"},
        recalibration={"script": "scripts/v11_recalibrate_decision_tree.py",
                       "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       "reason": "V10 audit finding 3: bounded Platt output (max 0.855 = threshold) made every call "
                                 "'Uncertain'", "tree_unchanged": True, "archived_original": arch.relative_to(ROOT).as_posix(),
                       "validation": validation},
        files={"model.npz": A.sha256_file(d / "model.npz"), "model.joblib": A.sha256_file(d / "model.joblib")},
        saved_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    new.setdefault("limitations", [])
    if isinstance(new["limitations"], list):
        new["limitations"] = list(new["limitations"]) + [
            "V11: calibration and threshold re-derived from the same leave-one-unit-out predictions as v10.0; the "
            "unit-validated numbers in recalibration.validation are the honest estimate."]
    (d / "model_card.json").write_text(json.dumps(new, indent=2, default=str))
    print(json.dumps(validation, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
