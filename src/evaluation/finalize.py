"""Common post-training step for every model: calibrate on validation, pick the threshold on validation, write the
per-patch predictions for validation and test (test scores are only *recorded* here; they play no role in any choice).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.common.paths import MANIFESTS, RESULTS
from src.evaluation.guards import CalibrationCollapsed, SafeCalibrator, check_predictions
from src.evaluation.metrics import choose_threshold

PRED_DIR = RESULTS / "predictions"


def patch_table() -> pd.DataFrame:
    pa = pd.read_csv(MANIFESTS / "patches_task_a.csv")
    pa["row"] = np.arange(len(pa))
    return pa


def finalize(model_id: str, family: str, y_val, s_val, s_test, test_rows, val_rows, pa: pd.DataFrame,
             infer_seconds_per_patch: float | None = None) -> dict:
    y_val = np.asarray(y_val).astype(int)
    chk = check_predictions(np.asarray(s_val, float))
    try:
        cal = SafeCalibrator().fit(s_val, y_val)
        p_val, p_test = cal(np.asarray(s_val, float)), cal(np.asarray(s_test, float))
        cal_d, cal_status = cal.to_dict(), "platt_on_validation"
    except CalibrationCollapsed as e:  # never replaced by a constant: the model is reported as not calibrated
        p_val = 1 / (1 + np.exp(-np.asarray(s_val, float)))
        p_test = 1 / (1 + np.exp(-np.asarray(s_test, float)))
        cal_d, cal_status = None, f"calibration_failed: {e}"
    thr = choose_threshold(y_val, p_val, "youden")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, idx, s, p in (("val", val_rows, s_val, p_val), ("test", test_rows, s_test, p_test)):
        sub = pa.iloc[idx]
        rows.append(pd.DataFrame({"row": sub.row.to_numpy(), "split": split, "upstream_index": sub.upstream_index.to_numpy(),
                                  "scan_id": sub.scan_id.to_numpy(), "y": sub.label.to_numpy(), "score": np.asarray(s, float),
                                  "prob": np.asarray(p, float)}))
    pd.concat(rows).to_csv(PRED_DIR / f"{model_id}.csv", index=False)
    return {"calibration": cal_d, "calibration_status": cal_status, "threshold": float(thr),
            "threshold_rule": "Youden J on validation (calibrated probability)",
            "validation_score_check": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in chk.items()
                                       if k in ("constant_scores", "ok")},
            "inference_seconds_per_patch": infer_seconds_per_patch, "family": family}


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0
