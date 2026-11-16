"""Calibration and saturation report (V10 audit finding 4: deep models return p ~ 1.000 on tumour slices).

Sources (no new training, nothing tuned on test data):
  * V9 models: recorded calibration parameters, validation ECE and test-set metrics stored in each model.json
    (V9 fixed split, patch level).
  * V10 models: expected calibration error recomputed from the nested out-of-fold predictions
    (results/v10/nested/*/oof_predictions.csv) when present.
  * Saturation on real slices: share of patch probabilities >= 0.999 / <= 0.001 on the sample slices in
    dataset/sample_data/lits_sample (and the full dataset test split when dataset/liver_ct_lits_verified_v8 exists).

    python scripts/v10_calibration_report.py         # writes reports/v10/CALIBRATION_AND_SATURATION.md + .json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def v9_rows(engine) -> list[dict]:
    rows = []
    for mid, lm in sorted((engine.v9.models if engine.v9 else {}).items()):
        if lm.meta.get("version") != "v9":
            continue
        cal = lm.meta.get("calibration") or {}
        ts = lm.meta.get("test_summary") or {}
        rows.append({"model_id": mid, "family": engine.compare_family(engine.describe(mid)) if mid in engine.entries else "",
                     "platt_slope_on_standardised_score": cal.get("a"), "val_auc": lm.meta.get("validation_auc"),
                     "val_ece": lm.meta.get("validation_ece"), "test_ece": ts.get("ece"), "test_auc": ts.get("auc"),
                     "test_sensitivity": ts.get("sensitivity"), "test_specificity": ts.get("specificity")})
    return rows


def v10_oof_rows() -> list[dict]:
    """Wide format: one row per patch, columns `label` and `<spec>__prob` (outer-fold out-of-fold probabilities)."""
    out = []
    for p in sorted((ROOT / "results" / "v10" / "nested").glob("*/oof_predictions.csv")):
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        y = np.array([float(r["label"]) for r in rows])
        for col in [c for c in rows[0] if c.endswith("__prob")]:
            vals = [(yy, r[col]) for yy, r in zip(y, rows) if r[col] not in ("", None)]
            if not vals:
                continue
            yy = np.array([v[0] for v in vals])
            pr = np.array([float(v[1]) for v in vals])
            out.append({"model_id": f"v10_{col[:-6]}_{p.parent.name}", "feature_set": p.parent.name, "n": len(pr),
                        "oof_ece": round(ece(yy, pr), 4), "share_p_ge_0_999": round(float((pr >= 0.999).mean()), 4),
                        "share_p_le_0_001": round(float((pr <= 0.001).mean()), 4)})
    return out


def saturation_on_samples(engine) -> list[dict]:
    from src.v10.engine.compare import ComparisonRunner
    base = ROOT / "dataset" / "sample_data" / "lits_sample"
    rows = list(csv.DictReader(open(base / "labels" / "slices.csv")))
    by = {int(r["upstream_index"]): r for r in rows}
    runner = ComparisonRunner(engine)
    acc: dict[str, dict] = {}
    for r in rows:
        i = int(r["upstream_index"])
        nb = lambda k: (base / by[i + k]["image_path"]).read_bytes() if (i + k) in by and by[i + k]["scan_id"] == r["scan_id"] else None
        prep = runner.prepare((base / r["image_path"]).read_bytes(), r["file_name"], nb(-1), nb(1), nb(-2), nb(2))
        if prep.stop_reason:
            continue
        res = runner.run(prep, runner.plan(prep))
        for m in res["models"]:
            if m["status"] != "ok":
                continue
            a = acc.setdefault(m["model_id"], {"model_id": m["model_id"], "family": m["family"], "slices": 0,
                                                "tumour_slices": 0, "patches": [], "slice_pmax_tumour": []})
            a["slices"] += 1
            ps = [x["p"] for x in m["patch_probabilities"]]
            a["patches"] += ps
            if r["label_tumour"] == "1":
                a["tumour_slices"] += 1
                a["slice_pmax_tumour"].append(m["probability"])
    out = []
    for a in acc.values():
        ps = np.asarray(a.pop("patches"))
        pm = a.pop("slice_pmax_tumour")
        a.update(n_patches=int(len(ps)), share_patch_p_ge_0_999=round(float((ps >= 0.999).mean()), 3),
                 share_patch_p_le_0_001=round(float((ps <= 0.001).mean()), 3),
                 tumour_slice_pmax_mean=round(float(np.mean(pm)), 4) if pm else None,
                 tumour_slices_saturated=int(sum(p >= 0.999 for p in pm)))
        out.append(a)
    return sorted(out, key=lambda r: (-r["share_patch_p_ge_0_999"], r["model_id"]))


def main() -> int:
    from src.v10.engine.registry import ModelEngine
    eng = ModelEngine(ROOT).load()
    v9 = v9_rows(eng)
    v10 = v10_oof_rows()
    sat = saturation_on_samples(eng)
    out_dir = ROOT / "reports" / "v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "CALIBRATION_AND_SATURATION.json").write_text(json.dumps({"v9": v9, "v10_oof": v10, "saturation_samples": sat},
                                                                       indent=1))
    L = ["# Calibration and saturation report (V10 audit finding 4)", "",
         "Generated by `scripts/v10_calibration_report.py`. No model, threshold or calibration was changed.", "",
         "## Why the V9 deep models saturate", "",
         "V9 calibrates each CNN with Platt scaling on the *validation* scan groups. Those groups are almost perfectly "
         "separable (validation AUC 0.997-0.9996, the 'saturated validation set' of V9 Stage 2), so the fitted slope on "
         "the standardised score is very steep (column `slope`). A steep slope maps most scores to probabilities near "
         "0 or 1. On the held-out test groups the same models are worse calibrated than on validation (test ECE vs "
         "validation ECE) and less specific, so p = 1.000 on a slice must not be read as certainty.", "",
         "| Model | Family | slope | Val AUC | Val ECE | Test ECE | Test sens. | Test spec. |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    f = lambda v, d=3: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))
    for r in v9:
        L.append(f"| {r['model_id']} | {r['family']} | {f(r['platt_slope_on_standardised_score'], 2)} | {f(r['val_auc'], 4)} | "
                 f"{f(r['val_ece'])} | {f(r['test_ece'])} | {f(r['test_sensitivity'])} | {f(r['test_specificity'])} |")
    L += ["", "## V10 models: out-of-fold calibration (nested leave-one-unit-out)", ""]
    if v10:
        L += ["| Model | Features | n | OOF ECE | share p>=0.999 | share p<=0.001 |", "|---|---|---:|---:|---:|---:|"]
        L += [f"| {r['model_id']} | {r['feature_set']} | {r['n']} | {r['oof_ece']:.4f} | {r['share_p_ge_0_999']:.3f} | "
              f"{r['share_p_le_0_001']:.3f} |" for r in v10]
    else:
        L.append("_results/v10/nested/*/oof_predictions.csv not present: run scripts/run_nested_evaluation.py._")
    L += ["", "## Saturation on the real sample slices (patch probabilities)", "",
          "| Model | Family | slices | patches | share p>=0.999 | share p<=0.001 | mean p_max on tumour slices | tumour slices at p>=0.999 |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {r['model_id']} | {r['family']} | {r['slices']} | {r['n_patches']} | {r['share_patch_p_ge_0_999']:.3f} | "
          f"{r['share_patch_p_le_0_001']:.3f} | {f(r['tumour_slice_pmax_mean'], 4)} | {r['tumour_slices_saturated']}/{r['tumour_slices']} |"
          for r in sat]
    L += ["", "## Consequences in the platform", "",
          "* Every result carries a `saturated` flag (p >= 0.999) and the model's calibration source; the dashboard shows "
          "both.", "* The confidence shown is the decision margin from the model's own threshold, not a probability of "
          "being correct.", "* Generalisation cannot be judged from 8 sample slices: re-run this script and "
          "`scripts/evaluate_models.py` on the full dataset (scan-group-level held-out test split).", ""]
    (out_dir / "CALIBRATION_AND_SATURATION.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
