#!/usr/bin/env python3
"""Write reports/v10/V10_NESTED_RESULTS.md from results/v10/nested/<features>/ (every number comes from the result
files; nothing is typed by hand).   python scripts/write_results_report.py"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NESTED = ROOT / "results" / "v10" / "nested"
FAMILY = {"classical": "classical", "control": "classical control", "quantum": "quantum (simulated)",
          "hybrid": "hybrid (simulated)"}


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def section(features: str) -> list[str]:
    d = NESTED / features
    r = json.loads((d / "results.json").read_text())
    man = json.loads((d / "run_manifest.json").read_text()) if (d / "run_manifest.json").exists() else {}
    comp = json.loads((d / "comparisons.json").read_text()) if (d / "comparisons.json").exists() else {}
    fi = r.get("feature_info") or {}
    dim = fi.get("dim")
    cache = sorted((ROOT / "results" / "v10" / "features").glob(f"{features}_*.npz"))
    note = ""
    if cache:
        import numpy as np
        actual = int(np.load(cache[0])["X"].shape[1])
        if dim is not None and actual != dim:
            note = (f" (`feature_info.dim` in results.json says {dim}: a stale label from the code version at run start; "
                    f"the cached feature matrix the run loaded has {actual} columns)")
        dim = actual
    L = [f"## Feature set `{features}`", "",
         f"Dataset `{r['dataset_version']}`, {r['n_patches']} patches, units {', '.join(r['units'])}; "
         f"primary metric over the {len(r['qualifying_units'])} units with ≥ 10 patches per class "
         f"({', '.join(r['qualifying_units'])}); inner {r['inner_k']}-fold; seeds {r['seeds']} for stochastic models. "
         f"Features: {fi.get('description', features)}, {dim if dim is not None else '?'} dimensions{note}, status "
         f"{fi.get('status', '?')}. "
         + (f"Analysis plan sha256 `{man.get('analysis_plan_sha256', '')[:16]}…`; " if man else "")
         + (f"run time {man.get('seconds', 0) / 60:.0f} min." if man.get("seconds") else ""), "",
         "| Model | Family | Mean within-unit AUC | Min – max over units | Seed SD | Pooled outer-fold AUC | "
         "Balanced acc. @ fold thr. | Sensitivity | Specificity | Brier | ECE | Params | Predict ms / patch |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(r["models"].items(), key=lambda kv: -kv[1]["primary_mean_within_unit_auc"])
    for k, v in rows:
        po, pt = v["pooled_oof"], v.get("pooled_threshold_metrics") or {}
        L.append(f"| `{k}` | {FAMILY.get(v['family'], v['family'])} | **{f(v['primary_mean_within_unit_auc'])}** | "
                 f"{f(v['within_unit_auc_min'])} – {f(v['within_unit_auc_max'])} | {f(v.get('mean_seed_sd_within_unit'))} | "
                 f"{f(po.get('roc_auc'))} | {f(pt.get('balanced_accuracy'))} | {f(pt.get('sensitivity'))} | "
                 f"{f(pt.get('specificity'))} | {f(po.get('brier'))} | {f(po.get('ece'))} | "
                 f"{v.get('n_parameters') or '—'} | {f(v.get('predict_ms_per_patch'), 4)} |")
    L += ["", "Per-unit within-unit AUC (outer folds; mean over seeds):", "",
          "| Model | " + " | ".join(r["units"]) + " |", "|---|" + "---|" * len(r["units"])]
    for k, v in rows:
        pu = v["per_unit"]
        L.append(f"| `{k}` | " + " | ".join(f(pu.get(u, {}).get("within_unit_auc")) for u in r["units"]) + " |")
    if comp.get("comparisons"):
        fam = comp.get("family")
        title = "Confirmatory comparisons (fixed in the analysis plan before the run)" if fam == "confirmatory" else \
            "Secondary comparisons (not part of the confirmatory family; exploratory)"
        L += ["", f"### {title} — exact sign-flip test over units, Holm correction within this table", "",
              "| Comparison | Mean AUC difference | Units favouring first / second | p (two-sided) | p (Holm) | "
              "Smallest attainable p | Reading |", "|---|---|---|---|---|---|---|"]
        for name, c in comp["comparisons"].items():
            L.append(f"| {name} | {c['mean_difference']:+.4f} | {c['units_favouring_first']} / "
                     f"{c['units_favouring_second']} | {c['p_two_sided']:.3f} | {c['p_holm']:.3f} | "
                     f"{c['smallest_attainable_p']:.4f} | {c['interpretation']} |")
    L.append("")
    return L


def main() -> None:
    feats = [p.name for p in sorted(NESTED.iterdir()) if (p / "results.json").exists()] if NESTED.exists() else []
    L = ["# V10 nested evaluation — results", "",
         "> This platform is intended for academic research and software experimentation only. It has not been "
         "clinically validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. "
         "Model outputs are experimental predictions and require independent clinical validation.", "",
         "Generated by `scripts/write_results_report.py` from `results/v10/nested/*/results.json`. Design "
         "(`configs/v10/analysis_plan.json`): leave one reconstructed scan unit out (8 outer folds); hyperparameters, "
         "Platt calibration and the Youden threshold chosen on grouped inner folds of the training units only; "
         "primary metric = unweighted mean of within-unit ROC AUC (tumour-vs-tumour-free pairs inside the held-out "
         "unit). Units are reconstructed scan groups, **not patients**. Quantum and hybrid models run on a classical "
         "NumPy state-vector simulator; no quantum hardware was used and no quantum advantage is claimed.", ""]
    if not feats:
        L.append("No nested results found.")
    for ft in feats:
        L += section(ft)
    L += ["## How to read this", "",
          "* The within-unit AUC asks: inside one held-out scan unit, does the model rank tumour patches above "
          "tumour-free patches? The pooled outer-fold AUC mixes units and is lower because score levels shift between "
          "units; together with the ECE values this shows the probabilities are not calibrated across units.",
          "* Differences that are not significant are not evidence of equivalence: with 7 qualifying units the "
          "smallest attainable two-sided p of the exact sign-flip test is 1/64.",
          "* These estimates are for 96 × 96 liver-tissue patches selected with expert masks, on one public dataset.",
          ""]
    out = ROOT / "reports" / "v10" / "V10_NESTED_RESULTS.md"
    out.write_text("\n".join(L))
    print(f"wrote {out.relative_to(ROOT)} ({', '.join(feats) or 'no results'})")


if __name__ == "__main__":
    main()
