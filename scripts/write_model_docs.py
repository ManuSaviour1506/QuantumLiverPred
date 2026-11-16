#!/usr/bin/env python3
"""Write MODEL_DOCUMENTATION.md: fixed descriptions of each model family plus tables generated from the model
registry export (artifacts/v10/MODEL_REGISTRY.json — run export_model_registry.py first).
    python scripts/write_model_docs.py"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = ("This platform is intended for academic research and software experimentation only. It has not been "
              "clinically validated and must not be used for diagnosis, treatment decisions or emergency medical "
              "decisions. Model outputs are experimental predictions and require independent clinical validation.")

PROSE = """
## How every model is served

All models share one interface (`src/v10/engine/interface.py`: `load_model`, `preprocess`, `predict`,
`predict_proba`, `get_metadata`, `get_metrics`) and one front end (`src/v10/engine/patch_pipeline.py`):

1. decode PNG / JPEG / DICOM / NIfTI (PNG and JPEG are recorded as having no DICOM / NIfTI metadata);
2. image quality control (size, dynamic range, constant images) and the V9 out-of-distribution detector
   (non-CT images are withheld, not scored);
3. a 96 × 96 grid with stride 64 over the body; the V9 liver-tissue gate keeps liver patches;
4. each model scores every liver patch; the slice call is the maximum patch probability against the model's threshold,
   with an "uncertain" band of ± 0.10 around it;
5. explanations: a patch probability map for every model, Grad-CAM for the V9 CNNs, coefficient contributions for linear
   models, numerical feature sensitivity for the others. Explanations show what drives the model's score, not where the
   tumour is.

A model is served only if its artifact exists, its SHA-256 matches the card, its preprocessing signature matches the
code and its training data are real. Models that need adjacent slices ask for them in the upload form; the five-slice
models cannot be used from the form and say so.

## V10 models (trained in this release)

**Task:** tumour vs tumour-free liver tissue in 96 × 96 patches, aggregated to a slice call.
**Data:** all 2,449 patches of `lits_verified_v8` (8 reconstructed scan units). **Configuration:** the inner-loop
selection made most often across the 8 outer folds of the nested evaluation. **Calibration and threshold:** Platt
scaling and Youden threshold fitted on leave-one-unit-out predictions. **Performance estimate:** the nested evaluation
(the final models are trained on every unit, so they have no held-out data of their own).

| Family | Models | Inputs |
|---|---|---|
| Classical | logistic regression, decision tree, random forest, extra trees, RBF-SVM, k-NN, histogram gradient boosting, small MLP | 40 radiomics features (intensity statistics, histogram bands, GLCM at distances 1 and 3, LBP histogram, gradient mean / SD / 90th percentile), standardised |
| Classical controls | logistic regression and RBF-SVM on PCA(4) | the same 4 PCA components the quantum models see |
| Quantum (simulated) | VQC: 4 qubits, angle encoding, brickwork entangler; grid 1–2 layers × readout `linear_z` / `z0`; Adam with cosine schedule, 30 epochs, adjoint gradients. QKSVM: 4-qubit fidelity kernel of a Z feature map (2 repetitions) with V9-style or product pair terms; grid bandwidth 0.25 / 0.5 / 1.0 × C; SVC on the precomputed kernel | 4 PCA components scaled to angles in [0, π] |
| Hybrid (simulated) | `hybrid_no_fusion` (classical encoder → VQC → head), `hybrid_feature_fusion` (VQC output concatenated with a classical branch), `late_fusion` (logistic regression + VQC, equal-weight standardised logits); ablations `hybrid_random_quantum` (frozen random circuit) and `hybrid_quantum_only` (evaluated, not deployed) | standardised radiomics |
| Classical twin | `hybrid_twin`: the hybrid with the quantum layer replaced by a classical layer of matched size — the control for `hybrid_no_fusion` | standardised radiomics |

All quantum circuits run on the project's NumPy state-vector simulator (execution mode `custom_numpy_simulator`);
no quantum hardware was used. The noise study (`results/v10/noise/noise_study.json`) re-scores the deployed VQC under
hypothetical depolarising noise (density-matrix simulation); it measures robustness of the deployed model, not
performance (the model saw every unit in training).

## V9 models (imported unchanged)

NumPy CNNs (2-D, 2.5-D with ablations a1–a4, 5-slice, 3-D), ImageNet backbones fine-tuned on CT (ResNet-50,
EfficientNet-B0, DenseNet-121; heads trained, backbones partially), radiomics models, and 4- / 8-qubit quantum and
hybrid models on CNN or EfficientNet embeddings with classical twins. They were trained on groups G03–G06, selected on
G02 / G09 / G10 and tested on G01 / G07 / G08 (patch level). Their numbers are V9's recorded test results and are not
comparable with the V10 nested estimates. Stage 2 of the audit found the V9 validation set saturated and the
embedding-based quantum models exploratory (their embeddings were learnt on the training groups).
""".strip()


def fmt(v, d=3):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))


def main() -> None:
    reg = json.loads((ROOT / "artifacts" / "v10" / "MODEL_REGISTRY.json").read_text())
    models = reg["models"]
    s = reg["summary"]
    L = ["# Model documentation", "", f"> {DISCLAIMER}", "",
         f"Registry: {s['models_servable']} servable of {s['models_total']} registered models (generated by "
         "`scripts/write_model_docs.py` from `artifacts/v10/MODEL_REGISTRY.json`; the live list is at "
         "`GET /api/v1/models` and on the Models page).", "", PROSE, "", "## Registry", "",
         "| Model id | Family | Version | Status | Training | Execution | Input | Evaluation | Primary metric | "
         "Threshold | Parameters | Artifact SHA-256 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"v10": 0, "v9": 1}
    for m in sorted(models, key=lambda m: (order.get(m.get("version", "")[:3], 2), m["status"] != "available",
                                           m.get("family", ""), m["id"])):
        ev = ((m.get("metrics") or {}).get("evaluations") or [{}])[0]
        met = ev.get("metrics") or {}
        if "primary_mean_within_unit_auc" in met:
            prim = f"{met['primary_mean_within_unit_auc']:.3f} mean within-unit AUC"
        elif met.get("auc") is not None:
            prim = f"{met['auc']:.3f} test AUC (V9 split)"
        else:
            prim = "—"
        ic = m.get("input_contract") or {}
        inp = ("slice + 2 neighbours each side" if ic.get("context_slices_each_side", 0) > 1 else
               "slice + neighbours" if ic.get("requires_context") else "single slice" if ic else "—")
        status = m["status"] + (f" ({m['reason']})" if m.get("reason") and m["status"] != "available" else "")
        L.append(f"| `{m['id']}` | {m.get('family')} | {m.get('version')} | {status} | {m.get('training_status')} | "
                 f"{m.get('execution_mode') or '—'} | {inp} | {ev.get('name', '—')} | {prim} | "
                 f"{fmt(m.get('threshold'))} | {m.get('parameters') or '—'} | "
                 f"`{(m.get('artifact_sha256') or '')[:12]}` |")
    L += ["", "Statuses: **available** = served; **legacy** = V8-era artifact kept for reproducibility, not offered; "
          "**not_trained** = recorded in V9 as not trained; other statuses carry the reason (for example a missing "
          "weight file). Full cards (configuration, calibration, training history, per-unit metrics, limitations) are "
          "in `artifacts/**/model_card.json` / `model.json` and on each model's page in the app.", ""]
    out = ROOT / "MODEL_DOCUMENTATION.md"
    out.write_text("\n".join(L))
    print(f"wrote {out.relative_to(ROOT)} ({len(models)} models)")


if __name__ == "__main__":
    main()
