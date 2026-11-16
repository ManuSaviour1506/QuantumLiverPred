# Reproducing the V10 results

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

## Inputs (verify before running)

| Input | SHA-256 |
|---|---|
| `patches/patches_v8.npz` of `liver_ct_lits_verified_v8` | `2a237bf320b5491e73a2c762e6169aef0d13824b9bfed9dac3758560680f30a1` |
| `manifests/patches_task_a.csv` | `9d4f42ca997f46d7970c90025de8d96dd01bb7ffc4a38ddaebf240445d3715c2` |
| Dataset version string (first 16 hex of the hash of the two hashes above) | `lits_verified_v8@2a9a4887bb322cfc` |
| `configs/v10/analysis_plan.json` (frozen design) | `9d63f113aeb48ca87035f8d52950c81413db0e4511beee46e62af2d92bc1ce4d` |
| Vendored V9 engine (`src/` minus `src/v10`), combined hash | `16254c33…` (see `reports/v10/research/V9_SOURCE_AUDIT.md`) |

Every file of the dataset folder (1,757 files) is listed with its SHA-256 in `dataset/manifests/lits_verified_v8_SHA256SUMS.txt`.

## Environment used

Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, scikit-image 0.26.0 on Linux x86-64
with 2 CPUs; no GPU, no PyTorch, no PennyLane / Qiskit. Each run writes its environment into its manifest
(`results/v10/nested/<features>/run_manifest.json`, `results/v10/runs/<run>/environment.json`). Floating-point results
can differ in the last digits on other CPUs / BLAS builds; model selection and metrics should agree to about 3 decimals.

## Commands, in the order they were run

```bash
export LTI_DATASET_DIR="/path/to/liver_ct_lits_verified_v8"
python scripts/validate_dataset.py --dataset "$LTI_DATASET_DIR"          # data checks (no training)
python scripts/v10_math_validation.py                                   # simulator / gradient / kernel checks
# nested evaluation (frozen plan; ~60 min for radiomics on 2 CPUs)
python scripts/run_nested_evaluation.py --dataset "$LTI_DATASET_DIR" --features radiomics
python scripts/run_nested_evaluation.py --dataset "$LTI_DATASET_DIR" --features imagenet_gap \
    --models logreg,svm_rbf,logreg_pca4,svm_rbf_pca4,vqc,qksvm
# final deployable models: configuration = the inner-loop selection made most often across the 8 outer folds;
# Platt calibration and threshold from leave-one-unit-out predictions; final fit on all units; seed 0
python scripts/train_classical.py --dataset "$LTI_DATASET_DIR" --features radiomics
python scripts/train_vqc.py --dataset "$LTI_DATASET_DIR" --features radiomics
python scripts/train_quantum_kernel.py --dataset "$LTI_DATASET_DIR" --features radiomics
python scripts/train_hybrid.py --dataset "$LTI_DATASET_DIR" --features radiomics
python scripts/run_noise_experiments.py --dataset "$LTI_DATASET_DIR"     # VQC under depolarising noise
python scripts/export_model_registry.py                                 # artifacts/v10/MODEL_REGISTRY.{json,md}
python scripts/evaluate_models.py --dataset "$LTI_DATASET_DIR"           # integrity check of served models
python scripts/write_results_report.py                                  # reports/v10/V10_NESTED_RESULTS.md
```

Seeds: outer folds are deterministic (one per unit); inner folds are seeded by the outer-fold index; stochastic models
(VQC, hybrids, MLP, forests) run with seeds 0–4 in the nested evaluation and seed 0 for the final fit.

## Where the outputs are

| Output | Path |
|---|---|
| Nested results per feature set (metrics, per-unit AUC, selected configurations, comparisons, out-of-fold predictions) | `results/v10/nested/<features>/` |
| Readable results report | `reports/v10/V10_NESTED_RESULTS.md` |
| Final models (card + arrays / joblib, SHA-256 in the card) | `artifacts/v10/<model_id>/` |
| Training run records (config, split, environment, log, metrics, out-of-fold predictions) | `results/v10/runs/<stamp>_<model_id>/` |
| Noise study | `results/v10/noise/noise_study.json` |
| Registry export and integrity evaluation | `artifacts/v10/MODEL_REGISTRY.{json,md}`, `results/v10/registry_evaluation.json` |
| V9 results (imported unchanged) | `results/v9/` |

## Known record-keeping issues

* A first `imagenet_gap` run with 8 specifications (adding `hybrid_no_fusion`, `hybrid_twin`) was stopped by the build
  environment after 5 models (its partial log is kept in `results/v10/nested/imagenet_stdout_interrupted_*`). It was
  re-run with the 6 specifications above; the 5 models present in both runs gave identical results to all printed
  digits (e.g. `logreg` 0.9271027810245599), which also shows the run is deterministic. The hybrids were not
  re-run on this feature set for lack of compute time; the hybrid comparison is part of the radiomics analysis.

* `configs/v10/analysis_plan.json` has `"frozen_utc": "2026-11-12T15:30:00Z"`. That value is wrong: the file was
  written at 14:46:11 UTC and the first outer-fold result was logged at 14:51:59 UTC. The file was not edited
  afterwards, so its hash still matches the hash in the run manifests.
* `results/v10/nested/radiomics/results.json` records `feature_info.dim = 44`; the feature matrix the run loaded has 40
  columns (the label came from the code version at run start and was corrected in the code afterwards).
