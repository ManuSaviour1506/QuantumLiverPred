# Liver Tumour Intelligence Platform — V11

> **This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.**

A research platform for liver-tumour detection on CT slices. It puts **classical machine learning, deep learning,
simulated quantum models and hybrid quantum-classical models** behind one secured web application. Upload a slice,
run every model family concurrently, and watch the results, timings and resource use arrive live. Models are then
benchmarked on labelled, scan-group-held-out data.

V11 is the consolidated final build of V9 → V10 → V10.1. It is the same project, not a rewrite. The V9 engine and its
trained models, the V10 models and the V10.1 comparison platform are all kept and integrated.

## Quick start (Windows, macOS Apple Silicon / Intel, Linux)

Prerequisite: Python 3.10+ (python.org installer on Windows/macOS; tick *Add python.exe to PATH* on Windows). Node.js 20+
is optional: a prebuilt web app is included.

```bash
python install.py --admin-email you@example.org     # Windows: install.bat --admin-email you@example.org
python start.py --open                              # Windows: start.bat        → http://localhost:8000
```

`install.py` performs eight steps and reports each one:

1. Detect the OS and CPU.
2. Create `.venv` and install `requirements.txt`.
3. Verify the three bundled ImageNet weights by SHA-256.
4. Check every model artifact.
5. Write `backend/.env` with a random secret.
6. Build the web app if Node 20+ is present, otherwise use `frontend/dist`.
7. Apply the database migrations and optionally create the admin.
8. Verify the bundled datasets by SHA-256, load the model engine and run one real four-family comparison as a self-test.

`python install.py --check` re-verifies an installation without changing it. Details and manual steps are in
[INSTALLATION.md](INSTALLATION.md); for problems see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Docker: `docker build -t lti-v11 . && docker run -p 8000:8000 -v lti-data:/var/lib/lti lti-v11` (all-in-one), or
`cp .env.example .env` then `docker compose up --build` (PostgreSQL + API + worker + nginx → http://localhost:8080).

Then sign in → **Image Upload** → pick `ct_lits_4278.png` from the samples (or upload your own PNG / JPEG / DICOM /
NIfTI slice, optionally with neighbouring slices) → **Run all models**.

## What V11 adds

| Area | V11 change |
|---|---|
| Measurement | Every model run records its load time (cold start), preprocessing / queue / lock-wait / inference / total time, and thread CPU time. Every comparison run samples process CPU % and resident memory (100 ms), detects a GPU (never used) and names the fastest and slowest model and family. |
| System monitor | New **System Monitor** page (`/app/system`, `GET /api/v1/system/metrics`): live CPU / memory chart, threads, uptime, active runs, per-model load times, hardware and execution mode. |
| Benchmarking engine | New **New evaluation runs** on the Benchmarking page (`/api/v1/benchmarks/runs`). Runs models on ground-truth-labelled slices through the live pipeline and reports accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, ECE, confusion and latency. Every slice is labelled by the role its scan group played for that model (held-out test / validation / in-sample), metrics are never pooled across roles, and results are stored apart from the historical recorded evaluations. CSV export. |
| Decision tree | `v10_decision_tree_radiomics` (always "Uncertain" at p = 0.855) is **fixed**, not just flagged. v10.1 recalibrates the unchanged tree with isotonic regression on its own leave-one-unit-out scores and re-selects the threshold (0.679) by Youden on those scores. Validated per held-out unit; v10.0 archived. |
| Registry | V9 models whose weight file is missing used to vanish from the registry. They are now listed as `missing_weights` with a reason (63 models tracked, was 61). Unloadable models are described from their own `model.json`. |
| Installer | Cross-platform `install.py` / `start.py` (+ `install.bat` / `start.bat`), root all-in-one `Dockerfile`, `TROUBLESHOOTING.md`. |
| Packaging | All three ImageNet backbone weights are bundled (156 MB). `models/` index per family, `dataset/dataset_manifest.json`, `MISSING_ARTIFACTS.md`, `COMPLETE_FILE_MANIFEST.json` (SHA-256 of every file). |
| Tests | Engine regression suite `tests/test_regression_cases.py`, backend tests for resources / load times / benchmark runs / system metrics, browser e2e extended to the new pages. Results: [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md). |

Status of every V10 audit finding: [reports/v11/V11_AUDIT_STATUS.md](reports/v11/V11_AUDIT_STATUS.md).

## Project layout (requested V11 structure → this project)

| Requested | Here | Notes |
|---|---|---|
| `frontend/` | `frontend/` | React 19 + TypeScript + Tailwind 4; `dist/` prebuilt |
| `backend/` | `backend/` | Starlette + Pydantic API, JWT auth, migrations, comparison + benchmark services |
| `models/{classical,deep_learning,quantum,hybrid}` | `models/` (index) → code in `src/`, trained artifacts in `artifacts/` | moving code would break every import and artifact hash; `models/model_index.json` maps each model to its code, artifact, SHA-256 and availability |
| `artifacts/{weights,checkpoints,configs,manifests}` | `artifacts/weights`, `artifacts/manifests`, `artifacts/{classical,quantum,hybrid,segmentation,ood,v10}` (per-model folders hold weights + config cards), `artifacts/archive/` (superseded versions) | training checkpoints: see MISSING_ARTIFACTS.md |
| `datasets/{CT,MRI,annotations,sample_data}` + `dataset_manifest.json` | `dataset/` (`sample_data/`, `manifests/`, `dataset_manifest.json`) and `data/` (labels, manifests) | full CT dataset not included (see below); no MRI data exist |
| `preprocessing/` | `src/preprocessing/`, `src/quality_control/`, `src/v10/engine/patch_pipeline.py` | |
| `evaluation/` | `src/evaluation/`, `src/v10/evaluation/` | |
| `benchmarking/` | `src/v10/engine/benchmark.py`, `backend/app/benchmarking/` | |
| `scripts/`, `tests/`, `reports/`, `docs/`, `deployment` | `scripts/`, `tests/` + `backend/tests/` + `frontend/tests` + `frontend/e2e`, `reports/`, `docs/`, `docker/` + `docker-compose.yml` + `render.yaml` | |
| root files | `requirements.txt`, `.env.example`, `docker-compose.yml`, `Dockerfile`, `README.md`, `COMPLETE_FILE_MANIFEST.json` | |

## What is included — and what is not

Included: all source code; the 45 trained models that load from this package (33 servable in comparisons — 21 on a
single slice, 12 with neighbouring slices — and 12 legacy models kept for reproducibility); all three ImageNet backbone weights; the OOD detector, liver gate and U-Net; 8 real CT slices with expert-mask
labels; 24 synthetic phantoms (software tests only); every label / manifest file; all reports.

**Not included**, and never faked or substituted (details and restore commands in
[MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md)):

* **7 large V9 files.** Four `model.npz` and three training checkpoints. Without them 16 models are unavailable: 8 V9
  comparison models and 8 V8-era legacy models. Restore with `python install.py --v9 /path/to/liver_tumour_project_v9`.
* **The full LiTS-derived CT dataset** `liver_ct_lits_verified_v8` (1,757 files; its SHA-256 list and slice manifest
  are included) and the derived 64 × 64 mask maps.
* **MRI data.** None exist in any version of the project.

## How it works — honestly

* **Concurrency.** The four family pipelines start together on four worker threads and run concurrently. Inside a
  family, models run one at a time (`family_parallel_model_sequential`), so the CPU-bound code does not oversubscribe
  the CPU. Every run reports the measured start spread and the overlap of all four families.
* **Quantum.** Every quantum circuit runs on an exact CPU state-vector simulator (`custom_numpy_simulator`), never on
  quantum hardware. **No GPU** is used or required.
* **Accuracy.** Accuracy comes only from labelled, scan-group-held-out evaluations (recorded, or new benchmark runs
  with their roles), never from one uploaded image. Model probability is not accuracy.
* **Front end and calls.** Every input passes decode → quality control → out-of-distribution check → 96-px patch grid
  → liver gate → U-Net liver validation. Slices without liver are withheld, not scored. Calls use each model's own
  threshold with a ±0.10 uncertainty band.

## Results (recorded evaluations)

**Task:** tumour vs tumour-free 96 × 96 liver-tissue patches (2,449 patches from 1,745 CT slices).

**Design:** leave one reconstructed scan unit out (8 outer folds; units are scan groups, *not patients*); selection,
calibration and threshold on inner folds only; 5 seeds for stochastic models.

**Primary metric:** mean within-unit ROC AUC.

| Model (radiomics features) | Mean within-unit AUC | Range over units |
|---|---|---|
| Logistic regression | 0.966 | 0.848 – 1.000 |
| RBF-SVM | 0.955 | 0.901 – 1.000 |
| Hybrid classical twin (control) | 0.936 | 0.827 – 0.998 |
| Hybrid quantum-classical (simulated) | 0.934 | 0.773 – 0.996 |
| Late fusion (simulated) | 0.922 | 0.844 – 1.000 |
| Logistic regression on 4 PCA components (control) | 0.894 | 0.638 – 0.998 |
| Quantum-kernel SVM, 4 qubits (simulated) | 0.893 | 0.641 – 1.000 |
| VQC, 4 qubits (simulated) | 0.878 | 0.672 – 0.996 |
| RBF-SVM on 4 PCA components (control) | 0.852 | 0.645 – 1.000 |

Confirmatory comparisons (exact sign-flip test, Holm-corrected) show **no significant difference and no evidence of a
quantum advantage**:

* VQC vs its classical control: Δ = −0.016 (p = 0.78)
* QKSVM vs its classical control: Δ = +0.041 (p = 0.56)
* Hybrid vs its classical twin: Δ = −0.001 (p = 0.81)

The V9 deep-learning models are served with their V9 fixed-split test results (test groups G01 / G07 / G08). Full
details: `reports/v10/V10_NESTED_RESULTS.md`, [RESEARCH_LIMITATIONS.md](RESEARCH_LIMITATIONS.md).

## Documentation

| Document | Contents |
|---|---|
| [INSTALLATION.md](INSTALLATION.md) | installer, manual setup on every OS, tests |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | common problems and fixes |
| [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md) | files not in this package and how to restore them |
| [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md) | V11 validation: what was executed, passed, not executed |
| [docs/COMPARISON_PLATFORM.md](docs/COMPARISON_PLATFORM.md) | comparison engine, timing / resource definitions, benchmark engine, API, SSE |
| [docs/MODEL_PIPELINE_MAP.md](docs/MODEL_PIPELINE_MAP.md), [models/](models/README.md) | every model → pipeline, artifact, availability |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | every endpoint (generated from OpenAPI, `docs/openapi.json`) |
| [DATASET_GUIDE.md](DATASET_GUIDE.md), [dataset/dataset_manifest.json](dataset/dataset_manifest.json) | data, licence, verification |
| [MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md) | every model: inputs, training, evaluation |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker, Compose, Render, Vercel |
| [V10_REPRODUCIBILITY.md](V10_REPRODUCIBILITY.md), [RESEARCH_LIMITATIONS.md](RESEARCH_LIMITATIONS.md) | reproduction, limits of the evidence |
| [reports/](reports/) | audit reports (V10 case review PDF, V10 / V11 audit status, calibration, liver gate) |

## Roles

| Role | Can |
|---|---|
| viewer | view models, comparisons, research results |
| researcher | + run comparisons and predictions, start benchmark runs, view own history, upload datasets |
| admin | + manage users and roles, register server-side datasets, see all users' runs (audit-logged) |

New registrations get the researcher role. The first administrator is created by `install.py --admin-email` or
`python -m app.manage create-admin`.

## Licence and attribution

Code: for academic research use by the project owner. CT data are derived from the LiTS benchmark via
`seongun/liver-tumor-classification` (CC BY-NC-SA 4.0). Cite P. Bilic et al., *Medical Image Analysis* 84, 102680
(2023). ImageNet weights: timm (Apache-2.0).
