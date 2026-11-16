# Research limitations

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

## 1. Data

* **One small, public, derived dataset.** 1,745 slices / 2,449 liver patches from a LiTS-derived Hugging Face dataset.
  No external test set, no other scanner, site or population. Nothing here estimates performance on new hospitals.
* **No patient identifiers.** The 10 scan groups are reconstructed from slice adjacency; evaluation is
  leave-one-reconstructed-unit-out over 8 units. These are not patients: the same patient may appear in two units,
  which would inflate every metric. Patient-level generalisation is not established.
* **Patch-level task.** The V10 models classify 96 × 96 liver-tissue patches (tumour vs tumour-free) chosen with the
  expert masks. Patches with 0–5 % tumour were discarded as ambiguous, which makes the task easier than real use.
  A slice call is derived from the maximum patch probability; slice-level and lesion-level performance of the V10
  models was not evaluated on held-out data. In this source the tumour-free slices contain no liver, so a slice-level
  "No tumour detected" on a slice with liver has never been validated against ground truth.
* **Pre-processed PNGs.** The images are windowed 8-bit PNGs without DICOM metadata. Hounsfield units, spacing and
  acquisition parameters are unknown; uploaded DICOM / NIfTI are windowed to match.
* **Small units.** Seven of the eight units qualify for the primary metric (≥ 10 patches per class); confidence
  intervals are wide and the sign-flip test with 7 units has a minimum two-sided p of 1/64 ≈ 0.016.

## 2. Models and evaluation

* **Probabilities are not well calibrated across scan units.** Platt scaling is fitted on inner-fold predictions,
  but on the held-out units the pooled expected calibration error is 0.10–0.23 (Brier 0.15–0.26) for the radiomics
  models, and the pooled outer-fold AUC (0.67–0.88) is well below the mean within-unit AUC (0.80–0.97): scores shift
  from one scan unit to the next. A probability of 0.8 therefore does not mean an 80 % chance of tumour, even on this
  dataset. The UI and reports call it "model probability" and show this caveat; it is not a clinical confidence.
* **Thresholds** (Youden J) were chosen on inner folds; the "uncertain" band (threshold ± 0.10) is a display
  convention, not a validated abstention rule.
* **Fixed analysis plan.** `configs/v10/analysis_plan.json` (sha256 `9d63f113…`) was written before the nested run
  (file time 14:46:11 UTC; first outer-fold result 14:51:59 UTC) and was not changed afterwards. Its `frozen_utc`
  field reads `15:30:00Z`, which is wrong (it was typed by hand); the file itself was left unchanged so that its hash
  still matches the hash recorded in every run manifest.
* **Non-significance is not equivalence.** Where a quantum or hybrid model is not significantly different from its
  classical control, that does not show they are equivalent; the study is too small for an equivalence claim.
* **No quantum advantage is claimed.** All quantum models run on a classical state-vector simulator (NumPy, exact,
  noiseless unless stated). Execution mode `custom_numpy_simulator` is recorded in every card, prediction and report.
  No quantum hardware was used. The noise study uses a density-matrix depolarising model with assumed error rates,
  not a calibrated device model.
* **4 qubits.** Quantum models see 4 PCA components (angle-encoded); their classical controls (`logreg_pca4`,
  `svm_rbf_pca4`) see the same 4 inputs. Higher-qubit circuits were not studied.
* **Selection on inner folds only**, but the model families, grids and feature sets were chosen by the developer
  after the V9 study on the same data, so the whole study is not a pristine hold-out.
* **Deep-learning V10 models.** The V9 NumPy CNNs and ImageNet backbones (ResNet-50, EfficientNet-B0,
  DenseNet-121) are served with their V9 fixed-split evaluation (test = G01, G07, G08). They were not re-evaluated
  with the nested design (too slow without a GPU / PyTorch here). The PyTorch training scripts are written but were
  **not executed** (PyTorch is not installed in the build environment).
* **V9 liver gate defect (known).** The V9 liver-tissue gate accepts some patches on slices without liver; slice-level
  calls on non-liver slices can therefore be wrong. The image QC and OOD detector catch non-CT images but not all
  non-liver CT slices.

## 3. Software stack substitutions (package installs were disabled while building)

| Requested | Used instead | Effect |
|---|---|---|
| FastAPI | Starlette + Pydantic with a typed routing layer (`backend/app/core/apikit.py`), OpenAPI 3.1, Swagger UI | same request validation and docs; FastAPI-specific plugins not available |
| SQLAlchemy + Alembic | DB-API layer (SQLite / psycopg) + versioned SQL migrations with checksums | SQL is hand-written and tested on SQLite and PostgreSQL 16 |
| psycopg (in tests) | test-only PostgreSQL wire client (`backend/tests/support/pgwire.py`) | the production driver psycopg 3 itself was not exercised here |
| Vite | esbuild + Tailwind 4 compiler (`frontend/scripts/build.mjs`) | no Vite plugins / HMR; dev server with live rebuild |
| React Router, TanStack Query, React Hook Form | small in-house router, query cache and form hook (`frontend/src/lib`) | fewer features (no nested route loaders, no devtools) |
| Recharts | SVG charts on `d3-scale` / `d3-shape` | bar and line charts only |
| lucide-react, shadcn/ui | Lucide icons via `react-icons/lu`; shadcn-style components (cva, clsx, tailwind-merge) | visually equivalent; no Radix primitives |
| PyTorch, MONAI, timm | V9 NumPy CNN / backbone implementations with real ImageNet weights | slower training; GPU not used |
| PennyLane, Qiskit | custom NumPy simulator (validated against analytic results and finite differences) | no hardware back-ends |
| bcrypt / passlib | Argon2id from `cryptography` | stronger KDF; different hash format |

## 4. Deployment

Docker images were not built and no cloud deployment was performed (no Docker daemon or accounts in the build
environment). The container start-up was reproduced without Docker and passed the smoke test; see
`DEPLOYMENT.md`. The S3 storage backend and SMTP mail backend were not tested against real services.

## 5. Not a medical device

No regulatory assessment, no prospective study, no reader study, no failure-mode analysis. Outputs must not be shown to
patients or used to make or change clinical decisions.
