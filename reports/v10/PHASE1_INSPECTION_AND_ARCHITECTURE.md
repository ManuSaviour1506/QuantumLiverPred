# Phase 1 — Inspection, final architecture and blockers

Date: 12 Nov 2026. Inspected: the V9 repository on the user's machine (`liver_tumour_project_v9`, 276 MB) and its
reports, the V10 research work added on 11–12 Nov (`src/v10`, `results/v10`, `reports/V10_*`), and the build
environment.

## 1. What V9 contains (verified)

| Item | Finding |
|---|---|
| V9 engine (`src/`, 50 Python files, ~5,100 lines) | NumPy-only: decoding of PNG / JPEG / DICOM / NIfTI with built-in readers, CT windowing, image QC, OOD detector, liver-tissue gate, NumPy CNNs (2-D, 2.5-D, 5-slice, 3-D), NumPy ImageNet backbones (ResNet-50, EfficientNet-B0, DenseNet-121) loading timm `.pth` weights without PyTorch, radiomics, 2.5-D U-Net, Grad-CAM, custom state-vector simulator, VQC, quantum-kernel SVM, hybrid models, classical twins |
| Trained artifacts (`artifacts/`, 235 MB, 102 files) | 46 models load with integrity (SHA-256) and self-tests in 6 s; 4 are recorded NOT TRAINED (ConvNeXt-Tiny, EfficientNet-B2, 3-D U-Net, nnU-Net). Includes ImageNet weights for ResNet-50 (102 MB), DenseNet-121 (32 MB), EfficientNet-B0 (21 MB) |
| Data in the folder | Labels (`data/labels/verified_slice_labels.csv`, 1,745 verified slices), manifests (patches, scan groups, splits), 64×64 expert maps. **No CT images**: `data/raw/scans` and `data/processed/` are absent (both git-ignored). No patient / volume IDs (Stage 3) |
| V9 backend | Starlette app (`backend/app/main.py`, SQLite history), 12 endpoints, no authentication |
| V9 frontend | React (JSX, no TypeScript) built with esbuild; 10 pages; built `dist/` included |
| V9 metrics | `results/v9/metrics_v9.csv`, `comparison_v9.json` (fixed split: train G03–G06, val G02/G09/G10, test G01/G07/G08; patch level). Stage 2 found that the validation set is saturated and some embeddings are exploratory |
| Stage 1–3 observations | All confirmed and carried forward unchanged: `vol_008` absent; ten reconstructed groups; the row-index label rule applied only to the 4,236 excluded images; scans missing; some audit findings only partly reproducible without the images |

A synthetic phantom (not a medical image) passed QC and was rejected by the V9 OOD detector as out of distribution in
0.1 s. That is the correct, real behaviour of the trained detector.

## 2. Build environment (verified, not assumed)

Network access for package installation is **off** in this environment (`pip` / `npm` installs are refused), and the
shell on the user's machine has no network either. Only pre-installed packages can be used and tested.

| Needed | Available here | Decision |
|---|---|---|
| FastAPI | **No** (Starlette 1.7, Pydantic 2.13, uvicorn 0.54 are) | API on **Starlette + Pydantic** (FastAPI is built on both). A small typed routing layer validates every request with Pydantic and generates OpenAPI 3.1; Swagger UI at `/docs` |
| SQLAlchemy, Alembic | **No** | Portable SQL data layer over DB-API; **versioned SQL migrations** with upgrade / downgrade / status and checksum protection |
| PostgreSQL | Server 16 binaries yes; Python driver (psycopg) **no** | Production and Docker use PostgreSQL through psycopg 3 (installed by the Docker image). Local tests run on SQLite; PostgreSQL compatibility is tested against a local PostgreSQL 16 server through a test-only wire-protocol client |
| bcrypt / passlib | **No** | **Argon2id** from `cryptography` (OWASP-recommended) |
| PyJWT, cryptography, python-multipart, reportlab, Pillow, NumPy, SciPy, scikit-learn, pandas, scikit-image | Yes | Used |
| PyTorch, torchvision, MONAI, timm | **No** | V9's NumPy backbones run ResNet-50 / EfficientNet-B0 / DenseNet-121 with real ImageNet weights. PyTorch training scripts are written for machines that have it and are marked **untested here** |
| PennyLane, Qiskit, XGBoost | **No** | Custom NumPy simulator only (execution mode reported as such); optional backends detected at runtime; XGBoost used only if installed |
| nibabel, pydicom | **No** | V9's built-in NIfTI and DICOM readers (uncompressed DICOM, `.nii`, `.nii.gz`) |
| Vite, React Router, TanStack Query, React Hook Form, Recharts, lucide-react, shadcn/Radix | **No** | **esbuild** bundler; small in-house router, query cache and form hook; SVG charts with `d3-scale` / `d3-shape`; Lucide icons through `react-icons/lu`; shadcn-style components built with `class-variance-authority`, `clsx`, `tailwind-merge` |
| React 19, TypeScript 6, Tailwind CSS 4, Zod 3, esbuild 0.28 | Yes | Used |
| Playwright + Chromium | Yes | Browser end-to-end tests against the live backend |
| Docker daemon | **No** (client only) | Dockerfiles and Compose written; `docker compose config` validation only; images cannot be built or run here |

Every substitution keeps the requested architecture's role (typed REST API with OpenAPI, relational database with
migrations, React + TypeScript + Tailwind SPA). Each one is listed again in `README.md` and `RESEARCH_LIMITATIONS.md`.
To use the exact requested libraries, allow package installs in the network settings and start a new chat.

## 3. Final architecture

```
Browser ── React 19 + TypeScript SPA (Tailwind 4, esbuild) ── nginx (static + /api proxy)
                                    │ REST + JWT
                         Starlette API (Pydantic, OpenAPI)  ── PostgreSQL 16 (SQLite for local tests)
                                    │                      ── object storage (local disk or S3-compatible)
                       job queue (DB-backed) → worker process / in-process thread pool
                                    │
                 Model engine (src/v10/engine): common interface, registry, integrity checks
                 ├── V9 engine adapter (vendored V9 code, byte-identical, 46 trained models)
                 ├── V10 classical (scikit-learn pipelines, joblib)
                 ├── V10 quantum (VQC, quantum-kernel SVM, feature maps; custom NumPy simulator)
                 ├── V10 hybrid (fusion / late fusion / ablations)
                 └── deep learning (V9 NumPy CNN + backbones; PyTorch modules when installed)
```

Model-serving rules: a model is served only if its artifact exists, its SHA-256 matches its card, its preprocessing
configuration matches the code, and its training data are real. Models trained on synthetic data are test-only and are
never offered for prediction unless `ALLOW_SYNTHETIC_MODELS=true` (tests only).

## 4. Blockers

| Blocker | Effect | What unblocks it |
|---|---|---|
| CT images absent (`scans` folder / `data/processed`) | No V10 real-data training or evaluation; V9 models still predict | Supply the scans folder; `scripts/v10_import_scans.py` builds the manifest |
| No patient / volume identifiers | Patient-level generalisation cannot be established | Original LiTS segmentation volumes + `scripts/v9/v9_recover_volume_ids.py` |
| No package installs | Requested FastAPI / SQLAlchemy / Alembic / Vite / Router / TanStack / RHF / Recharts / PyTorch not used | Allow package installs; the substitutions are documented one by one |
| No Docker daemon | Compose stack not run here | Run `docker compose up --build` on a machine with Docker |
| No quantum hardware or SDKs | Simulation only | Install PennyLane / Qiskit for simulator cross-checks; hardware needs provider credentials |
