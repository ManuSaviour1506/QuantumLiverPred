# V10 build — phase log

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

Per phase: files created / modified, tests run with results, blockers left. Test counts are those of the final run on
12 Nov 2026 (see `docs/QA_REPORT.md`); earlier intermediate runs are not repeated here. The V9 project folder was
read only: its 410 original files were re-verified against the SHA-256 snapshot taken before work began
(`sha256sum -c`, exit 0, 12 Nov 2026).

## Phase 1 — Inspect and plan

* **Created:** `reports/v10/PHASE1_INSPECTION_AND_ARCHITECTURE.md`; copies of the V9 / V10-research audit reports in
  `reports/v10/research/`.
* **Modified:** nothing in V9.
* **Tests:** V9 artifact load check (46 models load with integrity in 6 s); package inventory.
* **Blockers recorded:** no package installs (FastAPI, SQLAlchemy, Alembic, Vite, React Router, TanStack Query, React
  Hook Form, Recharts, PyTorch, PennyLane, Qiskit unavailable → substitutions); no Docker daemon; CT images not in the
  V9 folder (resolved later: the owner connected `liver_ct_lits_verified_v8`); no patient identifiers (permanent).

## Phase 2 — Backend foundation

* **Created:** `backend/app/` — `core/` (config, logging, errors, security, typed routing `apikit.py`, middleware),
  `db/` (database layer, migration runner, `migrations/0001_initial.sql` with 16 tables), `schemas/`, `services/`
  (auth, audit, mail, storage, jobs, models, predictions, reports, datasets, experiments, analytics), `api/v1/`
  (43 OpenAPI paths), `main.py`, `manage.py`, `worker.py`; `backend/tests/`.
* **Tests:** 42 integration / unit tests — **42 passed on SQLite, 42 passed on PostgreSQL 16** (final run).
* **Blockers:** FastAPI / SQLAlchemy / Alembic replaced (documented); psycopg not installable → PostgreSQL tested
  through a test-only wire client.

## Phase 3 — Python AI engine

* **Created:** `src/v10/engine/` (interface, artifacts, patch pipeline, explanations, V9 adapter, V10 model runtimes,
  registry), `src/v10/classical/`, `src/v10/quantum/` (VQC, QKSVM, optimiser, noise study), `src/v10/hybrid/`,
  `src/v10/features/`, `src/v10/evaluation/` (nested evaluation, specs), `src/v10/training/`,
  `src/v10/deep_learning/` (NumPy CNN / backbone training; PyTorch module); `configs/v10/analysis_plan.json`;
  training and evaluation scripts in `scripts/`.
* **Vendored unchanged:** the V9 engine in `src/` (combined hash verified) and the V9 artifacts (hash verified).
* **Trained (real data):** 16 V10 models on 2,449 patches — 10 classical / control, VQC, QKSVM, hybrid, hybrid
  feature fusion, late fusion, classical twin (`artifacts/v10/`).
* **Tests:** `python -m pytest tests` — **96 passed, 1 skipped** (PyTorch path).
* **Blockers:** PyTorch training not executable here (scripts written, untested); deep-learning models not
  re-evaluated with the nested design (CPU-only NumPy too slow); quantum = simulation only.

## Phase 4 — Frontend

* **Created:** `frontend/` — React 19 + TypeScript + Tailwind 4 SPA: landing, sign-in / registration / forgot and
  reset password, dashboard, imaging workspace, prediction detail, history, model registry and detail, comparison,
  datasets and dataset detail, research and experiments, settings, 404; UI kit; SVG charts; in-house router, query
  cache, form hook; esbuild build and dev server.
* **Tests:** **17 passed** (unit / render); production build passed; type check: 56 errors, all from the missing
  `@types/react` package.
* **Blockers:** Vite / React Router / TanStack Query / React Hook Form / Recharts / shadcn replaced (documented).

## Phase 5 — Integration

* **Connected:** every page to the API (no mock data; empty states when there are no data); real upload and
  prediction through the job queue; persisted history; PDF / CSV / JSON reports; model and experiment analytics from
  saved results.
* **Tests:** Playwright end-to-end — **9 passed** (real CNN prediction on CT with Grad-CAM and PDF; V10 comparison
  table asserted; no console / CSP errors).
* **Fixed in this phase's final pass:** bar charts covering their labels when the axis did not start at 0.

## Phase 6 — Dataset

* **Created:** `src/v10/data/` (safe ZIP ingestion, manifest reading / writing, validation with leakage checks, LiTS
  loader, synthetic phantoms for tests), `scripts/validate_dataset.py`, `generate_manifest.py`, `prepare_dataset.py`;
  `dataset/manifests/lits_verified_v8_manifest.csv` (1,745 rows), `lits_verified_v8_SHA256SUMS.txt` (1,757 files); `dataset/sample_data/lits_sample`
  (8 real slices, attributed); `dataset/sample_data/synthetic` (24 phantoms, test-only).
* **Validated:** the two dataset folders on the owner's computer are byte-identical (SHA-256 of all 1,757 files); the
  patch file and manifest used for training match the hashes recorded in every model card; the sample passes all
  checks; the full dataset passes `scripts/validate_dataset.py` on the owner's computer (1,745 / 1,745 images decoded, 1,745 / 1,745 pixel hashes match, 0 issues, no scan-group leakage; `results/v10/dataset_validation/lits_verified_v8_full.json`).
* **Documented missing data:** patient / volume identifiers (cannot be recovered from this source); tumour-free
  slices with liver (none exist in this source).

## Phase 7 — Deployment

* **Created:** `docker/backend.Dockerfile`, `docker/frontend.Dockerfile`, `docker/allinone.Dockerfile`,
  `docker/nginx.conf`, `docker-compose.yml`, `.env.example`, `.dockerignore`, `.gitignore`, `render.yaml`,
  `frontend/vercel.json`, `backend/requirements*.txt`, `requirements.txt`, `scripts/smoke_test_deployment.py`,
  `DEPLOYMENT.md`.
* **Modified:** `backend/app/core/middleware.py` (no duplicate `Cache-Control`), `backend/app/db/database.py`
  (`port` query parameter).
* **Tests:** `docker compose config` valid; container start-up reproduced without Docker in production mode on
  PostgreSQL 16 and the smoke test **passed** (12 checks incl. a real prediction).
* **Blockers:** images not built, Compose stack not run, Render / Vercel not deployed (no Docker daemon or accounts).
  Exact instructions in `DEPLOYMENT.md`.

## Phase 8 — QA and delivery

* **Ran:** every suite above on the final code; nested evaluation on real data (radiomics: 18 specifications;
  ImageNet embeddings: 8); final training; noise study; registry consistency check.
* **Fixed:** see "Defects found and fixed" in `docs/QA_REPORT.md`.
* **Documentation:** `README.md`, `docs/` (installation, deployment, API reference generated from OpenAPI, dataset
  guide, model documentation generated from the registry, reproducibility, limitations, security, QA report),
  `reports/v10/V10_NESTED_RESULTS.md` (generated from the result files).
* **Packaged:** `liver_tumour_intelligence_v10.zip` by `scripts/package_release.py` (large V9 weight files listed
  with SHA-256 in `artifacts/LARGE_FILES_MANIFEST.json` and restored by `scripts/v10_import_v9_artifacts.py`).
* **Remaining blockers:** Docker / cloud deployment not executed; PyTorch path untested; no quantum hardware; no
  patient-level or external validation possible with this dataset.

## Follow-up (13 Nov 2026) — dataset inside the project, setup scripts

* **Added:** the full dataset copy at `dataset/liver_ct_lits_verified_v8/` in the delivered folder (verified against
  `dataset/manifests/lits_verified_v8_SHA256SUMS.txt`); `setup.sh`, `run.sh`, `SETUP.md`, `backend/.env.example`
  (local template); `package_release.py --full` (full ZIP with dataset and all model files).
* **Modified:** scripts default to the in-project dataset; `DATASET_ROOTS` defaults to `dataset/`;
  `manage register-dataset --if-missing`; Compose mounts `dataset/` at `/data`; `backend/.env` (local) and the root
  `.env` (Docker) are no longer mixed; README, DATASET_GUIDE, INSTALLATION, DEPLOYMENT updated.
* **Tests:** backend 42 passed (SQLite) and 42 passed (PostgreSQL 16); engine 96 passed, 1 skipped; setup / run
  scripts as in `docs/QA_REPORT.md`.
