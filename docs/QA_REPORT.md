# QA report — V10 release

> **V11:** the current validation results (all suites re-run on the V11 build and on a clean extraction of the
> final ZIP) are in [../FINAL_TEST_REPORT.md](../FINAL_TEST_REPORT.md). This file is kept as the V10 record.

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

Latest run: 12 Nov 2026, build environment (Linux x86-64, 2 CPUs, Python 3.13.18, Node 22, PostgreSQL 16.15,
Playwright Chromium), with the 16 trained V10 models and the V9 models in place. Failed and skipped tests are listed;
nothing was removed to make a suite pass.

## Results by category

| Category | Suite / command | What it exercises | Result |
|---|---|---|---|
| **Unit — model engine and research code** | `python -m pytest tests` | simulator vs analytic states, gradients vs finite differences, feature maps, kernels (PSD, symmetry), noise channels, nested-evaluation logic (no leakage, sign-flip / Holm), data-integrity checks on the real labels and masks, engine interface, artifact integrity (tampered SHA-256 and preprocessing mismatch are refused), synthetic-model gating, V9 adapter | **96 passed, 1 skipped** (PyTorch training path: PyTorch not installed) |
| **Unit — backend helpers** | `backend/tests/test_units.py` (inside the backend run) | migrations up / down / status and checksum protection, SQL statement splitting, Argon2id hashing, access tokens, storage-key validation, ZIP ingestion safety, rate limiter, production settings require secrets, log redaction | included in the 42 below |
| **Integration — API over real HTTP, SQLite** | `cd backend && python -m pytest` | live uvicorn server on a temporary database: health / readiness / OpenAPI, registration, login, lock-out, refresh rotation and reuse detection, logout deny list, password reset by e-mail, roles, upload validation (wrong type, oversize, corrupt), real predictions with V9 and V10 models on real CT slices (CNN with context slices, VQC / QKSVM / hybrid / late fusion / classical twin), OOD withholding, history, CSV / PDF / JSON exports, artifact access control, deletion and retention purge, dataset ZIP upload with zip-slip / symlink rejection, validation jobs, experiments, analytics | **42 passed** (also 42 passed when run from the Docker image's file layout, as `docker compose exec backend python -m pytest` would) |
| **Integration — same suite on PostgreSQL 16** | `LTI_TEST_DATABASE_URL=postgresql://… python -m pytest` | the same 42 tests against a real PostgreSQL 16.15 server (through the test-only wire client; psycopg could not be installed) | **42 passed** |
| **Unit / render — frontend** | `cd frontend && npm test` | API client (URL building, structured errors, one refresh-and-retry on 401, sign-out on failed refresh, network errors), formatting, router matching, Zod form schemas, client-side image checks, query cache, chart rendering (incl. regression test for bars drawn over labels), UI components, disclaimer wording, landing page | **17 passed** |
| **Type check — frontend** | `npx tsc --noEmit -p tsconfig.offline.json` | TypeScript, non-strict offline profile | **56 errors, all caused by the missing `@types/react` package** (46 `IconBaseProps` / SVG attribute errors, 10 `key`-prop errors); no other errors. `@types/react` could not be installed here |
| **Build — frontend** | `npm run build` | production bundle | **passed** (app.js 441 KB, app.css 60 KB) |
| **End-to-end — browser** | `E2E_EXPECT_V10=1 npm run test:e2e` (Playwright, Chromium, against the live API serving `dist/`) | landing; registration with validation; protected routes; a real 2.5-D CNN prediction on three consecutive CT slices with Grad-CAM, PDF report; history and CSV export; models, comparison (asserts the V10 nested results and the V9 table), datasets, research and settings pages; phone-sized layout; logout; no console errors or CSP violations | **9 passed** (screenshots in `frontend/e2e/output/`) |
| **Local setup scripts** | `./setup.sh` (fresh and repeated), `./run.sh`, `./run.sh --dev` in a clean extract of the release (Linux, Python 3.13, Node 22; package installation skipped because the build environment has no network) | Python detection, model-file check and restore from V9, `backend/.env` generation, web-app build, migrations, dataset checksum + validation + registration (idempotent), admin creation; then the smoke test against `run.sh` | **passed** (the dataset step was exercised with an incomplete copy and correctly reported the missing files; registration tested separately). Not run on macOS itself: no macOS shell was available |
| **Deployment — configuration** | `docker compose config -q`; JSON / YAML parse of `frontend/vercel.json`, `render.yaml` | Compose file, env interpolation, health checks | **valid** |
| **Deployment — start-up reproduction** | container layout rebuilt under `/srv`-like tree; `ENVIRONMENT=production`; PostgreSQL 16; `python -m app.db.migrate upgrade`; the image's `uvicorn` CMD and `HEALTHCHECK`; `scripts/smoke_test_deployment.py --register --expect-production --image …` | migrations, production guards (start refused without `JWT_SECRET`), health and readiness, SPA with security headers, auth, synthetic models absent, real prediction (logistic regression V9: tumour slice called "Tumour detected"), PDF report, deletion, logout | **passed** |
| **Deployment — images and hosting** | `docker compose up --build`, Render, Vercel | — | **not run** (no Docker daemon, no cloud accounts in the build environment) |
| **Real-data — dataset** | `scripts/validate_dataset.py --dataset liver_ct_lits_verified_v8` (run on the owner's Mac: Python 3.10, NumPy 2.2.6, Pillow 12.3) | every one of the 1,745 slices decoded (all 512 × 512), pixel hashes against the labels file, labels, identifiers, duplicate slice ids, exact and near duplicates, unlisted images, scan-group leakage across splits | **passed**: 1,745 / 1,745 readable, 1,745 / 1,745 pixel hashes match, 0 issues, 0 exact duplicates, 0 unlisted images, 0 scan groups in more than one split; 300 near-duplicate groups (adjacent slices of one scan); warning: no patient identifiers (`results/v10/dataset_validation/lits_verified_v8_full.json`). The two dataset folders on that computer are byte-identical (SHA-256 of all 1,757 files) |
| **Real-data — nested evaluation** | `scripts/run_nested_evaluation.py` | 18 model specifications on radiomics (60 min), 6 on frozen ImageNet embeddings (32 min; a first 8-specification run was stopped by the environment and re-run — identical numbers for the models in both runs) | **completed**; results in `reports/v10/V10_NESTED_RESULTS.md` |
| **Real-data — registry consistency** | `scripts/evaluate_models.py` | every servable patch model scores real patches; probabilities finite and in [0, 1]; in-sample vs out-of-sample labelled | **39 of 39 scored models**: probabilities finite and in [0, 1] on 600 real patches; 2 five-slice V9 models skipped (the patch file stores one neighbour per side); `results/v10/registry_evaluation.json` |
| **Quantum hardware** | — | — | **not applicable**: no hardware access; all quantum results are classical simulations |

## Defects found and fixed during QA (this release)

* Horizontal bar charts drew bars from x = 0 even when the axis started at 0.5, covering the model names (Dashboard,
  Compare, Research). Fixed; regression test added.
* `evaluate_models.py` crashed on the five-slice V9 models and ran out of memory on the ImageNet backbones; now skips
  models that need two neighbours per side and scores in chunks.
* Loaded VQC / hybrid models reported default training settings (e.g. 40 epochs) in their circuit description;
  they now report none (the training settings are in the model card).
* `DATABASE_URL` ignored a `port` query parameter (unix-socket URLs); fixed.
* Duplicate `Cache-Control` header on artifact downloads; duplicate OpenAPI schema name (`RegisterIn`); fixed.
* Research data-integrity tests were silently skipped after a folder move (wrong project-root path); fixed — they now
  run and pass.

## Known issues (not fixed)

* V9 liver-tissue gate accepts some patches on non-liver slices (inherited; documented in every model card).
* Probabilities are not calibrated across scan units (pooled outer-fold ECE 0.10–0.23).
* `@types/react` missing → the type-check errors above.
* PyTorch training scripts, S3 storage and SMTP mail are implemented but untested here.

## V10.1 four-paradigm comparison (13 Nov 2026)

| Suite | Result | Notes |
|---|---|---|
| `backend/tests` (real HTTP, SQLite) | 57 passed | 42 earlier + 15 comparison tests: upload validation, samples with neighbours, run-all with concurrent family start and overlap, SSE event order, heart slice 4353 withheld, no-liver slice withheld, OOD phantom rejected, run-selected + failure isolation, cancellation, privacy, history / two-run diff / CSV / PDF, availability registry, benchmarks, two simultaneous runs give identical results to single runs |
| `tests/` (engine + research units) | 88 passed, 9 skipped | the V9 real-slice interface test skips in the lean release (2 large V9 files absent) instead of failing |
| `frontend/tests` | 20 passed | incl. the live-run reducer (independent family updates, failures, terminal events) |
| `frontend/e2e/compare_ui_test.py` (Chrome) | passed | sample upload → Run all models → four panels reach Completed live (SSE) → fastest / slowest family and concurrency measurement shown → history, registry, benchmarking pages; no console errors |
| TypeScript (`tsc --noEmit`) | `src/` clean | the test files still need `@types/node` (unchanged) |
