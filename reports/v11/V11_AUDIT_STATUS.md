# V11 — status of the V10 audit issues

Sources: `reports/v10_audit_2026-11-13/V10_case_review_and_audit.pdf` (case review),
`reports/v10/V10_UPGRADE_AUDIT.md` (V10.1 status) and the V11 consolidation.

Status terms:

* **Fixed** means fixed in code and covered by an executed test.
* **Mitigated** means the platform now detects and reports the problem, but a data or model limitation remains.
* **Open** means it cannot be done with the data in this package. The reason is given.

## Summary

| # | Issue | V10.1 status | V11 status | Test |
|---|---|---|---|---|
| 1 | Missing timm ImageNet backbone weights: models listed "available", every prediction crashed | Fixed (checked at start-up, `scripts/fetch_weights.py`) | **Fixed.** All three weights are now physically bundled (156 MB, SHA-256 verified by `install.py`, `fetch_weights.py --check` and at engine load) | `backend/tests/test_compare.py::test_availability_registry`, `tests/test_regression_cases.py::test_missing_backbone_weights_make_models_unavailable` |
| 2 | Liver gate accepted heart tissue on ct_lits_4353 → 5 models reported a tumour | Fixed on the sample (gate AND U-Net liver area ≥ 4,096 px) | **Fixed on the sample; full-data validation open.** 4353 and 4354 are never scored; all 6 tumour slices are analysed | `tests/test_regression_cases.py::test_liver_free_slices_are_never_scored[4353, 4354]`, backend `test_liver_free_heart_slice_is_not_analysed`. Open: the 4,096 px default must be re-checked on the full dataset (`scripts/v10_validate_liver_gate.py`); the dataset is not in this package |
| 3 | Decision tree returned p = 0.855 on every slice → always "Uncertain" | Mitigated (marked `non_deployable`) | **Fixed.** v10.1: unchanged tree, isotonic calibration fitted on its own leave-one-unit-out out-of-fold scores (2,449 patches), threshold re-selected by Youden J on those scores (0.6790, midway between attained values). Validated per held-out unit (each unit calibrated on the others), see table below. v10.0 archived in `artifacts/archive/v10_decision_tree_radiomics_v10.0/`; script `scripts/v11_recalibrate_decision_tree.py` | `tests/test_regression_cases.py::test_decision_tree_versions` (v10.0 detected non-deployable, v10.1 makes confident calls), backend `test_decision_tree_makes_confident_calls` |
| 4 | Deep models return p ≈ 1.000 on tumour slices (saturation) | Investigated, mitigated (`saturated` flag, decision margin, calibration report) | **Mitigated; recalibration open.** Cause documented in `reports/v10/CALIBRATION_AND_SATURATION.md`: V9 Platt slopes of 9–31 on near-separable validation groups; test ECE above validation ECE for every CNN (1.3–12×). V11 adds per-role ECE in the new benchmark engine. Re-calibrating the V9 CNNs needs their validation scans (full dataset), and doing it on the 8-slice sample would leak | `reports/v10/CALIBRATION_AND_SATURATION.md` |
| 5 | Context-dependent models could not run on the sample | Fixed in the workflow (neighbour uploads, automatic real neighbours for samples, 5-slice stacks) | **Fixed in the workflow; data limitation remains.** The benchmark engine also attaches real neighbours. cnn3d_v9 / cnn5s_v9 need two slices on each side, which the 8-slice sample does not provide for any slice | backend `test_run_selected_and_failure_isolation`; compatibility table |
| 6 | Missing V9 model artifacts | Reported in `artifacts/LARGE_FILES_MANIFEST.json`; restore script | **Open for 7 files (not in the source workspace); everything else consolidated.** All V9 model folders, the OOD detector, the liver gate, the U-Net and every V10 model are included. The 7 large V9 files (four `model.npz`, three training checkpoints) were not in the V10 folder this package was built from and were not fabricated. The 16 models that need them are listed as unavailable with the reason. `python install.py --v9 PATH` restores them with SHA-256 verification (`MISSING_ARTIFACTS.md`) | `tests/test_v10_engine.py::test_v9_models_follow_the_interface_on_real_slices` (asserts each missing file → `missing_weights`), `tests/test_regression_cases.py::test_models_with_missing_weight_files_are_reported_not_hidden` |
| 7 | Runtime dependency on files in the results directory (`results/v9/cnn_selection.json`) | Fixed (`artifacts/manifests/v9_cnn_selection.json`) | **Fixed.** No inference path reads `results/`. The only remaining reads are of recorded (historical) metrics shown on the Research / Benchmarking pages; they are optional and their absence does not affect any prediction | `tests/test_regression_cases.py::test_artifacts_self_contained` |
| 8 | Registry did not reflect runtime availability | Fixed (8 availability categories, dependencies, last success) | **Fixed, plus a V11 defect fix.** V9 models whose `model.npz` was missing were silently skipped by the V9 loader, so `efficientnet_b0_ft_v9` and `resnet50_ft_v9` were absent. They are now listed as `missing_weights` with the reason, and described from their own `model.json`. The registry tracks 63 models (was 61). Also new: per-model load time, `models/model_index.json` | `tests/test_regression_cases.py::test_models_with_missing_weight_files_are_reported_not_hidden`, backend `test_resources_and_load_times_measured` |
| 9 | OOD detection and liver-free image validation | Front end: QC → OOD (EfficientNet-B0 feature detector) → liver gate → U-Net liver area | **Fixed and verified on the available data.** Non-CT inputs and synthetic phantoms are rejected as out-of-distribution. Liver-free slices (gate AND U-Net ≥ 4,096 px) are withheld, never scored, and never counted as negatives in benchmarks. Full-data validation of the liver threshold is open (dataset not included). The sample has only 2 real liver-free slices and no tumour-free liver slice; see item 11 | `tests/test_regression_cases.py::test_ood_phantom_rejected`, `::test_liver_free_slices_are_never_scored`, backend `test_liver_free_heart_slice_is_not_analysed` |
| 10 | Regression testing for false-positive and false-negative cases (liver-free, tumour, OOD inputs) | Added (backend) | **Extended.** Engine-level suite (8 tests: liver-free ×2, no false negatives on the 6 tumour slices, OOD phantom, missing weights, decision-tree versions, self-contained artifacts, registry completeness) and browser e2e for the V11 pages | `tests/test_regression_cases.py`, `frontend/e2e/compare_ui_test.py` |

Further V11 items:

| # | Issue | V10.1 status | V11 status | Test |
|---|---|---|---|---|
| 11 | Only 2 real tumour-free (liver-free) slices; the case review padded the "not detected" set with labelled phantoms | — | **Mitigated; data limitation.** The new benchmark engine reports this instead of hiding it. The bundled sample has 6 tumour and 2 liver-free slices and **no tumour-free liver slice**, so specificity, ROC-AUC and PR-AUC on liver tissue cannot be estimated from it; the UI warns and the metrics show "—". Meaningful numbers need `liver_ct_lits_verified_v8` | backend `test_new_evaluation_benchmark_run` |
| 12 | No measurement of model loading, CPU, memory or GPU; no fastest / slowest model | — | **Fixed.** `load_ms` per model, `thread_cpu_ms` per model run, a process CPU % / RSS sampler per run (100 ms), GPU detection (never used), fastest / slowest model and family, a System Monitor page and `/system/metrics` | backend `test_resources_and_load_times_measured`, `test_system_metrics`; e2e |

## Decision tree v10.1 — validation (from the model card)

| | v10.0 (Platt) | v10.1 isotonic, in-sample | v10.1 isotonic, unit-validated |
|---|---|---|---|
| threshold | 0.855 | 0.679 | 0.679 |
| attainable probabilities | 0.103 – 0.855 | 0.216, 0.278, 0.437, 0.921 | per unit |
| calls (tumour / no tumour / uncertain) | 0 % / 59 % / 41 % | 41 % / 59 % / 0 % | 42 % / 58 % / 0 % |
| ROC-AUC | 0.813 | 0.836 (optimistic) | **0.763** |
| ECE | 0.106 | 0.000 (optimistic) | **0.130** |
| sensitivity / specificity at threshold | 0.667 / 0.926 | 0.667 / 0.926 | **0.667 / 0.891** |

The unit-validated column is the honest estimate: each held-out unit was calibrated by an isotonic fit on the other
units only. The decision tree remains the weakest V10 classical model (recorded nested AUC 0.795). The fix makes it
produce decisive calls; it does not make it a good model.

## Additional defects found and fixed during V11 validation

| Defect | Found by | Fix | Test |
|---|---|---|---|
| Token refresh shared the strict login rate limit (10/min per IP). Every page load refreshes the token, so about 10 reloads a minute from one address logged the user out (HTTP 429 on `/auth/refresh`) | browser e2e (original V10 flow) | `/auth/refresh` has its own bucket (`RATE_LIMIT_REFRESH`, default 60/min). Refresh tokens are single-use and rotated; login, registration and password reset stay at 10/min | `backend/tests/test_units.py::test_token_refresh_has_its_own_rate_bucket`; e2e |
| Sign-out button had no accessible name on small screens (its text is hidden below the `sm` breakpoint) | browser e2e (phone viewport) | `aria-label="Sign out"` | e2e "logout ends the session" |
| The original V10 browser e2e (`frontend/e2e/e2e.test.mjs`) still expected the pre-V10.1 navigation: `/app` showing "Welcome", the registry links and the "Prediction history" menu item | running it | Updated to the V10.1+ layout: overview at `/app`, personal dashboard at `/app/activity`, registry links by model id, "Execution History". The tested flows are unchanged | e2e |
| System Monitor chart labelled "seconds ago" but plotted time forwards | screenshot review | x axis = seconds relative to the latest sample (0 = now) | visual check |
| Measured-latency card said "No comparison run yet" after a run (only cold executions existed) | screenshot review | Message now distinguishes "no run yet" from "only cold-start executions so far" | visual check |
| Frontend `npm run typecheck` failed on test files (`node:test` types not installed) | running it | Typecheck covers the application sources; the tests are compiled and run by their own runner | `npm run typecheck` |
| `test_v9_models_follow_the_interface_on_real_slices` was skipped whenever any large V9 file was missing | test review | Now runs on every V9 model that can load and asserts that the missing ones are reported as `missing_weights` | `tests/test_v10_engine.py` |

## Still open

These cannot be done with the data in this package:

* liver-gate threshold validation on the full dataset;
* re-calibration of the V9 CNNs;
* full-dataset benchmark numbers;
* cnn3d / cnn5s on the bundled sample.

In addition, 16 models need the 7 V9 files that are not included (see `MISSING_ARTIFACTS.md`).
