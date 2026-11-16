# API reference — Liver Tumour Intelligence Platform API 11.0.0

Generated from the application's OpenAPI document by `scripts/export_api_docs.py`. Interactive docs: `/docs` (Swagger UI); machine-readable: `/openapi.json` and `docs/openapi.json`.

Auth column: **public** = no token; **user** = any signed-in user (Bearer access token); **researcher** / **admin** = role required.

## system

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| GET | `/health` | public | Liveness probe: the process is up. Does not touch the database. |  |  |
| GET | `/health/ready` | public | Readiness probe: database reachable and migrations applied; model engine state reported. |  |  |
| GET | `/api/v1/system/info` | public | Version, configuration summary (no secrets), optional packages and quantum execution modes. |  | SystemInfoOut |

## auth

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| POST | `/api/v1/auth/register` | public | Create a researcher account (disabled when ALLOW_REGISTRATION=false). | application/json `RegisterIn` | UserOut |
| POST | `/api/v1/auth/login` | public | Exchange e-mail and password for an access token (15 min) and a rotating refresh token. | application/json `LoginIn` | TokenOut |
| POST | `/api/v1/auth/refresh` | public | Rotate the refresh token and issue a new access token. Re-using an old refresh token revokes the session. | application/json `RefreshIn` | TokenOut |
| GET | `/api/v1/auth/me` | user | The signed-in user. |  | UserOut |
| POST | `/api/v1/auth/logout` | user | Revoke the current access token and (if given) the refresh-token family. | application/json `LogoutIn` | Message |
| POST | `/api/v1/auth/forgot-password` | public | E-mail a one-hour reset link (503 if e-mail delivery is not configured). Same answer for unknown addresses. | application/json `ForgotPasswordIn` | Message |
| POST | `/api/v1/auth/reset-password` | public | Set a new password with a reset token; all existing sessions are revoked. | application/json `ResetPasswordIn` | Message |

## users

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| GET | `/api/v1/users/me` | user | Profile of the signed-in user. |  | UserOut |
| PATCH | `/api/v1/users/me` | user | Update display name and preferences (theme, default model, page size). | application/json `UpdateMeIn` | UserOut |
| POST | `/api/v1/users/me/password` | user | Change password (requires the current password); other sessions are signed out. | application/json `ChangePasswordIn` | Message |
| GET | `/api/v1/users` | admin | All users (administrators only). |  | UserPage |
| PATCH | `/api/v1/users/{user_id}` | admin | Change a user's role or active flag (administrators only). | application/json `AdminUserUpdateIn` | UserOut |

## models

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| GET | `/api/v1/models` | user | Every registered model (V9 and V10) with status, deployment state and recorded metrics. |  | list[ModelOut] |
| GET | `/api/v1/models/availability` | user | Runtime availability of every registered model: category, dependencies (weights, upstream models), input |  |  |
| GET | `/api/v1/models/{model_id}` | user | One model: architecture, version, training data, artifact, input contract, metrics. |  | ModelOut |
| GET | `/api/v1/models/{model_id}/metrics` | user | Saved evaluation results with their split and evaluation unit. |  | ModelMetricsOut |
| GET | `/api/v1/models/{model_id}/status` | user | Load / integrity status and whether the model can serve predictions. |  | ModelStatusOut |
| GET | `/api/v1/models/{model_id}/metadata.json` | user | Download the model's metadata as a JSON file. |  | string |
| POST | `/api/v1/models/sync` | admin | Reload the model engine from disk and mirror the registry into the database (administrators). |  |  |

## predictions

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| POST | `/api/v1/predictions` | user | Queue a prediction. Returns 202 with the id; poll GET /api/v1/predictions/{id} until status is final. | multipart/form-data `object` | PredictionCreated |
| GET | `/api/v1/predictions` | user | Prediction history with search, filters, sorting and pagination (own records; admins may request all). |  | PredictionPage |
| GET | `/api/v1/predictions/export.csv` | user | CSV export of the filtered history (no images). |  | string |
| GET | `/api/v1/predictions/{prediction_id}` | user | Prediction detail: status, result, explanation metadata and artifact links. |  | PredictionOut |
| DELETE | `/api/v1/predictions/{prediction_id}` | user | Delete a prediction and its stored images. |  |  |
| GET | `/api/v1/predictions/{prediction_id}/artifacts/{kind}` | user | Preview, patch-probability map or Grad-CAM image of a prediction (owner or admin only). |  | string |
| GET | `/api/v1/predictions/{prediction_id}/report` | user | PDF research report of a completed prediction (not a clinical report). |  | string |

## datasets

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| POST | `/api/v1/datasets/upload` | user | Upload a dataset archive; it is stored privately and validated in the background. | multipart/form-data `object` | DatasetOut |
| POST | `/api/v1/datasets/register` | admin | Register a folder already on the server (must be inside DATASET_ROOTS); administrators only. | application/json `DatasetRegisterIn` | DatasetOut |
| GET | `/api/v1/datasets` | user | Datasets visible to the user (own and shared). |  | list[DatasetOut] |
| GET | `/api/v1/datasets/{dataset_id}` | user | Dataset with all versions and their validation reports. |  | DatasetOut |
| DELETE | `/api/v1/datasets/{dataset_id}` | user | Delete a dataset you own (uploaded files are removed; registered folders are never touched). |  |  |
| POST | `/api/v1/datasets/{dataset_id}/validate` | user | Queue validation of the latest (or a given) version. | application/json `ValidateIn` | JobAccepted |
| GET | `/api/v1/datasets/{dataset_id}/statistics` | user | Label, split and group distributions, image sizes, intensity summary, identifier availability. |  |  |
| GET | `/api/v1/datasets/{dataset_id}/files` | user | Manifest viewer (paginated, filterable). |  | DatasetFilesPage |

## experiments

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| GET | `/api/v1/experiments` | user | Recorded experiments (saved result files) and the user's own runs. |  | list[ExperimentOut] |
| POST | `/api/v1/experiments` | user | Queue an experiment: quantum_diagnostics, noise_study or nested_evaluation. | application/json `ExperimentCreateIn` | ExperimentOut |
| GET | `/api/v1/experiments/{experiment_id}` | user | Experiment status and configuration. |  | ExperimentOut |
| GET | `/api/v1/experiments/{experiment_id}/results` | user | Experiment results (metrics, per-unit tables, logs). |  | ExperimentResultsOut |
| GET | `/api/v1/experiments/{experiment_id}/report` | user | PDF experiment report. |  | string |

## analytics

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| GET | `/api/v1/analytics/overview` | user | Counts from the database and the model registry (user totals for administrators only). |  | OverviewOut |
| GET | `/api/v1/analytics/model-comparison` | user | Saved evaluation results of every trained and evaluated model, with split and evaluation unit. |  |  |
| GET | `/api/v1/analytics/model-comparison.csv` | user | Model comparison as CSV. |  | string |
| GET | `/api/v1/analytics/model-comparison.pdf` | user | Model comparison report (PDF). |  | string |
| GET | `/api/v1/analytics/prediction-distribution` | user | Predictions by label, model, status, day and probability bin. |  |  |

## research

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| GET | `/api/v1/research/quantum` | user | Quantum and hybrid models: qubits, depth, gate counts, feature map, readout, execution mode. |  |  |
| GET | `/api/v1/research/summary` | user | Nested-evaluation comparisons, hybrid ablations, noise study, Stage 2 and mathematical validation, run logs. |  |  |

## benchmarks

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| GET | `/api/v1/benchmarks/datasets` | user | Ground-truth-labelled slice sets available for new evaluations (with splits and scan groups). |  |  |
| POST | `/api/v1/benchmarks/runs` | user | Start a new evaluation: every (or the selected) model on a labelled dataset; metrics per evaluation role. | application/json `BenchmarkRunIn` |  |
| GET | `/api/v1/benchmarks/runs` | user | New-evaluation runs (newest first). |  |  |
| GET | `/api/v1/benchmarks/runs/{bench_id}` | user | Status, progress and results of a new-evaluation run. |  |  |
| POST | `/api/v1/benchmarks/runs/{bench_id}/cancel` | user | Stop a running evaluation after the current slice. |  |  |
| GET | `/api/v1/benchmarks/runs/{bench_id}/export.csv` | user | Per-model, per-role metrics of a new-evaluation run as CSV. |  | string |
| GET | `/api/v1/system/metrics` | user | Live process metrics of the API server (CPU %, memory, threads, GPU detection), engine load times and the |  |  |

## comparison

| Method | Path | Auth | Summary | Request | Response |
|---|---|---|---|---|---|
| POST | `/api/v1/upload` | user | Upload and validate an image: format, quality control, out-of-distribution check, liver gate + U-Net liver | multipart/form-data `object` |  |
| POST | `/api/v1/upload/sample` | user | Use a slice from the bundled dataset; its real neighbouring slices from the same scan are attached. | application/json `SampleUploadIn` |  |
| GET | `/api/v1/uploads/{upload_id}` | user | A validated upload with its analysis. |  |  |
| GET | `/api/v1/compare/samples` | user | Real CT sample slices (with ground truth and available neighbours) and synthetic OOD test phantoms. |  |  |
| POST | `/api/v1/compare/run-all` | user | Run every registered model: the four family pipelines start concurrently. Returns the run id; follow | application/json `RunAllIn` |  |
| POST | `/api/v1/compare/run-selected` | user | Run only the selected models (still grouped into concurrent family pipelines). | application/json `RunSelectedIn` |  |
| GET | `/api/v1/compare/{run_id}` | user | Run status and, when finished, the full result. |  |  |
| GET | `/api/v1/compare/{run_id}/stream` | user | Server-Sent Events: run_started, preprocessing_completed, plan, family_started, model_started, |  | string |
| POST | `/api/v1/compare/{run_id}/cancel` | user | Cancel a running comparison (models not yet started are skipped). |  |  |
| GET | `/api/v1/compare/{run_id}/export.csv` | user | Per-model and per-family results of a run as CSV. |  | string |
| GET | `/api/v1/compare/{run_id}/report` | user | PDF research report of a run (all four families, measured timings). |  | string |
| GET | `/api/v1/history` | user | Saved comparison runs (newest first). |  |  |
| GET | `/api/v1/history/compare` | user | Side-by-side difference of two runs (labels, probabilities, inference times). |  |  |
| GET | `/api/v1/benchmarks` | user | Recorded evaluation metrics per model and family (labelled by protocol) plus latencies measured in runs. |  |  |
| GET | `/api/v1/benchmarks/summary.csv` | user | Benchmark summary as CSV. |  | string |
| GET | `/api/v1/system/status` | user | Engine state, model availability counts, weights, liver validation, comparison settings and run statistics. |  |  |

## Errors

Every error has the same JSON shape: `{"error": {"code": "...", "message": "...", "request_id": "..."}}` (plus `details` for validation errors). Server paths, secrets and stack traces are never returned.
