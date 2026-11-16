# Real-time four-paradigm comparison platform

One uploaded CT slice → every eligible **Classical ML, Deep Learning, Quantum ML and Hybrid ML** model, executed with
four concurrent family pipelines, with measured predictions and timings, and the reason for every model that could not
run. Research and teaching use only; not a medical device.

## 1. What runs, and how

```
Upload (POST /api/v1/upload  or  /api/v1/upload/sample)
   │  format detection + decoding (PNG, JPEG, uncompressed DICOM, NIfTI)
   ▼
Shared front end — computed ONCE per run (src/v10/engine/compare.py: ComparisonRunner.prepare)
   decode → image QC → OOD check (EfficientNet-B0 features, Mahalanobis + residual)
   → 96-px patch grid (stride 64) → liver gate (patch classifier)
   → U-Net liver validation (new, audit finding 2) → liver patches + 3- / 5-slice stacks
   ▼
Eligibility plan — every registered model gets: eligible | unavailable (+ registry reason)
   | needs_context (neighbouring slices missing) | withheld / out_of_distribution / rejected (input stopped)
   ▼
Four family pipelines started at the same moment (ThreadPoolExecutor, 4 workers)
   ├─ Classical ML     ├─ Deep Learning     ├─ Quantum ML     ├─ Hybrid ML
   │  inside a family: one model at a time (COMPARE_MODEL_WORKERS_PER_FAMILY=1)
   │  shared features (radiomics, frozen backbones, CNN embeddings) computed once per run, under per-key locks
   ▼
Aggregation: per-family summary, agreement, timing ranking, family overlap (proof of concurrency)
   ▼
Live events (SSE) → dashboard panels update independently → result persisted (history, CSV, PDF, benchmarks)
```

### Scheduling — stated honestly

* The four **family** pipelines run concurrently. Each run reports `family_overlap`: when each family started and
  finished, the spread of the four start times (typically < 10 ms) and how long all four were running at the same time.
* Inside a family, models run **one after another** by default. They are CPU-bound NumPy / scikit-learn code; running all
  ~30 at once oversubscribes the CPU and makes the timings meaningless. `COMPARE_MODEL_WORKERS_PER_FAMILY` raises this.
* Every run carries `scheduling.strategy = "family_parallel_model_sequential"` and a plain-language description; the
  UI never claims that every model ran simultaneously.
* Python threads share the GIL; NumPy / BLAS release it for heavy array work, so family pipelines overlap in practice
  but are not four times faster than sequential execution. No GPU is used (none of the models needs one).

### Isolation and safety

* Every run has its own id, stored input, feature cache, cancel flag and event log — results of different uploads cannot
  mix (tested: two simultaneous runs give exactly the probabilities each gives alone).
* Model objects shared between runs are protected by per-model locks; models are loaded once at start-up and reused.
* HTTP handlers never block the event loop: uploads are validated in Starlette's thread pool, runs execute on a bounded
  pool (`COMPARE_MAX_CONCURRENT_RUNS`, default 2), the SSE generator only reads the in-memory event log.
* **Cancel** (`POST /compare/{id}/cancel`) and the **time limit** (`COMPARE_RUN_TIMEOUT_S`) stop models that have not
  started; a model already executing finishes (threads cannot be interrupted safely) and its result is still recorded.
* A failing model is caught and reported as `error` with its message; the other models and families continue.

## 2. Timing definitions (time.perf_counter, monotonic)

| Field | Meaning |
|---|---|
| `preprocessing.timings_ms` | shared front end: decode/QC, OOD, gate, U-Net liver validation, total |
| `queue_ms` | run start → model start (waiting for its family worker / earlier models) |
| `features_ms` | model-specific feature extraction; a feature shared with other models is paid by the first model that needs it |
| `inference_ms` | the model's own computation on prepared inputs — **used for all speed rankings** |
| `lock_wait_ms` | waiting for the model's lock (another run using the same model) |
| `total_ms` | queue + features + inference + slice-level decision |
| `cold_start` | first inference of that model in this server process |
| family `wall_ms` | family pipeline start → its last model finished |
| `load_ms` (V11) | model loading at server start: artifact read + SHA-256 integrity check + build + self-test (V9 models, measured around the vendored loader) or `load_model()` (V10 models); models are loaded once and reused, so runs never pay it |
| `resources.thread_cpu_ms` (V11) | CPU time of the worker thread while this model ran (`time.thread_time`); NumPy / BLAS helper threads are not included |
| run `resources` (V11) | sampled every 100 ms during the run: process CPU % = Δ process CPU time / Δ wall time / logical CPUs (all threads, incl. BLAS), resident memory (psutil; `/proc` on Linux; else peak RSS only), GPU detection (`nvidia-smi` presence — models never use a GPU) |

Exceptions, stated in the code: NumPy CNNs down-sample their stacks to 64 px inside the model, and the V9 radiomics
models compute their normalisation-specific features inside the model; for those, that work is part of `inference_ms`.

**Fastest / slowest family** = lowest / highest mean per-model `inference_ms` of completed models in that run.
**Fastest / slowest model** (V11) = lowest / highest `inference_ms` among the completed models of that run.
**Best performing model** = highest *recorded* primary metric (V10 nested mean within-unit AUC) among the models that ran
— never derived from the uploaded image.

## 3. Predictions

* Slice call per model = largest liver-patch probability vs the model's own threshold ± 0.10:
  Tumour detected / No tumour detected / Uncertain. Thresholds are unchanged from training.
* Other outcomes are never forced into a binary call: `needs_context`, `unavailable`, `withheld` (no / unconfirmed
  liver), `out_of_distribution`, `rejected` (QC), `error`, `cancelled`, `timeout`.
* `confidence` = decision margin: distance of p from the threshold scaled to [0, 1]. It is not a probability of being
  correct. `saturated` = p ≥ 0.999 (see `reports/v10/CALIBRATION_AND_SATURATION.md`).
* Calibration differs by source and is disclosed with every result: V9 = Platt on V9 validation scans; V10 = Platt on
  inner out-of-fold predictions; `v10_decision_tree_radiomics` v10.1 = isotonic regression on its leave-one-unit-out
  out-of-fold scores (V11; tree unchanged, v10.0 archived in `artifacts/archive/`).
* Family "majority" / hybrid "combined" call = majority vote of the completed models of the family, ties → Uncertain.
  It is a summary, not a validated ensemble. Agreement between models is not evidence of correctness.
* Quantum and hybrid models run on an exact state-vector simulator on the CPU; every result says `hardware: false`.

## 4. API (all under `/api/v1`, bearer token required)

| Method & path | Purpose |
|---|---|
| `POST /upload` (multipart: `file`, optional `prev_file`, `next_file`, `prev2_file`, `next2_file`, `modality`) | store + validate an image; returns QC, OOD, liver checks and per-model compatibility |
| `POST /upload/sample` `{sample_id}` | use a bundled real CT slice; its real neighbouring slices are attached |
| `GET /compare/samples` | real CT samples (ground truth, available context) and synthetic OOD phantoms |
| `POST /compare/run-all` `{upload_id}` | run every registered model → `202 {run_id, stream_url}` |
| `POST /compare/run-selected` `{upload_id, model_ids}` | run selected models (still four concurrent family pipelines) |
| `GET /compare/{run_id}` | status and full result |
| `GET /compare/{run_id}/stream` | Server-Sent Events (resume with `Last-Event-ID` or `?last_event_id=`) |
| `POST /compare/{run_id}/cancel` | cancel models not yet started |
| `GET /compare/{run_id}/export.csv`, `GET /compare/{run_id}/report` | CSV / PDF research report |
| `GET /history`, `GET /history/compare?a=&b=` | saved runs; side-by-side difference of two runs |
| `GET /benchmarks`, `GET /benchmarks/summary.csv` | recorded metrics by protocol + measured latencies (warm / cold) |
| `GET /models/availability` | registry with availability category, dependencies, input requirements, last success |
| `GET /system/status` | engine, weights, liver validation, availability counts, comparison settings, run statistics |
| `GET /system/metrics` (V11) | live process monitor (1 s samples of CPU %, RSS, threads for the last 5 min), active runs, engine load times, hardware |
| `GET /benchmarks/datasets` (V11) | labelled datasets available for new evaluations (bundled sample; full dataset when installed) |
| `POST /benchmarks/runs` `{dataset, split?, max_slices?, model_ids?}` (V11) | start a new evaluation run (researcher / admin) → `202 {id}` |
| `GET /benchmarks/runs`, `GET /benchmarks/runs/{id}` (V11) | runs with status, progress and the per-model, per-role metrics |
| `POST /benchmarks/runs/{id}/cancel`, `GET /benchmarks/runs/{id}/export.csv` (V11) | stop a run; per-slice CSV |

SSE events: `run_queued, run_started, preprocessing_completed, plan, family_started, model_started, model_completed,
model_failed, model_skipped, family_completed, cancel_requested, run_completed | run_cancelled | run_failed`.
The web client reads the stream with `fetch` (so the bearer token is sent), reconnects with the last event id and falls
back to polling `GET /compare/{run_id}` if streaming is blocked by a proxy.

The backend framework is the project's existing Starlette + Pydantic layer (`backend/app/core/apikit.py`), which plays
FastAPI's role (FastAPI is built on the same two libraries but was not installable in the build environment); the
OpenAPI document is generated at `/openapi.json`, Swagger UI at `/docs`.

## 5. New-evaluation benchmarking engine (V11)

`src/v10/engine/benchmark.py`, `backend/app/benchmarking/service.py`, migration `0003_benchmark_runs`.

* Runs the selected (default: all servable) models on ground-truth-labelled slices through **the same front end and
  concurrent family pipelines as the live comparison** (`ComparisonRunner` per slice; each slice's real neighbours from
  the same scan are attached, so context models run where the data allow).
* Ground truth per slice = expert-mask label (`label_tumour`). Datasets: the bundled 8-slice LiTS sample, and the full
  `liver_ct_lits_verified_v8` when installed (`dataset/` or `LTI_DATASET_DIR`), optionally one split and a slice cap.
* **Leakage control:** each model × slice result carries the role the slice's scan group played for that model —
  `test` (held out from training, threshold and calibration), `validation` (used to choose threshold / calibration —
  optimistic), `train` (in-sample; the final V10 models were fit on all units, so every slice is in-sample for them).
  Metrics are computed per role and never pooled; only `test` rows are estimates of generalisation.
* Metrics: accuracy, precision, recall (sensitivity), specificity, F1, confusion matrix on decisive calls; Uncertain
  counted separately; ROC-AUC, PR-AUC and ECE on the slice score (need both classes); coverage (withheld slices are
  not counted as negatives); median / p95 inference time; mean process CPU of the run. The patch-level recorded
  metrics are a different unit and are not comparable.
* Stored as benchmark runs with id, dataset, split, user and timestamps, apart from the historical (recorded)
  evaluations, which the Benchmarking page shows in a separate section. Nothing is tuned on these results.

## 6. System monitor (V11)

`ProcessMonitor` (backend) samples the API process every second: CPU % (process CPU time / wall time / logical
CPUs), resident memory, thread count. `GET /system/metrics` returns the last 5 minutes plus uptime, active runs, engine
load time, the slowest model loads and hardware (platform, Python, BLAS, GPU detection, quantum execution mode).
The **System Monitor** page polls it every 2 s.

## 7. Frontend pages

Overview · Image Upload · **Live Comparison** (2 × 2 family panels, live bars, timeline, outcome donut, ranking with
fastest / slowest model, prediction table, agreement, resource-utilisation chart, event log) · Model Results
(per-model table with CPU and load time + patch-probability overlay) · Benchmarking (new evaluation runs + historical
recorded evaluations + measured latency) · Model Registry · Datasets · Execution History (compare two runs) · Research
Reports · **System Monitor** · System Settings. The earlier
single-model pages (imaging workspace, prediction history, metric comparison, research module) remain available.

## 8. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `COMPARE_MAX_CONCURRENT_RUNS` | 2 | runs executing at once (each starts 4 family workers) |
| `COMPARE_MODEL_WORKERS_PER_FAMILY` | 1 | models at once inside one family |
| `COMPARE_RUN_TIMEOUT_S` | 300 | models not started by then → `timeout` |
| `LTI_LIVER_MIN_UNET_PX` | 4096 | U-Net liver-area minimum of the liver validation |

Comparison runs execute in the API process (also when `WORKER_MODE=external`), because the live stream reads their
in-memory event log; finished runs are persisted and can be replayed from any process.
