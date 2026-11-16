# Final test report — Liver Tumour Intelligence Platform V11

Date: 2026-11-13/14.

## Where the tests ran

Every result below was produced by actually running the code. The final validation pass ran on a **clean extraction
of the release-candidate ZIP**, built by `scripts/v11_package.py`, in a directory unrelated to the build tree. That
candidate differs from the delivered ZIP only in this report, `COMPLETE_FILE_MANIFEST.json` and `SHA256SUMS.txt`. The
delivered ZIP was then verified again (integrity, SHA-256 of every file, extraction, installer check, engine
self-test, server start); see the end of this report.

## Test environment (hardware limits)

| | |
|---|---|
| OS / CPU | Linux 6.18 x86-64 container, **2 logical CPUs**, 7 GB RAM, **no GPU** |
| Python | 3.13.18 — numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, starlette 1.7.0, pydantic 2.13.5, uvicorn 0.54.0, psutil 7.2.2 |
| Node.js | 22.23.4 (no npm registry access) |
| Browser | Chrome for Testing 153 (Playwright 1.63) |
| Database | SQLite (default) and PostgreSQL 16.15 |
| Network | none for package installs (pip / npm / Docker Hub unreachable) |

Timings in reports and screenshots reflect this 2-CPU machine. Quantum models are simulated on the CPU; nothing uses
a GPU.

## Results

| Area | What was run | Result |
|---|---|---|
| ZIP integrity | `unzip -t`; CRC + size + SHA-256 of every member against the manifest (`v11_package.py`); single top folder; no absolute or `..` member paths | **passed** — 717 members, no errors |
| Extraction / checksums | extract to a clean directory; `sha256sum -c SHA256SUMS.txt` | **passed** — every file matches |
| Hidden dependencies | scan of the extracted tree for build-machine / user paths | **passed**. No code or configuration references a path outside the project. Absolute paths of the earlier build machine appear only inside historical logs under `results/v10/` (run logs, stdout captures), which record where the original V10 experiments ran and are never read. No personal data |
| Clean installation | `python install.py --no-venv --skip-pip --skip-frontend --offline` on the extracted copy | **passed** — 14 OK, 2 WARN (expected: 7 V9 files and the full dataset not included), 0 FAIL. This covers: weights SHA-256, model artifacts, `backend/.env` with a random secret, migrations 0001–0003, the 55 bundled dataset files SHA-256 checked, the engine loaded (33 of 63 servable), and a **real four-family comparison** on `ct_lits_4278` (completed per family: classical 17, deep learning 6, quantum 4, hybrid 4; all four overlapped 214 ms) |
| Installer, administrator / check mode | `--admin-email` with `LTI_ADMIN_PASSWORD`; then `--check` | **passed** |
| Installer, virtual-environment path | `python install.py` (creates `.venv`, runs pip) | **partially executed.** `.venv` created and pip invoked; installing from PyPI was impossible here (no network). The installer stopped with a clear FAIL line and a hint, not a traceback |
| Backend startup | `python start.py` from the extracted copy | **passed** — engine ready after 5 s; `/health`, `/health/ready`, `/docs`, `/openapi.json` return 200 |
| Engine / regression / research tests | `python -m pytest tests` | **passed — 103 passed, 8 skipped.** Release candidate: 97 + 8. The 6 input-format tests (`tests/test_v11_input_formats.py`) were added afterwards and the whole suite was re-run on the final extraction. Skips with their reasons: PyTorch not installed (1); derived 64 × 64 mask maps not included (7) |
| Backend API tests, SQLite | `cd backend && python -m pytest` (real uvicorn server over HTTP, temporary DB) | **passed — 62 passed** |
| Backend API tests, PostgreSQL 16 | same suite with `LTI_TEST_DATABASE_URL=postgresql://…` | **passed — 62 passed**, through the test-only wire client (`tests/support/pgwire.py`), because psycopg could not be installed here |
| Frontend type check | `tsc --noEmit -p tsconfig.json` | **passed — 0 errors** |
| Frontend unit / render tests | compiled with `tsc` and run with `node --test` (what `npm test` runs through tsx) | **passed — 20 / 20** |
| Browser e2e, comparison platform + V11 pages | `frontend/e2e/compare_ui_test.py` against the extracted copy's server | **passed.** Covers: sign-in; sample upload; RUN ALL MODELS; live SSE updates in the 4 family panels; concurrency measurement; resource-utilisation card; fastest / slowest model; run details; registry; System Monitor with measured CPU / memory; a **new evaluation run started from the UI** completing with per-role metrics, kept apart from the historical results; no console errors |
| Browser e2e, original V10 user flow | `frontend/e2e/e2e.test.mjs` | **passed — 9 / 9.** Covers: landing; registration validation; protected routes; **real prediction with the 2.5-D CNN**, Grad-CAM, **PDF report**; history + **CSV export**; models, comparisons, datasets, research and settings with real data; phone layout; logout; no console / CSP errors. Run through a harness shim, because the npm `playwright` package is not installed here: the shim re-exports the playwright-core bundled with Python Playwright and launches the system Chrome. The test file itself is unchanged apart from the V10.1 navigation updates in `reports/v11/V11_AUDIT_STATUS.md` |
| New-evaluation benchmark (via the e2e) | 8 labelled sample slices × 33 servable models, live pipeline | **completed in 12.9 s** (process CPU 55.8 % of 2 CPUs). **0 false positives, 0 false negatives.** Both liver-free slices were withheld by every model. Uncertain calls: 21 slice-model pairs. Specificity / ROC-AUC / PR-AUC are "—", because the sample has no tumour-free liver slice; these need the full dataset |

### Functional checklist

| Area | Result |
|---|---|
| Frontend build | type check + fallback bundle built and served, e2e passed; `npm run build` itself not executed, see "Not executed" |
| Backend startup | passed |
| API functionality | passed (backend suite, e2e) |
| Database connection | passed (SQLite, PostgreSQL 16) |
| Model registry | passed (63 tracked; availability reasons) |
| Model weights | passed (3 backbones SHA-256 verified; 7 V9 files reported missing) |
| Classical / deep-learning / quantum / hybrid inference | passed (self-test comparison, e2e, backend tests, V9 interface test on real slices) |
| Four-family concurrent execution | passed (measured overlap; concurrent-run isolation test) |
| Live frontend updates | passed (SSE in e2e) |
| Performance measurement | passed (load, queue, inference, CPU, memory) |
| Dataset loading | passed (sample; full dataset absent) |
| Image preprocessing | passed. PNG in all suites; the same real slice as JPEG, DICOM (HU) and NIfTI (HU) decodes to the PNG's grey levels (mean difference < 2) and gives the same four-family outcome (`tests/test_v11_input_formats.py`) |
| OOD rejection | passed |
| Liver validation | passed on the sample |
| Report generation | passed (prediction PDF; comparison CSV / PDF) |
| Execution history | passed |
| Clean installation | passed without pip; the pip / npm / Docker parts are not executable here |

## Failed tests

**None in the final validation pass.** Earlier runs during the build found real defects. Each was fixed, then all
suites were re-run on a fresh extraction:

* token refresh shared the strict login rate limit, logging users out after about 10 page loads a minute;
* the sign-out button had no accessible name on phones;
* the original e2e was stale after the V10.1 navigation change;
* a typecheck failed on the test files;
* registry entries were silently dropped.

Details: `reports/v11/V11_AUDIT_STATUS.md`.

## Not executed (and why)

| Not executed | Reason | What covers it instead |
|---|---|---|
| `npm install` / `npm run build` (esbuild) | no npm registry access; the esbuild binary available here is macOS-only | the type check passes. `frontend/dist` was built with the container's fallback bundler (same sources, unminified, no source map) and passed both browser e2e suites. `install.py` rebuilds it with `npm run build` when Node 20+ is present |
| `pip install -r requirements.txt` into a fresh venv | no PyPI access | venv creation and the pip invocation were executed; the tested package versions are listed above and in `requirements.txt` |
| Docker image build, `docker compose up` | no Docker daemon | Dockerfiles reviewed; the same commands run in the tested start path |
| Windows and macOS runs | Linux-only environment | `install.py` / `start.py` use `pathlib`, `.venv\Scripts\python.exe` on Windows, CRLF `.bat` wrappers, no shell-specific steps; psutil supplies CPU / memory on macOS / Windows |
| psycopg driver against PostgreSQL | psycopg not installable | the same SQL ran through the test wire client (62 passed) |
| GPU / CUDA | no GPU; the models are CPU NumPy / scikit-learn by design | GPU detection is reported as "none" |
| Full-dataset benchmarks, liver-gate threshold validation on all 1,745 slices, retraining | `liver_ct_lits_verified_v8` not in the package | scripts included (`scripts/v10_validate_liver_gate.py`, benchmark engine with `LTI_DATASET_DIR`) |
| cnn3d_v9 / cnn5s_v9 on real data | need two neighbouring slices on each side; no bundled sample slice has them | they load and self-test; they run on any upload with enough neighbours |
| 16 models that need the 7 missing V9 files | files not in the source workspace | reported as unavailable; `install.py --v9 PATH` restores them |
| PyTorch training scripts | torch not installed (optional) | — |

## Missing artifacts

See `MISSING_ARTIFACTS.md`:

* 7 large V9 files (4 `model.npz` + 3 training checkpoints) → 16 models unavailable (8 V9 comparison models, 8 V8-era
  legacy models);
* the full CT dataset `liver_ct_lits_verified_v8` (its SHA-256 list and manifest are included);
* the derived mask maps `data/masks`;
* no MRI data exist.

## Remaining known issues

1. **Liver-validation threshold** (U-Net ≥ 4,096 px) is validated on the 8-slice sample only. It must be checked on
   the full dataset (`scripts/v10_validate_liver_gate.py`).
2. **Deep-model saturation:** V9 CNN probabilities are near 0 / 1 (Platt slopes 9–31); re-calibration needs the V9
   validation scans. Read the decision margin, not the probability.
3. **Sample class balance:** the bundled sample cannot measure specificity or AUC. Use the full dataset for meaningful
   new-evaluation numbers.
4. **Decision tree v10.1** now gives decisive calls, but it remains a weak model: unit-validated AUC 0.763.
5. **Units are reconstructed scan groups, not patients** (no patient IDs in the source data).
6. **Timings depend on hardware;** the numbers here come from a 2-CPU container.

## Final ZIP verification (after this report was written)

* `scripts/v11_package.py` re-checked every member (CRC, size, SHA-256 against `COMPLETE_FILE_MANIFEST.json`).
* The ZIP was extracted to a new clean directory and checked with `sha256sum -c SHA256SUMS.txt`.
* `python install.py --check`, the engine self-test comparison, the regression suite and a server start were run on
  that extraction.
* The ZIP's own SHA-256 is in `Liver_Tumour_Intelligence_Platform_V11_COMPLETE.zip.sha256`.
