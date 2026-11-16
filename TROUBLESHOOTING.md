# Troubleshooting

Start with `python install.py --check` (Windows: `install.bat --check`). It verifies the interpreter and packages,
weights, model artifacts, settings, web app, database, datasets and the model engine, and runs one real comparison as
a self-test. Every line it prints is OK, WARN or FAIL, with the reason.

## Installation

| Symptom | Cause | Fix |
|---|---|---|
| `Python 3.10 or newer required` | old interpreter | install Python 3.11–3.13 from python.org (Windows: tick *Add python.exe to PATH*), then `py -3 install.py` / `python3 install.py` |
| `python` not found (Windows) | Python not on PATH | use `py -3 install.py` or `install.bat`; or reinstall Python with *Add to PATH* |
| `No module named venv` / `ensurepip` (Debian / Ubuntu) | distro splits venv out | `sudo apt install python3-venv`, or `python3 install.py --no-venv` to use the current environment |
| pip fails building `opencv` / `scikit-image` | no wheel for your Python (very new or 32-bit) | use 64-bit Python 3.11–3.13; on Apple Silicon use the arm64 python.org build (not Rosetta) |
| `missing Python packages ...` after `--skip-pip` | packages not installed in that interpreter | run `python install.py` without `--skip-pip`, or `pip install -r requirements.txt` |
| corporate proxy / offline machine | pip cannot reach PyPI | `pip download -r requirements.txt -d wheels` on a connected machine, then `pip install --no-index --find-links wheels -r requirements.txt` and `python install.py --skip-pip --offline` |
| `npm build failed; falling back to the prebuilt web app` | Node < 20 or blocked npm registry | nothing to do: `frontend/dist` is included. To rebuild, install Node 20+ and run `cd frontend && npm install && npm run build` |
| Windows: `install.bat` closes immediately | run by double-click and failed | open *Command Prompt* in the project folder and run `install.bat` to read the messages |
| macOS: "cannot be opened because the developer cannot be verified" for a `.pth` / script | quarantine flag from the browser download | `xattr -dr com.apple.quarantine /path/to/Liver_Tumour_Intelligence_Platform_V11` |

## Weights and models

| Symptom | Cause | Fix |
|---|---|---|
| `FAIL efficientnet_b0 ... -> hash mismatch` / missing | weight file damaged or removed | `python scripts/fetch_weights.py` (downloads the official timm release and verifies SHA-256); or re-extract it from the ZIP |
| models shown as `missing_dependency` "EfficientNet-B0 ImageNet weights" | backbone weights missing | as above; the OOD detector and liver gate need EfficientNet-B0 |
| 16 models `missing_weights` / `missing_dependency` "fine-tuned EfficientNet-B0" | the 7 large V9 files are not in this package | `python install.py --v9 /path/to/liver_tumour_project_v9` — see MISSING_ARTIFACTS.md |
| models `requires_context` | 2.5-D / 3-D / 5-slice models need neighbouring slices | upload the previous / next slice (and two each side for 5-slice models); dataset samples attach their real neighbours automatically |
| a model `non_deployable` | its calibrated probability can never cross threshold + band | expected only for archived versions; the registry shows the reason |
| engine status "loading" for a minute after start | 63 models are loaded and self-tested at start-up (≈ 5–30 s depending on CPU) | wait; `/health/ready` reports `engine: ready` when done |
| an upload says "no liver" / "out of distribution" and no model runs | the front end withheld a slice that is not a liver CT slice (or has < 4,096 liver pixels by U-Net) | intended behaviour; liver-free slices are never scored. Use an axial abdominal CT slice through the liver |

## Running

| Symptom | Cause | Fix |
|---|---|---|
| `address already in use` on port 8000 | another program uses the port | `python start.py --port 8080` |
| sign-in works, then sessions end after a restart | no `JWT_SECRET` (random per process in development) | `install.py` writes one into `backend/.env`; keep that file |
| `JWT_SECRET must be set in production` | `ENVIRONMENT=production` without secrets | set `JWT_SECRET` (≥ 32 random chars) and `DATABASE_URL` |
| locked out after failed sign-ins | 5 failures → 15-minute lock | wait, or an admin resets it; `python -m app.manage create-admin` creates a new admin |
| blank page / old UI after an update | browser cache | hard-reload (Ctrl+Shift+R / Cmd+Shift+R) |
| live comparison shows no updates behind nginx / a proxy | proxy buffers Server-Sent Events | disable buffering for `/api/v1/compare/*/stream` (`proxy_buffering off;` — already set in `docker/nginx.conf`) |
| comparison queued for a long time | `COMPARE_MAX_CONCURRENT_RUNS` (default 2) runs already active | wait, or raise it in `backend/.env` (CPU-bound: more runs = slower runs) |
| System Monitor shows memory "n/a" | `psutil` not installed and no `/proc` (macOS / Windows) | `pip install psutil` (included in requirements.txt) |
| benchmark "New evaluation" offers only the 8-slice sample | the full dataset is not installed | copy `liver_ct_lits_verified_v8` into `dataset/` (or set `LTI_DATASET_DIR`) and restart |

## Docker

| Symptom | Fix |
|---|---|
| `POSTGRES_PASSWORD` required | `cp .env.example .env` and set `POSTGRES_PASSWORD` and `JWT_SECRET` |
| image build fails at `npm install` | network / proxy for the Node stage; or build only the API (`docker/backend.Dockerfile`) and serve `frontend/dist` |
| health check failing for ~90 s after start | model loading; the health check has a 90 s start period |
| full dataset not visible in the container | mount it: `-v /path/to/liver_ct_lits_verified_v8:/srv/dataset/liver_ct_lits_verified_v8:ro` (compose: `LTI_DATASET_HOST_DIR`) |

## Tests

| Symptom | Fix |
|---|---|
| 9 root tests skipped | expected without the large V9 files (1), PyTorch (1) or the derived masks (7); each skip prints its reason |
| `npm test` fails with `tsx` missing | `cd frontend && npm install` |
| browser e2e cannot find Chromium | `pip install playwright && playwright install chromium`, or `E2E_CHROME=/path/to/chrome` |

Still stuck? Collect `python install.py --check` output and the server log (the terminal running `start.py`). Requests
are logged with a request id, which the API also returns in every error response.
