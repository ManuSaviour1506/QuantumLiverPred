# Installation

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

## 1. Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 (tested 3.13) | 64-bit; python.org installer recommended on Windows and macOS |
| OS | Windows 10/11, macOS 12+ (Apple Silicon or Intel), Linux (x86-64 / arm64) | CPU only — no GPU or CUDA used |
| Node.js | 20+ (tested 22), **optional** | only to rebuild the web app; a prebuilt `frontend/dist` is included |
| PostgreSQL | 16, optional | SQLite is used when `DATABASE_URL` is empty |
| Disk / memory | ≈ 1 GB with the virtual environment / 1.5–2 GB RAM with all models loaded | |

## 2. Automated installation (recommended)

```bash
# macOS / Linux
python3 install.py --admin-email you@example.org
python3 start.py --open                       # → http://127.0.0.1:8000

# Windows (Command Prompt or PowerShell, in the project folder)
install.bat --admin-email you@example.org
start.bat                                     # opens http://127.0.0.1:8000
```

`install.py` steps (each reported as OK / WARN / FAIL):

| Step | What it does |
|---|---|
| 1 Environment | detects OS (Windows, macOS Apple Silicon / Intel, Linux), CPU, Python; notes that no GPU is needed |
| 2 Python | creates `.venv` (Windows: `.venv\Scripts\python.exe`), installs `requirements.txt`, checks every runtime package imports |
| 3 Weights | verifies the three bundled ImageNet weights by SHA-256 (`scripts/fetch_weights.py`; downloads only a missing file) |
| 4 Models | checks the model artifacts; `--v9 PATH` restores the 7 large V9 files from a V9 project (SHA-256 verified) |
| 5 Settings | writes `backend/.env` with a random `JWT_SECRET` (kept if it exists) |
| 6 Web app | Node 20+ → `npm install && npm run build`; otherwise uses the prebuilt `frontend/dist` |
| 7 Database | applies the migrations (SQLite `backend/data/lti.db` or `DATABASE_URL`); `--admin-email` creates an administrator (password prompt, or `LTI_ADMIN_PASSWORD`) |
| 8 Verification | SHA-256 of all bundled dataset files; the full dataset against its 1,757-file SHA-256 list when present; loads the model engine and runs one real four-family comparison on `ct_lits_4278.png` |

Options:

| Option | Effect |
|---|---|
| `--check` | verify an existing installation, change nothing |
| `--no-venv` | use the current interpreter (conda, system Python, an existing venv) |
| `--skip-pip` | do not install packages (they must already be importable) |
| `--skip-frontend` | do not rebuild the web app |
| `--offline` | do not download weights, only verify them |
| `--v9 PATH` | restore the large V9 model files from `PATH` |
| `--admin-email EMAIL` | create an administrator |

`start.py` options: `--port`, `--host` (use `0.0.0.0` for network access behind a TLS proxy), `--open`.

The original macOS / Linux scripts still work: `./setup.sh --admin-email you@example.org` then `./run.sh`.

## 3. Docker

```bash
docker build -t lti-v11 .
docker run --rm -p 8000:8000 -e JWT_SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))") \
  -v lti-data:/var/lib/lti lti-v11                                  # → http://localhost:8000
# full stack (PostgreSQL + API + worker + nginx):
cp .env.example .env            # set JWT_SECRET and POSTGRES_PASSWORD
docker compose up --build       # → http://localhost:8080
docker compose exec backend python -m app.manage create-admin --email you@example.org
```

## 4. Manual installation

```bash
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_weights.py --check
cp backend/.env.example backend/.env          # then set JWT_SECRET
cd backend
export PYTHONPATH="$(cd ..; pwd):$(pwd)"      # Windows (PowerShell): $env:PYTHONPATH="..;."
python -m app.db.migrate upgrade
python -m app.manage create-admin --email you@example.org
uvicorn app.main:app --port 8000
```

* API: http://localhost:8000/api/v1 — Swagger UI: http://localhost:8000/docs — health: `/health`, `/health/ready`.
* PostgreSQL: `DATABASE_URL=postgresql://user:password@localhost:5432/lti` and `pip install "psycopg[binary]"`.
* Web app development: `cd frontend && npm install && npm run dev` (http://localhost:5173, proxies the API).

## 5. Restoring files that are not in this package

See [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md).

```bash
python install.py --v9 /path/to/liver_tumour_project_v9                         # 7 large V9 model files
cp -R /path/to/liver_ct_lits_verified_v8 dataset/ && python install.py --check  # full CT dataset, verified
mkdir -p data/masks && cp /path/to/liver_tumour_project_v9/data/masks/*.npz data/masks/   # derived masks (tests only)
```

## 6. Tests

```bash
python -m pytest tests -q                         # engine, regression cases, research modules
cd backend && python -m pytest -q                 # API + comparison + benchmark tests over real HTTP (temporary DB)
LTI_TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/lti_test python -m pytest -q    # against PostgreSQL
cd frontend && npm test                           # frontend unit / render tests
npm run typecheck                                 # TypeScript (application sources)
# browser end-to-end against a running server (pip install playwright; playwright install chromium):
E2E_BASE_URL=http://127.0.0.1:8000 python frontend/e2e/compare_ui_test.py
# reports
python scripts/v10_calibration_report.py          # reports/v10/CALIBRATION_AND_SATURATION.md
python scripts/v10_validate_liver_gate.py         # reports/v10/LIVER_GATE_VALIDATION.md (full dataset if present)
python scripts/export_pipeline_map.py             # docs/MODEL_PIPELINE_MAP.md
python scripts/v11_build_model_index.py           # models/model_index.json + per-family READMEs
```

The latest results and what could not be executed are in [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md).

## 7. Optional packages

`backend/requirements-optional.txt`: `boto3` (S3 storage), `torch` + `torchvision` (PyTorch training scripts),
`pennylane` / `qiskit` (simulator cross-checks), `xgboost`, `pydicom` (compressed DICOM). All are detected at runtime;
`GET /api/v1/system/info` reports which are present.
