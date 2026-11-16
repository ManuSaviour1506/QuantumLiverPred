# Project setup — step by step

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

**V11: on any OS (Windows, macOS, Linux) the shortest path is `python install.py` then `python start.py` — see
[INSTALLATION.md](INSTALLATION.md).** This guide describes the original shell-script route.

This guide gets the complete platform running on your own computer (written for macOS; Linux is the same). For
servers and cloud hosting see [DEPLOYMENT.md](DEPLOYMENT.md); for details of each part see
[INSTALLATION.md](INSTALLATION.md).

## 0. What is in the folder

```
liver_tumour_intelligence_v10/
├── setup.sh, run.sh          one-time setup, start the platform
├── frontend/                 web app (React + TypeScript; prebuilt copy in frontend/dist)
├── backend/                  API server (Python), database migrations, tests
├── src/                      model engine: V9 engine (unchanged) + src/v10 (classical, quantum, hybrid, evaluation)
├── artifacts/                trained models: V9 models + artifacts/v10 (16 V10 models)
├── dataset/
│   ├── liver_ct_lits_verified_v8/   the CT dataset (1,745 slices, 2,449 patches) — only in the full folder / full ZIP
│   ├── manifests/            slice manifest + SHA-256 of every dataset file
│   └── sample_data/          8 real CT slices for quick tests, synthetic phantoms for software tests
├── results/, reports/        evaluation results and reports (V9 imported, V10 nested evaluation)
├── scripts/                  dataset, training, evaluation, packaging scripts
├── docker/, docker-compose.yml, render.yaml   deployment
└── *.md                      documentation
```

## 1. Install the prerequisites (once per computer)

| Tool | Version | macOS install | Check |
|---|---|---|---|
| Python | 3.10 or newer (3.12 recommended) | `brew install python@3.12` (or pyenv / conda) | `python3.12 --version` |
| Node.js | 20 or newer | `brew install node` | `node --version` |
| Internet access | for the first `pip install` / `npm install` | — | — |

The Python that ships with macOS (3.9) is too old; `setup.sh` picks the newest Python 3.10+ it can find.
Homebrew itself: <https://brew.sh>. Node is optional — without it the prebuilt web app in `frontend/dist` is used.

> **Move the folder first.** If you want the project somewhere else (for example `~/Downloads/` next to the V9
> project), move it *before* running setup — the Python environment created by setup is tied to the folder's path.
> If you move it later, delete `.venv` and run `./setup.sh` again.

## 2. Run the setup

```bash
cd ~/Downloads/liver_tumour_project_v9/liver_tumour_intelligence_v10      # or wherever the folder now is
./setup.sh --admin-email you@example.org
```

(If macOS says "permission denied": `chmod +x setup.sh run.sh` or run `bash setup.sh …`.)

What it does — every step is safe to repeat:

1. creates a Python environment in `.venv` and installs `requirements.txt` (a few minutes the first time);
2. checks the model files (if any large file is missing it tells you how to restore it from V9:
   `./setup.sh --skip-install --v9 ~/Downloads/liver_tumour_project_v9`);
3. writes `backend/.env` with a random secret, a local SQLite database and e-mails saved to `backend/data/outbox`;
4. installs the web-app packages and builds the web app;
5. creates the database tables;
6. verifies the dataset in `dataset/liver_ct_lits_verified_v8` (SHA-256 of all 1,757 files and a full validation of
   all 1,745 images, ~15 s) and registers it in the platform;
7. creates your administrator account (asks for a password: at least 10 characters with two character types).

## 3. Start the platform

```bash
./run.sh
```

Open <http://localhost:8000> and sign in. Stop with `Ctrl+C`. Other port: `PORT=8001 ./run.sh`.
Frontend development with live rebuild: `./run.sh --dev` (then <http://localhost:5173>).

## 4. First things to try

1. **Workspace** → upload `dataset/sample_data/lits_sample/images/test/ct_lits_4277.png` (a CT slice with tumour) →
   pick a model (for example *Logistic Regression — V10, radiomics* or *VQC (4 qubits, brickwork) — V10, radiomics*) → **Run prediction**.
   For the 2.5-D CNNs also add the previous and next slices (`ct_lits_4276.png`, `ct_lits_4278.png`).
2. Open the result: probability map, explanation, model version, dataset version, execution mode; download the PDF.
3. **Compare models** → V10 nested evaluation table and chart; **Research** → circuits, noise study, ablations.
4. **Datasets** → *LiTS verified (V8)*: label and group distribution, validation report, file list.
5. **History** → search, filter, export CSV.

## 5. Using the dataset for training and evaluation

All scripts use `dataset/liver_ct_lits_verified_v8` automatically (override with `--dataset` or `LTI_DATASET_DIR`):

```bash
source .venv/bin/activate
python scripts/validate_dataset.py                              # full dataset check (~15 s)
python scripts/run_nested_evaluation.py --features radiomics    # the V10 nested evaluation (~60 min on 2 CPUs)
python scripts/train_classical.py                               # retrain the final V10 classical models
python scripts/train_vqc.py; python scripts/train_quantum_kernel.py; python scripts/train_hybrid.py
python scripts/export_model_registry.py && python scripts/write_results_report.py
```

Retrained models replace the ones in `artifacts/v10/` (their run records go to `results/v10/runs/`); restart
`./run.sh` to serve them. [V10_REPRODUCIBILITY.md](V10_REPRODUCIBILITY.md) lists the exact sequence used for the
delivered results.

## 6. Tests

```bash
source .venv/bin/activate
python -m pytest tests -q                    # model engine + quantum simulator (unit)
(cd backend && python -m pytest -q)          # API over real HTTP (integration)
(cd frontend && npm test)                    # web app (unit / render)
```

## 7. Docker instead (optional)

With Docker Desktop: `cp .env.example .env`, set `JWT_SECRET` and `POSTGRES_PASSWORD` in `.env`, then
`docker compose up --build` → <http://localhost:8080>. The project's `dataset/` folder is mounted into the containers;
register the dataset once with the command in [DEPLOYMENT.md](DEPLOYMENT.md#a-docker-compose).

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Python 3.10+ not found` | `brew install python@3.12`, open a new terminal, re-run `./setup.sh` |
| `pip install` fails on a package | update pip (`.venv/bin/python -m pip install -U pip`) and re-run; check internet / proxy |
| A model shows "unavailable" | a model file is missing: `./setup.sh --skip-install --v9 ~/Downloads/liver_tumour_project_v9` |
| Port 8000 in use | `PORT=8001 ./run.sh` |
| "Folder is not inside a configured DATASET_ROOTS location" | the dataset must be a real folder inside `dataset/` (not a symlink), or set `DATASET_ROOTS` in `backend/.env` |
| Start over with an empty local database | stop the server, delete `backend/data/`, run `./setup.sh --skip-install` |
| Forgot-password e-mail | with the local settings it is saved as a file in `backend/data/outbox/` |
| Windows | use WSL2 (Ubuntu) and follow the Linux steps, or Docker Desktop |
