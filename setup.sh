#!/usr/bin/env bash
# One-time local setup of the Liver Tumour Intelligence Platform V11 (macOS / Linux; on any OS: python install.py).
#
#   ./setup.sh                                  # install, build, migrate, register the dataset
#   ./setup.sh --admin-email you@example.org    # ... and create an administrator (asks for a password)
#   ./setup.sh --v9 /path/to/liver_tumour_project_v9   # restore missing large model files from V9 first
#   ./setup.sh --skip-install                   # reuse the existing .venv and frontend/node_modules
#
# Afterwards start the platform with ./run.sh and open http://localhost:8000
# This platform is intended for academic research and software experimentation only. It has not been clinically
# validated and must not be used for diagnosis, treatment decisions or emergency medical decisions.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
ADMIN_EMAIL=""; V9_DIR=""; SKIP_INSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --admin-email) ADMIN_EMAIL="$2"; shift 2 ;;
    --v9) V9_DIR="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

step "1/7 Python (3.10 or newer)"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
[ -n "$PY" ] || die "Python 3.10+ not found. macOS: 'brew install python@3.12' (or pyenv install 3.12), then re-run."
echo "using $PY ($("$PY" --version))"
if [ "$SKIP_INSTALL" = 0 ] || [ ! -x .venv/bin/python ]; then
  [ -x .venv/bin/python ] || "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
fi
VPY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT:$ROOT/backend" PYTHONDONTWRITEBYTECODE=1

step "2/7 Model files"
if [ -n "$V9_DIR" ]; then "$VPY" scripts/v10_import_v9_artifacts.py --v9 "$V9_DIR"; fi
MISSING=$("$VPY" - <<'PY'
import json, pathlib
m = json.loads(pathlib.Path("artifacts/LARGE_FILES_MANIFEST.json").read_text()) if pathlib.Path("artifacts/LARGE_FILES_MANIFEST.json").exists() else {"files": []}
print(sum(1 for f in m["files"] if not pathlib.Path(f["path"]).exists()))
PY
)
if [ "$MISSING" != "0" ]; then
  warn "$MISSING large model files are missing (see artifacts/LARGE_FILES_MANIFEST.json); the models that need them will be shown as unavailable. Fix: ./setup.sh --skip-install --v9 /path/to/liver_tumour_project_v9"
else
  echo "all model files present"
fi

echo "ImageNet backbone weights (needed by the OOD detector, the liver gate and every patch model):"
if ! "$VPY" scripts/fetch_weights.py; then
  warn "backbone weights could not be downloaded or verified: every model will be reported as 'missing_dependency'. Copy the files listed in artifacts/manifests/weights_manifest.json into artifacts/weights/ (e.g. from the V9 project) and run: python scripts/fetch_weights.py --check"
fi

step "3/7 Local settings (backend/.env)"
if [ ! -f backend/.env ]; then
  SECRET="$("$VPY" -c 'import secrets; print(secrets.token_urlsafe(48))')"
  cat > backend/.env <<ENV
# Local development settings written by setup.sh. Never commit this file.
ENVIRONMENT=development
JWT_SECRET=$SECRET
# empty DATABASE_URL = SQLite file backend/data/lti.db; for PostgreSQL: postgresql://user:password@localhost:5432/lti
DATABASE_URL=
WORKER_MODE=inline
PRELOAD_MODELS=true
ALLOW_REGISTRATION=true
ALLOW_SYNTHETIC_MODELS=false
# password-reset e-mails are written to backend/data/outbox instead of being sent
MAIL_BACKEND=file
FRONTEND_BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:8000,http://localhost:5173
RETENTION_DAYS=30
LOG_JSON=false
ENV
  echo "wrote backend/.env (random JWT secret, SQLite database)"
else
  echo "backend/.env exists — kept"
fi

step "4/7 Web app (Node.js 20 or newer)"
if command -v node >/dev/null 2>&1 && node -e 'process.exit(+process.versions.node.split(".")[0] >= 20 ? 0 : 1)'; then
  if [ "$SKIP_INSTALL" = 0 ] || [ ! -d frontend/node_modules ]; then (cd frontend && npm install --no-audit --no-fund); fi
  (cd frontend && npm run build)
elif [ -f frontend/dist/index.html ]; then
  warn "Node.js 20+ not found — using the prebuilt web app in frontend/dist (install Node to rebuild: brew install node)"
else
  die "Node.js 20+ not found and no prebuilt frontend/dist. macOS: brew install node"
fi

step "5/7 Database migrations"
(cd backend && "$VPY" -m app.db.migrate upgrade && "$VPY" -m app.db.migrate status)

step "6/7 Dataset"
DS="$ROOT/dataset/liver_ct_lits_verified_v8"
if [ -f "$DS/patches/patches_v8.npz" ]; then
  if [ -f dataset/manifests/lits_verified_v8_SHA256SUMS.txt ]; then
    if command -v sha256sum >/dev/null 2>&1; then SUM="sha256sum"; else SUM="shasum -a 256"; fi
    if (cd "$DS" && $SUM -c --quiet "$ROOT/dataset/manifests/lits_verified_v8_SHA256SUMS.txt") > /tmp/lti_sha.log 2>&1; then
      echo "dataset files verified (SHA-256 of all 1,757 files)"
    else
      warn "$(grep -c 'FAILED' /tmp/lti_sha.log) dataset files are missing or differ from dataset/manifests/lits_verified_v8_SHA256SUMS.txt (list: /tmp/lti_sha.log)"
    fi
  fi
  if "$VPY" scripts/validate_dataset.py --out results/v10/dataset_validation/local_setup_check.json > /tmp/lti_validate.log 2>&1; then
    echo "dataset validation passed (report: results/v10/dataset_validation/local_setup_check.json)"
    (cd backend && "$VPY" -m app.manage register-dataset --path "$DS" --name "LiTS verified (V8)" --if-missing --validate-now)
  else
    tail -20 /tmp/lti_validate.log
    warn "dataset validation failed (details above) — not registered in the platform; fix the folder and re-run ./setup.sh --skip-install"
  fi
else
  warn "dataset/liver_ct_lits_verified_v8 not found — predictions work, training / evaluation scripts need the dataset (DATASET_GUIDE.md)"
fi

step "7/7 Administrator"
if [ -n "$ADMIN_EMAIL" ]; then
  (cd backend && "$VPY" -m app.manage create-admin --email "$ADMIN_EMAIL")
else
  echo "skipped (create one later: cd backend && ../.venv/bin/python -m app.manage create-admin --email you@example.org)"
  echo "or register in the web app — new accounts get the researcher role"
fi

printf '\n\033[1;32mSetup complete.\033[0m Start the platform with:  ./run.sh   then open http://localhost:8000\n'
