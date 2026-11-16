#!/usr/bin/env bash
# Start the platform locally: API + web app on http://localhost:8000 (Ctrl+C to stop). Run ./setup.sh once first.
#   ./run.sh            # serve the built web app and the API
#   ./run.sh --dev      # also start the frontend dev server with live rebuild on http://localhost:5173
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
[ -x .venv/bin/python ] || { echo "run ./setup.sh first" >&2; exit 1; }
export PYTHONPATH="$ROOT:$ROOT/backend" PYTHONDONTWRITEBYTECODE=1
PORT="${PORT:-8000}"
if [ "${1:-}" = "--dev" ]; then
  (cd frontend && BACKEND_URL="http://127.0.0.1:$PORT" PORT=5173 npm run dev) &
  DEV_PID=$!
  trap 'kill $DEV_PID 2>/dev/null' EXIT
  echo "frontend dev server: http://localhost:5173"
fi
echo "API + web app: http://localhost:$PORT   (API docs: http://localhost:$PORT/docs)"
cd backend
"$ROOT/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
