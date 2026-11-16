#!/usr/bin/env python3
"""Start the Liver Tumour Intelligence Platform V11 (API + web app) on Windows, macOS or Linux.

    python start.py                       # http://127.0.0.1:8000
    python start.py --port 8080 --open    # other port, open the browser
    python start.py --host 0.0.0.0        # listen on the network (put a TLS reverse proxy in front for real use)
Uses .venv when present (created by install.py), otherwise the current interpreter.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--open", action="store_true", help="open the web app in the default browser when ready")
    a = ap.parse_args()
    vpy = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    py = str(vpy) if vpy.exists() else sys.executable
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "backend")]), PYTHONDONTWRITEBYTECODE="1")
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"Liver Tumour Intelligence Platform V11 → {url}   (API docs {url}/docs, health {url}/health/ready)")
    print("Research use only — not a medical device. Stop with Ctrl+C.")
    if a.open:
        def opener():
            for _ in range(240):
                try:
                    if urllib.request.urlopen(url + "/health", timeout=1).status == 200:
                        webbrowser.open(url)
                        return
                except Exception:
                    time.sleep(0.5)
        threading.Thread(target=opener, daemon=True).start()
    try:
        return subprocess.call([py, "-m", "uvicorn", "app.main:app", "--host", a.host, "--port", str(a.port)],
                               cwd=ROOT / "backend", env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
