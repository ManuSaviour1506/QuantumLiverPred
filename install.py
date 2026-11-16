#!/usr/bin/env python3
"""Cross-platform installer for the Liver Tumour Intelligence Platform V11 (Windows, macOS Apple Silicon / Intel, Linux).

    python install.py                                   # full install (venv, dependencies, weights, web app, database, checks)
    python install.py --admin-email you@example.org     # ... and create an administrator (password asked, or LTI_ADMIN_PASSWORD)
    python install.py --skip-pip --no-venv              # use the current Python environment as it is
    python install.py --check                           # verify an existing installation only (no changes)
Windows: run it with `py install.py`. Afterwards start the platform with `python start.py` and open http://localhost:8000.

Every step reports what it found; nothing is silently skipped. CPU only: no GPU or CUDA is required or used.
Research and teaching use only; not a medical device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
OK, WARN, FAIL = "OK", "WARN", "FAIL"
COLOUR = {"OK": "\033[32mOK\033[0m", "WARN": "\033[33mWARN\033[0m", "FAIL": "\033[31mFAIL\033[0m"}
USE_COLOUR = sys.stdout.isatty() and not IS_WIN and not os.environ.get("NO_COLOR")
REPORT: list[tuple[str, str, str]] = []


def step(title: str):
    print(f"\n==> {title}")


def record(status: str, what: str, detail: str = ""):
    REPORT.append((status, what, detail))
    print(f"  [{COLOUR[status] if USE_COLOUR else status}] {what}" + (f" — {detail}" if detail else ""))


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def run(cmd, cwd=None, env=None, check=True, capture=False):
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "backend")])
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        e.update(env)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, env=e, text=True,
                       capture_output=capture)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stderr or ''}")
    return r


def environment():
    step("1/8 Environment")
    mach = platform.machine().lower()
    osname = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(platform.system(), platform.system())
    flavour = "Apple Silicon" if osname == "macOS" and mach in ("arm64", "aarch64") else (
        "Intel" if osname == "macOS" else mach)
    record(OK, "platform", f"{osname} {flavour}, Python {platform.python_version()}")
    if sys.version_info < (3, 10):
        record(FAIL, "Python 3.10 or newer required", sys.version.split()[0])
        raise SystemExit(1)
    smi = shutil.which("nvidia-smi")
    record(OK, "compute", "CPU execution (NumPy / scikit-learn); " + ("an NVIDIA GPU is present but not needed" if smi
                                                                     else "no GPU needed"))
    return osname


def python_env(args) -> Path:
    step("2/8 Python environment and dependencies")
    py = Path(sys.executable)
    if args.check:
        py = venv_python() if (venv_python().exists() and not args.no_venv) else py
        record(OK, "interpreter checked", str(py))
    elif not args.no_venv:
        if not venv_python().exists():
            print("  creating .venv ...")
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
        py = venv_python()
        record(OK, "virtual environment", str(py.relative_to(ROOT)))
    else:
        record(OK, "using the current interpreter", str(py))
    if not args.skip_pip and not args.check:
        run([py, "-m", "pip", "install", "--upgrade", "pip"], capture=True, check=False)   # best effort
        print("  installing requirements.txt (numpy, scipy, scikit-learn, scikit-image, opencv-headless, starlette, ...)")
        run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
        record(OK, "Python packages installed", "requirements.txt")
    missing = run([py, "-c", "import importlib.util as u,sys; m=[n for n in ('numpy','scipy','sklearn','skimage',"
                   "'cv2','PIL','pandas','joblib','starlette','uvicorn','pydantic','jwt','cryptography','reportlab') "
                   "if u.find_spec(n) is None]; print(','.join(m))"], capture=True).stdout.strip()
    if missing:
        record(FAIL, "missing Python packages", missing + " (run without --skip-pip)")
        raise SystemExit(1)
    record(OK, "required Python packages importable")
    return py


def weights(py: Path, args):
    step("3/8 ImageNet backbone weights (artifacts/weights)")
    cmd = [py, "scripts/fetch_weights.py"] + (["--check"] if args.check or args.offline else [])
    r = run(cmd, check=False, capture=True)
    for line in r.stdout.strip().splitlines():
        record(OK if line.startswith("OK") else WARN, line[5:].strip() if len(line) > 5 else line)
    if r.returncode != 0:
        record(WARN, "some backbone weights are missing or could not be downloaded",
               "dependent models will be listed as 'missing_dependency' (see TROUBLESHOOTING.md)")


def model_artifacts(py: Path, args):
    step("4/8 Model artifacts")
    if args.v9 and not args.check:
        r = run([py, "scripts/v10_import_v9_artifacts.py", "--v9", args.v9], check=False, capture=True)
        record(OK if r.returncode == 0 else FAIL, "restore V9 model files from " + args.v9,
               ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")
    man = json.loads((ROOT / "artifacts" / "LARGE_FILES_MANIFEST.json").read_text())
    absent = [f["path"] for f in man["files"] if not (ROOT / f["path"]).exists() and "artifacts/weights" not in f["path"]]
    if absent:
        record(WARN, f"{len(absent)} large V9 files not included in this package",
               "restore with: python scripts/v10_import_v9_artifacts.py --v9 /path/to/liver_tumour_project_v9 "
               "(see MISSING_ARTIFACTS.md); the models that need them are reported as unavailable")
    else:
        record(OK, "all V9 large model files present")
    art = ROOT / "artifacts"
    folders = {p.parent for p in art.rglob("model.json")} | {p.parent for p in art.rglob("model_card.json")}
    folders = {f for f in folders if "archive" not in f.relative_to(art).parts}
    record(OK, "model artifact folders found", f"{len(folders)} (the model engine verifies each one at start-up)")


def env_file(args):
    step("5/8 Local settings (backend/.env)")
    p = ROOT / "backend" / ".env"
    if p.exists():
        record(OK, "backend/.env exists — kept")
        return
    if args.check:
        record(WARN, "backend/.env missing (python install.py writes one)")
        return
    p.write_text("# Local settings written by install.py. Never commit this file.\n"
                 "ENVIRONMENT=development\n"
                 f"JWT_SECRET={secrets.token_urlsafe(48)}\n"
                 "DATABASE_URL=\nWORKER_MODE=inline\nPRELOAD_MODELS=true\nALLOW_REGISTRATION=true\n"
                 "ALLOW_SYNTHETIC_MODELS=false\nMAIL_BACKEND=file\nFRONTEND_BASE_URL=http://localhost:8000\n"
                 "CORS_ORIGINS=http://localhost:8000,http://localhost:5173\nRETENTION_DAYS=30\nLOG_JSON=false\n"
                 "COMPARE_MAX_CONCURRENT_RUNS=2\nCOMPARE_MODEL_WORKERS_PER_FAMILY=1\nCOMPARE_RUN_TIMEOUT_S=300\n")
    record(OK, "wrote backend/.env", "random JWT secret, SQLite database in backend/data/")


def frontend(args):
    step("6/8 Web app")
    node, npm = shutil.which("node"), shutil.which("npm") or shutil.which("npm.cmd")
    dist = ROOT / "frontend" / "dist" / "index.html"
    ok_node = False
    if node:
        v = subprocess.run([node, "-p", "process.versions.node"], capture_output=True, text=True).stdout.strip()
        ok_node = v.split(".")[0].isdigit() and int(v.split(".")[0]) >= 20
    if ok_node and npm and not args.skip_frontend and not args.check:
        try:
            run([npm, "install", "--no-audit", "--no-fund"], cwd=ROOT / "frontend")
            run([npm, "run", "build"], cwd=ROOT / "frontend")
            record(OK, "web app built", "frontend/dist (npm run build)")
            return
        except RuntimeError as e:
            record(WARN, "npm build failed; falling back to the prebuilt web app", str(e).splitlines()[0])
    if dist.exists():
        record(OK, "prebuilt web app present", "frontend/dist (Node.js 20+ only needed to rebuild)")
    else:
        record(FAIL, "no web app: install Node.js 20+ and re-run", "https://nodejs.org")


def database(py: Path, args):
    step("7/8 Database")
    if args.check:
        r = run([py, "-m", "app.db.migrate", "status"], cwd=ROOT / "backend", check=False, capture=True)
        record(OK if r.returncode == 0 else WARN, "migration status", (r.stdout or r.stderr).strip().splitlines()[-1]
               if (r.stdout or r.stderr).strip() else "")
        return
    run([py, "-m", "app.db.migrate", "upgrade"], cwd=ROOT / "backend")
    record(OK, "migrations applied", "SQLite backend/data/lti.db (or DATABASE_URL)")
    if args.admin_email:
        run([py, "-m", "app.manage", "create-admin", "--email", args.admin_email], cwd=ROOT / "backend")
        record(OK, "administrator created", args.admin_email)


def verify(py: Path):
    step("8/8 Verification (datasets, engine, one real comparison)")
    man = ROOT / "dataset" / "dataset_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        bad = [f["path"] for f in m["files"] if not (ROOT / f["path"]).exists() or
               hashlib.sha256((ROOT / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
        record(OK if not bad else FAIL, f"bundled datasets: {len(m['files'])} files SHA-256 verified" if not bad
               else f"{len(bad)} dataset files missing or modified", ", ".join(bad[:5]))
        env = os.environ.get("LTI_DATASET_DIR")
        full = Path(env) if env and Path(env).is_dir() else ROOT / "dataset" / "liver_ct_lits_verified_v8"
        if full.is_dir():
            sums = (ROOT / "dataset" / "manifests" / "lits_verified_v8_SHA256SUMS.txt").read_text().splitlines()
            bad = []
            for line in sums:
                digest, rel = line.split(None, 1)
                f = full / rel.strip().lstrip("*")
                if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
                    bad.append(rel.strip())
            record(OK if not bad else FAIL, f"full dataset: {len(sums) - len(bad)} of {len(sums)} files verified",
                   (", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")) if bad else str(full))
        else:
            record(WARN, "full LiTS-derived dataset not present",
                   "optional; see MISSING_ARTIFACTS.md (the bundled 8-slice sample is used meanwhile)")
    code = ("import sys,json; sys.path[:0]=['.','backend'];"
            "from src.v10.engine.registry import ModelEngine;from src.v10.engine.compare import ComparisonRunner;"
            "e=ModelEngine('.').load();s=e.summary();r=ComparisonRunner(e);"
            "b='dataset/sample_data/lits_sample/images/test/';rd=lambda n:open(b+n,'rb').read();"
            "p=r.prepare(rd('ct_lits_4278.png'),'x.png',rd('ct_lits_4277.png'),rd('ct_lits_4279.png'));"
            "o=r.run(p,r.plan(p));"
            "print(json.dumps({'servable':s['models_servable'],'total':s['models_total'],"
            "'families':{f:v['n_completed'] for f,v in o['families'].items()},'overlap':o['family_overlap'].get('all_four_overlap_ms')}))")
    r = run([py, "-c", code], check=False, capture=True)
    if r.returncode == 0:
        res = json.loads(r.stdout.strip().splitlines()[-1])
        record(OK, f"model engine: {res['servable']} of {res['total']} models servable")
        record(OK if all(v > 0 for v in res["families"].values()) else WARN,
               "real four-family comparison on ct_lits_4278", f"completed per family {res['families']}, "
               f"all four families overlapped {res['overlap']} ms")
    else:
        record(FAIL, "engine self-check failed", (r.stderr or "").strip().splitlines()[-1] if r.stderr else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--admin-email", help="create an administrator (password: prompt or LTI_ADMIN_PASSWORD)")
    ap.add_argument("--v9", metavar="PATH", help="restore the large V9 model files from a V9 project folder")
    ap.add_argument("--no-venv", action="store_true", help="install into / use the current Python environment")
    ap.add_argument("--skip-pip", action="store_true", help="do not install Python packages")
    ap.add_argument("--skip-frontend", action="store_true", help="do not rebuild the web app (use frontend/dist)")
    ap.add_argument("--offline", action="store_true", help="do not download weights; only verify them")
    ap.add_argument("--check", action="store_true", help="verify an existing installation without changing it")
    args = ap.parse_args()
    os.chdir(ROOT)
    try:
        environment()
        py = python_env(args)
        weights(py, args)
        model_artifacts(py, args)
        env_file(args)
        frontend(args)
        database(py, args)
        verify(py)
    except (RuntimeError, OSError) as e:          # a failed command: report it plainly instead of a traceback
        record(FAIL, "step failed", str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__)
        tail = [ln for ln in str(e).strip().splitlines()[1:] if ln.strip()][-6:]
        if tail:
            print("    " + "\n    ".join(tail))
        if "pip" in str(e):
            print("  hint: pip could not install the packages (no network / proxy / unsupported Python?) — "
                  "see TROUBLESHOOTING.md → Installation")
    except SystemExit as e:
        if e.code not in (0, None):
            print("Installation incomplete — see the FAIL lines above and TROUBLESHOOTING.md")
        raise
    fails = [r for r in REPORT if r[0] == FAIL]
    print("\nSummary: " + ", ".join(f"{sum(1 for r in REPORT if r[0] == s)} {n}" for s, n in ((OK, 'ok'), (WARN, 'warnings'), (FAIL, 'failed'))))
    if fails:
        print("Installation incomplete — see the FAIL lines above and TROUBLESHOOTING.md")
        return 1
    print("Ready. Start the platform with:  python start.py   then open http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
