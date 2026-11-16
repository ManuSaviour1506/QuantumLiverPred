#!/usr/bin/env python3
"""Build and verify the complete V11 ZIP.

    python scripts/v11_package.py --out ../Liver_Tumour_Intelligence_Platform_V11_COMPLETE.zip [--split-mb 45]

What it does:

1. Writes COMPLETE_FILE_MANIFEST.json (path, bytes, SHA-256, category of every packaged file) and SHA256SUMS.txt.
2. Zips the project under one top folder, Liver_Tumour_Intelligence_Platform_V11/, with normalised permissions.
   Scripts are executable.
3. Verifies the archive: every member's CRC is checked, and members, sizes and SHA-256 are compared with the manifest.
4. Writes <zip>.sha256.
5. With --split-mb, also writes numbered parts plus reassembly scripts that verify the SHA-256 (sh and PowerShell).

Included: everything, including the trained weights and the bundled datasets.

Excluded: secrets and local runtime data (.env, backend/.env, backend/data), virtual environments, node_modules,
caches (__pycache__, .pytest_cache), source maps and e2e screenshots. A file that is required but absent from the
source tree cannot be packaged. Those files are listed in MISSING_ARTIFACTS.md, never silently dropped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "Liver_Tumour_Intelligence_Platform_V11"
EXCLUDE_DIRS = {".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                ".git", ".idea", ".vscode"}
EXCLUDE_GLOBS = [".env", "backend/.env", "backend/data/*", "*.pyc", "*.pyo", ".DS_Store", "*.zip", "*.zip.[0-9]*",
                 "frontend/dist/assets/*.map", "frontend/e2e/output/*", "results/v10/features/*",
                 "results/v10/nested_quick/*", "_to_delete/*"]
EXECUTABLE = {"setup.sh", "run.sh", "install.py", "start.py"}
STORED_EXT = {".pth", ".npz", ".png", ".jpg", ".jpeg", ".pdf", ".joblib", ".gz", ".zip"}
MANIFEST, SUMS = "COMPLETE_FILE_MANIFEST.json", "SHA256SUMS.txt"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def category(rel: str) -> str:
    top = rel.split("/", 1)[0]
    if rel.startswith("artifacts/weights/"):
        return "model_weights (ImageNet backbones)"
    if rel.startswith("artifacts/"):
        return "model_artifacts"
    if top in ("dataset", "data"):
        return "dataset"
    if top == "models":
        return "model_index"
    if top in ("docker",) or rel in ("Dockerfile", "docker-compose.yml", "render.yaml", ".dockerignore"):
        return "deployment"
    if top in ("frontend", "backend", "src", "scripts", "tests", "configs", "docs", "reports", "results"):
        return {"src": "model_engine_source", "configs": "configuration", "results": "recorded_results"}.get(top, top)
    return "project_root"


def collect() -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not (Path(dirpath) / d).is_symlink())
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(ROOT).as_posix()
            if p.is_symlink() or rel in (MANIFEST, SUMS):
                continue
            if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(fn, g) for g in EXCLUDE_GLOBS):
                if rel not in (".env.example", "backend/.env.example"):
                    continue
            out.append(p)
    return out


def write_manifest(files: list[Path]) -> dict:
    entries = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p), "category": category(rel)})
    cats: dict[str, dict] = {}
    for e in entries:
        c = cats.setdefault(e["category"], {"files": 0, "bytes": 0})
        c["files"] += 1
        c["bytes"] += e["bytes"]
    missing = json.loads((ROOT / "artifacts" / "LARGE_FILES_MANIFEST.json").read_text())["files"]
    man = {"project": "Liver Tumour Intelligence Platform", "version": "11.0.0",
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "top_folder": TOP, "n_files": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
           "categories": dict(sorted(cats.items())),
           "not_included": {"large_v9_files": [f for f in missing if not (ROOT / f["path"]).exists()],
                            "full_dataset": "dataset/liver_ct_lits_verified_v8 (1,757 files; SHA-256 list in "
                                            "dataset/manifests/lits_verified_v8_SHA256SUMS.txt)",
                            "derived_masks": "data/masks/*.npz", "mri": "none exist",
                            "details": "MISSING_ARTIFACTS.md"},
           "excluded_by_rule": ["secrets and local runtime data (.env, backend/.env, backend/data)",
                                "virtual environments, node_modules, __pycache__ / test caches, source maps"],
           "files": entries}
    (ROOT / MANIFEST).write_text(json.dumps(man, indent=1) + "\n")
    lines = [f"{e['sha256']}  {e['path']}" for e in entries]
    lines.append(f"{sha256(ROOT / MANIFEST)}  {MANIFEST}")
    (ROOT / SUMS).write_text("\n".join(sorted(lines, key=lambda l: l.split('  ', 1)[1])) + "\n")
    return man


def build_zip(out: Path, files: list[Path]) -> None:
    allf = files + [ROOT / MANIFEST, ROOT / SUMS]
    tmp = out.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in allf:
            rel = p.relative_to(ROOT).as_posix()
            st = p.stat()
            ts = dt.datetime.fromtimestamp(max(st.st_mtime, 315532800)).timetuple()[:6]
            zi = zipfile.ZipInfo(f"{TOP}/{rel}", date_time=ts)
            mode = 0o755 if (p.name in EXECUTABLE or p.suffix == ".sh") else 0o644
            zi.external_attr = (0o100000 | mode) << 16
            zi.create_system = 3
            zi.compress_type = zipfile.ZIP_STORED if p.suffix.lower() in STORED_EXT else zipfile.ZIP_DEFLATED
            with open(p, "rb") as src, z.open(zi, "w", force_zip64=st.st_size > 2 ** 31) as dst:
                for b in iter(lambda: src.read(1 << 20), b""):
                    dst.write(b)
    tmp.replace(out)


def verify_zip(out: Path, man: dict) -> dict:
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        assert bad is None, f"CRC error in {bad}"
        names = {i.filename: i for i in z.infolist()}
        expected = {f"{TOP}/{e['path']}": e for e in man["files"]}
        assert set(expected) | {f"{TOP}/{MANIFEST}", f"{TOP}/{SUMS}"} == set(names), "member list differs from manifest"
        mism = []
        for name, e in expected.items():
            info = names[name]
            if info.file_size != e["bytes"]:
                mism.append(name)
                continue
            h = hashlib.sha256()
            with z.open(info) as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            if h.hexdigest() != e["sha256"]:
                mism.append(name)
        assert not mism, f"content differs from manifest: {mism[:5]}"
        tops = {n.split("/", 1)[0] for n in names}
        assert tops == {TOP}, tops
        assert not any(n.startswith("/") or ".." in n.split("/") for n in names), "unsafe member path"
    return {"members": len(names), "crc": "ok", "sha256_vs_manifest": "ok", "single_top_folder": TOP}


def split(out: Path, mb: int) -> list[Path]:
    size, parts, i = mb * 1024 * 1024, [], 1
    with open(out, "rb") as f:
        while True:
            b = f.read(size)
            if not b:
                break
            p = out.with_name(f"{out.name}.{i:03d}")
            p.write_bytes(b)
            parts.append(p)
            i += 1
    digest = sha256(out)
    (out.parent / "reassemble.sh").write_text(
        "#!/bin/sh\n# Rebuild the V11 ZIP from its numbered parts and verify its SHA-256 (macOS / Linux).\nset -e\n"
        f'cd "$(dirname "$0")"\ncat {out.name}.[0-9][0-9][0-9] > {out.name}\n'
        f'echo "{digest}  {out.name}" | (shasum -a 256 -c - 2>/dev/null || sha256sum -c -)\n'
        f'echo "OK: {out.name} rebuilt and verified. Unzip it and run: python3 install.py"\n')
    os.chmod(out.parent / "reassemble.sh", 0o755)
    names = ",".join(f"'{p.name}'" for p in parts)
    (out.parent / "reassemble.ps1").write_text(
        "# Rebuild the V11 ZIP from its numbered parts and verify its SHA-256 (Windows PowerShell).\n"
        "Set-Location -Path $PSScriptRoot\n"
        f"$out = Join-Path $PSScriptRoot '{out.name}'\n"
        "$fs = [System.IO.File]::Create($out)\n"
        f"foreach ($p in @({names})) {{ $b = [System.IO.File]::ReadAllBytes((Join-Path $PSScriptRoot $p)); "
        "$fs.Write($b, 0, $b.Length) }\n$fs.Close()\n"
        "$h = (Get-FileHash -Algorithm SHA256 $out).Hash.ToLower()\n"
        f"if ($h -ne '{digest}') {{ Write-Error \"SHA-256 mismatch: $h\"; exit 1 }}\n"
        f"Write-Output 'OK: {out.name} rebuilt and verified. Unzip it and run: py install.py'\n")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT.parent / f"{TOP}_COMPLETE.zip"))
    ap.add_argument("--split-mb", type=int, default=0, help="also write numbered parts of this size + reassembly scripts")
    a = ap.parse_args()
    for bad in (ROOT / ".env", ROOT / "backend" / ".env"):
        if bad.exists():
            print(f"note: {bad.relative_to(ROOT)} exists and is excluded (local secret)")
    files = collect()
    man = write_manifest(files)
    out = Path(a.out).resolve()
    build_zip(out, files)
    res = verify_zip(out, man)
    digest = sha256(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(json.dumps({"zip": str(out), "bytes": out.stat().st_size, "sha256": digest, "files": man["n_files"] + 2,
                      "uncompressed_bytes": man["total_bytes"], "categories": man["categories"], **res}, indent=1))
    if a.split_mb:
        parts = split(out, a.split_mb)
        print(f"split into {len(parts)} parts of ≤ {a.split_mb} MB + reassemble.sh / reassemble.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
