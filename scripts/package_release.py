#!/usr/bin/env python3
"""Build the release ZIP of the V10 project (V10 lean / full layouts).

V11: the complete V11 package (weights included, file manifest, verification, optional split) is built by
scripts/v11_package.py; this script is kept for the V10 lean / full layouts.

    python scripts/package_release.py [--out ../liver_tumour_intelligence_v10.zip] [--max-file-mb 4]
    python scripts/package_release.py --full [--out ../liver_tumour_intelligence_v10_full.zip]

Lean (default, for sharing): no dataset, no files above --max-file-mb (listed with SHA-256 for restoring).
--full (for your own backup / moving to another computer): also the in-project dataset
dataset/liver_ct_lits_verified_v8 (CC BY-NC-SA 4.0 — non-commercial use, attribution, share-alike), the derived masks
and every large model file. Both modes leave out local secrets and runtime data (.env, backend/data with the local
database and uploaded images), virtual environments, node_modules and caches.

Excluded: virtual environments, node_modules, caches, build output that is rebuilt (frontend/dist is kept), `.env`
files, local databases and storage, feature caches, the 64x64 expert-mask maps (dataset licence: not redistributed),
and every file larger than --max-file-mb. Each large file left out is listed with its size and SHA-256 in
artifacts/LARGE_FILES_MANIFEST.json together with how to restore it (scripts/v10_import_v9_artifacts.py copies and
verifies them from the V9 project). SHA256SUMS.txt lists every file that is in the ZIP.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = ROOT.name

EXCLUDE_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".git", ".idea",
                ".vscode"}
EXCLUDE_GLOBS = [".env", ".env.*", "*.pyc", "*.db", "*.sqlite3", ".DS_Store", "*.zip", "*.log.tmp",
                 "backend/data/*", "results/v10/features/*", "results/v10/nested_quick/*", "results/v10/runs/*/tmp/*",
                 "frontend/dist/assets/*.map", "frontend/e2e/output/report.pdf", "_to_delete/*"]
LEAN_ONLY_EXCLUDES = ["data/masks/*", "dataset/liver_ct_lits_verified_v8/*"]   # licence: not in the shareable ZIP
KEEP = {".env.example", "backend/.env.example"}
# left out like large files (listed with SHA-256 in the manifest): training checkpoints, which serving never reads
OMIT_TO_MANIFEST = ["artifacts/*/*/checkpoint_best.npz"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(rel: str, full: bool = False) -> bool:
    if rel in KEEP:
        return False
    name = rel.rsplit("/", 1)[-1]
    globs = EXCLUDE_GLOBS + ([] if full else LEAN_ONLY_EXCLUDES)
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in globs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-file-mb", type=float, default=4.0)
    ap.add_argument("--full", action="store_true", help="include the dataset and all large files (personal copy)")
    a = ap.parse_args()
    a.out = a.out or str(ROOT.parent / (f"{NAME}_full.zip" if a.full else f"{NAME}.zip"))
    limit = float("inf") if a.full else int(a.max_file_mb * 1024 * 1024)
    files, large = [], []
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS
                             and not (Path(dirpath) / d).is_symlink())
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(ROOT).as_posix()
            if p.is_symlink() or excluded(rel, a.full):
                continue
            size = p.stat().st_size
            if not a.full and (size > limit or any(fnmatch.fnmatch(rel, g) for g in OMIT_TO_MANIFEST)):
                large.append({"path": rel, "bytes": size, "sha256": sha256(p)})
                continue
            files.append(p)
    manifest = {
        "about": "Files left out of the release ZIP because of their size (or, for checkpoint_best.npz, because serving never reads training checkpoints). They are byte-identical copies of files in "
                 "the V9 project; restore and verify them with: python scripts/v10_import_v9_artifacts.py --v9 "
                 "/path/to/liver_tumour_project_v9 . Models whose files are missing are reported as unavailable by "
                 "the API (never replaced by placeholders). The ImageNet weights (artifacts/weights) are timm "
                 "releases (Apache-2.0) and are needed by the OOD detector, the liver gate and the backbone models.",
        "max_file_bytes_in_zip": limit,
        "files": large,
    }
    mpath = ROOT / "artifacts" / "LARGE_FILES_MANIFEST.json"
    if not a.full:
        mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    if mpath.exists() and mpath not in files:
        files.append(mpath)
    sums = "".join(f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files if p.name != "SHA256SUMS.txt")
    (ROOT / "SHA256SUMS.txt").write_text(sums)
    files = [p for p in files if p.name != "SHA256SUMS.txt"] + [ROOT / "SHA256SUMS.txt"]
    out = Path(a.out)
    stored = {".png", ".jpg", ".jpeg", ".npz", ".gz", ".zip", ".pdf"}   # --full: store these (little to gain, much faster)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in files:
            z.write(p, f"{NAME}/{p.relative_to(ROOT).as_posix()}",
                    compress_type=zipfile.ZIP_STORED if a.full and p.suffix.lower() in stored else zipfile.ZIP_DEFLATED)
    total = sum(p.stat().st_size for p in files)
    print(f"{out}: {len(files)} files, {total / 1e6:.1f} MB uncompressed, {out.stat().st_size / 1e6:.1f} MB zipped"
          + ("; full copy (dataset and all model files included)" if a.full else
             f"; {len(large)} large files listed in artifacts/LARGE_FILES_MANIFEST.json "
             f"({sum(f['bytes'] for f in large) / 1e6:.1f} MB)"))
    # secret scan of what went in
    bad = [p.relative_to(ROOT).as_posix() for p in files if p.name == ".env" or p.suffix in (".pem", ".key")]
    if bad:
        print("refusing: secret-like files included:", bad, file=sys.stderr)
        out.unlink()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
