#!/usr/bin/env python3
"""Import the trained V9 artifacts (and the V9 result files the engine reads) from a V9 project folder, verifying
every file by SHA-256. V9 is only read, never modified.

    python scripts/v10_import_v9_artifacts.py --v9 /path/to/liver_tumour_project_v9 [--skip-weights]

ImageNet weights (artifacts/weights, 150 MB; timm, Apache-2.0) are needed by the OOD detector, the liver gate and the
fine-tuned backbones. --skip-weights leaves them out (those models are then reported as FAILED, never faked).
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from _common import ROOT

RESULTS = ["results/v9/metrics_v9.csv", "results/v9/comparison_v9.json", "results/v9/cnn_selection.json",
           "results/v9/cnn_ablation.json", "results/v9/cnn_architectures.json", "results/v9/g01_analysis.json",
           "results/v9/RESULTS_V9.md", "results/v9/quantum/quantum_experiments_cnn.json",
           "results/v9/quantum/quantum_experiments_eff.json", "results/v9/quantum/inference_timing.json",
           "results/v9/segmentation/segmentation_report.json"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", required=True)
    ap.add_argument("--skip-weights", action="store_true")
    a = ap.parse_args()
    v9 = Path(a.v9).expanduser()
    if not (v9 / "artifacts" / "classical").is_dir():
        sys.exit(f"{v9} does not look like the V9 project (artifacts/classical missing)")
    files = [p for p in (v9 / "artifacts").rglob("*") if p.is_file()]
    if a.skip_weights:
        files = [p for p in files if "weights" not in p.relative_to(v9).parts]
    files += [v9 / r for r in RESULTS if (v9 / r).is_file()]
    manifest = {}
    for src in sorted(files):
        rel = src.relative_to(v9).as_posix()
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(src)
        if not dst.exists() or sha(dst) != digest:
            shutil.copy2(src, dst)
        if sha(dst) != digest:
            sys.exit(f"copy verification failed for {rel}")
        manifest[rel] = digest
    (ROOT / "artifacts" / "V9_IMPORT_MANIFEST.json").write_text(json.dumps(
        {"source": str(v9.name), "files": manifest}, indent=1))
    print(f"imported and verified {len(manifest)} files")
