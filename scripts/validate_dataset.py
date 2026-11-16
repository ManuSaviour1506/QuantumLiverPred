#!/usr/bin/env python3
"""Validate a dataset folder (files, readability, labels, identifiers, duplicates, leakage, masks, distributions).

    python scripts/validate_dataset.py --dataset DIR [--layout lits_verified|manifest] [--manifest FILE]
                                       [--no-images] [--out results/v10/dataset_validation/NAME.json]

Needs only NumPy and Pillow (runs on machines without SciPy). Exit code 0 = passed, 1 = problems found.
"""
import argparse
import json
import sys
from pathlib import Path

from _common import ROOT, default_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=default_dataset(),
                    help="dataset folder (default: LTI_DATASET_DIR, else dataset/liver_ct_lits_verified_v8)")
    ap.add_argument("--layout", default="lits_verified", choices=["lits_verified", "manifest"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--no-images", action="store_true", help="skip decoding images (existence checks only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.dataset:
        ap.error("no dataset: pass --dataset, set LTI_DATASET_DIR, or put it in dataset/liver_ct_lits_verified_v8")
    from src.v10.data import ingest
    root = Path(a.dataset).expanduser()
    if a.layout == "lits_verified":
        rows = ingest.lits_verified_manifest(root)
    else:
        man = Path(a.manifest) if a.manifest else ingest.find_manifest(root)
        if man is None:
            sys.exit("no manifest.csv found")
        rows, _, errors = ingest.read_manifest(man)
        for e in errors:
            print("manifest:", e)
    report, per_file = ingest.validate(rows, root, check_images=not a.no_images,
                                       progress=lambda i, n: print(f"  {i}/{n}", flush=True))
    report["dataset_version_sha256"] = ingest.content_hash(rows, per_file)
    out = Path(a.out) if a.out else ROOT / "results" / "v10" / "dataset_validation" / f"{root.name.replace(' ', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "problems": [dict(row_index=f["row_index"],
                                                                   image_path=rows[f["row_index"]]["image_path"],
                                                                   status=f["file_status"], issues=f["issues"])
                                                              for f in per_file if f["issues"]][:2000]}, indent=1))
    print(json.dumps({k: report[k] for k in ("passed", "rows", "file_status", "label_distribution", "warnings")},
                     indent=1))
    print(f"report: {out}")
    sys.exit(0 if report["passed"] else 1)
