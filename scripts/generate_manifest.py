#!/usr/bin/env python3
"""Generate a structured manifest (image_path, label, patient_id, scan_id, volume_id, slice_id, modality, mask_path,
split, data_source) for a dataset folder.

    python scripts/generate_manifest.py --dataset DIR --layout lits_verified [--out dataset/manifests/NAME.csv]

lits_verified: labels from labels/slices.csv (derived from the expert masks); patient and volume ids are left empty
because the source has none (they are never invented); scan_id = reconstructed scan group.
"""
import argparse
from pathlib import Path

from _common import ROOT, default_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=default_dataset(),
                    help="dataset folder (default: LTI_DATASET_DIR, else dataset/liver_ct_lits_verified_v8)")
    ap.add_argument("--layout", default="lits_verified", choices=["lits_verified"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.dataset:
        ap.error("no dataset: pass --dataset, set LTI_DATASET_DIR, or put it in dataset/liver_ct_lits_verified_v8")
    from src.v10.data import ingest
    root = Path(a.dataset).expanduser()
    rows = ingest.lits_verified_manifest(root)
    out = Path(a.out) if a.out else ROOT / "dataset" / "manifests" / "lits_verified_v8_manifest.csv"
    ingest.write_manifest(rows, out)
    print(f"{len(rows)} rows -> {out}  (content hash {ingest.content_hash(rows)[:16]})")
