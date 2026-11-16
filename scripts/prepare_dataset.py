#!/usr/bin/env python3
"""Prepare the processed data used by training: check the patch arrays against their manifest, record hashes and
cache the patch features (radiomics, frozen ImageNet EfficientNet-B0).

    python scripts/prepare_dataset.py --dataset DIR [--features radiomics,imagenet_gap]

Writes results/v10/features/<set>_<hash>.npz and dataset/processed/dataset_card.json (no images are copied).
"""
import argparse
import json

from _common import ROOT, dataset_arg, require_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    dataset_arg(ap)
    ap.add_argument("--features", default="radiomics,imagenet_gap")
    a = ap.parse_args()
    from src.v10.data.lits_dataset import load_patch_dataset
    from src.v10.features.extract import cached
    import numpy as np
    ds = load_patch_dataset(require_dataset(a.dataset))
    card = {"dataset_version": ds["dataset_version"], "files_sha256": ds["files"], "license": ds["license"],
            "citation": ds["citation"], "n_patches": int(len(ds["y"])), "positives": int(ds["y"].sum()),
            "units": {u: int((ds["units"] == u).sum()) for u in sorted(set(ds["units"]))}, "features": {}}
    for f in a.features.split(","):
        X = cached(f, ds["patches2d"], ROOT / "results" / "v10" / "features")
        card["features"][f] = {"shape": list(X.shape), "finite": bool(np.isfinite(X).all())}
    out = ROOT / "dataset" / "processed" / "dataset_card.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))
