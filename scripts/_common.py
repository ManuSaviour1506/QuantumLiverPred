"""Shared helpers for the training / evaluation scripts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


# where the dataset lives when it is kept inside the project folder (see DATASET_GUIDE.md)
BUNDLED_DATASET = ROOT / "dataset" / "liver_ct_lits_verified_v8"


def default_dataset() -> str | None:
    """--dataset default: LTI_DATASET_DIR if set, else the in-project copy if present, else None."""
    env = os.environ.get("LTI_DATASET_DIR")
    if env:
        return env
    return str(BUNDLED_DATASET) if (BUNDLED_DATASET / "patches" / "patches_v8.npz").exists() else None


def dataset_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", default=default_dataset(),
                    help="verified LiTS dataset folder (default: LTI_DATASET_DIR, else dataset/liver_ct_lits_verified_v8)")


def require_dataset(path: str | None) -> Path:
    if not path:
        sys.exit("error: no dataset: pass --dataset, set LTI_DATASET_DIR, or put the dataset in "
                 "dataset/liver_ct_lits_verified_v8 (see DATASET_GUIDE.md)")
    p = Path(path).expanduser()
    if not (p / "patches" / "patches_v8.npz").exists():
        sys.exit(f"error: {p} does not look like the verified LiTS dataset folder (patches/patches_v8.npz missing)")
    return p


def load_nested(features: str) -> dict | None:
    p = ROOT / "results" / "v10" / "nested" / features / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def train_specs(keys: list[str], features: str, dataset: Path, seed: int = 0) -> list[Path]:
    from src.v10.data.lits_dataset import load_patch_dataset
    from src.v10.evaluation import specs as S
    from src.v10.features.extract import cached
    from src.v10.training.final import train_final
    ds = load_patch_dataset(dataset)
    X = cached(features, ds["patches2d"], ROOT / "results" / "v10" / "features")
    nested = load_nested(features)
    if nested is None:
        print(f"warning: no nested evaluation for {features}; using the first grid entry (recorded in the card)")
    out = []
    by_key = {s.key: s for s in S.all_specs()}
    for k in keys:
        if k not in by_key:
            sys.exit(f"unknown model key {k!r}; choose from {sorted(by_key)}")
        out.append(train_final(ROOT, by_key[k], X, ds["y"], ds["units"], ds, features, nested, seed))
        print(f"saved {out[-1].relative_to(ROOT)}")
    return out
