#!/usr/bin/env python3
"""Nested, unit-grouped evaluation of classical, quantum and hybrid models on the real patch dataset.

    python scripts/run_nested_evaluation.py --dataset "/path/to/liver_ct_lits_verified_v8" --features radiomics
    python scripts/run_nested_evaluation.py --dataset ... --features imagenet_gap
    python scripts/run_nested_evaluation.py --dataset ... --features radiomics --models logreg,vqc --quick  (smoke run)

Writes results/v10/nested/<features>/: results.json, comparisons.json, oof_predictions.csv, log.txt, run_manifest.json.
The analysis plan (configs/v10/analysis_plan.json) is hashed into the manifest; the script refuses to run if it is
missing, so the plan cannot be written after seeing outer results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v10.data.lits_dataset import load_patch_dataset  # noqa: E402
from src.v10.evaluation import specs as S  # noqa: E402
from src.v10.evaluation.nested import compare, run_nested  # noqa: E402
from src.v10.features.extract import FEATURE_SETS, cached  # noqa: E402
from src.v10.utils import clean, environment  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sys.path.insert(0, str(ROOT / "scripts"))
    from _common import default_dataset
    ap.add_argument("--dataset", default=default_dataset(),
                    help="dataset folder (default: LTI_DATASET_DIR, else dataset/liver_ct_lits_verified_v8)")
    ap.add_argument("--features", default="radiomics", choices=["radiomics", "imagenet_gap"])
    ap.add_argument("--models", default="all", help="comma-separated spec keys, or 'all'")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--inner-k", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smoke test: 300-patch subsample, 1 seed (not a result)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not a.dataset:
        ap.error("no dataset: pass --dataset, set LTI_DATASET_DIR, or put it in dataset/liver_ct_lits_verified_v8")
    plan_p = ROOT / "configs" / "v10" / "analysis_plan.json"
    if not plan_p.is_file():
        print("analysis plan missing: refusing to run", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else ROOT / "results" / "v10" / ("nested_quick" if a.quick else "nested") / a.features
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    ds = load_patch_dataset(Path(a.dataset))
    X = cached(a.features, ds["patches2d"], ROOT / "results" / "v10" / "features", log=log)
    y, units = ds["y"], ds["units"]
    idx = np.arange(len(y))
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.quick:
        rng = np.random.default_rng(0)
        idx = np.sort(np.concatenate([rng.choice(np.where(units == u)[0], min(40, (units == u).sum()), replace=False)
                                      for u in np.unique(units)]))
        seeds = seeds[:1]
    specs = S.all_specs()
    if a.models != "all":
        keep = set(a.models.split(","))
        specs = [s for s in specs if s.key in keep]
        for s in specs:
            if s.fixed_from and s.fixed_from not in keep:
                s.fixed_from = None
    if a.quick:
        import src.v10.evaluation.specs as SS
        for s in specs:
            s.grid = s.grid[:2]
    log(f"dataset {ds['dataset_version']}: {len(idx)} patches, {len(np.unique(units[idx]))} units, features "
        f"{a.features} {X.shape[1]}-d; {len(specs)} model specs; seeds {seeds}")
    res = run_nested(X[idx], y[idx], units[idx], specs, seeds, a.inner_k, log)
    oof = res.pop("_oof")
    conf = compare(res, S.CONFIRMATORY, "confirmatory" if a.features == "radiomics" else "secondary")
    rows = pd.DataFrame({"patch_index": idx, "unit": units[idx], "group": ds["groups"][idx], "label": y[idx]})
    for k, v in oof.items():
        rows[f"{k}__score"], rows[f"{k}__prob"] = v["score"], v["prob"]
    rows.to_csv(out / "oof_predictions.csv", index=False)
    res.update({"features": a.features, "feature_info": FEATURE_SETS[a.features], "dataset_version":
                ds["dataset_version"], "n_patches": int(len(idx)), "quick": a.quick,
                "evaluation_unit": "reconstructed scan-group unit (8 units; G06+G07 and G08+G09 merged)",
                "not_patient_level": True})
    (out / "results.json").write_text(json.dumps(clean(res), indent=1))
    (out / "comparisons.json").write_text(json.dumps(clean(conf), indent=1))
    manifest = {"script": "scripts/run_nested_evaluation.py", "args": vars(a), "seconds": round(time.time() - t0, 1),
                "analysis_plan_sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest(),
                "dataset_files_sha256": ds["files"], "dataset_version": ds["dataset_version"],
                "license": ds["license"], "environment": environment(), "seeds": seeds}
    (out / "run_manifest.json").write_text(json.dumps(clean(manifest), indent=1))
    log(f"done in {manifest['seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
