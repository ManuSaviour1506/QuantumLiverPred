#!/usr/bin/env python3
"""Noise study of the deployed V10 VQC (density-matrix simulation; hypothetical noise parameters).

    python scripts/run_noise_experiments.py --dataset DIR [--model v10_vqc_radiomics] [--levels 0,0.001,0.005,0.01,0.03]
                                            [--readout 0.0] [--shots 0] [--samples 400]
Writes results/v10/noise/noise_study.json.
"""
import argparse
import json

from _common import ROOT, dataset_arg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    dataset_arg(ap)
    ap.add_argument("--model", default="v10_vqc_radiomics")
    ap.add_argument("--levels", default="0,0.001,0.005,0.01,0.03")
    ap.add_argument("--readout", type=float, default=0.0)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--samples", type=int, default=400)
    a = ap.parse_args()
    from pathlib import Path
    from src.v10.quantum import noise_study
    from src.v10.utils import clean
    X = y = None
    if a.dataset:
        from src.v10.data.lits_dataset import load_patch_dataset
        from src.v10.features.extract import cached
        ds = load_patch_dataset(Path(a.dataset))
        X, y = cached("radiomics", ds["patches2d"], ROOT / "results" / "v10" / "features"), ds["y"]
    res = {"exact": noise_study.run(ROOT / "artifacts" / "v10" / a.model, X, y,
                                    [float(v) for v in a.levels.split(",")], a.readout, 0, a.samples)}
    res["with_readout_and_shots"] = noise_study.run(ROOT / "artifacts" / "v10" / a.model, X, y, [0.0, 0.01], 0.02,
                                                    a.shots or 1024, min(a.samples, 200))
    out = ROOT / "results" / "v10" / "noise" / "noise_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(res), indent=1))
    for r in res["exact"]["summary"]:
        print(r)
    print(f"-> {out}")
