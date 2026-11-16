#!/usr/bin/env python3
"""Registry consistency evaluation: load every servable patch model, score the real patch dataset per unit and report
AUC per unit, labelling each unit as in-sample or out-of-sample FOR THAT MODEL.

    python scripts/evaluate_models.py --dataset DIR [--models id1,id2] [--max-patches 600]

V9 models: units G01, G07, G08 were V9's test groups (out-of-sample); others were training / validation (in-sample).
V10 final models: trained on all units (every unit in-sample) — their honest estimate is the nested evaluation.
This script is a software / integrity check, not a new performance estimate.
"""
import argparse
import json

from _common import ROOT, dataset_arg, require_dataset

V9_TEST = {"G01", "G07", "G08"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    dataset_arg(ap)
    ap.add_argument("--models", default=None)
    ap.add_argument("--max-patches", type=int, default=600)
    a = ap.parse_args()
    import numpy as np
    from src.v10.data.lits_dataset import load_patch_dataset
    from src.v10.engine.registry import ModelEngine
    from src.v10.engine.v10_models import V10PatchModel
    from src.v10.engine.v9_adapter import V9Model
    from src.v10.evaluation.metrics import roc_auc
    from src.v10.utils import clean
    ds = load_patch_dataset(require_dataset(a.dataset))
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(ds["y"]), min(a.max_patches, len(ds["y"])), replace=False))
    eng = ModelEngine(ROOT).load()
    wanted = set(a.models.split(",")) if a.models else None
    out = {}
    for d in eng.list():
        if not d.get("servable") or (wanted and d["id"] not in wanted):
            continue
        m = eng.get(d["id"])
        contract = d.get("input_contract") or {}
        if contract.get("context_slices_each_side", 0) > 1:
            out[d["id"]] = {"skipped": "needs two adjacent slices on each side; the patch dataset stores one on each side"}
            print(d["id"], "skipped:", out[d["id"]]["skipped"])
            continue
        if isinstance(m, V9Model):
            stacks = ds["stacks25d"][idx]   # in chunks: the ImageNet backbones need ~1 GB per 100 patches at 224 px
            p = np.concatenate([m.predict_proba(stacks[i:i + 48]) for i in range(0, len(stacks), 48)])
        elif isinstance(m, V10PatchModel) and m.input_kind == "features":
            from src.v10.features.extract import cached
            X = cached(m.card["feature_set"], ds["patches2d"], ROOT / "results" / "v10" / "features")
            p = m.predict_proba(X[idx])
        else:
            continue
        rows = {}
        for g in sorted(set(ds["groups"][idx])):
            s = ds["groups"][idx] == g
            yy = ds["y"][idx][s]
            if len(np.unique(yy)) == 2:
                rows[g] = {"auc": roc_auc(yy, p[s]), "n": int(s.sum()),
                           "sample": ("out-of-sample (V9 test group)" if g in V9_TEST else "in-sample")
                           if d["source"] == "v9" else "in-sample (final V10 model)"}
        out[d["id"]] = {"finite": bool(np.all(np.isfinite(p))), "in_unit_interval": bool(np.all((p >= 0) & (p <= 1))),
                        "groups": rows}
        print(d["id"], {g: round(r["auc"], 3) for g, r in rows.items()})
    path = ROOT / "results" / "v10" / "registry_evaluation.json"
    path.write_text(json.dumps(clean(out), indent=1))
    print(f"-> {path}")
