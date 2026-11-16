"""Validate the two-stage liver check (patch gate AND U-Net liver area) against the expert masks.

For every verified slice whose image is present, records whether the front end lets the slice through (liver
confirmed) and compares it with the mask label `liver_present`. Sweeps the U-Net minimum liver area so the
default (LTI_LIVER_MIN_UNET_PX, 4096 px) can be re-chosen on TRAIN/VALIDATION groups only and then checked once
on the TEST groups — never chosen on test data.

    python scripts/v10_validate_liver_gate.py                       # full dataset if present, else the 8 samples
    python scripts/v10_validate_liver_gate.py --limit 400 --split validation
Output: reports/v10/LIVER_GATE_VALIDATION.json and .md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SWEEP = [0, 1024, 2048, 3072, 4096, 5120, 6144, 8192]


def rows_and_root():
    full = ROOT / "dataset" / "liver_ct_lits_verified_v8"
    if full.is_dir():
        return list(csv.DictReader(open(ROOT / "data" / "labels" / "verified_slice_labels.csv"))), full, "path"
    base = ROOT / "dataset" / "sample_data" / "lits_sample"
    rows = list(csv.DictReader(open(base / "labels" / "slices.csv")))
    for r in rows:
        r["liver_present"] = "1" if int(r["liver_px_512"]) > 0 else "0"
    return rows, base, "image_path"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="", help="train | validation | test (default: all)")
    a = ap.parse_args()
    from src.inference.registry import Registry
    from src.v10.engine import patch_pipeline as PP
    reg = Registry().load()
    rows, base, col = rows_and_root()
    if a.split:
        rows = [r for r in rows if r.get("split") == a.split]
    if a.limit:
        rows = rows[:a.limit]
    recs = []
    for r in rows:
        p = base / r[col]
        if not p.is_file():
            continue
        fe = PP.run(reg, p.read_bytes(), p.name, validate_liver=False)
        gate_pass = bool(fe.liver_idx is not None and len(fe.liver_idx) and not fe.stop_reason)
        unet_px = None
        if gate_pass:
            unet_px = PP.liver_validation(reg, fe.image, min_px=0)["unet_liver_px"]
        recs.append({"file": p.name, "split": r.get("split"), "liver_present": int(r.get("liver_present") or 0),
                     "gate_pass": gate_pass, "unet_liver_px": unet_px})
    out = {"n": len(recs), "source": str(base.relative_to(ROOT)), "sweep": []}
    y = np.array([x["liver_present"] for x in recs])
    for t in SWEEP:
        accept = np.array([x["gate_pass"] and (x["unet_liver_px"] or 0) >= t for x in recs])
        tp, fp = int((accept & (y == 1)).sum()), int((accept & (y == 0)).sum())
        fn, tn = int((~accept & (y == 1)).sum()), int((~accept & (y == 0)).sum())
        out["sweep"].append({"min_unet_liver_px": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "liver_free_wrongly_analysed": fp, "liver_slices_withheld": fn})
    out["records"] = recs
    rep = ROOT / "reports" / "v10"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "LIVER_GATE_VALIDATION.json").write_text(json.dumps(out, indent=1))
    L = ["# Liver-region validation sweep", "", f"Slices: {out['n']} from `{out['source']}`"
         + (f", split = {a.split}" if a.split else ""), "",
         "Gate = V9 patch liver gate (unchanged). U-Net = V9 2.5-D U-Net liver area. A slice is analysed only if the gate "
         "keeps >= 1 patch AND the U-Net liver area >= the minimum.", "",
         "| min U-Net liver px | liver slices analysed | liver-free slices analysed (bad) | liver slices withheld | liver-free withheld |",
         "|---:|---:|---:|---:|---:|"]
    L += [f"| {s['min_unet_liver_px']} | {s['tp']} | {s['fp']} | {s['fn']} | {s['tn']} |" for s in out["sweep"]]
    (rep / "LIVER_GATE_VALIDATION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
