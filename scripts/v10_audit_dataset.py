"""Stage 3: dataset integrity audit of the V9 data files (read-only). Outputs go to results/v10/dataset/.

  python scripts/v10_audit_dataset.py [--root .]

Checks: files present, labels and verification evidence, masks, Task A patches, liver-gate patches, split and groups,
manifest, near-duplicates, group structure and boundaries, intensity by group, evidence of related groups, a proposed
set of evaluation units (related groups merged), and a search for volume identifiers such as vol_008.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v10.data import validation as V  # noqa: E402
from src.v10.evaluation.grouped_cv import merge_related  # noqa: E402
from src.v10.utils import run_manifest, write_json  # noqa: E402


def search_volume_ids(root: Path) -> dict:
    pat = re.compile(r"vol[_-]?0*\d{1,4}", re.I)
    hits, scanned = {}, 0
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in (".py", ".md", ".csv", ".json", ".txt", ".log", ".out"):
            continue
        if "results/v10" in p.as_posix() or "reports" in p.parts or p.name.startswith("test_v10") or "v10_" in p.name or "/v10/" in p.as_posix():
            continue
        scanned += 1
        try:
            found = set(m.group(0) for m in pat.finditer(p.read_text(errors="ignore")))
        except Exception:
            continue
        if found:
            hits[str(p.relative_to(root))] = sorted(found)[:10]
    return {"pattern": pat.pattern, "text_files_scanned": scanned, "files_with_matches": hits,
            "vol_008_found": any("008" in h for v in hits.values() for h in v)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = root / "results" / "v10" / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = V.Data(root)
    checks = {"files_present": V.check_files_present(d), "labels": V.check_labels(d), "masks": V.check_masks(d),
              "task_a_patches": V.check_patches(d), "gate_patches": V.check_gate_patches(d), "split": V.check_split(d),
              "manifest": V.check_manifest(d), "near_duplicates": V.check_near_duplicates(d)}
    struct = V.group_structure(d)
    rel = V.relation_evidence(d, struct)
    intensity = V.intensity_by_group(d)
    disc = V.series_discontinuities(d)
    vol = search_volume_ids(root)
    # proposed evaluation units: merge groups with direct evidence of a shared scan (identical padding run crossing them)
    groups = sorted(d.labels.group_id.unique())
    merges = [tuple(r["groups"]) for r in rel if r["identical_padding_run_crosses_boundary"]]
    unit_of = merge_related(groups, merges)
    pa = d.patches
    units = []
    for u in sorted(set(unit_of.values())):
        gs = [g for g in groups if unit_of[g] == u]
        sub = pa[pa.group_id.isin(gs)]
        units.append({"unit": u, "groups": gs, "patches": int(len(sub)), "tumour_patches": int(sub.label.sum()),
                      "tumour_free_patches": int((sub.label == 0).sum()), "share_of_all_patches": round(len(sub) / len(pa), 4),
                      "v9_splits": sorted(d.labels[d.labels.group_id.isin(gs)].split.unique().tolist())})
    proposal = {"rule": "merge two groups when a run of identical upstream padding signatures crosses their boundary "
                        "(data/labels/verification/scan_signature_audit.json); merging is the leakage-safe direction",
                "merges": [list(m) for m in merges], "units": units,
                "not_assessable": [r["groups"] for r in rel if not r["padding_signature_audited"]],
                "note": ("padding signatures were audited only at boundaries between different V9 splits; for the other "
                         "boundaries no evidence exists either way, and the per-slice padding counts are not in the repository")}
    write_json(out / "integrity.json", {"checks": checks, "group_structure": struct, "relation_evidence": rel,
                                         "intensity_by_group": intensity, "series_discontinuities": disc, "volume_id_search": vol, "proposed_units": proposal})
    pd.DataFrame(struct["groups"]).to_csv(out / "groups.csv", index=False)
    pd.DataFrame(struct["boundaries"]).to_csv(out / "group_boundaries.csv", index=False)
    pd.DataFrame(struct["runs"]).to_csv(out / "tumour_runs.csv", index=False)
    pd.DataFrame(struct["within_run_liver_iou"]["below_0_8"]).to_csv(out / "within_run_low_iou.csv", index=False)
    pd.DataFrame(checks["gate_patches"]["affected_patches"]).to_csv(out / "gate_defect_patches.csv", index=False)
    pd.DataFrame(units).to_csv(out / "proposed_units.csv", index=False)
    pd.DataFrame(disc["largest_intensity_steps"]).to_csv(out / "series_discontinuities.csv", index=False)
    # G01 split at the discontinuity: how many Task A patches lie on each side
    if disc["extreme_on_both"]:
        cut = disc["extreme_on_both"][0]
        gp = pa[pa.group_id == cut["group"]]
        write_json(out / "g01_discontinuity.json", {"step": cut, "patches_before": int((gp.upstream_index <= cut["from"]).sum()),
                                                    "patches_after": int((gp.upstream_index >= cut["to"]).sum()),
                                                    "decision": "keep the group as one evaluation unit (merging is leakage-safe; the two parts may be one patient)"})
    inputs = ["data/labels/verified_slice_labels.csv", "data/manifests/split_v8.csv", "data/manifests/scan_groups.csv",
              "data/manifests/patches_task_a.csv", "data/manifests/patches_gate.csv", "data/manifests/v9_slices.csv",
              "data/manifests/manifest_v8.csv", "data/manifests/duplicates.csv", "data/masks/lits_tumour_slices_masks64.npz",
              "data/labels/verification/upstream_metadata_labels_3986.txt", "data/labels/verification/mask_chunk0.txt",
              "data/labels/verification/mask_chunk1.txt", "data/labels/verification/scan_signature_audit.json"]
    write_json(out / "run_manifest.json", run_manifest(root, inputs, {"none": "deterministic checks"},
                                                       {"seconds": round(time.time() - t0, 1), "command": "python scripts/v10_audit_dataset.py"}))
    for k, v in checks.items():
        print(f"{k:16s} pass={v.get('pass')}")
    print("units:", [(u["unit"], u["patches"]) for u in units])


if __name__ == "__main__":
    main()
