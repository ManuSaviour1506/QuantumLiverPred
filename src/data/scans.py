"""Scan (patient) grouping for the verified LiTS-derived slices.

The upstream dataset provides no patient, volume or slice-position field. Grouping is reconstructed as follows and
is deliberately conservative (merging two scans into one group is safe; splitting one scan across train/test is not):

1. Tumour slices form contiguous index runs whose consecutive slices are near-identical neighbours (median
   correlation 0.98-0.998). The dataset contains no tumour-free liver slices and no-tumour slices lie outside the
   liver, so a scan's tumour slices cannot be interrupted by its own no-tumour slices: each run is one scan.
2. Neighbouring runs have different body outlines (body-mask IoU 0.26-0.53 at the run boundary), supporting that
   they are different patients. Runs separated by <= MERGE_GAP no-tumour slices are nevertheless merged into one group,
   so any residual ambiguity cannot leak a patient across splits.
3. No-tumour slices are attached to the nearest tumour run (the boundary is placed in the middle of each no-tumour run).

If exact per-scan signatures become available (data/labels/verification/scan_signatures.csv with columns
upstream_index,signature) they replace step 1-3: consecutive slices with equal signature belong to the same scan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import LABELS

MERGE_GAP = 10


def positive_runs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """(start_upstream_index, end_upstream_index) of contiguous tumour runs (no missing index inside)."""
    d = df.sort_values("upstream_index")
    idx, lab = d.upstream_index.to_numpy(), d.label_tumour.to_numpy()
    runs, s = [], None
    for k in range(len(idx)):
        if lab[k] == 1 and s is None:
            s = k
        if s is not None and (k == len(idx) - 1 or lab[k + 1] == 0 or idx[k + 1] - idx[k] > 1):
            runs.append((int(idx[s]), int(idx[k])))
            s = None
    return runs


def assign_groups(df: pd.DataFrame, merge_gap: int = MERGE_GAP) -> pd.DataFrame:
    sig_path = LABELS / "verification" / "scan_signatures.csv"
    d = df.sort_values("upstream_index").reset_index(drop=True).copy()
    if sig_path.exists():
        sig = pd.read_csv(sig_path).set_index("upstream_index")["signature"]
        s = d.upstream_index.map(sig)
        d["scan_id"] = (s != s.shift()).cumsum().map(lambda k: f"S{k:02d}")
        d["group_id"] = d["scan_id"]
        d["grouping_method"] = "upstream per-scan signature"
        return d
    runs = positive_runs(d)
    # merge runs separated by few negatives
    groups = [[runs[0]]]
    for r in runs[1:]:
        if r[0] - groups[-1][-1][1] - 1 <= merge_gap:
            groups[-1].append(r)
        else:
            groups.append([r])
    run_id = {}
    for gi, g in enumerate(groups):
        for ri, r in enumerate(g):
            run_id[r] = (gi, ri)
    centres = [((g[0][0] + g[-1][1]) / 2.0, g[0][0], g[-1][1]) for g in groups]
    gid, sid = [], []
    for i in d.upstream_index:
        hit = next((r for r in runs if r[0] <= i <= r[1]), None)
        if hit is not None:
            gi, ri = run_id[hit]
        else:  # negative slice: nearest group boundary (midpoint rule between neighbouring groups)
            dist = [0 if s <= i <= e else min(abs(i - s), abs(i - e)) for _, s, e in centres]
            gi, ri = int(np.argmin(dist)), -1
        gid.append(f"G{gi + 1:02d}")
        sid.append(f"G{gi + 1:02d}-R{ri + 1}" if ri >= 0 else f"G{gi + 1:02d}-neg")
    d["group_id"], d["scan_id"] = gid, sid
    d["grouping_method"] = f"tumour-run reconstruction (runs <= {merge_gap} slices apart merged)"
    return d
