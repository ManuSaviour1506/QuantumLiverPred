"""Read-only integrity checks of the V9 dataset files. Nothing here writes into data/.

Every check returns a dict with "pass" (True / False / None when it could not run) and the evidence it used.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

CELL, P_CELLS = 8, 12                      # 96 px patches on 8 px cells (V9 src/data/patches.py, PATCH = 96)
TUMOUR_MIN, LIVER_MIN_T, LIVER_MIN_H, MARGIN = 0.05, 0.40, 0.50, 1
UNMATCHED = [322, 324, 325, 326, 328, 329, 330] + list(range(333, 351)) + list(range(668, 698))


class Data:
    def __init__(self, root):
        self.root = Path(root)
        r = self.root / "data"
        self.labels = pd.read_csv(r / "labels" / "verified_slice_labels.csv")
        self.split = pd.read_csv(r / "manifests" / "split_v8.csv")
        self.groups = pd.read_csv(r / "manifests" / "scan_groups.csv")
        self.patches = pd.read_csv(r / "manifests" / "patches_task_a.csv")
        self.gate = pd.read_csv(r / "manifests" / "patches_gate.csv")
        self.slices9 = pd.read_csv(r / "manifests" / "v9_slices.csv")
        self.manifest = pd.read_csv(r / "manifests" / "manifest_v8.csv")
        self.excluded = pd.read_csv(r / "labels" / "excluded_unverified.csv")
        dpath = r / "manifests" / "duplicates.csv"
        self.duplicates = pd.read_csv(dpath) if dpath.exists() else None
        z = np.load(r / "masks" / "lits_tumour_slices_masks64.npz")
        self.masks, self.mask_index = z["masks"], z["upstream_index"]
        self.ver = r / "labels" / "verification"
        self.report = json.loads((r / "manifests" / "dataset_report.json").read_text())


def _chi2(table: np.ndarray) -> tuple[float, int]:
    t = table.astype(float)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    return float(((t - e) ** 2 / np.where(e > 0, e, 1)).sum()), int((t.shape[0] - 1) * (t.shape[1] - 1))


def check_files_present(d: Data) -> dict:
    root = d.root
    scans = root / "data" / "raw" / "scans"
    out = {"scans_folder": scans.exists(), "processed_patches_v8": (root / "data" / "processed" / "patches_v8.npz").exists(),
           "processed_v9_5slice": (root / "data" / "processed" / "v9_task_a_5s.npz").exists(),
           "processed_v9_seg": (root / "data" / "processed" / "v9_seg.npz").exists(),
           "feature_cache": (root / "artifacts" / "cache").exists()}
    referenced = len(d.manifest)
    present = int(sum((scans / p).exists() for p in d.manifest.path)) if scans.exists() else 0
    out.update({"images_referenced_by_manifest": referenced, "images_present": present,
                "pass": None if not scans.exists() else present == referenced,
                "note": "image files are not part of the repository; image-level checks cannot run" if not scans.exists() else ""})
    return out


def check_labels(d: Data) -> dict:
    s = (d.ver / "upstream_metadata_labels_3986.txt").read_text().strip()
    sha = hashlib.sha256(s.encode()).hexdigest()
    vt = d.labels
    match = all(int(s[i]) == y for i, y in zip(vt.upstream_index, vt.label_tumour))
    idx_expected = [i for i in range(100, 1900) if i not in set(UNMATCHED)]
    order_ok = vt.sort_values("file_number").upstream_index.tolist() == idx_expected
    tumour_idx = set(vt[vt.label_tumour == 1].upstream_index)
    res = {"rows": int(len(vt)), "tumour": int(vt.label_tumour.sum()), "no_tumour": int((vt.label_tumour == 0).sum()),
           "label_values": sorted(vt.label_tumour.unique().tolist()), "missing_values": int(vt.isna().sum().sum()),
           "upstream_string_sha256": sha,
           "upstream_string_sha256_matches_record": sha == "a7df4ea68cb36e6fdfa1ec618c9958956c06da10b8207cbf8a2774943c1624b0",
           "labels_equal_upstream_string": bool(match), "file_number_to_index_mapping_monotonic": bool(order_ok),
           "mask_index_equals_tumour_slices": set(d.mask_index.tolist()) == tumour_idx,
           "split_file_identical_to_label_file": bool(d.labels.equals(d.split)),
           "folder_not_used_as_label": "source_folder_not_a_label" in vt.columns}
    # mask transfer chunks decode to the stored masks
    b64 = "".join((d.ver / f"mask_chunk{i}.txt").read_text().strip() for i in (0, 1))
    pk = np.frombuffer(zlib.decompress(base64.b64decode(b64), -15), np.uint8)
    dd = np.stack([(pk >> sft) & 3 for sft in (0, 2, 4, 6)], 1).reshape(-1, 4096)
    c3 = np.zeros_like(dd)
    c3[0] = dd[0]
    for k in range(1, len(dd)):
        c3[k] = (dd[k] + c3[k - 1]) % 3
    res["mask_chunks_decode_to_stored_masks"] = bool(np.array_equal(c3.reshape(-1, 64, 64), d.masks))
    # modulo / index rules
    mod = {}
    for name, v in (("upstream_index", vt.upstream_index.to_numpy()), ("file_number", vt.file_number.to_numpy()),
                    ("row_position", np.arange(len(vt)))):
        mod[name] = {str(k): _chi2(pd.crosstab(v % k, vt.label_tumour).to_numpy())[0] for k in range(2, 11)}
    res["label_vs_index_modulo_chi2"] = mod
    res["max_residue_rate_deviation"] = float(max(abs(vt.groupby(vt.upstream_index % k).label_tumour.mean() - vt.label_tumour.mean()).max()
                                                  for k in range(2, 11)))
    y = vt.sort_values("upstream_index").label_tumour.to_numpy()
    res["label_changes_along_index"] = int((np.diff(y) != 0).sum())
    res["pass"] = all([res["upstream_string_sha256_matches_record"], res["labels_equal_upstream_string"], order_ok,
                       res["mask_index_equals_tumour_slices"], res["mask_chunks_decode_to_stored_masks"],
                       res["missing_values"] == 0, res["label_values"] == [0, 1]])
    return res


def check_masks(d: Data) -> dict:
    M = d.masks
    tc = (M == 2).reshape(len(M), -1).sum(1)
    lc = (M >= 1).reshape(len(M), -1).sum(1)
    vt = d.labels.set_index("upstream_index")
    tp = vt.loc[d.mask_index, "tumour_px_512"].to_numpy()
    lp = vt.loc[d.mask_index, "liver_px_512"].to_numpy()
    # tumour cells touching liver or other tumour cells (4-neighbourhood) - isolated tumour cells would be suspicious
    iso = 0
    for m in M:
        t = m == 2
        nb = np.zeros_like(t)
        nb[1:] |= m[:-1] >= 1
        nb[:-1] |= m[1:] >= 1
        nb[:, 1:] |= m[:, :-1] >= 1
        nb[:, :-1] |= m[:, 1:] >= 1
        iso += int((t & ~nb).sum())
    res = {"masks": int(len(M)), "shape": list(M.shape[1:]), "values": sorted(np.unique(M).tolist()),
           "tumour_cells_min_median_max": [int(tc.min()), int(np.median(tc)), int(tc.max())],
           "liver_cells_min_median_max": [int(lc.min()), int(np.median(lc)), int(lc.max())],
           "slices_without_tumour_cells": int((tc == 0).sum()),
           "corr_tumour_pixels_vs_cells": float(np.corrcoef(tp, tc * 64)[0, 1]),
           "corr_liver_pixels_vs_cells": float(np.corrcoef(lp, lc * 64)[0, 1]),
           "isolated_tumour_cells": iso,
           "no_tumour_slices_with_liver_annotation": int(((d.labels.label_tumour == 0) & (d.labels.liver_present == 1)).sum())}
    res["pass"] = res["values"] == [0, 1, 2] and res["slices_without_tumour_cells"] == 0 and res["corr_tumour_pixels_vs_cells"] > 0.99
    return res


def _frac(m, y, x, margin=0):
    h = P_CELLS // 2
    cy, cx = y // CELL + h, x // CELL + h
    y0, x0 = cy - h - margin, cx - h - margin
    sub = m[max(y0, 0):max(y0 + P_CELLS + 2 * margin, 0), max(x0, 0):max(x0 + P_CELLS + 2 * margin, 0)]
    a = (P_CELLS + 2 * margin) ** 2
    return (sub == 2).sum() / a, (sub >= 1).sum() / a, m[cy, cx]


def check_patches(d: Data) -> dict:
    pa = d.patches
    mb = {int(u): m for u, m in zip(d.mask_index, d.masks)}
    bad_frac = bad_rule = bad_centre = 0
    for r in pa.itertuples():
        m = mb[int(r.upstream_index)]
        ft, fl, c = _frac(m, r.y, r.x)
        bad_frac += abs(ft - r.tumour_frac) > 1e-4 or abs(fl - r.liver_frac) > 1e-4
        if r.label == 1:
            bad_rule += not (ft >= TUMOUR_MIN and fl >= LIVER_MIN_T)
            bad_centre += c != 2
        else:
            bad_rule += not (_frac(m, r.y, r.x, MARGIN)[0] == 0 and fl >= LIVER_MIN_H)
            bad_centre += c != 1
    sl = d.labels.set_index("upstream_index")
    group_ok = bool((pa.group_id.to_numpy() == sl.loc[pa.upstream_index, "group_id"].to_numpy()).all()
                    and (pa.split.to_numpy() == sl.loc[pa.upstream_index, "split"].to_numpy()).all()
                    and (pa.scan_id.to_numpy() == sl.loc[pa.upstream_index, "scan_id"].to_numpy()).all())
    res = {"patches": int(len(pa)), "tumour": int(pa.label.sum()), "tumour_free_liver": int((pa.label == 0).sum()),
           "fractions_not_reproduced": int(bad_frac), "label_rule_violations": int(bad_rule),
           "centre_cell_mismatches": int(bad_centre), "duplicate_coordinates": int(pa.duplicated(["upstream_index", "y", "x"]).sum()),
           "patches_only_from_tumour_slices": set(pa.upstream_index) <= set(d.mask_index.tolist()),
           "group_split_scan_consistent_with_slice_table": group_ok, "missing_values": int(pa.isna().sum().sum()),
           "patches_per_slice_mean": float(len(pa) / pa.upstream_index.nunique())}
    res["pass"] = (bad_frac == 0 and bad_rule == 0 and bad_centre == 0 and res["duplicate_coordinates"] == 0
                   and res["patches_only_from_tumour_slices"] and group_ok and res["missing_values"] == 0)
    return res


def check_gate_patches(d: Data) -> dict:
    g = d.gate
    pos = g[g.label == 1]
    pa = d.patches
    key = set(zip(pa.upstream_index, pa.y, pa.x))
    pos_are_task_a = all((u, y, x) in key for u, y, x in zip(pos.upstream_index, pos.y, pos.x))
    mb = {int(u): m for u, m in zip(d.mask_index, d.masks)}
    neg = g[g.label == 0]
    bad = []
    for r in neg.itertuples():
        m = mb.get(int(r.upstream_index))
        if m is not None and _frac(m, r.y, r.x, MARGIN)[1] > 0:
            bad.append({"upstream_index": int(r.upstream_index), "y": int(r.y), "x": int(r.x), "group": r.group_id,
                        "split": r.split, "liver_fraction_in_patch": float(_frac(m, r.y, r.x, 0)[1])})
    split_ok = bool((g.split.to_numpy() == d.labels.set_index("upstream_index").loc[g.upstream_index, "split"].to_numpy()).all())
    at_edge = [b for b in bad if b["x"] == 0 or b["y"] == 0]
    res = {"patches": int(len(g)), "liver": int(len(pos)), "other": int(len(neg)), "liver_patches_are_task_a_patches": pos_are_task_a,
           "other_patches_touching_liver": len(bad), "of_which_at_image_edge_x0_or_y0": len(at_edge),
           "other_patches_by_split": pd.Series([b["split"] for b in bad]).value_counts().to_dict() if bad else {},
           "liver_fraction_range": [min(b["liver_fraction_in_patch"] for b in bad), max(b["liver_fraction_in_patch"] for b in bad)] if bad else None,
           "defect_cause": ("V9 src/data/patches.py::sample_gate_negatives tests m[cy-half-1:cy+half+1, cx-half-1:cx+half+1] for liver; "
                            "for cx = half (x = 0) or cy = half (y = 0) the start index is -1, the slice is empty and the liver "
                            "test passes vacuously") if at_edge else None,
           "affected_patches": bad, "split_consistent": split_ok, "task_a_affected": False}
    res["pass"] = pos_are_task_a and not bad and split_ok
    return res


def check_split(d: Data) -> dict:
    vt = d.labels
    per_group = vt.groupby("group_id").split.nunique()
    pa = d.patches
    gstats = vt.groupby("group_id").agg(n_slices=("label_tumour", "size"), n_tumour_slices=("label_tumour", "sum"),
                                         first_index=("upstream_index", "min"), last_index=("upstream_index", "max")).reset_index()
    stored = d.groups.sort_values("group_id").reset_index(drop=True)
    groups_reproduce = bool((gstats[["n_slices", "n_tumour_slices", "first_index", "last_index"]].to_numpy()
                             == stored[["n_slices", "n_tumour_slices", "first_index", "last_index"]].to_numpy()).all())
    s9 = d.slices9.set_index("upstream_index")
    s9_ok = bool((s9.loc[vt.upstream_index, "group_id"].to_numpy() == vt.group_id.to_numpy()).all()
                 and (s9.loc[vt.upstream_index, "split"].to_numpy() == vt.split.to_numpy()).all()
                 and (s9.loc[vt.upstream_index, "label_tumour"].to_numpy() == vt.label_tumour.to_numpy()).all())
    nb_cross = 0
    for col in ("prev_index", "next_index"):
        other = s9.loc[s9[col].to_numpy(), "group_id"].to_numpy()
        nb_cross += int((other != s9.group_id.to_numpy()).sum())
    res = {"groups": int(len(per_group)), "groups_in_more_than_one_split": int((per_group > 1).sum()),
           "patch_groups_in_more_than_one_split": int((pa.groupby("group_id").split.nunique() > 1).sum()),
           "scan_groups_csv_reproduced": groups_reproduce, "v9_slice_table_consistent": s9_ok,
           "v9_neighbours_crossing_groups": nb_cross}
    res["pass"] = res["groups_in_more_than_one_split"] == 0 and res["patch_groups_in_more_than_one_split"] == 0 and groups_reproduce and s9_ok and nb_cross == 0
    return res


def check_manifest(d: Data) -> dict:
    m = d.manifest
    lits = m[m.category == "ct_lits"]
    res = {"rows": int(len(m)), "categories": m.category.value_counts().to_dict(),
           "sizes": {f"{w}x{h}": int(c) for (w, h), c in m.groupby(["width", "height"]).size().items()},
           "ct_lits_all_512": bool(((lits.width == 512) & (lits.height == 512)).all()),
           "ct_lits_channels_identical": bool(lits.channels_identical.all()),
           "qc_status": m.qc_status.value_counts().to_dict(), "exact_duplicates": int(m.exact_duplicate.sum()),
           "ct_lits_unique_pixel_hashes": int(lits.pixel_sha12.nunique()), "missing_values_in_key_columns":
               int(m[["path", "file_name", "category", "pixel_sha12", "label_status"]].isna().sum().sum()),
           "verified_files_in_manifest": int(d.labels.file_name.isin(lits.file_name).sum()),
           "excluded_rows": int(len(d.excluded)), "excluded_categories": d.excluded.category.value_counts().to_dict()}
    res["pass"] = (res["ct_lits_all_512"] and res["exact_duplicates"] == 0 and res["ct_lits_unique_pixel_hashes"] == len(lits)
                   and res["verified_files_in_manifest"] == len(d.labels) and res["missing_values_in_key_columns"] == 0)
    return res


def check_near_duplicates(d: Data) -> dict:
    if d.duplicates is None:
        return {"pass": None, "note": "duplicates.csv missing"}
    du = d.duplicates
    vt = d.labels
    p2 = {c: dict(zip(vt.path, vt[c])) for c in ("upstream_index", "group_id", "split", "label_tumour")}
    both = du[du.a.isin(p2["group_id"]) & du.b.isin(p2["group_id"])].copy()
    for c in ("upstream_index", "group_id", "split", "label_tumour"):
        both[c + "_a"], both[c + "_b"] = both.a.map(p2[c]), both.b.map(p2[c])
    dist = (both.upstream_index_a - both.upstream_index_b).abs()
    res = {"pairs_total": int(len(du)), "pairs_between_verified_slices": int(len(both)),
           "verified_with_unverified": int((du.a.isin(p2["group_id"]) ^ du.b.isin(p2["group_id"])).sum()),
           "index_distance_max": int(dist.max()) if len(dist) else None,
           "pairs_crossing_groups": int((both.group_id_a != both.group_id_b).sum()),
           "pairs_crossing_splits": int((both.split_a != both.split_b).sum()),
           "pairs_with_different_labels": int((both.label_tumour_a != both.label_tumour_b).sum()),
           "threshold": "32x32 thumbnail correlation >= 0.999 (computed by V9 prepare_data.py; not recomputable without images)"}
    res["pass"] = res["pairs_crossing_groups"] == 0 and res["pairs_crossing_splits"] == 0
    return res


def tumour_runs(vt: pd.DataFrame) -> list[tuple[int, int]]:
    d = vt.sort_values("upstream_index")
    idx, lab = d.upstream_index.to_numpy(), d.label_tumour.to_numpy()
    runs, s = [], None
    for k in range(len(idx)):
        if lab[k] == 1 and s is None:
            s = k
        if s is not None and (k == len(idx) - 1 or lab[k + 1] == 0 or idx[k + 1] - idx[k] > 1):
            runs.append((int(idx[s]), int(idx[k])))
            s = None
    return runs


def group_structure(d: Data) -> dict:
    vt, pa = d.labels, d.patches
    liver = {int(u): (m >= 1) for u, m in zip(d.mask_index, d.masks)}

    def iou(a, b):
        u = (a | b).sum()
        return float((a & b).sum() / u) if u else float("nan")
    runs = tumour_runs(vt)
    g_of = dict(zip(vt.upstream_index, vt.group_id))
    sp_of = dict(zip(vt.upstream_index, vt.split))
    present = set(vt.upstream_index)
    run_rows = []
    for a, b in runs:
        run_rows.append({"start": a, "end": b, "length": b - a + 1, "group": g_of[a], "split": sp_of[a],
                         "liver_cells_start": int(liver[a].sum()), "liver_cells_end": int(liver[b].sum())})
    within = []
    for a, b in runs:
        for i in range(a, b):
            within.append({"index": i, "next": i + 1, "group": g_of[i], "liver_iou": iou(liver[i], liver[i + 1])})
    w = np.array([r["liver_iou"] for r in within])
    boundaries = []
    for (a0, a1), (b0, b1) in zip(runs[:-1], runs[1:]):
        boundaries.append({"prev_run_end": a1, "next_run_start": b0, "gap_slices": b0 - a1 - 1,
                           "gap_missing_indices": sum(1 for i in range(a1 + 1, b0) if i not in present),
                           "group_prev": g_of[a1], "group_next": g_of[b0], "split_prev": sp_of[a1], "split_next": sp_of[b0],
                           "same_group": g_of[a1] == g_of[b0], "liver_iou_across": iou(liver[a1], liver[b0])})
    gt = pd.crosstab(pa.group_id, pa.label)
    rows = []
    for g in sorted(vt.group_id.unique()):
        s = vt[vt.group_id == g]
        rows.append({"group": g, "split": s.split.iloc[0], "slices": int(len(s)), "tumour_slices": int(s.label_tumour.sum()),
                     "first_index": int(s.upstream_index.min()), "last_index": int(s.upstream_index.max()),
                     "tumour_runs": int(sum(1 for r in runs if g_of[r[0]] == g)),
                     "patches": int(gt.loc[g].sum()) if g in gt.index else 0,
                     "tumour_patches": int(gt.loc[g, 1]) if g in gt.index else 0,
                     "tumour_free_patches": int(gt.loc[g, 0]) if g in gt.index else 0})
    gdf = pd.DataFrame(rows)
    gdf["tumour_share_patches"] = (gdf.tumour_patches / gdf.patches).round(4)
    chi, dof = _chi2(gt.to_numpy())
    return {"groups": rows, "runs": run_rows, "boundaries": boundaries,
            "within_run_liver_iou": {"n": int(len(w)), "median": float(np.median(w)), "p5": float(np.percentile(w, 5)),
                                     "min": float(w.min()), "below_0_8": [r for r in within if r["liver_iou"] < 0.8]},
            "label_vs_group_chi2": {"chi2": chi, "dof": dof,
                                    "note": "patch counts are not independent; descriptive only"}}


def series_discontinuities(d: Data, top: int = 10) -> dict:
    """Steps between index-adjacent tumour slices (both inside one tumour run): change in whole-slice mean intensity
    (manifest) and liver-mask IoU (expert maps). A step that is extreme on both measures suggests two different series
    joined in one run. Steps across no-tumour slices are not used: their content differs (no liver), so intensity jumps
    there do not indicate a change of scan."""
    man = d.manifest.set_index("file_name")
    vt = d.labels.sort_values("upstream_index").reset_index(drop=True)
    mu = man.loc[vt.file_name, "mean"].to_numpy()
    idx, y, g = vt.upstream_index.to_numpy(), vt.label_tumour.to_numpy(), vt.group_id.to_numpy()
    liver = {int(u): m >= 1 for u, m in zip(d.mask_index, d.masks)}
    rows = []
    for k in range(len(idx) - 1):
        if y[k] == 1 and y[k + 1] == 1 and idx[k + 1] - idx[k] == 1:
            a, b = liver[int(idx[k])], liver[int(idx[k + 1])]
            rows.append({"from": int(idx[k]), "to": int(idx[k + 1]), "group": g[k], "delta_mean_intensity": float(mu[k + 1] - mu[k]),
                         "liver_iou": float((a & b).sum() / max((a | b).sum(), 1)), "liver_area_ratio": float(b.sum() / max(a.sum(), 1))})
    R = pd.DataFrame(rows)
    R["abs_delta"] = R.delta_mean_intensity.abs()
    by_int = R.sort_values("abs_delta", ascending=False).head(top)
    by_iou = R.sort_values("liver_iou").head(top)
    both = R[(R.abs_delta >= R.abs_delta.quantile(0.99)) & (R.liver_iou <= R.liver_iou.quantile(0.01))]
    return {"steps": int(len(R)), "abs_delta_median": float(R.abs_delta.median()), "abs_delta_p99": float(R.abs_delta.quantile(0.99)),
            "liver_iou_median": float(R.liver_iou.median()), "liver_iou_p1": float(R.liver_iou.quantile(0.01)),
            "largest_intensity_steps": by_int.drop(columns="abs_delta").to_dict("records"),
            "lowest_liver_iou_steps": by_iou.drop(columns="abs_delta").to_dict("records"),
            "extreme_on_both": both.drop(columns="abs_delta").to_dict("records")}


def intensity_by_group(d: Data) -> list[dict]:
    m = d.manifest.set_index("file_name")
    vt = d.labels.copy()
    vt["mean"] = m.loc[vt.file_name, "mean"].to_numpy()
    vt["std"] = m.loc[vt.file_name, "std"].to_numpy()
    out = []
    for (g, y), s in vt.groupby(["group_id", "label_tumour"]):
        out.append({"group": g, "label_tumour": int(y), "slices": int(len(s)), "mean_intensity_median": float(s["mean"].median()),
                    "mean_intensity_iqr": [float(s["mean"].quantile(0.25)), float(s["mean"].quantile(0.75))],
                    "std_intensity_median": float(s["std"].median())})
    return out


def relation_evidence(d: Data, structure: dict) -> list[dict]:
    """Evidence that two reconstructed groups might share a patient or volume. Recorded padding-signature runs come from
    data/labels/verification/scan_signature_audit.json (only boundaries between different splits were audited there)."""
    audit = json.loads((d.ver / "scan_signature_audit.json").read_text())
    pad = {}
    for r in audit.get("runs_of_identical_padding_crossing_split_boundaries", []):
        gs = [s.split(" ")[0] for s in r["groups"]]
        pad[tuple(sorted(gs))] = r
    out = []
    order = sorted(d.labels.group_id.unique())
    sp = dict(zip(d.labels.group_id, d.labels.split))
    by_pair = {}
    for b in structure["boundaries"]:
        if not b["same_group"]:
            by_pair[(b["group_prev"], b["group_next"])] = b
    for g1, g2 in zip(order[:-1], order[1:]):
        b = by_pair.get((g1, g2))
        key = tuple(sorted((g1, g2)))
        audited = sp[g1] != sp[g2]
        out.append({"groups": [g1, g2], "splits": [sp[g1], sp[g2]], "adjacent_in_index_order": True,
                    "gap_between_tumour_runs": b["gap_slices"] if b else None,
                    "liver_iou_across_boundary": b["liver_iou_across"] if b else None,
                    "padding_signature_audited": audited,
                    "identical_padding_run_crosses_boundary": key in pad,
                    "padding_run": pad.get(key)})
    return out
