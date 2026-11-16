#!/usr/bin/env python3
"""Write dataset/dataset_manifest.json: every dataset / annotation file bundled with this installation (path, bytes,
SHA-256, role), the full dataset's expected location, verification list and acquisition route, and an explicit record
of what does NOT exist (MRI). Nothing is fabricated: missing data are reported as missing.

    python scripts/v11_build_dataset_manifest.py
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "dataset" / "liver_ct_lits_verified_v8"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def role(rel: str) -> str:
    if rel.startswith("dataset/sample_data/lits_sample/images/"):
        return "ct_slice (real, LiTS-derived, CC BY-NC-SA 4.0)"
    if rel.startswith("dataset/sample_data/synthetic/images/"):
        return "synthetic_phantom (generated, not a medical image; software tests / OOD checks only)"
    if rel.endswith("slices.csv") or rel.startswith("data/labels/"):
        return "annotation (slice labels derived from the expert segmentation masks)"
    if rel.startswith("data/manifests/") or rel.startswith("dataset/manifests/") or rel.endswith("manifest.csv"):
        return "manifest"
    return "documentation"


def main() -> int:
    files = []
    for base in ("dataset/sample_data", "dataset/manifests", "data/labels", "data/manifests"):
        for p in sorted((ROOT / base).rglob("*")):
            if p.is_file() and p.name != ".gitkeep":
                rel = p.relative_to(ROOT).as_posix()
                files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha(p), "role": role(rel)})
    sample_rows = list(csv.DictReader(open(ROOT / "dataset/sample_data/lits_sample/labels/slices.csv")))
    full_present = FULL.is_dir()
    n_sums = sum(1 for _ in open(ROOT / "dataset/manifests/lits_verified_v8_SHA256SUMS.txt"))
    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "about": "Datasets and annotations of the Liver Tumour Intelligence Platform V11. 'files' lists every bundled "
                 "file with its SHA-256 (install.py verifies them). Research use only; not for clinical use.",
        "structure_mapping": {
            "datasets/CT": ["dataset/sample_data/lits_sample (bundled, 8 real CT slices)",
                            "dataset/liver_ct_lits_verified_v8 (full CT dataset, " +
                            ("PRESENT" if full_present else "NOT INCLUDED in this package — see full_dataset") + ")"],
            "datasets/MRI": [],
            "datasets/annotations": ["dataset/sample_data/lits_sample/labels/slices.csv",
                                     "data/labels/verified_slice_labels.csv (all 1,745 slices of the full dataset)",
                                     "data/labels/excluded_unverified.csv", "data/manifests/*.csv",
                                     "masks/ inside the full dataset (expert segmentation masks; not included)"],
            "datasets/sample_data": ["dataset/sample_data"],
            "datasets/dataset_manifest.json": "dataset/dataset_manifest.json (this file)"},
        "bundled": {
            "lits_sample": {"path": "dataset/sample_data/lits_sample", "modality": "CT", "n_slices": len(sample_rows),
                            "n_tumour": sum(int(r["label_tumour"]) for r in sample_rows),
                            "n_liver_free": sum(1 for r in sample_rows if int(r["liver_px_512"]) == 0),
                            "groups": sorted({r["group_id"] for r in sample_rows}),
                            "licence": "CC BY-NC-SA 4.0 (inherited from LiTS via the Hugging Face source)",
                            "ground_truth": "label_tumour = any tumour pixel in the expert mask; liver_px_512 / "
                                            "tumour_px_512 = mask pixel counts at 512x512"},
            "synthetic": {"path": "dataset/sample_data/synthetic", "modality": "none (generated phantoms)",
                          "n_images": len(list((ROOT / "dataset/sample_data/synthetic/images").glob("*.png"))),
                          "use": "software tests and OOD-rejection checks only; never used to train or evaluate a "
                                 "reported model"}},
        "full_dataset": {
            "id": "liver_ct_lits_verified_v8", "modality": "CT", "present": full_present,
            "expected_path": "dataset/liver_ct_lits_verified_v8 (or set LTI_DATASET_DIR)",
            "contents": "1,745 axial CT slices (512x512 PNG, 476 with tumour) in 10 reconstructed scan groups, 2,449 "
                        "96x96 training patches, expert masks, labels, verification records",
            "n_files": n_sums,
            "verification": "dataset/manifests/lits_verified_v8_SHA256SUMS.txt (SHA-256 of every file); "
                            "cd dataset/liver_ct_lits_verified_v8 && sha256sum -c ../manifests/"
                            "lits_verified_v8_SHA256SUMS.txt   (macOS: shasum -a 256 -c ...)",
            "slice_manifest": "dataset/manifests/lits_verified_v8_manifest.csv",
            "source": "pixel-matched to the Hugging Face dataset seongun/liver-tumor-classification, derived from the "
                      "LiTS benchmark; labels re-derived from the expert masks",
            "citation": "P. Bilic et al., The Liver Tumor Segmentation Benchmark (LiTS), Medical Image Analysis 84, "
                        "102680 (2023), doi:10.1016/j.media.2022.102680",
            "licence": "CC BY-NC-SA 4.0",
            "how_to_restore": "copy the folder from the V9/V10 project (dataset/liver_ct_lits_verified_v8) into "
                              "dataset/, verify it with the SHA-256 list, then: python install.py --check "
                              "(see DATASET_GUIDE.md and MISSING_ARTIFACTS.md)",
            "why_not_included": None if full_present else
            "the full dataset folder was not present in the source workspace this package was built from (the V10 "
            "folder held the lean release); it was not fabricated or replaced"},
        "mri": {"present": False,
                "note": "No MRI data and no MRI-trained model exist in any version of this project. An upload marked "
                        "MRI is flagged as out of scope (every model is CT-trained) and is expected to be withheld by "
                        "the out-of-distribution check; nothing was fabricated."},
        "patient_level": "The source has no patient or volume identifiers; units are reconstructed scan groups "
                         "(never reported as patients).",
        "files": files,
    }
    (ROOT / "dataset" / "dataset_manifest.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(files)} files; full dataset present: {full_present}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
