"""Loader for the verified LiTS-derived dataset folder (`liver_ct_lits_verified_v8`).

Reads the patch arrays used by V8/V9 (`patches/patches_v8.npz`) and their manifest (`manifests/patches_task_a.csv`),
checks that both agree row by row (labels, counts), maps reconstructed scan groups to evaluation units
(G06+G07 and G08+G09 merged: shared upstream padding signatures, Stage 3) and records SHA-256 hashes so every result
can be traced to the exact files. Nothing is relabelled or invented: labels come from the expert masks via the files.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_OF_GROUP = {"G01": "G01", "G02": "G02", "G03": "G03", "G04": "G04", "G05": "G05", "G06": "G06+G07",
                 "G07": "G06+G07", "G08": "G08+G09", "G09": "G08+G09", "G10": "G10"}
LICENSE = "CC BY-NC-SA 4.0 (derived from seongun/liver-tumor-classification, itself derived from LiTS)"
CITATION = ('P. Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)", Medical Image Analysis 84, 102680 '
            '(2023), doi:10.1016/j.media.2022.102680')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_dir() -> Path | None:
    env = os.environ.get("LTI_DATASET_DIR")
    return Path(env).expanduser() if env else None


def load_patch_dataset(root: Path) -> dict:
    root = Path(root)
    npz_p, csv_p = root / "patches" / "patches_v8.npz", root / "manifests" / "patches_task_a.csv"
    for p in (npz_p, csv_p):
        if not p.is_file():
            raise FileNotFoundError(f"dataset file missing: {p.relative_to(root)}")
    z = np.load(npz_p)
    man = pd.read_csv(csv_p)
    X2, X3, y = z["task_a_2d"], z["task_a_25d"], z["task_a_label"].astype(int)
    if not (len(X2) == len(X3) == len(y) == len(man)):
        raise ValueError("patch arrays and manifest have different row counts")
    if not np.array_equal(y, man["label"].to_numpy().astype(int)):
        raise ValueError("labels in patches_v8.npz disagree with manifests/patches_task_a.csv")
    groups = man["group_id"].astype(str).to_numpy()
    units = np.array([UNIT_OF_GROUP[g] for g in groups])
    version = hashlib.sha256((sha256_file(npz_p) + sha256_file(csv_p)).encode()).hexdigest()[:16]
    return {"patches2d": X2, "stacks25d": X3, "y": y, "groups": groups, "units": units, "manifest": man,
            "files": {"patches_v8.npz": sha256_file(npz_p), "patches_task_a.csv": sha256_file(csv_p)},
            "dataset_version": f"lits_verified_v8@{version}", "license": LICENSE, "citation": CITATION}
