"""One patient(scan)-independent train / validation / test split shared by every model.

The split is chosen once, deterministically, before any model is trained: an exhaustive search over assignments of
scan groups to splits, minimising the deviation from the target share of tumour slices (60 / 15 / 25 %) subject to
>= 3 scan groups in every split. It is saved to data/manifests/split_v8.csv and every
training / evaluation script reads that file. The test split is never used for model selection, early stopping,
calibration or threshold choice.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TARGET = {"train": 0.60, "val": 0.15, "test": 0.25}
SPLITS = ("train", "val", "test")


def choose_split(groups: pd.DataFrame, weight_col: str = "n_tumour_slices", min_groups: int = 3) -> dict[str, str]:
    g = groups.sort_values("group_id").reset_index(drop=True)
    w = g[weight_col].to_numpy(float)
    tot = w.sum()
    best, best_key = None, None
    for assign in itertools.product(range(3), repeat=len(g)):
        a = np.array(assign)
        counts = [(a == s).sum() for s in range(3)]
        if min(counts) < min_groups:
            continue
        share = [w[a == s].sum() / tot for s in range(3)]
        dev = sum(abs(share[s] - TARGET[SPLITS[s]]) for s in range(3))
        key = (round(dev, 6), -min(counts), assign)
        if best_key is None or key < best_key:
            best, best_key = a, key
    if best is None:
        raise ValueError("no valid split: too few scan groups")
    return {gid: SPLITS[s] for gid, s in zip(g.group_id, best)}


def assert_no_leakage(df: pd.DataFrame, group_col: str = "group_id", split_col: str = "split") -> None:
    per = df.groupby(group_col)[split_col].nunique()
    bad = per[per > 1]
    if len(bad):
        raise AssertionError(f"patient leakage: groups in more than one split: {list(bad.index)}")
