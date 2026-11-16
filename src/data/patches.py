"""Patch datasets built from the expert LiTS masks (64x64 cells of 8x8 pixels on the 512x512 slice).

Task A - tumour vs tumour-free liver tissue (the task every classical / quantum / hybrid model is compared on):
  tumour patch   96x96 px centred on a tumour cell, tumour fraction >= TUMOUR_MIN and liver(+tumour) >= LIVER_MIN_T
  healthy patch  96x96 px centred on a tumour-free liver cell, NO tumour cell inside the patch or a 1-cell (8 px)
                 margin around it, liver fraction >= LIVER_MIN_H
  patches with 0 < tumour fraction < TUMOUR_MIN are ambiguous and never used.
Task G - liver-tissue gate (supporting model used at inference to find the liver):
  liver patches  = Task A patches;  other patches = body patches with no liver (from no-tumour slices, which contain
  no liver, and from liver-free regions of tumour slices).
2.5-D context: the same window in the previous and next slice when that slice is a real neighbour (same tumour run);
otherwise the current slice is repeated and `has_neighbours` is False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.ct import PATCH

CELL = 8
P_CELLS = PATCH // CELL          # 16
TUMOUR_MIN = 0.05
LIVER_MIN_T = 0.40
LIVER_MIN_H = 0.50
HEALTHY_MARGIN = 1
PER_TUMOUR = 3
PER_HEALTHY = 4
MIN_SEP_CELLS = 5


def _frac(m, cy, cx, margin=0):
    y0, x0 = cy - P_CELLS // 2 - margin, cx - P_CELLS // 2 - margin
    y1, x1 = y0 + P_CELLS + 2 * margin, x0 + P_CELLS + 2 * margin
    sub = m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
    area = (P_CELLS + 2 * margin) ** 2
    return (sub == 2).sum() / area, (sub >= 1).sum() / area


def _pick(cands, rng, k, sep):
    rng.shuffle(cands)
    out = []
    for c in cands:
        if all(max(abs(c[0] - o[0]), abs(c[1] - o[1])) >= sep for o in out):
            out.append(c)
        if len(out) == k:
            break
    return out


def sample_task_a(masks: np.ndarray, upstream_index: np.ndarray, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    half = P_CELLS // 2
    for m, ui in zip(masks, upstream_index):
        tum, hea = [], []
        for cy in range(half, 64 - half + 1):
            for cx in range(half, 64 - half + 1):
                v = m[cy, cx] if cy < 64 and cx < 64 else 0
                if v == 2:
                    ft, fl = _frac(m, cy, cx)
                    if ft >= TUMOUR_MIN and fl >= LIVER_MIN_T:
                        tum.append((cy, cx, ft, fl))
                elif v == 1:
                    ft, fl = _frac(m, cy, cx)
                    ft_ext, _ = _frac(m, cy, cx, margin=HEALTHY_MARGIN)
                    if ft_ext == 0 and fl >= LIVER_MIN_H:
                        hea.append((cy, cx, ft, fl))
        for lab, cands, k in ((1, tum, PER_TUMOUR), (0, hea, PER_HEALTHY)):
            for cy, cx, ft, fl in _pick(cands, rng, k, MIN_SEP_CELLS):
                rows.append({"upstream_index": int(ui), "y": int((cy - half) * CELL), "x": int((cx - half) * CELL),
                             "label": lab, "tumour_frac": round(float(ft), 4), "liver_frac": round(float(fl), 4)})
    return pd.DataFrame(rows)


def body_cells(img: np.ndarray, thr: int = 10) -> np.ndarray:
    return (img > thr).reshape(64, CELL, 64, CELL).mean((1, 3)) >= 0.5


def sample_gate_negatives(images: dict, masks_by_index: dict, slice_labels: pd.DataFrame, seed: int = 43,
                          per_negative_slice: int = 1, per_tumour_slice: int = 2) -> pd.DataFrame:
    """Body patches without liver. images: upstream_index -> 512x512 uint8."""
    rng = np.random.default_rng(seed)
    half = P_CELLS // 2
    rows = []
    for ui, y in zip(slice_labels.upstream_index, slice_labels.label_tumour):
        img = images[ui]
        body = body_cells(img)
        m = masks_by_index.get(ui)
        cands = []
        for cy in range(half, 64 - half + 1, 2):
            for cx in range(half, 64 - half + 1, 2):
                sub_body = body[cy - half:cy + half, cx - half:cx + half].mean()
                if sub_body < 0.6:
                    continue
                if m is not None and (m[cy - half - 1:cy + half + 1, cx - half - 1:cx + half + 1] >= 1).any():
                    continue
                cands.append((cy, cx, 0.0, 0.0))
        k = per_tumour_slice if y == 1 else per_negative_slice
        for cy, cx, _, _ in _pick(cands, rng, k, MIN_SEP_CELLS):
            rows.append({"upstream_index": int(ui), "y": int((cy - half) * CELL), "x": int((cx - half) * CELL),
                         "label": 0, "tumour_frac": 0.0, "liver_frac": 0.0})
    return pd.DataFrame(rows)


def neighbour_indices(ui: int, present: set, run_of: dict) -> tuple[int, int, bool]:
    """Previous / next upstream index inside the same tumour run (real neighbours), else the slice itself."""
    r = run_of.get(ui)
    prev_ = ui - 1 if (ui - 1) in present and run_of.get(ui - 1) == r and r is not None else ui
    next_ = ui + 1 if (ui + 1) in present and run_of.get(ui + 1) == r and r is not None else ui
    return prev_, next_, (prev_ != ui and next_ != ui)
