"""Segmentation metrics on 64 x 64 expert-resolution maps (0 background, 1 liver, 2 tumour).

Volume metrics stack the slices of one reconstructed scan group in upstream order. Lesions are 3-D connected
components (26-connectivity) of the tumour class; a reference lesion counts as detected when any predicted tumour cell
overlaps it, and a predicted lesion with no overlap is a false positive.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

STRUCT3 = np.ones((3, 3, 3), bool)


def overlap_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    s = 2 * tp + fp + fn
    return {"dice": 2 * tp / s if s else 1.0, "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
            "precision": tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0),
            "recall": tp / (tp + fn) if (tp + fn) else 1.0, "tp_cells": tp, "fp_cells": fp, "fn_cells": fn}


def lesion_metrics(pred_t: np.ndarray, ref_t: np.ndarray) -> dict:
    """pred_t, ref_t: (D, H, W) boolean tumour volumes of one scan."""
    rl, nr = ndi.label(ref_t, STRUCT3)
    pl, npred = ndi.label(pred_t, STRUCT3)
    detected = sum(1 for k in range(1, nr + 1) if pred_t[rl == k].any())
    fp = sum(1 for k in range(1, npred + 1) if not ref_t[pl == k].any())
    sizes = ndi.sum(np.ones_like(rl), rl, range(1, nr + 1)) if nr else np.array([])
    det_flags = [bool(pred_t[rl == k].any()) for k in range(1, nr + 1)]
    return {"reference_lesions": int(nr), "detected_lesions": int(detected), "predicted_lesions": int(npred),
            "false_positive_lesions": int(fp), "lesion_sensitivity": detected / nr if nr else None,
            "lesion_sizes_cells": [int(s) for s in sizes], "lesion_detected": det_flags}


def postprocess(pred: np.ndarray, min_tumour_cells: int = 0, tumour_in_liver: bool = True) -> np.ndarray:
    """pred (D, H, W) labels. Remove 3-D tumour components smaller than min_tumour_cells; optionally keep tumour only
    where it belongs to the largest 3-D liver-or-tumour component (the liver)."""
    out = pred.copy()
    if tumour_in_liver and (out >= 1).any():
        lab, n = ndi.label(out >= 1, STRUCT3)
        if n > 1:
            sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            out[(lab != keep) & (lab > 0)] = 0
    if min_tumour_cells > 0:
        lab, n = ndi.label(out == 2, STRUCT3)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < min_tumour_cells:
                out[m] = 1  # small tumour blob inside liver becomes liver
    return out


def scan_report(pred: np.ndarray, ref: np.ndarray) -> dict:
    """pred, ref: (D, H, W) label volumes of one scan group."""
    liver = overlap_metrics(pred >= 1, ref >= 1)
    tumour = overlap_metrics(pred == 2, ref == 2)
    les = lesion_metrics(pred == 2, ref == 2)
    ref_slice = (ref == 2).any((1, 2))
    pred_slice = (pred == 2).any((1, 2))
    return {"liver": liver, "tumour": tumour, "lesions": les,
            "slice_tumour_detection": {"tp": int((ref_slice & pred_slice).sum()), "fn": int((ref_slice & ~pred_slice).sum()),
                                       "fp": int((~ref_slice & pred_slice).sum()), "tn": int((~ref_slice & ~pred_slice).sum())},
            "n_slices": int(len(ref)), "tumour_cells_reference": int((ref == 2).sum()), "tumour_cells_predicted": int((pred == 2).sum())}
