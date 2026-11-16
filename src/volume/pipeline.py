"""Whole CT-volume analysis (V9). Research output only - NOT a diagnosis.

volume (ordered axial slices) -> canonical 512 x 512 grey (HU windowed when HU are available)
  -> 2.5-D U-Net on every slice (previous / current / next) -> 64 x 64 liver / tumour maps
  -> 3-D post-processing (tumour kept inside the main liver component, minimum lesion size chosen on validation)
  -> 3-D lesions (26-connected components): slice range, bounding box in pixels, size in voxels and mm^3 when the
     file supplies voxel spacing
  -> 2.5-D CNN tumour probability for 96 x 96 patches inside the predicted liver (stride 48); a lesion is
     'confirmed' when a patch overlapping it reaches the classifier's validation threshold
  -> per-slice and per-volume summaries, overlay images.
Volume figures use the segmentation resolution: one cell = 8 x 8 pixels of the 512 x 512 slice.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from src.preprocessing.ct import TRAINING_WINDOW, apply_window, resize_u8
from src.preprocessing.formats import decode, read_nifti_bytes
from src.segmentation.metrics import STRUCT3, postprocess

CELL = 8


class VolumeInputError(ValueError):
    pass


@dataclass
class Volume:
    slices: np.ndarray              # (D, 512, 512) uint8, canonical grey
    spacing_mm: tuple | None        # (dz, dy, dx) after resampling to 512 x 512, or None if unknown
    source: str
    notes: list = field(default_factory=list)


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _to512(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape
    s = max(h, w)
    if h != w:
        pad = np.zeros((s, s), img.dtype)
        pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
        img = pad
    return resize_u8(img, 512), s / 512.0


def load_volume(data: bytes, filename: str) -> Volume:
    """NIfTI volume (.nii / .nii.gz, HU windowed with the training window) or a .zip of ordered slices
    (PNG / JPEG / DICOM; natural filename order)."""
    name = (filename or "").lower()
    notes = []
    if name.endswith((".nii", ".nii.gz")) or data[:2] == b"\x1f\x8b":
        a, meta = read_nifti_bytes(data)
        while a.ndim > 3:
            a = a[..., 0]
        if a.ndim != 3:
            raise VolumeInputError("NIfTI file is not a 3-D volume")
        lo, hi = float(a.min()), float(a.max())
        hu = lo >= -3100 and hi <= 4100 and lo < -200
        out, scale = [], 1.0
        for z in range(a.shape[2]):
            img = a[:, :, z].T
            g = apply_window(img, *TRAINING_WINDOW) if hu else np.clip((img - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
            g, scale = _to512(g)
            out.append(g)
        dx, dy, dz = meta["pixdim"]
        spacing = (abs(dz), abs(dy) * scale, abs(dx) * scale) if min(abs(dx), abs(dy), abs(dz)) > 0 else None
        notes.append("HU windowed with the training window (level %g, width %g)" % TRAINING_WINDOW if hu else
                     "intensities not in a CT HU range: min-max scaled (results unreliable)")
        if spacing is None:
            notes.append("voxel spacing missing in the NIfTI header: sizes reported in voxels only")
        return Volume(np.stack(out), spacing, "nifti", notes)
    if name.endswith(".zip") or data[:2] == b"PK":
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise VolumeInputError(f"not a readable zip: {e}")
        names = sorted([n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n
                        and n.lower().endswith((".png", ".jpg", ".jpeg", ".dcm", ".dicom"))], key=_natural_key)
        if len(names) < 3:
            raise VolumeInputError("the zip needs at least 3 slice images (PNG, JPEG or DICOM)")
        if len(names) > 1000:
            raise VolumeInputError("more than 1000 slices")
        out = []
        for n in names:
            d = decode(zf.read(n), n)
            g = apply_window(d.pixels, *TRAINING_WINDOW) if d.kind == "hu" else np.asarray(d.pixels, np.uint8)
            out.append(_to512(g)[0])
        notes.append(f"{len(names)} slices ordered by file name")
        notes.append("voxel spacing unknown for a zip of slices: sizes reported in voxels only "
                     "(upload a NIfTI volume to get mm^3)")
        return Volume(np.stack(out), None, "zip", notes)
    raise VolumeInputError("upload a NIfTI volume (.nii / .nii.gz) or a .zip of ordered slices")


def unet_inputs(vol: np.ndarray) -> np.ndarray:
    small = np.stack([np.asarray(Image.fromarray(s).resize((128, 128), Image.BOX)) for s in vol])
    D = len(small)
    prev = np.concatenate([small[:1], small[:-1]])
    nxt = np.concatenate([small[1:], small[-1:]])
    return (np.stack([prev, small, nxt], 1).astype(np.float32) / 255.0) if D else np.zeros((0, 3, 128, 128), np.float32)


def _down64(stack):
    return np.stack([np.asarray(Image.fromarray(c).resize((64, 64), Image.LANCZOS)) for c in stack])


def analyse_volume(vol: Volume, unet, postproc: dict, cnn=None, cnn_threshold: float | None = None, cnn_calibrator=None,
                   cnn_slices: int = 3, stride: int = 48, overlay_limit: int = 12) -> dict:
    t0 = time.perf_counter()
    V = vol.slices
    D = len(V)
    probs = unet.predict_proba(unet_inputs(V))
    raw = probs.argmax(1).astype(np.uint8)
    seg = postprocess(raw, int(postproc.get("min_tumour_cells", 0)), bool(postproc.get("tumour_in_liver", True)))
    t_seg = time.perf_counter() - t0

    # 2.5-D CNN on patches inside the predicted liver
    patch_rows = []
    if cnn is not None:
        half = cnn_slices // 2
        crops, meta = [], []
        for z in range(D):
            liver = seg[z] >= 1
            if liver.sum() < 4:
                continue
            for y in range(0, 512 - 96 + 1, stride):
                for x in range(0, 512 - 96 + 1, stride):
                    cell = liver[y // CELL:(y + 96) // CELL, x // CELL:(x + 96) // CELL]
                    if cell.mean() < 0.4:
                        continue
                    zs = [min(max(z + d, 0), D - 1) for d in range(-half, half + 1)]
                    crops.append(_down64(np.stack([V[k][y:y + 96, x:x + 96] for k in zs])))
                    meta.append((z, y, x))
        if crops:
            Xp = np.stack(crops).astype(np.float32) / 255.0
            s = cnn.decision_function(Xp)
            p = cnn_calibrator(s) if cnn_calibrator is not None else 1 / (1 + np.exp(-s))
            for (z, y, x), pi in zip(meta, p):
                patch_rows.append({"slice": int(z), "y": int(y), "x": int(x), "p_tumour": float(pi)})
    t_cnn = time.perf_counter() - t0 - t_seg

    # lesions
    lab, n = ndi.label(seg == 2, STRUCT3)
    cell_mm3 = None
    if vol.spacing_mm:
        dz, dy, dx = vol.spacing_mm
        cell_mm3 = dz * dy * dx * CELL * CELL
    lesions = []
    for k in range(1, n + 1):
        zz, yy, xx = np.where(lab == k)
        cells = int(len(zz))
        les = {"id": k, "slices": [int(zz.min()), int(zz.max())], "n_slices": int(zz.max() - zz.min() + 1),
               "bbox_px": [int(yy.min() * CELL), int(xx.min() * CELL), int((yy.max() + 1) * CELL), int((xx.max() + 1) * CELL)],
               "centroid_px": [float(zz.mean()), float((yy.mean() + 0.5) * CELL), float((xx.mean() + 0.5) * CELL)],
               "voxels_512": cells * CELL * CELL, "cells": cells,
               "volume_mm3": round(cells * cell_mm3, 1) if cell_mm3 else None,
               "mean_tumour_probability": float(probs[:, 2][lab == k].mean())}
        if patch_rows:
            over = [r["p_tumour"] for r in patch_rows if zz.min() <= r["slice"] <= zz.max()
                    and (lab[r["slice"], r["y"] // CELL:(r["y"] + 96) // CELL, r["x"] // CELL:(r["x"] + 96) // CELL] == k).any()]
            les["max_classifier_probability"] = float(max(over)) if over else None
            les["classifier_confirmed"] = bool(over and cnn_threshold is not None and max(over) >= cnn_threshold)
        lesions.append(les)
    lesions.sort(key=lambda r: -r["cells"])

    per_slice = []
    for z in range(D):
        pr = [r["p_tumour"] for r in patch_rows if r["slice"] == z]
        per_slice.append({"slice": z, "liver_cells": int((seg[z] >= 1).sum()), "tumour_cells": int((seg[z] == 2).sum()),
                          "max_patch_probability": float(max(pr)) if pr else None, "patches": len(pr),
                          "lesion_ids": sorted({int(v) for v in np.unique(lab[z]) if v})})
    liver_cells, tumour_cells = int((seg >= 1).sum()), int((seg == 2).sum())
    confirmed = [l for l in lesions if l.get("classifier_confirmed")]
    summary = {
        "n_slices": D, "slices_with_liver": int(sum(1 for s in per_slice if s["liver_cells"])),
        "slices_with_tumour": int(sum(1 for s in per_slice if s["tumour_cells"])),
        "lesions": len(lesions), "lesions_confirmed_by_classifier": len(confirmed) if cnn is not None else None,
        "liver_volume_mm3": round(liver_cells * cell_mm3, 1) if cell_mm3 else None,
        "tumour_volume_mm3": round(tumour_cells * cell_mm3, 1) if cell_mm3 else None,
        "liver_voxels_512": liver_cells * CELL * CELL, "tumour_voxels_512": tumour_cells * CELL * CELL,
        "tumour_burden_fraction": round(tumour_cells / liver_cells, 4) if liver_cells else None,
        "finding": ("tumour suspected (research output)" if (confirmed if cnn is not None else lesions)
                    else ("no tumour found by the models (research output)" if liver_cells else "no liver found")),
        "spacing_mm": list(vol.spacing_mm) if vol.spacing_mm else None,
    }
    order = sorted(range(D), key=lambda z: -per_slice[z]["tumour_cells"])[:overlay_limit]
    overlays = {int(z): overlay_png(V[z], seg[z]) for z in sorted(order) if per_slice[z]["tumour_cells"]}
    return {"summary": summary, "lesions": lesions, "per_slice": per_slice, "patches": patch_rows,
            "segmentation": seg, "tumour_probability": probs[:, 2], "overlays_png": overlays,
            "timing_s": {"segmentation": round(t_seg, 2), "classifier": round(t_cnn, 2), "total": round(time.perf_counter() - t0, 2)},
            "notes": vol.notes + ["research output only - not a diagnosis; segmentation at 8 x 8-pixel resolution"]}


def overlay_png(img: np.ndarray, seg64: np.ndarray) -> bytes:
    rgb = np.stack([img] * 3, -1).astype(np.float32)
    big = np.kron(seg64, np.ones((CELL, CELL), np.uint8))
    liver, tum = big == 1, big == 2
    rgb[liver] = rgb[liver] * 0.75 + np.array([40, 160, 220]) * 0.25
    rgb[tum] = rgb[tum] * 0.45 + np.array([230, 60, 60]) * 0.55
    edge = tum ^ ndi.binary_erosion(tum, iterations=2)
    rgb[edge] = [255, 80, 80]
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buf, "PNG", optimize=True)
    return buf.getvalue()
