"""V9 services behind the new endpoints: slice segmentation, Grad-CAM, whole-volume analysis.

Every output is computed by a trained model; when a model is missing the service raises ServiceUnavailable instead of
returning anything made up. Research use only - not a diagnosis.
"""
from __future__ import annotations

import base64
import io
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from src.explain.gradcam import gradcam, heat_overlay
from src.inference.pipeline import DISCLAIMER, InputError, _decode_to_canonical
from src.inference.registry import Registry, _down64
from src.preprocessing.ct import PATCH, crop
from src.volume.nifti_write import nifti_bytes
from src.volume.pipeline import CELL, Volume, VolumeInputError, analyse_volume, load_volume, overlay_png, unet_inputs


class ServiceUnavailable(RuntimeError):
    pass


def _b64png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _slices(data, filename, prev, nxt):
    _, can, qc = _decode_to_canonical(data, filename)
    if qc["status"] != "ok":
        raise InputError(422, "image failed quality control: " + ", ".join(qc["reasons"]))
    imgs, note = [can.image], "single slice: the U-Net's neighbour channels repeat it"
    if prev is not None and nxt is not None:
        _, cp, _ = _decode_to_canonical(prev, None)
        _, cn, _ = _decode_to_canonical(nxt, None)
        imgs, note = [cp.image, can.image, cn.image], "previous and next slices used as context"
    return imgs, note, can


def segment_slice(reg: Registry, data: bytes, filename: str | None, prev: bytes | None = None, nxt: bytes | None = None,
                  focus: str = "tumour") -> dict:
    if reg.unet is None:
        raise ServiceUnavailable("the V9 U-Net is not trained / not loaded")
    t0 = time.perf_counter()
    imgs, note, can = _slices(data, filename, prev, nxt)
    vol = np.stack(imgs)
    X = unet_inputs(vol)
    k = 1 if len(imgs) == 3 else 0
    prob = reg.unet.predict_proba(X[k:k + 1])[0]
    seg = prob.argmax(0).astype(np.uint8)
    post = reg.unet.meta.get("postprocessing", {})
    from src.segmentation.metrics import postprocess
    seg = postprocess(seg[None], int(post.get("min_tumour_cells", 0)), bool(post.get("tumour_in_liver", True)))[0]
    lab, n = ndi.label(seg == 2)
    lesions = []
    for j in range(1, n + 1):
        yy, xx = np.where(lab == j)
        lesions.append({"id": j, "cells": int(len(yy)), "pixels_512": int(len(yy)) * CELL * CELL,
                        "bbox_px": [int(yy.min() * CELL), int(xx.min() * CELL), int((yy.max() + 1) * CELL), int((xx.max() + 1) * CELL)],
                        "mean_tumour_probability": round(float(prob[2][lab == j].mean()), 4)})
    lesions.sort(key=lambda r: -r["cells"])
    heat = (np.kron(prob[2], np.ones((CELL, CELL))) * 255).astype(np.uint8)
    out = {"disclaimer": DISCLAIMER, "model": "unet25d_v9", "focus": focus, "context": note,
           "image_notes": can.notes, "liver_cells": int((seg >= 1).sum()), "tumour_cells": int((seg == 2).sum()),
           "liver_pixels_512": int((seg >= 1).sum()) * CELL * CELL, "tumour_pixels_512": int((seg == 2).sum()) * CELL * CELL,
           "tumour_share_of_liver": round(float((seg == 2).sum() / max((seg >= 1).sum(), 1)), 4),
           "lesions": lesions, "map64": seg.tolist(), "resolution": "64 x 64 cells (8 x 8 pixels of the 512 x 512 slice)",
           "overlay_png": base64.b64encode(overlay_png(can.image, seg)).decode("ascii"),
           "tumour_probability_png": _b64png(heat), "postprocessing": post,
           "finding": ("tumour region segmented (research output)" if lesions else
                       ("liver found, no tumour region segmented" if (seg >= 1).any() else "no liver found in this slice")),
           "processing_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return out


def gradcam_slice(reg: Registry, data: bytes, filename: str | None, prev: bytes | None = None, nxt: bytes | None = None,
                  model_id: str | None = None) -> dict:
    """Grad-CAM over the liver patches of one slice (liver from the U-Net when available, else the V8 liver gate)."""
    mid = model_id or reg.selected_cnn
    lm = reg.models.get(mid)
    if lm is None or lm.kind != "numpy_cnn":
        raise ServiceUnavailable(f"Grad-CAM needs a loaded NumPy CNN; '{mid}' is not available")
    t0 = time.perf_counter()
    imgs, note, can = _slices(data, filename, prev, nxt)
    img = can.image
    if lm.context_slices == 2:
        raise InputError(422, "choose a 2-D or 3-slice CNN for Grad-CAM on an upload (5-slice models need 4 neighbours)")
    if lm.context_slices == 1 and len(imgs) != 3:
        raise InputError(422, "this CNN reads adjacent slices: upload the previous and next slices too")
    # candidate patches (stride 48) inside the liver
    if reg.unet is not None:
        seg = reg.unet.predict(unet_inputs(np.stack(imgs))[1 if len(imgs) == 3 else 0:][:1])[0]
        liver = seg >= 1
        src = "U-Net liver mask"
    else:
        raise ServiceUnavailable("Grad-CAM needs the U-Net to locate the liver")
    pos = [(y, x) for y in range(0, 512 - PATCH + 1, 48) for x in range(0, 512 - PATCH + 1, 48)
           if liver[y // CELL:(y + PATCH) // CELL, x // CELL:(x + PATCH) // CELL].mean() >= 0.4]
    if not pos:
        return {"model_id": mid, "status": "no_liver", "detail": "no liver found in this slice", "liver_source": src}
    stacks = np.stack([np.stack([crop(s, y, x, PATCH) for s in imgs]) for (y, x) in pos])
    X = _down64(stacks).astype(np.float32) / 255.0
    if not lm.needs_context:
        X = X[:, (1 if len(imgs) == 3 else 0):][:, :1]
    cams, logits = gradcam(lm.m, X)
    p = lm.calibrator(logits) if lm.calibrator else 1 / (1 + np.exp(-logits))
    heat = np.zeros((512, 512), np.float32)
    for (y, x), c, pi in zip(pos, cams, p):
        big = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((PATCH, PATCH), Image.BILINEAR)) / 255.0
        heat[y:y + PATCH, x:x + PATCH] = np.maximum(heat[y:y + PATCH, x:x + PATCH], big * float(pi))
    k = int(np.argmax(p))
    return {"disclaimer": DISCLAIMER, "model_id": mid, "model_name": lm.meta.get("name"), "liver_source": src, "context": note,
            "method": "Grad-CAM on the last convolutional block; each patch's map is weighted by its calibrated probability",
            "caution": "a heat map shows which pixels influenced the score, not that the model is medically correct",
            "threshold": lm.threshold, "patches": [{"y": int(y), "x": int(x), "p": round(float(v), 4)} for (y, x), v in zip(pos, p)],
            "top_patch": {"y": int(pos[k][0]), "x": int(pos[k][1]), "p": round(float(p[k]), 4)},
            "overlay_png": _b64png(heat_overlay(img, heat / max(heat.max(), 1e-6))),
            "processing_ms": round((time.perf_counter() - t0) * 1000, 1)}


def analyze_volume(reg: Registry, data: bytes, filename: str, include_mask: bool = True) -> dict:
    if reg.unet is None:
        raise ServiceUnavailable("the V9 U-Net is not trained / not loaded")
    try:
        vol = load_volume(data, filename)
    except VolumeInputError as e:
        raise InputError(422, str(e))
    lm = reg.models.get(reg.selected_cnn)
    cnn = lm.m if lm is not None else None
    res = analyse_volume(vol, reg.unet, reg.unet.meta.get("postprocessing", {}), cnn=cnn,
                         cnn_threshold=lm.threshold if lm else None, cnn_calibrator=lm.calibrator if lm else None,
                         cnn_slices=lm.meta["config"]["in_channels"] if lm else 3)
    seg = res["segmentation"]
    D = len(seg)
    # 3-D view: tumour cells and the liver surface (subsampled) as [slice, row, col] at 64 x 64 resolution
    tz, ty, tx = np.where(seg == 2)
    surf = (seg >= 1) & ~ndi.binary_erosion(seg >= 1, np.ones((3, 3, 3), bool))
    lz, ly, lx = np.where(surf)
    step = max(1, len(lz) // 6000)
    out = {"disclaimer": DISCLAIMER, "source": vol.source, "notes": res["notes"], "summary": res["summary"],
           "lesions": res["lesions"], "per_slice": res["per_slice"], "timing_s": res["timing_s"],
           "models": {"segmentation": "unet25d_v9", "classifier": reg.selected_cnn},
           "overlays_png": {str(z): base64.b64encode(b).decode("ascii") for z, b in res["overlays_png"].items()},
           "slice_previews_png": {str(z): _b64png(np.asarray(Image.fromarray(vol.slices[z]).resize((256, 256), Image.LANCZOS)))
                                  for z in sorted(set(np.linspace(0, D - 1, min(D, 24)).round().astype(int)))},
           "points3d": {"shape": [int(D), 64, 64],
                        "tumour": np.stack([tz, ty, tx], 1)[: 20000].tolist(),
                        "liver_surface": np.stack([lz, ly, lx], 1)[::step].tolist()},
           "spacing_mm": list(vol.spacing_mm) if vol.spacing_mm else None}
    if include_mask:
        sp = vol.spacing_mm
        xyz = np.transpose(seg, (2, 1, 0)).astype(np.uint8)
        spacing_xyz = (sp[2] * CELL, sp[1] * CELL, sp[0]) if sp else (1.0, 1.0, 1.0)
        out["mask_nifti_b64"] = base64.b64encode(nifti_bytes(xyz, spacing_xyz, compress=True)).decode("ascii")
        out["mask_note"] = "labels 0 background, 1 liver, 2 tumour; 64 x 64 in-plane (8 x 8-pixel cells)" + \
            ("" if sp else "; spacing unknown (set to 1 mm)")
    return out
