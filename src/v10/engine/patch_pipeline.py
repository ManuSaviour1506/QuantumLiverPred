"""Shared front end for patch-level models (identical to V9's inference front end, built from the same V9 functions).

bytes -> format detection + decoding (PNG / JPEG / DICOM / NIfTI) -> image QC -> canonical 512x512 grey image
-> whole-slice OOD check (V9 detector) -> 96-px patch grid, stride 64 -> body filter -> V9 liver-tissue gate
-> liver patches (+ adjacent-slice stacks when supplied).

Known limitation carried over from V9: the liver gate was trained with 43 mislabelled edge patches (Stage 3) and can
accept non-liver tissue on slices without liver, so slice-level "No tumour detected" is not validated.

V10 liver-region validation (audit finding 2): the patch gate alone accepted heart tissue on ct_lits_4353 (a chest
slice with no liver in the expert mask). After the gate, the V9 2.5-D U-Net (liver + tumour segmentation) must also
find a liver region of at least LIVER_MIN_UNET_PX pixels (default 4096 px = 64 cells of 8x8 px, 1.6 % of the slice);
otherwise the slice is withheld as "liver region not confirmed". The default was set with reference to the audit
sample (U-Net liver area: tumour slices 6,976-37,248 px, heart slice 4353 3,200 px) and has NOT been validated on the
full dataset: run scripts/v10_validate_liver_gate.py on the full dataset before relying on it.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.v10.engine.interface import InputRejected

PATCH, STRIDE = 96, 64
LIVER_MIN_UNET_PX = int(os.environ.get("LTI_LIVER_MIN_UNET_PX", "4096"))


@dataclass
class FrontEnd:
    image: np.ndarray | None = None
    preview_png: bytes | None = None
    meta: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    ood: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)
    liver_idx: np.ndarray | None = None
    patches: np.ndarray | None = None          # liver patches (N, 96, 96)
    stacks: np.ndarray | None = None           # (N, 3, 96, 96) when prev/next supplied
    stacks5: np.ndarray | None = None          # (N, 5, 96, 96) when two slices on each side are supplied
    liver_check: dict = field(default_factory=dict)
    stop_reason: str | None = None
    timings_ms: dict = field(default_factory=dict)
    context_note: str | None = None


def decode(data: bytes, filename: str | None):
    from src.preprocessing import formats
    from src.preprocessing.ct import to_canonical
    from src.quality_control.image_qc import ImageSpec, check_image
    try:
        dec = formats.decode(data, filename)
    except formats.UnsupportedFormat as e:
        raise InputRejected(415, str(e))
    except formats.CorruptImage as e:
        raise InputRejected(422, str(e))
    can = to_canonical(dec)
    disp = dec.pixels if dec.kind == "display_u8" else None
    qc = check_image(disp if disp is not None else can.image, ImageSpec())
    return dec, can, qc


def preview(img: np.ndarray, size: int = 384) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).resize((size, size), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def liver_validation(registry, img: np.ndarray, prev_img: np.ndarray | None = None,
                     next_img: np.ndarray | None = None, min_px: int | None = None) -> dict:
    """Second, independent liver check with the U-Net. status: confirmed | rejected | unavailable."""
    min_px = LIVER_MIN_UNET_PX if min_px is None else int(min_px)
    unet = getattr(registry, "unet", None)
    if unet is None:
        return {"status": "unavailable", "method": "U-Net not loaded; patch gate only", "min_px": min_px}
    from src.volume.pipeline import CELL, unet_inputs
    t0 = time.perf_counter()
    if prev_img is not None and next_img is not None:
        vol, k, ctx = np.stack([prev_img, img, next_img]), 1, "previous and next slices"
    else:
        vol, k, ctx = img[None], 0, "single slice (neighbour channels repeat it)"
    from src.segmentation.metrics import postprocess
    prob = unet.predict_proba(unet_inputs(vol)[k:k + 1])[0]          # same decoding as the V9 segmentation service
    post = unet.meta.get("postprocessing", {})
    seg = postprocess(prob.argmax(0).astype(np.uint8)[None], int(post.get("min_tumour_cells", 0)),
                      bool(post.get("tumour_in_liver", True)))[0]
    liver_px = int((seg >= 1).sum()) * CELL * CELL
    ok = liver_px >= min_px
    return {"status": "confirmed" if ok else "rejected", "unet_liver_px": liver_px, "min_px": min_px,
            "unet_tumour_px": int((seg == 2).sum()) * CELL * CELL, "context": ctx,
            "method": "liver gate (patch) AND U-Net liver area >= min_px", "mask64": seg.astype(np.uint8),
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


def patch_liver_coverage(mask64: np.ndarray | None, positions: list) -> list[float]:
    if mask64 is None:
        return []
    from src.volume.pipeline import CELL
    return [round(float((mask64[y // CELL:(y + PATCH) // CELL, x // CELL:(x + PATCH) // CELL] >= 1).mean()), 3)
            for (y, x) in positions]


def run(registry, data: bytes, filename: str | None, prev: bytes | None = None, nxt: bytes | None = None,
        prev2: bytes | None = None, nxt2: bytes | None = None, validate_liver: bool = True) -> FrontEnd:
    from src.inference.features import imagenet_features
    from src.inference.registry import PatchContext
    from src.preprocessing.ct import crop, patch_grid, resize_u8
    fe = FrontEnd()
    t0 = time.perf_counter()
    dec, can, qc = decode(data, filename)
    img = can.image
    fe.image = img
    fe.meta = {"format": dec.meta.get("format"), "hu_available": bool(dec.meta.get("hu_available")),
               "original_shape": list(map(int, dec.pixels.shape)), "window": can.window, "notes": can.notes,
               "sha256_prefix": hashlib.sha256(data).hexdigest()[:16]}
    fe.quality = {"status": qc["status"], "reasons": qc["reasons"], "warnings": qc["warnings"],
                  "stats": {k: round(float(v), 4) for k, v in qc["stats"].items()}}
    fe.preview_png = preview(img)
    fe.timings_ms["decode_qc"] = round((time.perf_counter() - t0) * 1000, 1)
    if qc["status"] != "ok":
        fe.stop_reason = "image failed quality control: " + ", ".join(qc["reasons"])
        return fe
    t1 = time.perf_counter()
    _, mid = imagenet_features(resize_u8(img, 224)[None])
    if registry.ood is not None:
        sc = registry.ood.score(mid)
        out_dist = bool(sc["out_of_domain"][0])
        fe.ood = {"status": "out_of_distribution" if out_dist else "in_distribution",
                  "d2_ratio": round(float(sc["d2_ratio"][0]), 3), "residual_ratio": round(float(sc["residual_ratio"][0]), 3),
                  "rule": "ratio > 1 means beyond the 99th percentile of unseen training scans"}
    else:
        fe.ood = {"status": "unavailable"}
    fe.timings_ms["ood"] = round((time.perf_counter() - t1) * 1000, 1)
    if fe.ood["status"] == "out_of_distribution":
        fe.stop_reason = "image is unlike the training CT slices (out-of-distribution)"
        return fe
    t2 = time.perf_counter()
    body = img > 10
    keep = [(y, x) for (y, x) in patch_grid(STRIDE, PATCH) if body[y:y + PATCH, x:x + PATCH].mean() >= 0.6]
    fe.positions = keep
    patches = np.stack([crop(img, y, x, PATCH) for (y, x) in keep]) if keep else np.zeros((0, PATCH, PATCH), np.uint8)
    liver_idx = np.arange(len(keep))
    if registry.gate is not None and len(keep):
        pl = registry.gate.prob(PatchContext(patches, None, None).imagenet_gap())
        liver_idx = np.where(pl >= registry.gate.threshold)[0]
        fe.gate = {"status": "ok", "n_candidates": len(keep), "n_liver_patches": int(len(liver_idx)),
                   "threshold": registry.gate.threshold}
    else:
        fe.gate = {"status": "unavailable", "n_candidates": len(keep)}
    fe.liver_idx = liver_idx
    fe.timings_ms["gate"] = round((time.perf_counter() - t2) * 1000, 1)
    if len(liver_idx) == 0:
        fe.stop_reason = "no liver tissue detected in this slice; the models only analyse liver tissue"
        return fe
    cp = cn = None
    if prev is not None and nxt is not None:
        try:
            _, cp, _ = decode(prev, None)
            _, cn, _ = decode(nxt, None)
            if cp.image.shape != img.shape or cn.image.shape != img.shape:
                raise InputRejected(422, "context slices have a different size from the target slice")
        except InputRejected as e:
            cp = cn = None
            fe.context_note = f"context slices unusable: {e.message}"
    if validate_liver:
        t3 = time.perf_counter()
        lc = liver_validation(registry, img, cp.image if cp is not None else None, cn.image if cn is not None else None)
        mask = lc.pop("mask64", None)
        lc["gate_patch_liver_coverage"] = patch_liver_coverage(mask, [keep[i] for i in liver_idx])
        fe.liver_check = lc
        fe.timings_ms["liver_validation"] = round((time.perf_counter() - t3) * 1000, 1)
        if lc["status"] == "rejected":
            fe.stop_reason = (f"liver region not confirmed: the patch gate kept {len(liver_idx)} patch(es) but the U-Net "
                              f"liver segmentation found {lc['unet_liver_px']} px (< {lc['min_px']} px); the models "
                              "only analyse liver tissue")
            return fe
    fe.patches = patches[liver_idx]
    if cp is not None and cn is not None:
        fe.stacks = np.stack([np.stack([crop(cp.image, *keep[i], PATCH), crop(img, *keep[i], PATCH),
                                        crop(cn.image, *keep[i], PATCH)]) for i in liver_idx])
        fe.context_note = "previous and next slices supplied"
        if prev2 is not None and nxt2 is not None:
            try:
                _, cp2, _ = decode(prev2, None)
                _, cn2, _ = decode(nxt2, None)
                fe.stacks5 = np.stack([np.stack([crop(cp2.image, *keep[i], PATCH), fe.stacks[j][0], fe.stacks[j][1],
                                                 fe.stacks[j][2], crop(cn2.image, *keep[i], PATCH)])
                                       for j, i in enumerate(liver_idx)])
                fe.context_note = "two slices above and two below supplied"
            except InputRejected as e:
                fe.context_note = f"outer context slices unusable: {e.message}"
    elif prev is None or nxt is None:
        fe.context_note = "no adjacent slices supplied"
    return fe


def png_to_b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")
