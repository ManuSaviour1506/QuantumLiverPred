"""End-to-end inference for one uploaded CT slice (separate from all training code).

bytes -> format detection + decoding (PNG / JPEG / DICOM / NIfTI) -> image quality control -> canonical 512x512 grey
image (HU windowing only when DICOM/NIfTI provide HU) -> OOD check on the whole slice -> patch grid (96 px, stride 64)
-> body filter -> liver-tissue gate -> tumour models on liver patches -> slice-level call per model.

Slice-level call (per model, using that model's validation-chosen patch threshold t and band +/- 0.1):
  "Tumour detected"                  some liver patch has calibrated probability >= t + 0.1
  "No tumour detected"               every analysed liver patch < t - 0.1
  "Uncertain / unable to classify"   otherwise, or no liver tissue found, or image failed QC / is out of distribution
Limits stated with every result: the models discriminate tumour vs tumour-free LIVER TISSUE (validated at patch level on
held-out scans). A slice-level "No tumour detected" has NOT been validated because the dataset has no tumour-free
liver slices. Research use only.
"""
from __future__ import annotations

import base64
import hashlib
import io
import time
import uuid

import numpy as np
from PIL import Image

from src.inference.features import imagenet_features
from src.inference.registry import ROLE, PatchContext, Registry
from src.preprocessing import formats
from src.preprocessing.ct import PATCH, crop, patch_grid, resize_u8, to_canonical
from src.quality_control.image_qc import ImageSpec, check_image

BAND = 0.10
STRIDE = 64
LABELS = ("Tumour detected", "No tumour detected", "Uncertain / unable to classify")
DISCLAIMER = ("Research prototype. Not a medical device and not validated for clinical use. Outputs are model scores on "
              "LiTS-derived training data and must not be used for diagnosis or treatment decisions.")


class InputError(ValueError):
    def __init__(self, status: int, message: str, detail: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail or {}


def _decode_to_canonical(data: bytes, filename: str | None):
    try:
        dec = formats.decode(data, filename)
    except formats.UnsupportedFormat as e:
        raise InputError(415, str(e))
    except formats.CorruptImage as e:
        raise InputError(422, str(e))
    disp = dec.pixels if dec.kind == "display_u8" else None
    can = to_canonical(dec)
    qc = check_image(disp if disp is not None else can.image, ImageSpec())
    return dec, can, qc


def analyse(reg: Registry, data: bytes, filename: str | None = None, family: str = "all",
            prev: bytes | None = None, nxt: bytes | None = None, model_ids: list[str] | None = None,
            prev2: bytes | None = None, nxt2: bytes | None = None, version: str = "v9") -> dict:
    t_start = time.perf_counter()
    dec, can, qc = _decode_to_canonical(data, filename)
    img = can.image
    out = {"request_id": uuid.uuid4().hex[:12], "disclaimer": DISCLAIMER,
           "image": {"format": dec.meta.get("format"), "hu_available": bool(dec.meta.get("hu_available")),
                     "original_shape": list(map(int, dec.pixels.shape)), "notes": can.notes,
                     "sha256_prefix": hashlib.sha256(data).hexdigest()[:16], "window": can.window},
           "quality": {"status": qc["status"], "reasons": qc["reasons"], "warnings": qc["warnings"],
                       "stats": {k: round(float(v), 4) for k, v in qc["stats"].items()}}}
    buf = io.BytesIO()
    Image.fromarray(img).resize((384, 384), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    out["preview_png"] = base64.b64encode(buf.getvalue()).decode("ascii")
    ids = model_ids or reg.by_family(family, version)
    if qc["status"] != "ok":
        out["ood"] = {"status": "not_evaluated"}
        out["models"] = [_withheld(reg, m, "image failed quality control: " + ", ".join(qc["reasons"])) for m in ids]
        return _finish(out, t_start)

    # ---- OOD on the whole slice
    t0 = time.perf_counter()
    _, mid = imagenet_features(resize_u8(img, 224)[None])
    if reg.ood is not None:
        sc = reg.ood.score(mid)
        ood = {"status": "out_of_distribution" if bool(sc["out_of_domain"][0]) else "in_distribution",
               "d2_ratio": round(float(sc["d2_ratio"][0]), 3), "residual_ratio": round(float(sc["residual_ratio"][0]), 3),
               "rule": "ratio > 1 means beyond the 99th percentile of unseen training scans"}
    else:
        ood = {"status": "unavailable"}
    ood["seconds"] = round(time.perf_counter() - t0, 3)
    out["ood"] = ood
    if ood["status"] == "out_of_distribution":
        out["models"] = [_withheld(reg, m, "image is unlike the training CT slices (out-of-distribution)") for m in ids]
        return _finish(out, t_start)

    # ---- patch grid, body filter, liver gate
    grid = patch_grid(STRIDE, PATCH)
    body = img > 10
    keep = [(y, x) for (y, x) in grid if body[y:y + PATCH, x:x + PATCH].mean() >= 0.6]
    patches = np.stack([crop(img, y, x, PATCH) for (y, x) in keep]) if keep else np.zeros((0, PATCH, PATCH), np.uint8)
    ctx_all = PatchContext(patches, None, reg.ft_model())
    gate = {"status": "unavailable", "n_candidates": len(keep)}
    liver_idx = np.arange(len(keep))
    if reg.gate is not None and len(keep):
        pl = reg.gate.prob(ctx_all.imagenet_gap())
        liver_idx = np.where(pl >= reg.gate.threshold)[0]
        gate = {"status": "ok", "n_candidates": len(keep), "n_liver_patches": int(len(liver_idx)),
                "threshold": reg.gate.threshold, "p_liver": [round(float(v), 4) for v in pl]}
    out["liver_gate"] = gate
    out["patch_grid"] = {"patch_px": PATCH, "stride_px": STRIDE, "positions": [list(p) for p in keep]}
    if len(liver_idx) == 0:
        out["models"] = [_withheld(reg, m, "no liver tissue detected in this slice; the models only analyse liver tissue")
                         for m in ids]
        return _finish(out, t_start)

    lp = patches[liver_idx]
    stacks = None
    ctx_note = None
    if prev is not None and nxt is not None:
        try:
            _, cp, qp = _decode_to_canonical(prev, None)
            _, cn, qn = _decode_to_canonical(nxt, None)
            stacks = np.stack([np.stack([crop(cp.image, *keep[i], PATCH), crop(img, *keep[i], PATCH),
                                         crop(cn.image, *keep[i], PATCH)]) for i in liver_idx])
            ctx_note = "previous and next slices supplied"
        except InputError as e:
            ctx_note = f"context slices unusable: {e.message}"
    stacks5 = None
    if stacks is not None and prev2 is not None and nxt2 is not None:
        try:
            _, cp2, _ = _decode_to_canonical(prev2, None)
            _, cn2, _ = _decode_to_canonical(nxt2, None)
            stacks5 = np.stack([np.stack([crop(cp2.image, *keep[i], PATCH), stacks[j][0], stacks[j][1], stacks[j][2],
                                          crop(cn2.image, *keep[i], PATCH)]) for j, i in enumerate(liver_idx)])
            ctx_note = "two slices above and two below supplied"
        except InputError as e:
            ctx_note = f"outer context slices unusable: {e.message}"
    emb = reg.models[reg.selected_cnn].m if reg.selected_cnn in reg.models else None
    ctx = PatchContext(lp, stacks if stacks is not None else np.repeat(lp[:, None], 3, 1), reg.ft_model(), stacks5, emb)
    results = []
    for mid_ in ids:
        results.append(_run_model(reg, mid_, ctx, liver_idx, keep, stacks is not None, ctx_note, stacks5 is not None))
    out["models"] = results
    return _finish(out, t_start)


def _card(reg, mid):
    lm = reg.models.get(mid)
    meta = lm.meta if lm else {}
    st = reg.status.get(mid, {"status": "not_trained"})
    name = meta.get("name") or V9_PLACEHOLDER_NAMES.get(mid, mid)
    return {"model_id": mid, "name": name, "family": st.get("family"), "role": reg.role(mid) or ("planned" if mid in V9_PLACEHOLDER_NAMES else None),
            "version": reg.version(mid), "quantum_backend": meta.get("quantum_backend"), "test_metrics": meta.get("test_summary"),
            "calibrated": bool(lm and lm.calibrator is not None)}


def _withheld(reg, mid, reason):
    st = reg.status.get(mid, {}).get("status", "not_trained")
    card = _card(reg, mid)
    if st != "loaded":
        card.update(status="NOT TRAINED" if st == "not_trained" else "FAILED", label=None, probability=None,
                    reason=reg.status.get(mid, {}).get("detail"))
    else:
        card.update(status="withheld", label=LABELS[2], probability=None, reason=reason)
    return card


V9_PLACEHOLDER_NAMES = {"convnext_tiny_ft_v9": "ConvNeXt-Tiny (fine-tuned)", "efficientnet_b2_ft_v9": "EfficientNet-B2 (fine-tuned)",
                        "unet3d_v9": "3-D U-Net", "nnunet_v9": "nnU-Net"}


def _run_model(reg, mid, ctx, liver_idx, keep, has_context, ctx_note, has_context5=False):
    card = _card(reg, mid)
    st = reg.status.get(mid, {}).get("status", "not_trained")
    if st != "loaded":
        card.update(status="NOT TRAINED" if st == "not_trained" else "FAILED", label=None, probability=None,
                    reason=reg.status.get(mid, {}).get("detail"))
        return card
    lm = reg.models[mid]
    if lm.needs_context and not has_context:
        card.update(status="needs_context", label=LABELS[2], probability=None,
                    reason="this model reads adjacent slices; upload the previous and next slices to run it")
        return card
    if lm.context_slices == 2 and not has_context5:
        card.update(status="needs_context", label=LABELS[2], probability=None,
                    reason="this model reads 5 slices; upload two slices above and two below to run it")
        return card
    t0 = time.perf_counter()
    s, p = lm.probabilities(ctx)
    t, pmax = lm.threshold, float(np.max(p))
    if pmax >= t + BAND:
        label = LABELS[0]
    elif pmax < t - BAND:
        label = LABELS[1]
    else:
        label = LABELS[2]
    k = int(np.argmax(p))
    card.update(status="ok", label=label, probability=round(pmax, 4), threshold=round(t, 4), band=BAND,
                n_liver_patches=int(len(p)), top_patch={"y": int(keep[liver_idx[k]][0]), "x": int(keep[liver_idx[k]][1])},
                patch_probabilities=[{"y": int(keep[i][0]), "x": int(keep[i][1]), "p": round(float(v), 4)}
                                     for i, v in zip(liver_idx, p)],
                processing_ms=round((time.perf_counter() - t0) * 1000, 1),
                context=ctx_note if lm.needs_context else None,
                probability_note=("calibrated (Platt scaling on validation scans)" if lm.calibrator is not None else
                                  "uncalibrated logistic transform of the model score"))
    return card


def _finish(out, t_start):
    out["processing_ms_total"] = round((time.perf_counter() - t_start) * 1000, 1)
    return out
