"""Explanation visualisations that are technically supported by each model family.

* patch_map_png      the model's own patch probabilities drawn as coloured squares on the slice (a display of the
                     scores, not a segmentation and not a validated tumour localisation)
* gradcam (V9)       Grad-CAM of the NumPy CNNs (src/inference/v9_services.gradcam_slice) — shows which pixels
                     influenced the score, not that the model is medically correct
* linear_contributions  per-feature contributions (standardised value x coefficient) for linear models
* sensitivity        d logit / d input for quantum and hybrid models (exact adjoint gradients), plus a finite
                     perturbation check
No explanation is ever drawn for a model that did not produce a score, and none is presented as clinical localisation.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

CAUTION = ("Explanation visualisation of the model's scores. It is not a segmentation, not a validated tumour "
           "localisation and not a diagnosis.")


def patch_map_png(img: np.ndarray, positions: list, probs: list[float], patch: int = 96, threshold: float | None = None,
                  size: int = 384) -> bytes:
    base = Image.fromarray(img).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for (y, x), p in zip(positions, probs):
        p = float(np.clip(p, 0, 1))
        colour = (int(255 * p), int(90 + 80 * (1 - p)), int(255 * (1 - p)), int(50 + 110 * p))
        d.rectangle([x + 2, y + 2, x + patch - 3, y + patch - 3], fill=colour,
                    outline=(255, 255, 255, 200) if threshold is not None and p >= threshold else None, width=2)
    out = Image.alpha_composite(base, over).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def linear_contributions(coef: np.ndarray, z: np.ndarray, names: list[str], top: int = 8) -> list[dict]:
    c = np.asarray(coef, float).ravel() * np.asarray(z, float).ravel()
    order = np.argsort(-np.abs(c))[:top]
    return [{"feature": names[i] if i < len(names) else f"f{i}", "contribution": float(c[i])} for i in order]


def numeric_sensitivity(fn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, float)
    base = fn(x[None])[0]
    out = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        out[i] = (fn((x + e)[None])[0] - base) / eps
    return out
