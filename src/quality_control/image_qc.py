"""Image quality checks shared by dataset ingestion and inference (ported from V7, adapted to CT).

check_image(array, spec) returns {"status": "ok" | "reject", "reasons": [...], "warnings": [...], "stats": {...}}.
A rejected image is never classified. Checks:
  corrupted / undecodable (read_image), non-finite values, unsupported dimensions or channel count,
  empty or all-black, near-constant, saturated, noise-like (no spatial structure).
CT slices are single-channel; RGB files whose three channels are identical are accepted as greyscale, true colour
images are rejected (a colour photo or an MRI phase composite is not a CT slice).
Out-of-domain detection on deep features is in src/quality_control/ood.py (needs the backbone).
Thresholds are fixed constants (below), not fitted to any test image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

BLACK_MAX = 8                 # max pixel value <= this -> all-black
CONSTANT_STD = 2.0            # grey-level SD below this -> empty / constant
ZERO_FRACTION = 0.98          # >= this fraction of pixels at 0 -> empty
SATURATED_FRACTION = 0.50     # >= this fraction at 255 -> saturated
NOISE_AUTOCORR = 0.50         # lag-1 spatial autocorrelation below this -> noise-like (anatomy is ~0.9+)
PHASE_IDENTICAL_FRAC = 0.995  # >= this fraction of pixels equal across channels -> single phase
BRIGHT_BORDER = 160           # mean grey level of the outer 5 % frame above this -> not a CT-like dark (air) background


@dataclass
class ImageSpec:
    """The validated training-data format of the V8 CT models (LiTS-derived w_sample slices)."""
    shape: tuple = (512, 512)
    channels: int = 1             # greyscale CT; identical-channel RGB is accepted and reduced to one channel
    distinct_phases: bool = False
    allow_resize: bool = True     # other sizes are resampled to 512 x 512 and a warning is attached
    min_side: int = 128
    max_side: int = 2048
    notes: list = field(default_factory=lambda: ["axial CT slice, ~150 HU wide window, 8-bit, 512x512"])


def read_image(path) -> tuple[np.ndarray | None, str | None]:
    """Decode with PIL (PNG, JPEG, TIFF, BMP). Returns (array, error)."""
    try:
        with Image.open(path) as im:
            im.load()
            mode = im.mode
            a = np.array(im.convert("RGB") if mode in ("P", "RGBA", "LA", "CMYK", "YCbCr") else im)
        return a, None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        return None, f"corrupted_or_unreadable: {type(e).__name__}: {e}"


def lag1_autocorr(g: np.ndarray) -> float:
    g = g.astype(np.float64)
    g = g - g.mean()
    v = (g ** 2).mean()
    if v < 1e-12:
        return 1.0
    return float(0.5 * ((g[:, 1:] * g[:, :-1]).mean() + (g[1:, :] * g[:-1, :]).mean()) / v)


def check_image(a: np.ndarray | None, spec: ImageSpec | None = None, read_error: str | None = None) -> dict:
    spec = spec or ImageSpec()
    reasons, warnings = [], []
    if a is None:
        return {"status": "reject", "reasons": [read_error or "corrupted_or_unreadable"], "warnings": [], "stats": {}}
    a = np.asarray(a)
    if a.size == 0:
        return {"status": "reject", "reasons": ["empty_array"], "warnings": [], "stats": {}}
    if not np.issubdtype(a.dtype, np.number) or not np.all(np.isfinite(a)):
        return {"status": "reject", "reasons": ["non_finite_or_non_numeric"], "warnings": [], "stats": {}}
    if a.ndim == 2:
        ch = 1
    elif a.ndim == 3:
        ch = a.shape[2]
    else:
        return {"status": "reject", "reasons": [f"unsupported_ndim_{a.ndim}"], "warnings": [], "stats": {}}
    h, w = a.shape[:2]
    if a.dtype != np.uint8:
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / (hi - lo + 1e-12) * 255, 0, 255).astype(np.uint8)
        warnings.append("rescaled_to_8bit")
    g = a.astype(np.float64).mean(2) if a.ndim == 3 else a.astype(np.float64)
    stats = {"height": int(h), "width": int(w), "channels": int(ch), "min": float(g.min()), "max": float(g.max()),
             "mean": float(g.mean()), "std": float(g.std()), "zero_fraction": float((g <= 0).mean()),
             "saturated_fraction": float((g >= 255).mean()), "lag1_autocorr": lag1_autocorr(g)}
    if (h, w) != tuple(spec.shape):
        if spec.allow_resize and spec.min_side <= min(h, w) and max(h, w) <= spec.max_side:
            warnings.append(f"resized_from_{h}x{w}")
        else:
            reasons.append(f"unsupported_dimensions_{h}x{w}_expected_{spec.shape[0]}x{spec.shape[1]}")
    if ch == 3 and spec.channels == 1:
        same = ((a[..., 0] == a[..., 1]) & (a[..., 1] == a[..., 2])).mean()
        stats["identical_channel_fraction"] = float(same)
        if same < PHASE_IDENTICAL_FRAC:
            reasons.append("colour_image_not_a_ct_slice")
    elif ch != spec.channels:
        reasons.append(f"unsupported_channels_{ch}_expected_{spec.channels}")
    elif spec.distinct_phases and ch == 3:
        same = ((a[..., 0] == a[..., 1]) & (a[..., 1] == a[..., 2])).mean()
        stats["identical_channel_fraction"] = float(same)
        if same >= PHASE_IDENTICAL_FRAC:
            reasons.append("single_phase_input_channels_identical")
    if stats["max"] <= BLACK_MAX:
        reasons.append("all_black")
    elif stats["std"] < CONSTANT_STD or stats["zero_fraction"] >= ZERO_FRACTION:
        reasons.append("empty_or_constant")
    fh, fw = max(1, h // 20), max(1, w // 20)
    frame = np.concatenate([g[:fh].ravel(), g[-fh:].ravel(), g[:, :fw].ravel(), g[:, -fw:].ravel()])
    stats["border_mean"] = float(frame.mean())
    if stats["border_mean"] > BRIGHT_BORDER and stats["max"] > BLACK_MAX:
        reasons.append("bright_border_not_ct_like (CT slices have a dark air background; inverted or non-CT image?)")
    if stats["saturated_fraction"] >= SATURATED_FRACTION:
        reasons.append("saturated")
    if stats["std"] >= CONSTANT_STD and stats["lag1_autocorr"] < NOISE_AUTOCORR:
        reasons.append("noise_like_no_spatial_structure")
    return {"status": "reject" if reasons else "ok", "reasons": reasons, "warnings": warnings, "stats": stats}


def check_file(path, spec: ImageSpec | None = None) -> dict:
    a, err = read_image(path)
    r = check_image(a, spec, err)
    r["path"] = str(Path(path))
    return r
