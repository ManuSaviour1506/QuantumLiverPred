"""CT preprocessing: Hounsfield windowing, canonical 512x512 grey image, body mask, patch grid.

Training images (LiTS-derived `w_sample` slices) are 512x512 8-bit images that were windowed by the dataset author.
They contain exactly 151 distinct grey levels, i.e. a window ~150 HU wide; the window level is not documented
upstream. For DICOM / NIfTI inputs (real HU) the default TRAINING_WINDOW reproduces that width with an estimated level
of 30 HU (a standard liver window). Uploads in another windowing still run, but the OOD detector is expected to flag
them; nothing here pretends to know the exact upstream level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

CANONICAL = 512           # training slices are 512 x 512
PATCH = 96                # patch side in canonical pixels (~3/16 of the slice width, ~6-7 cm)

WINDOWS = {
    "training_estimate": (30.0, 150.0),   # (level, width) - see module docstring
    "liver": (60.0, 160.0),
    "soft_tissue": (40.0, 400.0),
}
TRAINING_WINDOW = WINDOWS["training_estimate"]


def apply_window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    lo = level - width / 2.0
    return np.clip(np.round((np.asarray(hu, np.float64) - lo) / width * 255.0), 0, 255).astype(np.uint8)


def resize_u8(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    im = Image.fromarray(img)
    method = Image.BICUBIC if size > max(img.shape) else Image.LANCZOS
    return np.asarray(im.resize((size, size), method))


@dataclass
class Canonical:
    image: np.ndarray        # uint8 (512, 512)
    notes: list
    window: tuple | None     # window applied (HU inputs only)


def to_canonical(decoded, window: tuple | None = None) -> Canonical:
    """DecodedImage -> 512x512 uint8 in the training grey-level convention."""
    notes = list(decoded.meta.get("warnings", []))
    if decoded.kind == "hu":
        level, width = window or TRAINING_WINDOW
        g = apply_window(decoded.pixels, level, width)
        notes.append(f"HU windowed with level {level:g}, width {width:g}")
        win = (level, width)
    else:
        g = np.asarray(decoded.pixels, np.uint8)
        win = None
    h, w = g.shape
    if h != w:  # pad to square (no anatomy is cropped)
        s = max(h, w)
        pad = np.zeros((s, s), np.uint8)
        pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
        g = pad
        notes.append(f"padded from {h}x{w} to square")
    if g.shape[0] != CANONICAL:
        notes.append(f"resized from {g.shape[0]}x{g.shape[1]} to {CANONICAL}x{CANONICAL}")
        g = resize_u8(g, CANONICAL)
    return Canonical(g, notes, win)


def body_mask(img: np.ndarray, thr: int = 10) -> np.ndarray:
    """Rough body region on a 64x64 grid (fraction of non-background pixels per 8x8 cell >= 0.3)."""
    cells = (img > thr).reshape(64, 8, 64, 8).mean((1, 3))
    return cells >= 0.3


def patch_grid(stride: int = 96, size: int = PATCH, canonical: int = CANONICAL) -> list[tuple[int, int]]:
    """Top-left corners of a regular patch grid covering the canonical slice."""
    starts = list(range(0, canonical - size + 1, stride))
    if starts[-1] != canonical - size:
        starts.append(canonical - size)
    return [(y, x) for y in starts for x in starts]


def crop(img: np.ndarray, y: int, x: int, size: int = PATCH) -> np.ndarray:
    """Crop with zero padding where the window leaves the image."""
    H, W = img.shape[:2]
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + size, H), min(x + size, W)
    if y1 > y0 and x1 > x0:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return out


def to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(gray, np.uint8)[..., None], 3, axis=-1)


# --------------------------------------------------------------------------------------------------- augmentation
def augment_patch(img: np.ndarray, rng: np.random.Generator, max_rot: float = 12.0, max_shift: int = 8,
                  scale: tuple = (0.92, 1.08), gamma: tuple = (0.9, 1.1), noise: float = 2.0) -> np.ndarray:
    """Label-preserving CT augmentation for uint8 patches (H, W) or (H, W, C).

    Small rotation / scaling / translation, gamma and Gaussian noise. No left-right flip (it would move the liver to
    the wrong side of the body)."""
    a = np.asarray(img)
    chans = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[-1])]
    ang = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale)
    dx, dy = rng.integers(-max_shift, max_shift + 1, 2)
    g = rng.uniform(*gamma)
    out = []
    for ch in chans:
        im = Image.fromarray(ch.astype(np.uint8))
        w, h = im.size
        im = im.rotate(ang, resample=Image.BILINEAR, translate=(int(dx), int(dy)))
        if abs(sc - 1) > 1e-3:
            nw = int(round(w * sc))
            im2 = im.resize((nw, nw), Image.BILINEAR)
            off = (nw - w) // 2
            if nw >= w:
                im = im2.crop((off, off, off + w, off + h))
            else:
                canvas = Image.new("L", (w, h))
                canvas.paste(im2, (-off, -off))
                im = canvas
        x = np.asarray(im, np.float64) / 255.0
        x = np.power(x, g) * 255.0 + rng.normal(0, noise, x.shape)
        out.append(np.clip(x, 0, 255).astype(np.uint8))
    return out[0] if a.ndim == 2 else np.stack(out, -1)
