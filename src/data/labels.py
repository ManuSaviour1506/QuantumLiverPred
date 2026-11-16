"""Verified CT labels for scans.zip.

Provenance (recorded in data/labels/verification/verification_log.json):
  * The 1,745 `ct_lits_####.png` files in scans.zip were downloaded by liver_final/backend/scripts/
    build_real_liver_dataset.py from seongun/liver-tumor-classification (Hugging Face; LiTS-derived; CC BY-NC-SA 4.0),
    `images/w_sample/{i}.png` for i = 100..1899. The folder they were saved into (normal/steatosis/...) was chosen
    by brightness rules and is NOT a label.
  * V8 matched every file to its upstream slice by SHA-256 of the decoded pixels: all 1,745 match exactly, in order;
    the 55 upstream indices without a file are the downloads that failed in the original script.
  * Upstream labels ("Tumor present" vs "Normal") were re-derived from the upstream expert masks (LiTS: 0 background,
    127 liver, 255 tumour): label == (tumour pixels > 0) for all 1,745 slices (100 % agreement).
  * Masks of the 476 tumour slices are stored as 64x64 three-class maps (data/masks/lits_tumour_slices_masks64.npz).
Every no-tumour slice has an EMPTY upstream mask (no liver annotated): those slices lie outside the liver. A slice-level
"tumour vs no tumour" model on these labels would therefore learn liver presence. V8 uses the masks to build the
genuine task (tumour vs tumour-free liver tissue); see src/data/patches.py.

The 4,236 `ct_slice_####.png` files (SofiaVouzika/Liver_Tumor: undocumented yes/no labels, no patient ids; subtype
folders assigned by row number) and the 5 synthetic drawings are excluded from supervised training and evaluation.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np
import pandas as pd

from src.common.paths import LABELS, MASKS

VER = LABELS / "verification"
FIRST_INDEX = 100
LAST_INDEX = 1899
FIRST_FILE_NUMBER = 4237
UNMATCHED = [322, 324, 325, 326, 328, 329, 330] + list(range(333, 351)) + list(range(668, 698))
SOURCE = "seongun/liver-tumor-classification (Hugging Face), derived from LiTS (doi:10.1016/j.media.2022.102680)"
LICENSE = "CC BY-NC-SA 4.0 (non-commercial research use, attribution required)"


def upstream_labels() -> str:
    s = (VER / "upstream_metadata_labels_3986.txt").read_text().strip()
    if len(s) != 3986 or set(s) - {"0", "1"}:
        raise ValueError("upstream label string corrupted")
    return s


def file_number_to_upstream() -> dict[int, int]:
    """ct_lits file number -> upstream index (monotonic; the 55 failed downloads are skipped)."""
    idx = [i for i in range(FIRST_INDEX, LAST_INDEX + 1) if i not in set(UNMATCHED)]
    return {FIRST_FILE_NUMBER + k: i for k, i in enumerate(idx)}


def positive_pixel_counts() -> list[tuple[int, int]]:
    """(tumour_px, liver_px incl. tumour) for the 476 tumour slices in upstream order (exact counts at 512x512)."""
    out = []
    for e in (VER / "positive_slice_pixel_counts_base36.txt").read_text().strip().split(";"):
        t, l = e.split(",")
        out.append((int(t, 36), int(l, 36)))
    return out


def decode_mask_chunks() -> np.ndarray:
    """Rebuild the 64x64 three-class masks from the transferred, checksum-verified chunks (same as the npz)."""
    b64 = "".join((VER / f"mask_chunk{i}.txt").read_text().strip() for i in (0, 1))
    pk = np.frombuffer(zlib.decompress(base64.b64decode(b64), -15), np.uint8)
    d = np.stack([(pk >> s) & 3 for s in (0, 2, 4, 6)], 1).reshape(-1, 4096)
    c3 = np.zeros_like(d)
    c3[0] = d[0]
    for k in range(1, len(d)):
        c3[k] = (d[k] + c3[k - 1]) % 3
    return c3.reshape(-1, 64, 64).astype(np.uint8)


def load_masks() -> tuple[np.ndarray, np.ndarray]:
    with np.load(MASKS / "lits_tumour_slices_masks64.npz", allow_pickle=False) as z:
        return z["masks"], z["upstream_index"]


def verified_table() -> pd.DataFrame:
    """One row per verified ct_lits slice."""
    lab = upstream_labels()
    m = file_number_to_upstream()
    counts = iter(positive_pixel_counts())
    rows = []
    for num in sorted(m):
        i = m[num]
        y = int(lab[i])
        t, l = next(counts) if y == 1 else (0, 0)
        rows.append({"file_number": num, "file_name": f"ct_lits_{num:04d}.png", "upstream_index": i,
                     "label_tumour": y, "tumour_px_512": t, "liver_px_512": l, "liver_present": int(l > 0),
                     "label_source": "upstream expert mask (LiTS) - tumour pixels > 0",
                     "verification": "pixel-exact match to upstream w_sample; mask-derived label agrees"})
    df = pd.DataFrame(rows)
    if len(df) != 1745 or int(df.label_tumour.sum()) != 476:
        raise ValueError("verified table does not reproduce the recorded counts (1745 slices, 476 tumour)")
    return df
