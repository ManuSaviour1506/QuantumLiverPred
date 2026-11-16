"""Minimal NIfTI-1 writer (single file .nii / .nii.gz) for label maps and test volumes."""
from __future__ import annotations

import gzip
import struct

import numpy as np

_DT = {np.dtype(np.uint8): (2, 8), np.dtype(np.int16): (4, 16), np.dtype(np.float32): (16, 32)}


def nifti_bytes(vol_xyz: np.ndarray, spacing_xyz=(1.0, 1.0, 1.0), compress: bool = False) -> bytes:
    """vol_xyz: array indexed (x, y, z) (NIfTI order). Returns the file contents."""
    a = np.asarray(vol_xyz)
    if a.dtype not in _DT:
        a = a.astype(np.float32)
    code, bits = _DT[a.dtype]
    dim = [a.ndim] + list(a.shape) + [1] * (7 - a.ndim)
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, *dim)
    struct.pack_into("<h", hdr, 70, code)
    struct.pack_into("<h", hdr, 72, bits)
    struct.pack_into("<8f", hdr, 76, 1.0, *[float(s) for s in spacing_xyz], 1, 1, 1, 1)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<2f", hdr, 112, 1.0, 0.0)
    struct.pack_into("<h", hdr, 252, 0)
    hdr[344:348] = b"n+1\x00"
    raw = bytes(hdr) + b"\x00" * 4 + a.astype(a.dtype.newbyteorder("<")).tobytes(order="F")
    return gzip.compress(raw) if compress else raw
