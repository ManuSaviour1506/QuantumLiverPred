"""CT input formats: detection by content (magic bytes, not file extension) and decoding.

Supported
  PNG / JPEG     8-bit display images. Hounsfield units are NOT recoverable from these (no rescale metadata), so they
                 are treated as already-windowed images. No HU value is ever invented for them.
  DICOM          Built-in reader for uncompressed little-endian transfer syntaxes (implicit / explicit VR). If pydicom
                 is installed it is used instead (and also handles compressed transfer syntaxes).
                 HU = stored_value * RescaleSlope + RescaleIntercept, only when those tags are present.
  NIfTI-1        .nii / .nii.gz, built-in reader; scl_slope / scl_inter applied. CT NIfTI volumes (e.g. LiTS) store HU.
                 A 3-D volume is reduced to one axial slice (the requested index, or the middle slice).

decode(data, filename) -> DecodedImage(pixels, kind, meta)
  kind == "hu"           pixels are float32 Hounsfield units
  kind == "display_u8"   pixels are uint8 grey levels (already windowed by whoever exported the image)
"""
from __future__ import annotations

import gzip
import io
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

NIFTI_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8, 512: np.uint16,
                768: np.uint32}
UNCOMPRESSED_TS = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}  # implicit VR LE, explicit VR LE
MAX_BYTES = 64 * 1024 * 1024


class UnsupportedFormat(ValueError):
    pass


class CorruptImage(ValueError):
    pass


@dataclass
class DecodedImage:
    pixels: np.ndarray
    kind: str                      # "hu" | "display_u8"
    meta: dict = field(default_factory=dict)


def detect_format(data: bytes, filename: str | None = None) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(data) > 132 and data[128:132] == b"DICM":
        return "dicom"
    if data[:2] == b"\x1f\x8b":
        try:
            head = gzip.GzipFile(fileobj=io.BytesIO(data)).read(352)
        except OSError:
            return "unknown"
        return "nifti" if _is_nifti(head) else "unknown"
    if _is_nifti(data[:352]):
        return "nifti"
    if filename and filename.lower().endswith((".dcm", ".dicom")) and len(data) > 8:
        return "dicom_raw"  # DICOM without the 128-byte preamble: try the parser
    return "unknown"


def _is_nifti(h: bytes) -> bool:
    if len(h) < 348:
        return False
    return any(struct.unpack(e + "i", h[:4])[0] == 348 for e in "<>") and h[344:348] in (b"n+1\x00", b"ni1\x00")


# --------------------------------------------------------------------------------------------------- raster
def decode_raster(data: bytes, fmt: str) -> DecodedImage:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            mode = im.mode
            if mode in ("I;16", "I;16B", "I"):
                a = np.asarray(im, dtype=np.float64)
                lo, hi = np.percentile(a, [0.5, 99.5])
                g = np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
                warn = ["16-bit image rescaled to 8 bits (no HU metadata)"]
            else:
                rgb = np.asarray(im.convert("RGB"))
                same = bool((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all())
                g = rgb[..., 0].copy() if same else np.asarray(im.convert("L"))
                warn = [] if same or mode == "L" else ["colour image converted to greyscale"]
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise CorruptImage(f"{fmt} could not be decoded: {type(e).__name__}: {e}") from e
    return DecodedImage(g, "display_u8", {"format": fmt, "mode": mode, "hu_available": False,
                                          "hu_note": "PNG/JPEG carry no rescale metadata; HU not recoverable",
                                          "warnings": warn})


# --------------------------------------------------------------------------------------------------- NIfTI
def read_nifti_bytes(raw: bytes) -> tuple[np.ndarray, dict]:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    for end in ("<", ">"):
        if struct.unpack(end + "i", raw[:4])[0] == 348:
            break
    else:
        raise CorruptImage("not a NIfTI-1 file (sizeof_hdr != 348)")
    dim = struct.unpack(end + "8h", raw[40:56])
    datatype = struct.unpack(end + "h", raw[70:72])[0]
    pixdim = struct.unpack(end + "8f", raw[76:108])
    vox_offset = int(struct.unpack(end + "f", raw[108:112])[0])
    slope, inter = struct.unpack(end + "2f", raw[112:120])
    if datatype not in NIFTI_DTYPES:
        raise UnsupportedFormat(f"unsupported NIfTI datatype {datatype}")
    if not 1 <= dim[0] <= 7:
        raise CorruptImage("bad NIfTI dimensions")
    shape = tuple(int(d) for d in dim[1:1 + dim[0]])
    n = int(np.prod(shape))
    dt = np.dtype(NIFTI_DTYPES[datatype]).newbyteorder(end)
    if vox_offset + n * dt.itemsize > len(raw):
        raise CorruptImage("NIfTI file is truncated")
    a = np.frombuffer(raw, dt, count=n, offset=vox_offset).reshape(shape, order="F").astype(np.float64)
    scaled = slope not in (0.0, 1.0) or inter != 0.0
    if scaled:
        a = a * (slope or 1.0) + inter
    return a, {"shape": shape, "pixdim": [float(p) for p in pixdim[1:4]], "scl_slope": float(slope),
               "scl_inter": float(inter), "datatype": int(datatype)}


def decode_nifti(data: bytes, slice_index: int | None = None) -> DecodedImage:
    a, meta = read_nifti_bytes(data)
    while a.ndim > 3:
        a = a[..., 0]
    if a.ndim == 3:
        z = a.shape[2] // 2 if slice_index is None else int(np.clip(slice_index, 0, a.shape[2] - 1))
        img = a[:, :, z].T  # NIfTI (x, y) -> image rows = y
        meta["slice_used"], meta["n_slices"] = int(z), int(a.shape[2])
    else:
        img = a.T
    lo, hi = float(np.min(a)), float(np.max(a))      # judged on the whole volume: one slice may contain no air
    looks_hu = lo >= -3100 and hi <= 4100 and lo < -200  # air present, CT range
    meta.update({"format": "nifti", "hu_available": bool(looks_hu), "warnings": []})
    if not looks_hu:
        meta["warnings"].append("NIfTI intensities are not in a CT Hounsfield range; treated as display values")
        g = np.clip((img - np.percentile(img, 0.5)) / (np.ptp(img) + 1e-9) * 255, 0, 255).astype(np.uint8)
        return DecodedImage(g, "display_u8", meta)
    return DecodedImage(img.astype(np.float32), "hu", meta)


# --------------------------------------------------------------------------------------------------- DICOM
_VR_LONG = {b"OB", b"OD", b"OF", b"OL", b"OW", b"SQ", b"UC", b"UN", b"UR", b"UT", b"OV", b"SV", b"UV"}
_TAGS = {(0x0002, 0x0010): "TransferSyntaxUID", (0x0008, 0x0060): "Modality", (0x0028, 0x0002): "SamplesPerPixel",
         (0x0028, 0x0004): "PhotometricInterpretation", (0x0028, 0x0010): "Rows", (0x0028, 0x0011): "Columns",
         (0x0028, 0x0100): "BitsAllocated", (0x0028, 0x0101): "BitsStored", (0x0028, 0x0103): "PixelRepresentation",
         (0x0028, 0x1050): "WindowCenter", (0x0028, 0x1051): "WindowWidth", (0x0028, 0x1052): "RescaleIntercept",
         (0x0028, 0x1053): "RescaleSlope", (0x0028, 0x0030): "PixelSpacing", (0x0028, 0x0008): "NumberOfFrames",
         (0x7FE0, 0x0010): "PixelData"}
_US = {"Rows", "Columns", "BitsAllocated", "BitsStored", "PixelRepresentation", "SamplesPerPixel"}


def _parse_dicom(data: bytes) -> dict:
    pos = 132 if data[128:132] == b"DICM" else 0
    out, ts, n = {}, "1.2.840.10008.1.2.1", len(data)
    while pos + 8 <= n:
        group, elem = struct.unpack("<HH", data[pos:pos + 4])
        explicit = group == 0x0002 or ts != "1.2.840.10008.1.2"
        vr = data[pos + 4:pos + 6]
        if explicit and vr.isalpha() and vr.isupper():
            if vr in _VR_LONG:
                length = struct.unpack("<I", data[pos + 8:pos + 12])[0]
                hdr = 12
            else:
                length = struct.unpack("<H", data[pos + 6:pos + 8])[0]
                hdr = 8
        else:
            vr = b""
            length = struct.unpack("<I", data[pos + 4:pos + 8])[0]
            hdr = 8
        start = pos + hdr
        name = _TAGS.get((group, elem))
        if length == 0xFFFFFFFF:
            if name == "PixelData":
                raise UnsupportedFormat("encapsulated (compressed) DICOM pixel data needs pydicom")
            # undefined-length sequence: skip to the sequence delimiter
            end = data.find(b"\xfe\xff\xdd\xe0", start)
            if end < 0:
                raise CorruptImage("unterminated DICOM sequence")
            pos = end + 8
            continue
        if start + length > n:
            raise CorruptImage("DICOM element extends beyond end of file")
        val = data[start:start + length]
        if name:
            if name == "PixelData":
                out[name] = val
            elif name in _US:
                out[name] = struct.unpack("<H", val[:2])[0] if len(val) >= 2 else None
            else:
                out[name] = val.decode("ascii", "ignore").strip("\x00 ").strip()
            if name == "TransferSyntaxUID":
                ts = out[name]
        pos = start + length
    out["_transfer_syntax"] = ts
    return out


def _first_float(s, default=None):
    if s is None or s == "":
        return default
    try:
        return float(str(s).split("\\")[0])
    except ValueError:
        return default


def decode_dicom(data: bytes) -> DecodedImage:
    try:
        import pydicom  # optional, preferred when available
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ds.pixel_array.astype(np.float64)
        tags = {"Modality": getattr(ds, "Modality", ""), "RescaleSlope": getattr(ds, "RescaleSlope", None),
                "RescaleIntercept": getattr(ds, "RescaleIntercept", None),
                "WindowCenter": getattr(ds, "WindowCenter", None), "WindowWidth": getattr(ds, "WindowWidth", None),
                "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", ""), "reader": "pydicom"}
    except ImportError:
        t = _parse_dicom(data)
        if t["_transfer_syntax"] not in UNCOMPRESSED_TS:
            raise UnsupportedFormat(f"DICOM transfer syntax {t['_transfer_syntax']} is compressed; install pydicom")
        if "PixelData" not in t or not t.get("Rows") or not t.get("Columns"):
            raise CorruptImage("DICOM file has no readable pixel data")
        bits, signed, spp = t.get("BitsAllocated", 16), t.get("PixelRepresentation", 0), t.get("SamplesPerPixel", 1)
        if spp != 1:
            raise UnsupportedFormat("multi-sample (colour) DICOM is not a CT slice")
        dt = {8: np.int8 if signed else np.uint8, 16: np.int16 if signed else np.uint16,
              32: np.int32 if signed else np.uint32}.get(bits)
        if dt is None:
            raise UnsupportedFormat(f"BitsAllocated={bits} not supported")
        count = t["Rows"] * t["Columns"]
        buf = t["PixelData"]
        if len(buf) < count * np.dtype(dt).itemsize:
            raise CorruptImage("DICOM pixel data is truncated")
        arr = np.frombuffer(buf, dtype=np.dtype(dt).newbyteorder("<"), count=count).reshape(t["Rows"], t["Columns"])
        arr = arr.astype(np.float64)
        tags = {k: t.get(k) for k in ("Modality", "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth",
                                       "PhotometricInterpretation")}
        tags["reader"] = "built-in"
    if arr.ndim == 3:  # multi-frame: middle frame
        arr = arr[arr.shape[0] // 2]
    slope, inter = _first_float(tags.get("RescaleSlope")), _first_float(tags.get("RescaleIntercept"))
    meta = {"format": "dicom", "modality": tags.get("Modality") or "", "reader": tags.get("reader"),
            "rescale_slope": slope, "rescale_intercept": inter,
            "window_center": _first_float(tags.get("WindowCenter")), "window_width": _first_float(tags.get("WindowWidth")),
            "warnings": []}
    if str(tags.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        meta["warnings"].append("MONOCHROME1 (inverted) photometric interpretation")
    if meta["modality"] and meta["modality"].upper() != "CT":
        meta["warnings"].append(f"DICOM modality is {meta['modality']}, not CT")
    if slope is None or inter is None:
        meta["hu_available"] = False
        meta["warnings"].append("RescaleSlope/RescaleIntercept missing: HU not computed")
        g = np.clip((arr - np.percentile(arr, 0.5)) / (np.ptp(arr) + 1e-9) * 255, 0, 255).astype(np.uint8)
        return DecodedImage(g, "display_u8", meta)
    meta["hu_available"] = True
    return DecodedImage((arr * slope + inter).astype(np.float32), "hu", meta)


# --------------------------------------------------------------------------------------------------- entry point
def decode(data: bytes, filename: str | None = None, slice_index: int | None = None) -> DecodedImage:
    if not data:
        raise CorruptImage("empty upload")
    if len(data) > MAX_BYTES:
        raise UnsupportedFormat(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    fmt = detect_format(data, filename)
    if fmt in ("png", "jpeg"):
        return decode_raster(data, fmt)
    if fmt in ("dicom", "dicom_raw"):
        return decode_dicom(data)
    if fmt == "nifti":
        return decode_nifti(data, slice_index)
    raise UnsupportedFormat("unrecognised file type (expected PNG, JPEG, DICOM or NIfTI)")


def write_minimal_dicom(pixels: np.ndarray, slope: float = 1.0, intercept: float = -1024.0, modality: str = "CT") -> bytes:
    """Write an uncompressed explicit-VR little-endian DICOM (used by tests and the demo fixtures)."""
    px = np.asarray(pixels, dtype=np.int16)

    def el(group, elem, vr, value: bytes):
        if len(value) % 2:
            value += b"\x00" if vr in (b"OB", b"UI") else b" "
        if vr in _VR_LONG:
            return struct.pack("<HH", group, elem) + vr + b"\x00\x00" + struct.pack("<I", len(value)) + value
        return struct.pack("<HH", group, elem) + vr + struct.pack("<H", len(value)) + value

    ts = b"1.2.840.10008.1.2.1"
    meta = el(0x0002, 0x0010, b"UI", ts)
    meta = el(0x0002, 0x0000, b"UL", struct.pack("<I", len(meta))) + meta
    body = (el(0x0008, 0x0060, b"CS", modality.encode()) + el(0x0028, 0x0002, b"US", struct.pack("<H", 1)) +
            el(0x0028, 0x0004, b"CS", b"MONOCHROME2") + el(0x0028, 0x0010, b"US", struct.pack("<H", px.shape[0])) +
            el(0x0028, 0x0011, b"US", struct.pack("<H", px.shape[1])) + el(0x0028, 0x0100, b"US", struct.pack("<H", 16)) +
            el(0x0028, 0x0101, b"US", struct.pack("<H", 16)) + el(0x0028, 0x0103, b"US", struct.pack("<H", 1)) +
            el(0x0028, 0x1052, b"DS", f"{intercept:g}".encode()) + el(0x0028, 0x1053, b"DS", f"{slope:g}".encode()) +
            el(0x7FE0, 0x0010, b"OW", px.astype("<i2").tobytes()))
    return b"\x00" * 128 + b"DICM" + meta + body


def write_minimal_nifti(volume: np.ndarray, slope: float = 1.0, inter: float = 0.0) -> bytes:
    """Write a NIfTI-1 single-file volume (int16), x/y/z order; used by tests."""
    v = np.asarray(volume, dtype=np.int16)
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    dims = [v.ndim] + list(v.shape) + [1] * (7 - v.ndim)
    struct.pack_into("<8h", hdr, 40, *dims)
    struct.pack_into("<h", hdr, 70, 4)       # int16
    struct.pack_into("<h", hdr, 72, 16)      # bitpix
    struct.pack_into("<8f", hdr, 76, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<2f", hdr, 112, slope, inter)
    hdr[344:348] = b"n+1\x00"
    return bytes(hdr) + v.tobytes(order="F")
