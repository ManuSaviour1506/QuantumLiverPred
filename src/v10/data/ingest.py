"""Dataset ingestion and validation (NumPy + Pillow + the V9 format readers only, so it also runs on machines without
SciPy or scikit-learn).

Manifest columns (CSV): image_path (required), label (required for training), patient_id, scan_id, volume_id,
slice_id, modality, mask_path, split, data_source. Paths are relative to the dataset root and must stay inside it.

Checks: file existence and readability, corrupted images, image dimensions, intensity statistics, missing / empty
labels, label values, duplicate paths, exact duplicates (SHA-256) and near duplicates (8x8 average hash), identifier
format and availability (patient / scan / volume / slice), slice ids unique within their volume / scan, leakage of a
patient / scan / volume across splits, mask existence, size and values, class distribution (overall, per split, per
group) and image files present on disk but missing from the manifest. Nothing is ever relabelled or invented.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

COLUMNS = ["image_path", "label", "patient_id", "scan_id", "volume_id", "slice_id", "modality", "mask_path", "split",
           "data_source"]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".dcm", ".nii", ".nii.gz")
ID_RE = re.compile(r"^[A-Za-z0-9._:+\-()/ ]{1,64}$")
MAX_ZIP_FILES = 20000
MAX_ZIP_UNCOMPRESSED = 4 * 1024 ** 3


class IngestError(ValueError):
    pass


def safe_rel(path: str) -> str:
    p = PurePosixPath(str(path).replace("\\", "/").strip())
    if p.is_absolute() or ".." in p.parts or not str(p) or str(p).startswith("~"):
        raise IngestError(f"unsafe path {path!r}")
    return str(p)


def safe_extract(zip_path: Path, dest: Path) -> list[str]:
    """Extract a zip without zip-slip, symlinks, absurd sizes or file counts."""
    dest = Path(dest).resolve()
    out = []
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > MAX_ZIP_FILES:
            raise IngestError(f"archive has more than {MAX_ZIP_FILES} files")
        if sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
            raise IngestError("archive expands beyond 4 GB")
        for i in infos:
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise IngestError(f"symbolic links are not allowed ({i.filename})")
            name = safe_rel(i.filename)
            if name.startswith("__MACOSX/") or PurePosixPath(name).name.startswith("._"):
                continue
            target = (dest / name).resolve()
            if dest not in target.parents:
                raise IngestError(f"path escapes the dataset folder ({i.filename})")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            out.append(name)
    return out


def find_manifest(root: Path) -> Path | None:
    for cand in ("manifest.csv", "manifests/manifest.csv"):
        if (root / cand).is_file():
            return root / cand
    hits = sorted(root.rglob("manifest.csv"))
    return hits[0] if hits else None


def read_manifest(path: Path) -> tuple[list[dict], list[str], list[str]]:
    errors = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        if "image_path" not in cols:
            raise IngestError("manifest must have an 'image_path' column")
        rows = []
        for i, r in enumerate(reader):
            row = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            try:
                row["image_path"] = safe_rel(row.get("image_path", ""))
            except IngestError as e:
                errors.append(f"row {i + 2}: {e}")
                continue
            rows.append(row)
    if not rows:
        raise IngestError("manifest has no usable rows")
    return rows, cols, errors


def lits_verified_manifest(root: Path) -> list[dict]:
    """Manifest for the verified LiTS-derived folder from its own labels/slices.csv (labels come from the expert
    masks; patient and volume identifiers do not exist in this source and are left empty)."""
    rows = []
    with open(Path(root) / "labels" / "slices.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"image_path": r["image_path"], "label": "tumour" if r["label_tumour"] == "1" else "no_tumour",
                         "patient_id": "", "scan_id": r["group_id"], "volume_id": "", "slice_id": r["upstream_index"],
                         "modality": "CT", "mask_path": "", "split": r["split"],
                         "data_source": "LiTS via seongun/liver-tumor-classification (CC BY-NC-SA 4.0)",
                         "_expected_sha12": r.get("pixel_sha12", "")})
    return rows


def _ahash(img: np.ndarray) -> str:
    from PIL import Image
    a = np.asarray(Image.fromarray(img).resize((8, 8), Image.BILINEAR), float)
    bits = (a > a.mean()).ravel()
    return "".join("1" if b else "0" for b in bits)


def read_image(path: Path) -> np.ndarray:
    from src.preprocessing import formats
    data = Path(path).read_bytes()
    dec = formats.decode(data, Path(path).name)
    px = np.asarray(dec.pixels)
    if px.ndim == 3:
        px = px[..., :3].mean(-1)
    return px


def validate(rows: list[dict], root: Path, check_images: bool = True, label_set: set[str] | None = None,
             progress=None) -> tuple[dict, list[dict]]:
    root = Path(root)
    per_file = []
    sha_groups, ahash_groups = defaultdict(list), defaultdict(list)
    sizes, means, stds = Counter(), [], []
    issues_count = Counter()
    seen_paths = Counter(r["image_path"] for r in rows)
    for i, r in enumerate(rows):
        issues = []
        status = "ok"
        p = root / r["image_path"]
        info = {"sha256": None, "width": None, "height": None}
        if seen_paths[r["image_path"]] > 1:
            issues.append("duplicate manifest path")
        lab = r.get("label", "")
        if not lab:
            issues.append("missing label")
        elif label_set and lab not in label_set:
            issues.append(f"label not in the allowed set: {lab}")
        for idc in ("patient_id", "scan_id", "volume_id", "slice_id"):
            v = r.get(idc, "")
            if v and not ID_RE.match(v):
                issues.append(f"invalid {idc} format")
        if not p.is_file():
            status = "missing"
            issues.append("file not found")
        elif not r["image_path"].lower().endswith(IMAGE_EXT):
            status = "unsupported"
            issues.append("unsupported file extension")
        elif check_images:
            try:
                raw = p.read_bytes()
                info["sha256"] = hashlib.sha256(raw).hexdigest()
                img = read_image(p)
                g = np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
                info["height"], info["width"] = int(img.shape[0]), int(img.shape[1])
                sizes[f"{img.shape[1]}x{img.shape[0]}"] += 1
                means.append(float(img.mean()))
                stds.append(float(img.std()))
                if float(img.std()) < 1e-6:
                    issues.append("constant image (no signal)")
                    status = "warning"
                sha_groups[info["sha256"]].append(r["image_path"])
                ahash_groups[_ahash(g)].append(r["image_path"])
                exp = r.get("_expected_sha12")
                if exp:
                    pix = hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()[:12]
                    info["pixel_sha12_matches"] = (pix == exp)
            except Exception as e:  # corrupted or unreadable
                status = "corrupt"
                issues.append(f"unreadable: {type(e).__name__}")
        mp = r.get("mask_path", "")
        if mp:
            mpath = root / mp
            if not mpath.is_file():
                issues.append("mask file not found")
            elif check_images and info["width"]:
                try:
                    m = read_image(mpath)
                    if m.shape[:2] != (info["height"], info["width"]):
                        issues.append("mask size differs from image size")
                    vals = np.unique(m)
                    if len(vals) > 16:
                        issues.append("mask has more than 16 distinct values (not a label map)")
                except Exception as e:
                    issues.append(f"mask unreadable: {type(e).__name__}")
        if issues and status == "ok":
            status = "warning"
        for it in issues:
            issues_count[it.split(":")[0]] += 1
        per_file.append({"row_index": i, "file_status": status, "issues": issues, **info})
        if progress and i % 200 == 0:
            progress(i, len(rows))
    exact_dups = [v for v in sha_groups.values() if len(v) > 1]
    near_dups = [v for v in ahash_groups.values() if len(v) > 1]
    report = summarise(rows, per_file, sizes, means, stds, issues_count, exact_dups, near_dups, root)
    return report, per_file


def _leakage(rows: list[dict], key: str) -> dict:
    by = defaultdict(set)
    for r in rows:
        if r.get(key) and r.get("split"):
            by[r[key]].add(r["split"])
    leaked = {k: sorted(v) for k, v in by.items() if len(v) > 1}
    return {"checked": bool(by), "ids_in_multiple_splits": len(leaked), "examples": dict(list(leaked.items())[:10])}


def summarise(rows, per_file, sizes, means, stds, issues_count, exact_dups, near_dups, root) -> dict:
    n = len(rows)
    status = Counter(f["file_status"] for f in per_file)
    labels = Counter(r.get("label") or "(missing)" for r in rows)
    splits = Counter(r.get("split") or "(none)" for r in rows)
    groups = Counter(r.get("scan_id") or "(none)" for r in rows)
    by_split = defaultdict(Counter)
    for r in rows:
        by_split[r.get("split") or "(none)"][r.get("label") or "(missing)"] += 1
    ids = {k: sum(1 for r in rows if r.get(k)) for k in ("patient_id", "scan_id", "volume_id", "slice_id")}
    slice_dupe = 0
    seen = defaultdict(set)
    for r in rows:
        unit = r.get("volume_id") or r.get("scan_id")
        if unit and r.get("slice_id"):
            if r["slice_id"] in seen[unit]:
                slice_dupe += 1
            seen[unit].add(r["slice_id"])
    listed = {r["image_path"] for r in rows}
    unlisted = []
    if Path(root).is_dir():
        for p in Path(root).rglob("*"):
            if p.is_file() and str(p).lower().endswith(IMAGE_EXT):
                rel = str(p.relative_to(root)).replace("\\", "/")
                if rel not in listed and not rel.startswith("masks/"):
                    unlisted.append(rel)
    sha_checks = [f.get("pixel_sha12_matches") for f in per_file if "pixel_sha12_matches" in f]
    warnings = []
    if ids["patient_id"] == 0:
        warnings.append("No patient identifiers: patient-level evaluation and leakage control are not possible; "
                        "evaluate by scan / group only and do not describe results as patient-level.")
    if status.get("missing"):
        warnings.append(f"{status['missing']} manifest files are missing on disk.")
    if labels.get("(missing)"):
        warnings.append(f"{labels['(missing)']} rows have no label.")
    if exact_dups:
        warnings.append(f"{sum(len(g) for g in exact_dups)} files are byte-identical duplicates ({len(exact_dups)} groups).")
    ok = status.get("missing", 0) == 0 and status.get("corrupt", 0) == 0 and labels.get("(missing)", 0) == 0
    return {
        "passed": bool(ok), "rows": n, "file_status": dict(status), "issue_counts": dict(issues_count),
        "label_distribution": dict(labels), "split_distribution": dict(splits), "group_distribution": dict(groups),
        "label_by_split": {k: dict(v) for k, v in by_split.items()},
        "identifier_availability": ids, "duplicate_slice_ids_within_unit": slice_dupe,
        "leakage": {k: _leakage(rows, k) for k in ("patient_id", "scan_id", "volume_id")},
        "exact_duplicate_groups": len(exact_dups), "exact_duplicate_examples": exact_dups[:5],
        "near_duplicate_groups": len(near_dups), "near_duplicate_examples": near_dups[:5],
        "image_sizes": dict(sizes.most_common(10)),
        "intensity": {"mean_of_means": float(np.mean(means)) if means else None,
                      "sd_of_means": float(np.std(means)) if means else None,
                      "mean_of_sds": float(np.mean(stds)) if stds else None},
        "unlisted_images": len(unlisted), "unlisted_examples": unlisted[:10],
        "pixel_hash_verification": {"checked": len(sha_checks), "matched": int(sum(1 for v in sha_checks if v))}
        if sha_checks else None,
        "warnings": warnings,
    }


def content_hash(rows: list[dict], per_file: list[dict] | None = None) -> str:
    h = hashlib.sha256()
    for i, r in enumerate(rows):
        h.update("|".join(r.get(c, "") for c in COLUMNS).encode())
        if per_file and per_file[i].get("sha256"):
            h.update(per_file[i]["sha256"].encode())
    return h.hexdigest()


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def manifest_csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()
