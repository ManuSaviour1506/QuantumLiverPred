"""ImageNet backbone weights: manifest, verification and download.

The OOD detector and liver gate run a frozen EfficientNet-B0, so without its weights NO patch model can produce a
prediction. Older releases loaded the weights lazily at the first prediction, so the engine listed models as available
and then every request crashed (V10 audit finding 1). The engine now verifies the files at start-up and marks every
dependent model unavailable with the reason.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "artifacts" / "manifests" / "weights_manifest.json"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_CACHE: dict[str, dict] = {}


def check(full_hash: bool = True) -> dict[str, dict]:
    """arch -> {"ok": bool, "reason": str | None, "file": str}. Hashes are computed once per process (file size and
    mtime key the cache)."""
    out = {}
    for f in manifest()["files"]:
        p = ROOT / "artifacts" / "weights" / f["file"]
        rec = {"file": f"artifacts/weights/{f['file']}", "ok": False, "reason": None, "required_by": f["required_by"]}
        if not p.is_file():
            rec["reason"] = f"missing weight file artifacts/weights/{f['file']} (run: python scripts/fetch_weights.py)"
        elif p.stat().st_size != f["bytes"]:
            rec["reason"] = f"artifacts/weights/{f['file']} has {p.stat().st_size} bytes, expected {f['bytes']}"
        else:
            key = f"{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}"
            if full_hash:
                if key not in _CACHE:
                    _CACHE[key] = {"sha": sha256(p)}
                if _CACHE[key]["sha"] != f["sha256"]:
                    rec["reason"] = f"SHA-256 of artifacts/weights/{f['file']} does not match the manifest"
                else:
                    rec["ok"] = True
            else:
                rec["ok"] = True
        out[f["arch"]] = rec
    return out


def download(arch: str | None = None, log=print) -> dict[str, dict]:
    """Download missing / wrong files from their official release URLs and verify them."""
    import urllib.request
    wdir = ROOT / "artifacts" / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    status = check()
    for f in manifest()["files"]:
        if (arch and f["arch"] != arch) or status[f["arch"]]["ok"]:
            continue
        tmp = wdir / (f["file"] + ".part")
        log(f"downloading {f['file']} ({f['bytes'] / 1e6:.0f} MB) from {f['url']}")
        with urllib.request.urlopen(f["url"], timeout=120) as r, open(tmp, "wb") as out:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                out.write(b)
        got = sha256(tmp)
        if got != f["sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{f['file']}: SHA-256 mismatch after download ({got})")
        tmp.replace(wdir / f["file"])
        log(f"verified {f['file']}")
    return check()
