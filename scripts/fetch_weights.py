"""Download (if missing) and verify the ImageNet backbone weights listed in artifacts/manifests/weights_manifest.json.

    python scripts/fetch_weights.py            # download missing files, verify all
    python scripts/fetch_weights.py --check    # verify only; exit code 1 if anything is missing or wrong
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v10.engine import weights  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arch", choices=["efficientnet_b0", "densenet121", "resnet50"])
    a = ap.parse_args()
    st = weights.check() if a.check else weights.download(a.arch)
    bad = 0
    for arch, r in st.items():
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {arch:16s} {r['file']}" + ("" if r["ok"] else f"  -> {r['reason']}"))
        bad += not r["ok"] and (a.arch in (None, arch))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
