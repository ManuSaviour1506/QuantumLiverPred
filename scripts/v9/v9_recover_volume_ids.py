"""Recover genuine LiTS volume identifiers for the verified slices (needs the original LiTS segmentation volumes).

  python scripts/v9_recover_volume_ids.py --lits /path/to/LiTS   (folder with segmentation-<k>.nii[.gz]; MSD Task03_Liver
                                                                   labelsTr/liver_<k>.nii.gz also works)

Why masks and not pixels: the windowing / resizing that produced the PNG slices is undocumented, so CT intensities
cannot be matched exactly. The expert label maps can: every verified tumour slice has a 64 x 64 three-class map
(argmax of background / liver / tumour per 8 x 8 cell of the 512 x 512 mask). The same map is computed for every axial
slice of every LiTS segmentation volume, in each of the 8 in-plane orientations (NIfTI axis conventions differ), and a
verified slice is assigned to a volume when its map matches exactly or with >= 0.97 cell agreement and the match is
unique. Tumour runs (contiguous upstream indices) must map to contiguous slices of one volume; conflicting runs are
reported, never forced.

Output: data/labels/verification/recovered_volume_ids.csv and scan_signatures.csv (upstream_index, signature =
'lits-<k>'), which src/data/scans.py uses instead of the reconstructed groups when present. Then rerun
scripts/prepare_data.py and the V9 scripts to rebuild every split by real volume.

Status in this build: the original LiTS volumes were not reachable (no network access), so the script has been run only
on synthetic volumes assembled from the verified maps (tests/test_v9_recovery.py), which checks the matching logic.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import LABELS, MANIFESTS  # noqa: E402
from src.data import labels as L  # noqa: E402
from src.preprocessing.formats import read_nifti_bytes  # noqa: E402


def cell_map(mask512: np.ndarray) -> np.ndarray:
    """(512, 512) labels {0, 1, 2} -> (64, 64) argmax of class counts per 8 x 8 cell (ties -> higher class)."""
    m = mask512.reshape(64, 8, 64, 8)
    counts = np.stack([(m == k).sum((1, 3)) for k in range(3)], -1)
    # argmax with ties resolved towards the higher class, as in the transfer encoding
    return (2 - np.argmax(counts[..., ::-1], -1)).astype(np.uint8)


def orientations(a: np.ndarray):
    for k in range(4):
        r = np.rot90(a, k)
        yield f"rot{k}", r
        yield f"rot{k}_flip", r[:, ::-1]


def volume_maps(seg_xyz: np.ndarray) -> dict:
    """All slice maps of one segmentation volume, for each orientation: {orientation: (Z, 64, 64)}."""
    Z = seg_xyz.shape[2]
    out = {}
    for z in range(Z):
        s = np.asarray(seg_xyz[:, :, z]).astype(np.int16)
        if s.shape != (512, 512):
            from PIL import Image
            s = np.asarray(Image.fromarray(s.astype(np.uint8)).resize((512, 512), Image.NEAREST))
        for name, o in orientations(s):
            out.setdefault(name, []).append(cell_map(np.clip(o, 0, 2)))
    return {k: np.stack(v) for k, v in out.items()}


def match(verified: dict[int, np.ndarray], volumes: dict[str, dict], min_agree: float = 0.97) -> pd.DataFrame:
    rows = []
    keys = [(vid, o) for vid, d in volumes.items() for o in d]
    for u, m in verified.items():
        best = []
        for vid, o in keys:
            V = volumes[vid][o]
            agree = (V == m[None]).mean((1, 2))
            z = int(np.argmax(agree))
            if agree[z] >= min_agree:
                best.append((float(agree[z]), vid, o, z))
        best.sort(reverse=True)
        if not best:
            rows.append({"upstream_index": u, "volume_id": None, "orientation": None, "slice": None, "agreement": None, "status": "no match"})
            continue
        top = best[0]
        unique = len({b[1] for b in best if b[0] >= top[0] - 1e-9}) == 1
        rows.append({"upstream_index": u, "volume_id": top[1], "orientation": top[2], "slice": top[3], "agreement": top[0],
                     "status": "matched" if unique else "ambiguous"})
    return pd.DataFrame(rows)


def check_runs(df: pd.DataFrame) -> list[str]:
    """Contiguous upstream indices should map to one volume and consecutive slices."""
    problems = []
    d = df.dropna(subset=["volume_id"]).sort_values("upstream_index")
    prev = None
    for r in d.itertuples():
        if prev is not None and r.upstream_index == prev.upstream_index + 1 and r.volume_id == prev.volume_id:
            if abs(int(r.slice) - int(prev.slice)) != 1:
                problems.append(f"{prev.upstream_index}->{r.upstream_index}: slices {prev.slice}->{r.slice} not adjacent in {r.volume_id}")
        prev = r
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lits", required=True, help="folder with LiTS segmentation-<k>.nii(.gz) or MSD labelsTr/liver_<k>.nii.gz")
    ap.add_argument("--min-agree", type=float, default=0.97)
    a = ap.parse_args(argv)
    root = Path(a.lits)
    files = sorted([p for p in root.rglob("*.nii*") if re.search(r"(segmentation-\d+|labelsTr/liver_\d+)", str(p))])
    if not files:
        raise SystemExit(f"no LiTS segmentation volumes found under {root}")
    masks, idx = L.load_masks()
    verified = {int(u): m for u, m in zip(idx, masks)}
    volumes = {}
    for p in files:
        k = re.findall(r"(\d+)", p.name)[-1]
        a3, _ = read_nifti_bytes(p.read_bytes())
        volumes[f"lits-{k}"] = volume_maps(a3)
        print(f"{p.name}: {a3.shape}", flush=True)
    df = match(verified, volumes, a.min_agree)
    out = LABELS / "verification"
    df.to_csv(out / "recovered_volume_ids.csv", index=False)
    probs = check_runs(df)
    ok = df[df.status == "matched"]
    print(f"matched {len(ok)}/{len(df)} tumour slices to {ok.volume_id.nunique()} volumes; ambiguous {int((df.status == 'ambiguous').sum())}; "
          f"run conflicts {len(probs)}")
    if probs:
        print("\n".join(probs[:20]))
        raise SystemExit("run conflicts found - signatures not written")
    # propagate to every verified slice: no-tumour slices take the volume of the nearest matched tumour slice
    sl = pd.read_csv(MANIFESTS / "split_v8.csv").sort_values("upstream_index")
    mapped = dict(zip(ok.upstream_index, ok.volume_id))
    us = np.array(sorted(mapped))
    sig = []
    for u in sl.upstream_index:
        j = us[np.argmin(np.abs(us - u))]
        sig.append({"upstream_index": int(u), "signature": mapped[int(j)] if u in mapped else mapped[int(j)] + "?"})
    pd.DataFrame(sig).to_csv(out / "scan_signatures.csv", index=False)
    print(f"wrote {out / 'scan_signatures.csv'} (no-tumour slices attached to the nearest matched slice, marked '?')")


if __name__ == "__main__":
    main()
