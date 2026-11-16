"""V10 dataset integrity checks on the repository's manifests, labels and masks (read-only)."""
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
# data/masks (64 x 64 maps derived from the LiTS expert masks) is not redistributed in the release ZIP
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "labels" / "verified_slice_labels.csv").exists() or not any((ROOT / "data" / "masks").glob("*.npz")),
    reason="dataset labels / derived masks not present (the masks are not in the release ZIP; copy data/masks from V9)")


@pytest.fixture(scope="module")
def data():
    from src.v10.data.validation import Data
    return Data(ROOT)


def test_labels(data):
    from src.v10.data.validation import check_labels
    r = check_labels(data)
    assert r["pass"], r
    assert max(max(v.values()) for v in r["label_vs_index_modulo_chi2"].values()) < 5.0   # no index / modulo rule
    assert r["label_changes_along_index"] < 50


def test_masks(data):
    from src.v10.data.validation import check_masks
    assert check_masks(data)["pass"]


def test_patches(data):
    from src.v10.data.validation import check_patches
    r = check_patches(data)
    assert r["pass"], r


def test_gate_patches_known_v9_edge_defect(data):
    """Documents a V9 defect: liver-gate 'other' patches at the left image edge can contain liver (see the Stage 3 report).
    The test asserts the defect is exactly the edge-slicing case, so any other kind of gate-label error would fail it."""
    from src.v10.data.validation import check_gate_patches
    r = check_gate_patches(data)
    assert r["liver_patches_are_task_a_patches"] and r["split_consistent"]
    assert r["other_patches_touching_liver"] == r["of_which_at_image_edge_x0_or_y0"]
    if r["other_patches_touching_liver"]:
        assert r["pass"] is False and r["defect_cause"]


def test_split_has_no_group_leakage(data):
    from src.v10.data.validation import check_split
    r = check_split(data)
    assert r["pass"], r


def test_manifest(data):
    from src.v10.data.validation import check_manifest
    assert check_manifest(data)["pass"]


def test_near_duplicates_stay_inside_groups(data):
    from src.v10.data.validation import check_near_duplicates
    r = check_near_duplicates(data)
    assert r["pass"] is not False
