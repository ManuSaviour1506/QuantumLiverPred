"""V11: the same real CT slice uploaded as JPEG, DICOM and NIfTI goes through the full front end and all four model
families, and gives the same outcome as the PNG.

The DICOM / NIfTI files are written with the project's own minimal writers. The pixel values are Hounsfield units,
reconstructed from the PNG by inverting the training window (level 30, width 150); background is set to air
(-1024 HU). The readers must re-window them to the PNG's grey levels.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
SAMPLE = ROOT / "dataset" / "sample_data" / "lits_sample" / "images" / "test"
pytestmark = pytest.mark.skipif(not (SAMPLE / "ct_lits_4278.png").exists() or
                                not (ROOT / "artifacts" / "weights" / "efficientnet_b0_ra-3dd342df.pth").exists(),
                                reason="needs the bundled LiTS sample and the EfficientNet-B0 weights")


@pytest.fixture(scope="module")
def runner():
    from src.v10.engine.compare import ComparisonRunner
    from src.v10.engine.registry import ModelEngine
    return ComparisonRunner(ModelEngine(ROOT).load())


def _png(name: str) -> np.ndarray:
    return np.array(Image.open(SAMPLE / name).convert("L"))


def _hu(u8: np.ndarray) -> np.ndarray:
    hu = np.round(u8.astype(np.float64) / 255.0 * 150.0 - 45.0).astype(np.int16)
    hu[u8 == 0] = -1024
    return hu


def _encode(fmt: str, u8: np.ndarray) -> tuple[bytes, str]:
    from src.preprocessing import formats
    if fmt == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(u8).save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "slice.jpg"
    if fmt == "dicom":
        return formats.write_minimal_dicom(_hu(u8), slope=1.0, intercept=0.0), "slice.dcm"
    return formats.write_minimal_nifti(_hu(u8).T[:, :, None]), "slice.nii"


@pytest.mark.parametrize("fmt,kind", [("jpeg", "display_u8"), ("dicom", "hu"), ("nifti", "hu")])
def test_formats_decode_to_the_same_canonical_slice(fmt, kind):
    from src.preprocessing import formats
    from src.preprocessing.ct import to_canonical
    u8 = _png("ct_lits_4278.png")
    data, name = _encode(fmt, u8)
    dec = formats.decode(data, name)
    assert dec.kind == kind
    can = to_canonical(dec).image
    assert can.shape == (512, 512)
    assert np.abs(can.astype(int) - u8.astype(int)).mean() < 2.0


@pytest.mark.parametrize("fmt", ["jpeg", "dicom", "nifti"])
def test_formats_run_through_all_four_families(runner, fmt):
    files = ("ct_lits_4277.png", "ct_lits_4278.png", "ct_lits_4279.png")
    (prev, pn), (cur, cn), (nxt, nn) = (_encode(fmt, _png(f)) for f in files)
    prep = runner.prepare(cur, cn, prev, nxt)
    assert not prep.stop_reason, f"{fmt}: withheld ({prep.stop_reason})"
    res = runner.run(prep, runner.plan(prep))
    done = {f: v["n_completed"] for f, v in res["families"].items()}
    assert all(n > 0 for n in done.values()), done
    fn = [m["model_id"] for m in res["models"] if m["status"] == "ok" and m["label"] == "No tumour detected"]
    assert not fn, f"{fmt}: false negatives {fn}"
    assert res["agreement"]["tumour"] >= 10
