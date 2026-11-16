"""Regression suite for the V10 audit cases (engine level, no server). Runs on the bundled real CT sample.

False positives  ct_lits_4353 (heart, no liver in the mask) and ct_lits_4354 (no liver) must never be scored.
False negatives  no servable model may call 'No tumour detected' on the six expert-labelled tumour slices.
OOD              synthetic phantoms are rejected before any model runs.
Availability     missing backbone weights -> dependent models unavailable (never "available" then crashing).
Decision tree    v10.0 (archived) is detected as non-deployable; v10.1 (recalibrated) makes confident calls.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "sample_data" / "lits_sample"
pytestmark = pytest.mark.skipif(not (SAMPLE / "labels" / "slices.csv").exists() or
                                not (ROOT / "artifacts" / "weights" / "efficientnet_b0_ra-3dd342df.pth").exists(),
                                reason="real CT sample or EfficientNet-B0 weights missing")


@pytest.fixture(scope="module")
def engine():
    from src.v10.engine.registry import ModelEngine
    return ModelEngine(ROOT).load()


@pytest.fixture(scope="module")
def runner(engine):
    from src.v10.engine.compare import ComparisonRunner
    return ComparisonRunner(engine)


def _rows():
    return list(csv.DictReader(open(SAMPLE / "labels" / "slices.csv")))


def _inputs(row, rows):
    by = {int(r["upstream_index"]): r for r in rows if r["scan_id"] == row["scan_id"]}
    i = int(row["upstream_index"])
    rd = lambda k: (SAMPLE / by[i + k]["image_path"]).read_bytes() if (i + k) in by else None
    prev, nxt = rd(-1), rd(1)
    if prev is None or nxt is None:
        prev = nxt = None
    return (SAMPLE / row["image_path"]).read_bytes(), row["file_name"], prev, nxt


@pytest.mark.parametrize("name", ["ct_lits_4353.png", "ct_lits_4354.png"])
def test_liver_free_slices_are_never_scored(runner, name):
    rows = _rows()
    row = next(r for r in rows if r["file_name"] == name)
    assert int(row["liver_px_512"]) == 0
    prep = runner.prepare(*_inputs(row, rows))
    assert prep.stop_reason, "liver-free slice reached the tumour models"
    res = runner.run(prep, runner.plan(prep))
    assert res["agreement"]["completed"] == 0
    assert not any(m.get("label") == "Tumour detected" for m in res["models"])


def test_no_false_negative_on_tumour_slices(runner):
    rows = _rows()
    for row in [r for r in rows if r["label_tumour"] == "1"]:
        prep = runner.prepare(*_inputs(row, rows))
        assert not prep.stop_reason, f"{row['file_name']} withheld: {prep.stop_reason}"
        res = runner.run(prep, runner.plan(prep))
        fn = [m["model_id"] for m in res["models"] if m["status"] == "ok" and m["label"] == "No tumour detected"]
        assert not fn, f"{row['file_name']}: false negatives {fn}"
        assert res["agreement"]["tumour"] >= 10


def test_ood_phantom_rejected(runner):
    p = ROOT / "dataset" / "sample_data" / "synthetic" / "images" / "synthetic_phantom_phantom_B_005.png"
    prep = runner.prepare(p.read_bytes(), p.name)
    assert prep.stop_status == "out_of_distribution"
    res = runner.run(prep, runner.plan(prep))
    assert all(m["status"] in ("out_of_distribution", "unavailable") for m in res["models"])


def test_missing_backbone_weights_make_models_unavailable(monkeypatch):
    from src.v10.engine import weights as W
    from src.v10.engine.compare import ComparisonRunner
    from src.v10.engine.interface import ModelUnavailable
    from src.v10.engine.registry import ModelEngine
    real = W.check()
    fake = {k: dict(v, ok=False, reason="missing weight file (simulated)") for k, v in real.items()}
    monkeypatch.setattr(W, "check", lambda full_hash=True: fake)
    eng = ModelEngine(ROOT).load()
    ds = eng.list(include_legacy=True)
    assert not any(d.get("servable") for d in ds if d.get("family") != "segmentation")
    assert {d["availability"] for d in ds if d.get("status") == "missing_dependency"} == {"missing_dependency"}
    assert eng.summary()["ood_detector"] is False
    with pytest.raises(ModelUnavailable):
        ComparisonRunner(eng).prepare((SAMPLE / "images" / "test" / "ct_lits_4278.png").read_bytes(), "x.png")


def test_decision_tree_versions(engine):
    from src.v10.engine import artifacts as A
    from src.v10.engine.v10_models import V10PatchModel
    d = engine.describe("v10_decision_tree_radiomics")
    assert d["servable"] and d["version"] == "v10.1"
    m = engine.get("v10_decision_tree_radiomics")
    lo, hi = m.attainable_probability_range()
    assert hi >= float(m.card["threshold"]) + 0.10 and lo < float(m.card["threshold"]) - 0.10
    arch = ROOT / "artifacts" / "archive" / "v10_decision_tree_radiomics_v10.0"
    if arch.exists():
        old = V10PatchModel(engine, arch, A.load_card(arch))
        old.load_model()
        assert old.deployability() and "non-deployable" in old.deployability()


def test_artifacts_self_contained():
    """Serving must not depend on results/ (audit finding 6) and weights must match the manifest (finding 1)."""
    from src.v10.engine import weights as W
    assert (ROOT / "artifacts" / "manifests" / "v9_cnn_selection.json").is_file()
    man = json.loads((ROOT / "artifacts" / "manifests" / "weights_manifest.json").read_text())
    st = W.check()
    for f in man["files"]:
        p = ROOT / "artifacts" / "weights" / f["file"]
        if p.exists():
            assert st[f["arch"]]["ok"], st[f["arch"]]["reason"]


def test_models_with_missing_weight_files_are_reported_not_hidden(engine):
    """V11 registry fix: every model folder with a model.json is tracked. A V9 model whose model.npz is absent is
    listed as missing_weights (with the reason and the artifact it needs) instead of silently disappearing."""
    listed = {d["id"]: d for d in engine.list(include_legacy=True)}
    for fam in ("classical", "quantum", "hybrid"):
        for mj in sorted((ROOT / "artifacts" / fam).glob("*/model.json")):
            if mj.parent.name == "liver_gate":      # front-end component (engine summary: liver_gate), not a classifier
                continue
            assert mj.parent.name in listed, f"{mj.parent.name} has a model.json but is not in the registry"
    for mid, d in listed.items():
        npz = ROOT / (d.get("artifact_path") or "x") / "model.npz"
        if d.get("source") == "v9" and d.get("artifact_path") and not npz.exists() and d.get("status") != "not_trained":
            assert d["availability"] in ("missing_weights", "missing_dependency", "legacy"), (mid, d["availability"])
            assert not d["servable"]
            assert d.get("reason"), mid
