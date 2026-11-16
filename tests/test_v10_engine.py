"""Model engine tests: common interface, persistence round trips, integrity refusal, synthetic gating, V9 adapters on
real patches, NumPy deep-learning training smoke runs (tiny subsets, not results)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from src.v10.engine import artifacts as A  # noqa: E402
from src.v10.engine.interface import ModelInput, ModelUnavailable  # noqa: E402
from src.v10.engine.registry import ModelEngine  # noqa: E402
from src.v10.evaluation import specs as S  # noqa: E402
from src.v10.evaluation.nested import run_nested  # noqa: E402
from src.v10.training.final import train_final  # noqa: E402

SAMPLE = ROOT / "dataset" / "sample_data" / "lits_sample"
HAVE_V9 = (ROOT / "artifacts" / "classical" / "liver_gate" / "model.json").exists()


def _toy(n=160, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(int)
    units = np.array([f"U{i % 4}" for i in range(n)])
    return X, y, units


def _ds():
    return {"dataset_version": "toy@test", "license": "test"}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    r = tmp_path_factory.mktemp("root")
    (r / "artifacts" / "v10").mkdir(parents=True)
    return r


@pytest.mark.parametrize("key", ["logreg", "vqc", "qksvm", "hybrid_no_fusion", "late_fusion"])
def test_train_save_load_round_trip(tmp_root, key, monkeypatch):
    X, y, units = _toy()
    spec = {s.key: s for s in S.all_specs()}[key]
    monkeypatch.setattr(S, "EPOCHS", 3, raising=False)
    if key in ("vqc", "hybrid_no_fusion", "late_fusion"):
        spec.grid = spec.grid[:1]
    d = train_final(tmp_root, spec, X, y, units, _ds(), "radiomics", None, seed=0, model_id=f"t_{key}")
    card = A.load_card(d)
    A.verify(d, card)
    assert card["training_data"] == "real" and card["preprocessing_signature"]
    from src.v10.engine.v10_models import V10PatchModel

    class E:  # minimal engine stand-in (no V9 front end needed for feature-level calls)
        allow_synthetic, v9 = False, None
    card["feature_dim"] = X.shape[1]
    m = V10PatchModel(E(), d, card)
    m.load_model()
    p1, p2 = m.predict_proba(X[:20]), m.predict_proba(X[:20])
    assert p1.shape == (20,) and np.all((p1 >= 0) & (p1 <= 1)) and np.array_equal(p1, p2)   # deterministic


def test_tampered_artifact_is_refused(tmp_root):
    X, y, units = _toy()
    spec = {s.key: s for s in S.all_specs()}["logreg"]
    d = train_final(tmp_root, spec, X, y, units, _ds(), "radiomics", None, model_id="t_tamper")
    with open(d / "model.npz", "ab") as f:
        f.write(b"x")
    with pytest.raises(A.ArtifactError):
        A.verify(d, A.load_card(d))


def test_preprocessing_signature_mismatch_is_refused(tmp_root):
    X, y, units = _toy()
    spec = {s.key: s for s in S.all_specs()}["logreg"]
    d = train_final(tmp_root, spec, X, y, units, _ds(), "radiomics", None, model_id="t_sig")
    card = json.loads((d / "model_card.json").read_text())
    card["preprocessing_signature"] = "0" * 20
    (d / "model_card.json").write_text(json.dumps(card))
    from src.v10.engine.v10_models import V10PatchModel

    class E:
        allow_synthetic, v9 = False, None
    with pytest.raises(ModelUnavailable):
        V10PatchModel(E(), d, A.load_card(d)).load_model()


def test_synthetic_models_are_gated(tmp_path):
    from src.v10.data.synthetic import train_synthetic_model
    (tmp_path / "artifacts" / "v10").mkdir(parents=True)
    train_synthetic_model(tmp_path / "artifacts" / "v10")
    eng = ModelEngine(tmp_path, allow_synthetic=False).load()
    d = eng.describe("synthetic_test_logreg")
    assert d["servable"] is False and d["status"] == "synthetic_only"
    with pytest.raises(ModelUnavailable):
        eng.get("synthetic_test_logreg")
    eng2 = ModelEngine(tmp_path, allow_synthetic=True).load()
    from src.v10.data.synthetic import phantom, png_bytes
    out = eng2.predict("synthetic_test_logreg", ModelInput(png_bytes(phantom(np.random.default_rng(1), 1)), "x.png"))
    assert out.status == "ok" and 0 <= out.probability <= 1 and "SYNTHETIC" in out.notes[0]


def test_nested_runner_never_leaks_units():
    X, y, units = _toy(120)
    spec = {s.key: s for s in S.all_specs()}["logreg"]
    res = run_nested(X, y, units, [spec], [0], inner_k=2, log=lambda m: None)
    assert sorted(res["models"]["logreg"]["per_unit"]) == ["U0", "U1", "U2", "U3"]
    assert 0.5 < res["models"]["logreg"]["primary_mean_within_unit_auc"] <= 1


@pytest.mark.skipif(not HAVE_V9, reason="V9 artifacts not imported")
def test_v9_models_follow_the_interface_on_real_slices():
    import json
    man = ROOT / "artifacts" / "LARGE_FILES_MANIFEST.json"
    missing = [f["path"] for f in json.loads(man.read_text())["files"] if not (ROOT / f["path"]).exists()] \
        if man.exists() else []
    # V11: with the large V9 files absent the test still runs on every V9 model that can load, and checks that the
    # models needing a missing file are reported as unavailable (never served, never crashing).
    eng = ModelEngine(ROOT).load()
    servable = [d for d in eng.list() if d["servable"] and d["source"] == "v9"]
    assert len(servable) >= (20 if not missing else 15)
    if missing:
        by = {d["id"]: d for d in eng.list(include_legacy=True)}
        for f in missing:
            mid = f.split("/")[-2]
            if f.endswith("model.npz"):
                assert mid in by, f"{mid} missing from the registry"
                assert not by[mid]["servable"] and by[mid]["availability"] == "missing_weights", (mid, by[mid])
    for d in servable:
        md = eng.get(d["id"]).get_metadata()
        assert md["execution_mode"] in ("classical_cpu_numpy", "custom_numpy_simulator")
        if d["family"] in ("quantum", "hybrid"):
            assert md["execution_mode"] == "custom_numpy_simulator"
    if not (SAMPLE / "images" / "test" / "ct_lits_4278.png").exists():
        pytest.skip("real CT sample missing")
    b = lambda n: (SAMPLE / "images" / "test" / n).read_bytes()
    inp = ModelInput(b("ct_lits_4278.png"), "x.png", b("ct_lits_4277.png"), b("ct_lits_4279.png"))
    backbone = "densenet121_ft_v9" if missing else "efficientnet_b0_ft_v9"
    for mid in ("logreg_radiomics_v9", "vqc_v9_cnn", "hybrid_v9_cnn", backbone):
        out = eng.predict(mid, inp)
        assert out.status == "ok" and 0 <= out.probability <= 1
        assert out.details["ood"]["status"] == "in_distribution"
        assert len(out.details["patch_probabilities"]) == out.details["n_liver_patches"] > 0
        again = eng.predict(mid, inp)
        assert again.probability == out.probability                                   # deterministic


@pytest.mark.skipif(not HAVE_V9, reason="V9 artifacts (CNN recipe, ImageNet weights) not imported")
def test_numpy_deep_learning_training_smoke(tmp_path):
    """Real forward/backward training on a tiny random tensor set: verifies the code path, not performance."""
    from src.v10.deep_learning.numpy_training import train_backbone, train_numpy_cnn
    rng = np.random.default_rng(0)
    n = 48
    ds = {"stacks25d": rng.integers(0, 255, (n, 3, 96, 96), dtype=np.uint8),
          "patches2d": rng.integers(0, 255, (n, 96, 96), dtype=np.uint8),
          "y": np.array([0, 1] * (n // 2)), "units": np.array(["A", "B", "C"] * (n // 3)),
          "dataset_version": "random-smoke", "training_data": "synthetic"}
    (tmp_path / "artifacts").mkdir()
    shutil.copytree(ROOT / "artifacts" / "classical" / "cnn25d_v9_a4_res_se_focal",
                    tmp_path / "artifacts" / "classical" / "cnn25d_v9_a4_res_se_focal")
    d = train_numpy_cnn(tmp_path, ds, "smoke_cnn", ["B"], ["C"], max_epochs=1, subset=16, log=lambda *a: None)
    card = A.load_card(d)
    A.verify(d, card)
    assert card["kind"] == "v10_patch_numpy_cnn" and card["requires_context"]
    d2 = train_backbone(tmp_path, ds, "efficientnet_b0", "smoke_eff", ["B"], ["C"], max_epochs=1, subset=12,
                        n_views=1, log=lambda *a: None)
    A.verify(d2, A.load_card(d2))
    from src.v10.engine.v10_models import V10PatchModel

    class E:
        allow_synthetic, v9 = True, None
    for dd in (d, d2):
        m = V10PatchModel(E(), dd, A.load_card(dd))
        m.load_model()


def test_pytorch_path_reports_availability():
    from src.v10.deep_learning import torch_training as T
    if not T.available():
        pytest.skip("PyTorch / torchvision not installed in this environment (training path not executed)")
    net, _ = T.build("cnn25d", pretrained=False)
    assert net is not None
