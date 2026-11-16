"""Model registry, experiments (background jobs) and analytics endpoints."""
import pytest

from conftest import wait_for
from test_datasets import _validated

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v10.data.synthetic import dataset_zip, make_dataset  # noqa: E402


def test_model_registry(live, researcher, admin):
    models = researcher.get("/api/v1/models").json()
    ids = {m["id"]: m for m in models}
    syn = ids["synthetic_test_logreg"]
    assert syn["servable"] and syn["training_status"].startswith("SYNTHETIC")
    for m in models:
        if m["family"] in ("quantum", "hybrid") and m["servable"]:
            assert m["execution_mode"] == "custom_numpy_simulator"
        if m["status"] in ("not_trained", "failed"):
            assert not m["servable"] and m["reason"]
    if "cnn25d_v9_a4_res_se_focal" in ids:
        m = ids["cnn25d_v9_a4_res_se_focal"]
        assert m["servable"] and m["input_contract"]["requires_context"]
        met = researcher.get("/api/v1/models/cnn25d_v9_a4_res_se_focal/metrics").json()
        ev = met["evaluations"][0]
        assert ev["split"]["test"] == ["G01", "G07", "G08"] and ev["metrics"]["auc"] > 0.5
        assert researcher.get("/api/v1/models/cnn25d_v9_a4_res_se_focal/status").json()["servable"]
    nt = [m for m in models if m["status"] == "not_trained"]
    assert all(m["deployment_status"] == "not_deployed" for m in nt)
    r = researcher.get("/api/v1/models/synthetic_test_logreg/metadata.json")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert researcher.get("/api/v1/models/nope").status_code == 404
    assert researcher.post("/api/v1/models/sync").status_code == 403
    assert admin.post("/api/v1/models/sync").json()["synced"] >= len(models)
    with live["app"].state.db.session() as c:
        assert c.scalar("SELECT COUNT(*) FROM model_versions") >= len(models)


def test_quantum_diagnostics_experiment(researcher, viewer):
    r = researcher.post("/api/v1/experiments", json={"name": "diag", "kind": "quantum_diagnostics",
                                                      "config": {"n_qubits": 3, "layers": 1, "n_init": 20}})
    assert r.status_code == 202, r.text
    eid = r.json()["id"]
    e = wait_for(lambda: (lambda j: j if j["status"] in ("succeeded", "failed") else None)(
        researcher.get(f"/api/v1/experiments/{eid}").json()), 120)
    assert e["status"] == "succeeded", e
    res = researcher.get(f"/api/v1/experiments/{eid}/results").json()["results"]
    assert res["circuit"]["qubits"] == 3 and res["dead_parameters"]["count"] == 0
    assert "hardware" in res["execution_mode"]
    assert researcher.get(f"/api/v1/experiments/{eid}/report").content[:4] == b"%PDF"
    assert researcher.post("/api/v1/experiments", json={"name": "bad", "kind": "quantum_diagnostics",
                                                         "config": {"n_qubits": 30}}).status_code == 422
    assert viewer.post("/api/v1/experiments", json={"name": "v", "kind": "quantum_diagnostics",
                                                     "config": {}}).status_code == 403
    assert viewer.get(f"/api/v1/experiments/{eid}").status_code == 404


def test_nested_evaluation_on_uploaded_dataset(researcher):
    z = dataset_zip(make_dataset(12, 4, seed=5))
    did = researcher.post("/api/v1/datasets/upload", files={"file": ("syn.zip", z, "application/zip")},
                          data={"name": "phantoms for nested"}).json()["id"]
    _validated(researcher, did)
    r = researcher.post("/api/v1/experiments", json={"name": "nested", "kind": "nested_evaluation",
                                                      "dataset_id": did,
                                                      "config": {"models": ["logreg", "qksvm"], "seeds": [0],
                                                                 "inner_k": 2}})
    assert r.status_code == 202, r.text
    eid = r.json()["id"]
    e = wait_for(lambda: (lambda j: j if j["status"] in ("succeeded", "failed") else None)(
        researcher.get(f"/api/v1/experiments/{eid}").json()), 300)
    assert e["status"] == "succeeded", e
    res = researcher.get(f"/api/v1/experiments/{eid}/results").json()["results"]
    assert {r["model"] for r in res["summary"]} == {"logreg", "qksvm"}
    assert res["results"]["units"] == ["SYN01", "SYN02", "SYN03", "SYN04"]


def test_recorded_experiments(researcher):
    exps = researcher.get("/api/v1/experiments").json()
    rec = [e for e in exps if e["source"] == "recorded"]
    assert rec, "recorded experiments (V9 comparison, Stage 2, math validation) should be imported"
    assert all(e["status"] == "succeeded" for e in rec)


def test_noise_study_requires_v10_vqc(live, researcher):
    r = researcher.post("/api/v1/experiments", json={"name": "noise", "kind": "noise_study",
                                                      "config": {"model_id": "v10_vqc_radiomics",
                                                                 "p2_levels": [0.0, 0.01], "n_samples": 60}})
    eid = r.json()["id"]
    e = wait_for(lambda: (lambda j: j if j["status"] in ("succeeded", "failed") else None)(
        researcher.get(f"/api/v1/experiments/{eid}").json()), 300)
    vqc_dir = Path(live["settings"].project_root) / "artifacts" / "v10" / "v10_vqc_radiomics"
    if vqc_dir.exists():
        assert e["status"] == "succeeded", e
        rows = researcher.get(f"/api/v1/experiments/{eid}/results").json()["results"]["summary"]
        assert rows[0]["p2"] == 0.0 and abs(rows[0]["mean_abs_logit_change"]) < 1e-9
    else:
        assert e["status"] == "failed" and "artifact" in e["error"]


def test_analytics(live, researcher, admin):
    ov = researcher.get("/api/v1/analytics/overview").json()
    assert ov["users_total"] is None and ov["models_servable"] >= 1
    ova = admin.get("/api/v1/analytics/overview").json()
    assert ova["users_total"] >= 3
    cmp_ = researcher.get("/api/v1/analytics/model-comparison").json()
    for row in cmp_["rows"]:
        assert row["evaluation"] and row["evaluation_unit"] and row["roc_auc"] is not None
    assert all("synthetic" not in (r["evaluation"] or "").lower() or r["model_id"] == "synthetic_test_logreg"
               for r in cmp_["rows"])
    assert researcher.get("/api/v1/analytics/model-comparison.csv").text.startswith("model_id,")
    assert researcher.get("/api/v1/analytics/model-comparison.pdf").content[:4] == b"%PDF"
    dist = researcher.get("/api/v1/analytics/prediction-distribution").json()
    assert dist["scope"] == "mine" and len(dist["probability_histogram"]) == 10
    q = researcher.get("/api/v1/research/quantum").json()
    assert all(m["hardware"] is False for m in q["models"])
    s = researcher.get("/api/v1/research/summary").json()
    assert "No quantum hardware" in s["execution_note"]
