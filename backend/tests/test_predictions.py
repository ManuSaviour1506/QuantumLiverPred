"""Prediction workflow over HTTP: upload validation, queued job, real model inference, history, artifacts, report,
export, authorization and deletion. Real-CT tests use the 8-slice sample (or LTI_TEST_DATASET_DIR)."""
import io

import numpy as np
import pytest
from PIL import Image

from conftest import poll_prediction

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v10.data.synthetic import phantom, png_bytes  # noqa: E402


def _png(seed=0, cls=1):
    return png_bytes(phantom(np.random.default_rng(seed), cls))


def _models(client, servable=True):
    return {m["id"]: m for m in client.get(f"/api/v1/models?servable_only={'true' if servable else 'false'}").json()}


def test_upload_validation(researcher):
    m = "synthetic_test_logreg"
    r = researcher.post("/api/v1/predictions", files={"file": ("notes.txt", b"hello world", "text/plain")},
                        data={"model_id": m})
    assert r.status_code == 415
    bad_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    r = researcher.post("/api/v1/predictions", files={"file": ("x.png", bad_png, "image/png")}, data={"model_id": m})
    assert r.status_code == 422 and r.json()["error"]["code"] == "corrupt_image"
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (6 * 1024 * 1024)
    r = researcher.post("/api/v1/predictions", files={"file": ("big.png", big, "image/png")}, data={"model_id": m})
    assert r.status_code == 413
    r = researcher.post("/api/v1/predictions", files={"file": ("x.png", _png(), "image/png")},
                        data={"model_id": "no_such_model"})
    assert r.status_code == 404
    r = researcher.post("/api/v1/predictions", data={"model_id": m})
    assert r.status_code == 415  # not multipart


def test_viewer_cannot_predict(viewer):
    r = viewer.post("/api/v1/predictions", files={"file": ("x.png", _png(), "image/png")},
                    data={"model_id": "synthetic_test_logreg"})
    assert r.status_code == 403


def test_synthetic_model_end_to_end(live, researcher, admin):
    """Full flow with the software-test model (independent of real data)."""
    r = researcher.post("/api/v1/predictions", files={"file": ("../../etc/passwd.png", _png(3, 1), "image/png")},
                        data={"model_id": "synthetic_test_logreg"})
    assert r.status_code == 202, r.text
    pid = r.json()["id"]
    p = poll_prediction(researcher, pid)
    assert p["status"] == "succeeded", p
    assert p["label"] in ("Synthetic class A", "Synthetic class B")
    assert 0.0 <= p["probability"] <= 1.0
    assert p["input_name"] == "passwd.png"            # path components stripped; never used for storage
    assert p["disclaimer"].startswith("This platform is intended for academic research")
    storage = Path(live["settings"].storage_dir).resolve()
    assert any(storage.rglob(f"{pid}/input.png"))
    assert not Path("/etc/passwd.png").exists()
    # another user cannot see it; admin can
    other = __import__("conftest").make_user(live, "other@example.org")
    assert other.get(f"/api/v1/predictions/{pid}").status_code == 404
    assert other.delete(f"/api/v1/predictions/{pid}").status_code == 404
    assert admin.get(f"/api/v1/predictions/{pid}").status_code == 200
    # history, search, export
    page = researcher.get("/api/v1/predictions?q=passwd&sort=-created_at").json()
    assert any(i["id"] == pid for i in page["items"])
    assert researcher.get("/api/v1/predictions?sort=bogus").status_code == 422
    csv_text = researcher.get("/api/v1/predictions/export.csv").text
    assert pid in csv_text and "academic research" in csv_text
    pdf = researcher.get(f"/api/v1/predictions/{pid}/report")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    img = researcher.get(f"/api/v1/predictions/{pid}/artifacts/preview")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert other.get(f"/api/v1/predictions/{pid}/artifacts/preview").status_code == 404
    # delete removes record and files
    assert researcher.delete(f"/api/v1/predictions/{pid}").status_code == 204
    assert researcher.get(f"/api/v1/predictions/{pid}").status_code == 404
    assert not any(storage.rglob(f"{pid}/input.png"))


def test_real_ct_v9_cnn_with_context(live, researcher):
    ds = live["dataset"]
    models = _models(researcher)
    if ds is None or "cnn25d_v9_a4_res_se_focal" not in models:
        pytest.skip("real CT sample or V9 artifacts not available")
    d = ds / "images" / "test"
    r = researcher.post("/api/v1/predictions", data={"model_id": "cnn25d_v9_a4_res_se_focal"},
                        files={"file": ("ct_lits_4278.png", (d / "ct_lits_4278.png").read_bytes(), "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "context_required"
    r = researcher.post("/api/v1/predictions", data={"model_id": "cnn25d_v9_a4_res_se_focal", "explain": "true"},
                        files={"file": ("ct_lits_4278.png", (d / "ct_lits_4278.png").read_bytes(), "image/png"),
                               "prev_file": ("p.png", (d / "ct_lits_4277.png").read_bytes(), "image/png"),
                               "next_file": ("n.png", (d / "ct_lits_4279.png").read_bytes(), "image/png")})
    assert r.status_code == 202, r.text
    p = poll_prediction(researcher, r.json()["id"])
    assert p["status"] == "succeeded", p
    res = p["result"]
    assert res["details"]["ood"]["status"] == "in_distribution"
    assert p["label"] in ("Tumour detected", "No tumour detected", "Uncertain / unable to classify")
    assert 0 <= p["probability"] <= 1 and p["model_version"] == "v9"
    assert p["execution_mode"] == "classical_cpu_numpy"
    kinds = {a["kind"] for a in p["artifacts"]}
    assert {"preview", "patch_map"} <= kinds
    assert "gradcam" in kinds and res["explanation"]["gradcam"]["status"] == "ok"
    assert all(0 <= x["p"] <= 1 for x in res["details"]["patch_probabilities"])


def test_real_ct_quantum_model_labelled_as_simulator(live, researcher):
    ds = live["dataset"]
    models = _models(researcher)
    qid = next((m for m in ("v10_vqc_radiomics", "vqc_v9_cnn") if m in models), None)
    if ds is None or qid is None:
        pytest.skip("real CT sample or quantum artifacts not available")
    d = ds / "images" / "val"
    files = {"file": ("ct.png", (d / "ct_lits_4650.png").read_bytes(), "image/png"),
             "prev_file": ("p.png", (d / "ct_lits_4649.png").read_bytes(), "image/png"),
             "next_file": ("n.png", (d / "ct_lits_4651.png").read_bytes(), "image/png")}
    r = researcher.post("/api/v1/predictions", data={"model_id": qid, "explain": "true"}, files=files)
    assert r.status_code == 202, r.text
    p = poll_prediction(researcher, r.json()["id"])
    assert p["status"] == "succeeded", p
    assert p["execution_mode"] == "custom_numpy_simulator"
    assert "hardware" not in (p["execution_mode"] or "")


def test_out_of_distribution_input_is_withheld(live, researcher):
    models = _models(researcher)
    if "logreg_radiomics_v9" not in models:
        pytest.skip("V9 artifacts not available")
    r = researcher.post("/api/v1/predictions", data={"model_id": "logreg_radiomics_v9"},
                        files={"file": ("phantom.png", _png(7, 0), "image/png")})
    p = poll_prediction(researcher, r.json()["id"])
    assert p["status"] == "succeeded"
    assert p["probability"] is None and p["label"] == "Uncertain / unable to classify"
    assert p["result"]["details"]["ood"]["status"] == "out_of_distribution"


V10_FAMILY_MODELS = ["v10_logreg_radiomics", "v10_qksvm_radiomics", "v10_hybrid_no_fusion_radiomics",
                     "v10_late_fusion_radiomics", "v10_hybrid_twin_radiomics"]


@pytest.mark.parametrize("model_id", V10_FAMILY_MODELS)
def test_real_ct_v10_models(live, researcher, model_id):
    """Every V10 family serves a real prediction on a real CT tumour slice, with its provenance recorded."""
    ds = live["dataset"]
    models = _models(researcher)
    if ds is None or model_id not in models:
        pytest.skip(f"real CT sample or {model_id} artifact not available")
    m = models[model_id]
    assert m["training_status"] == "TRAINED" and m["version"] == "v10.0"
    img = (ds / "images" / "test" / "ct_lits_4278.png").read_bytes()
    r = researcher.post("/api/v1/predictions", data={"model_id": model_id, "explain": "true"},
                        files={"file": ("slice.png", img, "image/png")})
    assert r.status_code == 202, r.text
    p = poll_prediction(researcher, r.json()["id"])
    assert p["status"] == "succeeded", p
    assert p["model_version"] == "v10.0"
    assert (p["dataset_version"] or "").startswith("lits_verified_v8@")
    expected = "classical_cpu_numpy" if model_id in ("v10_logreg_radiomics", "v10_hybrid_twin_radiomics") \
        else "custom_numpy_simulator"
    assert p["execution_mode"] == expected
    assert 0.0 <= p["probability"] <= 1.0 and p["label"]
    meta = researcher.get(f"/api/v1/models/{model_id}/metrics").json()
    ev = (meta.get("evaluations") or [{}])[0]
    assert "Nested" in ev.get("name", "") and ev.get("metrics", {}).get("primary_mean_within_unit_auc")
    researcher.delete(f"/api/v1/predictions/{p['id']}")
