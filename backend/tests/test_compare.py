"""Four-paradigm comparison over real HTTP: upload validation, samples with neighbouring slices, run-all with the four
families started concurrently, live SSE stream, independent failures, context handling, liver-free and OOD inputs,
history, run-to-run comparison, exports, cancellation, benchmarks and the availability registry."""
import json
import time

import numpy as np
import pytest

from conftest import wait_for

FAMILIES = {"classical", "deep_learning", "quantum", "hybrid"}


def _run(client, upload_id, selected=None, timeout=240):
    if selected:
        r = client.post("/api/v1/compare/run-selected", json={"upload_id": upload_id, "model_ids": selected})
    else:
        r = client.post("/api/v1/compare/run-all", json={"upload_id": upload_id})
    assert r.status_code == 202, r.text
    rid = r.json()["run_id"]

    def done():
        j = client.get(f"/api/v1/compare/{rid}").json()
        return j if j["status"] in ("completed", "cancelled", "failed") else None
    return rid, wait_for(done, timeout)


def _sample(client, sid):
    r = client.post("/api/v1/upload/sample", json={"sample_id": sid})
    assert r.status_code == 201, r.text
    return r.json()


def _events(client, rid):
    r = client.get(f"/api/v1/compare/{rid}/stream", stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    evs, cur = [], {}
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("event: "):
            cur["event"] = line[7:]
        elif line.startswith("data: "):
            cur["data"] = json.loads(line[6:])
        elif line == "" and cur:
            evs.append(cur)
            cur = {}
    return evs


@pytest.fixture(scope="module")
def tumour_run(researcher, live):
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    up = _sample(researcher, "ct_lits_4278.png")          # middle slice of a 3-slice run: adjacent context attached
    rid, run = _run(researcher, up["id"])
    return up, rid, run


def test_samples_list_real_and_synthetic(researcher, live):
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    s = {x["id"]: x for x in researcher.get("/api/v1/compare/samples").json()}
    assert s["ct_lits_4278.png"]["context"]["adjacent"] is True
    assert s["ct_lits_4277.png"]["context"]["adjacent"] is False
    assert any(x["category"] == "synthetic_phantom" for x in s.values())


def test_upload_validation_and_compatibility(researcher, live):
    r = researcher.post("/api/v1/upload", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    up = _sample(researcher, "ct_lits_4278.png")
    an = up["analysis"]
    assert an["quality"]["status"] == "ok" and an["ood"]["status"] == "in_distribution"
    assert an["liver_check"]["status"] == "confirmed"
    assert an["context"]["adjacent_slices"] is True and an["context"]["five_slice_context"] is False
    comp = {c["model_id"]: c for c in an["compatibility"]}
    assert comp["cnn25d_v9_a4_res_se_focal"]["eligible"] is True          # 2.5-D: neighbours supplied
    assert comp["cnn3d_v9"]["skip_status"] == "needs_context"              # 5-slice model: not enough context
    assert comp["v10_decision_tree_radiomics"]["eligible"] is True               # V11: recalibrated v10.1 (finding 3)
    assert set(an["eligible_by_family"]) == FAMILIES


def test_run_all_four_families_concurrently(tumour_run):
    up, rid, run = tumour_run
    assert run["status"] == "completed"
    res = run["result"]
    assert set(res["families"]) == FAMILIES
    for f, s in res["families"].items():
        assert s["n_completed"] >= 1, f                                     # every family returns real results
    ov = res["family_overlap"]
    assert ov["start_spread_ms"] < 500                                      # all four started together
    assert ov["all_four_overlap_ms"] > 0                                    # and ran at the same time
    assert res["scheduling"]["strategy"] == "family_parallel_model_sequential"
    ok = [m for m in res["models"] if m["status"] == "ok"]
    for m in ok:
        assert 0.0 <= m["probability"] <= 1.0
        t = m["timings_ms"]
        assert t["inference"] >= 0 and t["total"] >= t["inference"]
        assert m["label"] in ("Tumour detected", "No tumour detected", "Uncertain / unable to classify")
    q = [m for m in ok if m["family"] == "quantum"]
    assert q and all(m["quantum"]["hardware"] is False for m in q)
    assert "simulat" in res["quantum_execution"]
    assert res["ranking"]["fastest_family"] in FAMILIES
    # models that could not run are listed with a reason, never silently dropped
    for m in res["models"]:
        if m["status"] != "ok":
            assert m.get("reason"), m["model_id"]


def test_stream_replays_or_streams_events(researcher, tumour_run):
    _, rid, _ = tumour_run
    evs = _events(researcher, rid)
    names = [e["event"] for e in evs]
    assert "run_completed" in names


def test_live_stream_has_family_and_model_events(researcher, live):
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    up = _sample(researcher, "ct_lits_4650.png")
    r = researcher.post("/api/v1/compare/run-all", json={"upload_id": up["id"]})
    rid = r.json()["run_id"]
    evs = _events(researcher, rid)
    names = [e["event"] for e in evs]
    for need in ("run_started", "preprocessing_completed", "plan", "family_started", "model_started",
                 "model_completed", "family_completed", "run_completed"):
        assert need in names, need
    starts = [e for e in evs if e["event"] == "family_started"]
    assert {e["data"]["family"] for e in starts} == FAMILIES
    first_done = names.index("family_completed")
    assert all(names.index("family_started") <= first_done for _ in starts)
    # all four families were started before any family finished
    last_start = max(i for i, n in enumerate(names) if n == "family_started")
    assert last_start < first_done


def test_liver_free_heart_slice_is_not_analysed(researcher, live):
    """Audit finding 2: ct_lits_4353 (heart, no liver in the mask) must not be scored as liver."""
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    up = _sample(researcher, "ct_lits_4353.png")
    assert up["analysis"]["liver_check"]["status"] == "rejected"
    _, run = _run(researcher, up["id"])
    res = run["result"]
    assert res["agreement"]["completed"] == 0
    assert all(m["status"] in ("withheld", "unavailable") for m in res["models"])
    assert not any(m.get("label") == "Tumour detected" for m in res["models"])


def test_no_liver_slice_withheld(researcher, live):
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    up = _sample(researcher, "ct_lits_4354.png")
    _, run = _run(researcher, up["id"])
    assert run["result"]["agreement"]["completed"] == 0
    assert "no liver" in run["result"]["preprocessing"]["stopped"]


def test_ood_phantom_rejected(researcher):
    up = _sample(researcher, "synthetic_phantom_phantom_A_000.png")
    assert up["analysis"]["ood"]["status"] == "out_of_distribution"
    _, run = _run(researcher, up["id"])
    res = run["result"]
    assert res["agreement"]["completed"] == 0
    assert all(m["status"] in ("out_of_distribution", "unavailable") for m in res["models"])


def test_run_selected_and_failure_isolation(researcher, live):
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    up = _sample(researcher, "ct_lits_4649.png")                # edge slice: no neighbours
    sel = ["v10_logreg_radiomics", "cnn2d_v9", "v10_vqc_radiomics", "v10_late_fusion_radiomics",
           "cnn25d_v9_a4_res_se_focal", "no_such_model"]
    _, run = _run(researcher, up["id"], sel)
    ms = {m["model_id"]: m for m in run["result"]["models"]}
    assert set(ms) == set(sel)
    assert ms["no_such_model"]["status"] == "unavailable"
    assert ms["cnn25d_v9_a4_res_se_focal"]["status"] == "needs_context"
    for mid in ("v10_logreg_radiomics", "cnn2d_v9", "v10_vqc_radiomics", "v10_late_fusion_radiomics"):
        assert ms[mid]["status"] == "ok", ms[mid]


def test_history_compare_export_report(researcher, tumour_run, live):
    _, rid, _ = tumour_run
    h = researcher.get("/api/v1/history").json()
    assert any(i["id"] == rid for i in h["items"])
    other = [i["id"] for i in h["items"] if i["id"] != rid and i["status"] == "completed"]
    if other:
        d = researcher.get(f"/api/v1/history/compare?a={rid}&b={other[0]}").json()
        assert "models" in d and "families" in d
    r = researcher.get(f"/api/v1/compare/{rid}/export.csv")
    assert r.status_code == 200 and "inference_ms" in r.text
    r = researcher.get(f"/api/v1/compare/{rid}/report")
    assert r.status_code == 200 and r.content[:4] == b"%PDF"


def test_runs_are_private(viewer, tumour_run):
    _, rid, _ = tumour_run
    assert viewer.get(f"/api/v1/compare/{rid}").status_code == 404
    r = viewer.post("/api/v1/upload/sample", json={"sample_id": "ct_lits_4278.png"})
    assert r.status_code == 403


def test_cancel(researcher, live):
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    up = _sample(researcher, "ct_lits_4651.png")
    r = researcher.post("/api/v1/compare/run-all", json={"upload_id": up["id"]})
    rid = r.json()["run_id"]
    c = researcher.post(f"/api/v1/compare/{rid}/cancel")
    assert c.status_code == 200
    run = wait_for(lambda: (lambda j: j if j["status"] in ("completed", "cancelled", "failed") else None)(
        researcher.get(f"/api/v1/compare/{rid}").json()), 240)
    assert run["status"] in ("cancelled", "completed")          # completed if it finished before the request landed
    if run["status"] == "cancelled":
        assert any(m["status"] == "cancelled" for m in run["result"]["models"])


def test_availability_registry(researcher):
    a = researcher.get("/api/v1/models/availability").json()
    by = {m["id"]: m for m in a["models"]}
    assert by["v10_decision_tree_radiomics"]["availability"] == "available"        # v10.1 recalibration
    assert by["v10_decision_tree_radiomics"]["version"] == "v10.1"
    assert by["efficientnet_b0_ft"]["availability"] in ("missing_weights", "available")
    assert by["cnn3d_v9"]["availability"] == "requires_context"
    for m in a["models"]:
        if m["availability"] in ("available", "requires_context"):
            assert m["servable"] is True
            assert all(d["ok"] or d.get("optional") for d in m["dependencies"])
        else:
            assert m["servable"] is False and m["reason"]


def test_benchmarks_and_status(researcher, tumour_run):
    b = researcher.get("/api/v1/benchmarks").json()
    assert set(b["families"]) == FAMILIES
    assert b["latency"]["n_measurements"] > 0
    assert any(e["protocol"] == "v10_nested" for e in b["evaluations"])
    assert researcher.get("/api/v1/benchmarks/summary.csv").status_code == 200
    s = researcher.get("/api/v1/system/status").json()
    assert s["engine"]["state"] == "ready" and set(s["families"]) == FAMILIES
    assert s["stats"]["completed_runs"] >= 1


def test_concurrent_runs_do_not_mix_results(researcher, live):
    """Two runs on different inputs at the same time give exactly the probabilities each gives alone."""
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    ua, ub = _sample(researcher, "ct_lits_4278.png"), _sample(researcher, "ct_lits_4650.png")
    _, alone_a = _run(researcher, ua["id"])
    _, alone_b = _run(researcher, ub["id"])
    ra = researcher.post("/api/v1/compare/run-all", json={"upload_id": ua["id"]}).json()["run_id"]
    rb = researcher.post("/api/v1/compare/run-all", json={"upload_id": ub["id"]}).json()["run_id"]

    def both():
        a, b = (researcher.get(f"/api/v1/compare/{r}").json() for r in (ra, rb))
        return (a, b) if all(x["status"] == "completed" for x in (a, b)) else None
    a, b = wait_for(both, 300)
    for run, alone in ((a, alone_a), (b, alone_b)):
        p1 = {m["model_id"]: m["probability"] for m in run["result"]["models"] if m["status"] == "ok"}
        p0 = {m["model_id"]: m["probability"] for m in alone["result"]["models"] if m["status"] == "ok"}
        assert p1 == p0
    assert a["result"]["upload"]["id"] == ua["id"] and b["result"]["upload"]["id"] == ub["id"]



def test_resources_and_load_times_measured(tumour_run):
    _, _, run = tumour_run
    res = run["result"]
    r = res["resources"]
    assert r["cpu_count"] >= 1 and r["process_cpu_seconds"] >= 0 and r["gpu"]["used"] is False
    ok = [m for m in res["models"] if m["status"] == "ok"]
    assert all(m["resources"]["thread_cpu_ms"] >= 0 for m in ok)
    assert all(m.get("load_ms") is not None for m in ok)          # loaded once at start-up, time recorded
    assert res["ranking"]["fastest_model"] and res["ranking"]["slowest_model"]


def test_decision_tree_makes_confident_calls(tumour_run):
    """Audit finding 3: the recalibrated tree (v10.1) is no longer stuck in the uncertainty band."""
    _, _, run = tumour_run
    dt = {m["model_id"]: m for m in run["result"]["models"]}["v10_decision_tree_radiomics"]
    assert dt["status"] == "ok" and dt["label"] in ("Tumour detected", "No tumour detected")


def test_new_evaluation_benchmark_run(researcher, live):
    if live["dataset"] is None:
        pytest.skip("no real CT sample")
    ds = {d["id"]: d for d in researcher.get("/api/v1/benchmarks/datasets").json()}
    assert ds["lits_sample"]["n_slices"] == 8
    r = researcher.post("/api/v1/benchmarks/runs", json={"dataset": "lits_sample",
                                                        "model_ids": ["logreg_radiomics_v9", "v10_logreg_radiomics"]})
    assert r.status_code == 202, r.text
    bid = r.json()["id"]
    run = wait_for(lambda: (lambda j: j if j["status"] in ("completed", "failed", "cancelled") else None)(
        researcher.get(f"/api/v1/benchmarks/runs/{bid}").json()), 300)
    assert run["status"] == "completed", run.get("error")
    ms = {m["model_id"]: m for m in run["result"]["models"]}
    assert set(ms["logreg_radiomics_v9"]["by_role"]) == {"test", "validation"}     # G01 test, G02 validation (V9 split)
    assert set(ms["v10_logreg_radiomics"]["by_role"]) == {"train"}                # V10 final fit on all units: in-sample
    t = ms["logreg_radiomics_v9"]["by_role"]["test"]
    assert t["n_slices"] == 5 and t["n_tumour"] == 3 and t["n_not_scored"] == 2   # 2 liver-free slices not scored
    assert researcher.get(f"/api/v1/benchmarks/runs/{bid}/export.csv").status_code == 200


def test_system_metrics(researcher):
    m = researcher.get("/api/v1/system/metrics").json()
    assert m["cpu_count"] >= 1 and "samples" in m and m["engine"]["models_loaded"] > 0
