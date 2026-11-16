"""Dashboard analytics and research summaries, computed from the database, the model registry and saved result
files only (empty states, never placeholder numbers)."""
from __future__ import annotations

import json
from pathlib import Path

from app.core.security import iso, utcnow
from app.db.database import loads
from app.services import audit

METRIC_KEYS = ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1",
               "mcc", "log_loss", "brier", "ece")


def overview(state, user: dict) -> dict:
    is_admin = user["role"] == "admin"
    with state.db.session() as c:
        users = int(c.scalar("SELECT COUNT(*) FROM users") or 0) if is_admin else None
        p_total = int(c.scalar("SELECT COUNT(*) FROM predictions") or 0) if is_admin else None
        mine = int(c.scalar("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user["id"],)) or 0)
        scope = "" if is_admin else " WHERE user_id = ?"
        params = [] if is_admin else [user["id"]]
        by_status = {r["status"]: int(r["n"]) for r in
                     c.all("SELECT status, COUNT(*) AS n FROM predictions" + scope + " GROUP BY status", params)}
        ds_where = "" if is_admin else " WHERE owner_id = ? OR visibility = 'shared'"
        datasets = int(c.scalar("SELECT COUNT(*) FROM datasets" + ds_where, params) or 0)
        files = int(c.scalar("SELECT COALESCE(SUM(v.n_files), 0) FROM dataset_versions v JOIN datasets d ON "
                             "d.id = v.dataset_id WHERE v.version = (SELECT MAX(version) FROM dataset_versions "
                             "WHERE dataset_id = d.id)" + ("" if is_admin else " AND (d.owner_id = ? OR "
                                                                                "d.visibility = 'shared')"),
                             params) or 0)
        exps = int(c.scalar("SELECT COUNT(*) FROM experiments" + ("" if is_admin else
                                                                  " WHERE source = 'recorded' OR owner_id = ?"),
                            params) or 0)
        pending = int(c.scalar("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')") or 0)
        recent = audit.recent(c, None if is_admin else user["id"], 12)
    eng = state.engine.status()
    models = []
    if state.engine.engine is not None:
        models = state.model_service.list()
    fam: dict[str, int] = {}
    for m in models:
        if m["servable"]:
            fam[m["family"]] = fam.get(m["family"], 0) + 1
    return {"users_total": users, "predictions_total": p_total if p_total is not None else mine,
            "predictions_mine": mine, "predictions_by_status": by_status, "models_registered": len(models),
            "models_servable": sum(1 for m in models if m["servable"]), "models_by_family": fam,
            "datasets_total": datasets, "dataset_files_total": files, "experiments_total": exps,
            "jobs_pending": pending, "recent_activity": recent,
            "engine": {k: eng.get(k) for k in ("state", "error", "models_servable", "v9_engine", "load_seconds")},
            "generated_at": iso(utcnow())}


def _metric(met: dict, key: str):
    alias = {"roc_auc": ("roc_auc", "auc", "primary_mean_within_unit_auc")}.get(key, (key,))
    for a in alias:
        if met.get(a) is not None:
            return met[a]
    return None


def model_comparison(state) -> dict:
    rows = []
    for m in state.model_service.list():
        evs = (m.get("metrics") or {}).get("evaluations") or []
        ev = next((e for e in evs if e.get("metrics") and "completed" in str(e.get("status", "")) or
                   e.get("status") == "recorded V9 result"), None)
        if ev is None or not m["servable"]:
            continue
        met = ev["metrics"]
        primary = met.get("primary_mean_within_unit_auc")
        rows.append({
            "model_id": m["id"], "name": m["name"], "family": m["family"], "architecture": m["architecture"],
            "source": m["source"], "version": m["version"], "role": m.get("role"),
            "evaluation": ev.get("name"), "split": ev.get("split"), "evaluation_unit": ev.get("evaluation_unit"),
            "primary_metric": "mean within-unit ROC-AUC" if primary is not None else "test ROC-AUC (V9 split)",
            "roc_auc": primary if primary is not None else _metric(met, "roc_auc"),
            "pooled_roc_auc": met.get("roc_auc") if primary is not None else None,
            "metrics": {k: _metric(met, k) for k in METRIC_KEYS},
            "auc_ci95": met.get("auc_ci95"), "per_unit_auc": ev.get("per_unit_auc") or ev.get("per_group_auc"),
            "train_seconds": met.get("train_seconds") or met.get("fit_seconds_mean"),
            "inference_ms_per_patch": met.get("inference_ms_per_patch") or met.get("predict_ms_per_patch"),
            "size_bytes": m.get("size_bytes"), "parameters": (m.get("metadata") or {}).get("parameters"),
            "execution_mode": m.get("execution_mode"), "caveats": ev.get("caveats", [])})
    rows.sort(key=lambda r: (r["source"] != "v10", -(r["roc_auc"] or 0)))
    return {"rows": rows, "notes": [
        "Only trained and evaluated models are listed.",
        "V9 rows: recorded results on V9's fixed split (test groups G01, G07, G08; patch level).",
        "V10 rows: nested leave-one-unit-out evaluation; primary metric = mean within-unit ROC-AUC over units with "
        ">= 10 patches per class. The two evaluations are not directly comparable.",
        "No row is patient-level; no quantum advantage is claimed."], "generated_at": iso(utcnow())}


def prediction_distribution(state, user: dict, all_users: bool) -> dict:
    scope_all = all_users and user["role"] == "admin"
    where = "" if scope_all else " WHERE user_id = ?"
    params = [] if scope_all else [user["id"]]
    with state.db.session() as c:
        by_label = c.all("SELECT COALESCE(label, '(none)') AS label, COUNT(*) AS n FROM predictions" + where +
                         " GROUP BY label ORDER BY n DESC", params)
        by_model = c.all("SELECT model_id, COUNT(*) AS n FROM predictions" + where +
                         " GROUP BY model_id ORDER BY n DESC LIMIT 15", params)
        by_status = c.all("SELECT status, COUNT(*) AS n FROM predictions" + where + " GROUP BY status", params)
        by_day = c.all("SELECT SUBSTR(created_at, 1, 10) AS day, COUNT(*) AS n FROM predictions" + where +
                       " GROUP BY SUBSTR(created_at, 1, 10) ORDER BY day DESC LIMIT 30", params)
        probs = [r["probability"] for r in c.all("SELECT probability FROM predictions" + where +
                                                  (" AND" if where else " WHERE") + " probability IS NOT NULL", params)]
    hist = [0] * 10
    for p in probs:
        hist[min(9, int(float(p) * 10))] += 1
    return {"scope": "all users" if scope_all else "mine", "by_label": by_label, "by_model": by_model,
            "by_status": by_status, "by_day": list(reversed(by_day)),
            "probability_histogram": [{"bin": f"{i / 10:.1f}–{(i + 1) / 10:.1f}", "n": n} for i, n in enumerate(hist)],
            "n_with_probability": len(probs)}


# ------------------------------------------------------------------------------------------- research
def _read(root: Path, rel: str):
    p = root / rel
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _compact(o, depth=0, max_list=12):
    if depth > 4:
        return "…"
    if isinstance(o, dict):
        return {k: _compact(v, depth + 1, max_list) for k, v in list(o.items())[:30]}
    if isinstance(o, list):
        return [_compact(v, depth + 1, max_list) for v in o[:max_list]]
    if isinstance(o, str) and len(o) > 400:
        return o[:400] + "…"
    return o


def quantum_models(state) -> list[dict]:
    out = []
    for m in state.model_service.list():
        if m["family"] not in ("quantum", "hybrid") or not m["servable"]:
            continue
        md = m.get("metadata") or {}
        circ = md.get("circuit") or {}
        conf = md.get("configuration") or {}
        gc = circ.get("gate_counts") or {}
        out.append({"model_id": m["id"], "name": m["name"], "family": m["family"], "source": m["source"],
                    "execution_mode": m["execution_mode"],
                    "hardware": False,
                    "qubits": circ.get("qubits") or circ.get("n_qubits") or conf.get("n_qubits"),
                    "layers": circ.get("layers") or conf.get("n_layers"),
                    "depth": gc.get("depth") or circ.get("depth"),
                    "two_qubit_depth": gc.get("two_qubit_depth"),
                    "one_qubit_gates": gc.get("one_qubit") or gc.get("1q"),
                    "two_qubit_gates": gc.get("two_qubit") or gc.get("2q") or circ.get("two_qubit_gates_per_state"),
                    "trainable_parameters": circ.get("trainable_circuit_parameters") or md.get("parameters"),
                    "feature_map": circ.get("feature_map") or conf.get("feature_map"),
                    "readout": circ.get("readout") or conf.get("readout"),
                    "entangler": circ.get("entangler"), "raw": _compact({"circuit": circ, "configuration": conf})})
    return out


def research_summary(state) -> dict:
    root = Path(state.settings.project_root)
    nested = {}
    for fs in ("radiomics", "imagenet_gap"):
        r = _read(root, f"results/v10/nested/{fs}/results.json")
        c = _read(root, f"results/v10/nested/{fs}/comparisons.json")
        if r:
            nested[fs] = {"models": [{"key": k, "name": v["name"], "family": v["family"],
                                      "mean_within_unit_auc": v["primary_mean_within_unit_auc"],
                                      "min_unit_auc": v["within_unit_auc_min"], "max_unit_auc": v["within_unit_auc_max"],
                                      "pooled_auc": (v.get("pooled_oof") or {}).get("roc_auc"),
                                      "seed_sd": v["mean_seed_sd_within_unit"], "n_parameters": v["n_parameters"],
                                      "fit_seconds": v["fit_seconds_mean"], "predict_ms": v["predict_ms_per_patch"],
                                      "per_unit": {u: x["within_unit_auc"] for u, x in v["per_unit"].items()}}
                                     for k, v in r["models"].items()],
                          "qualifying_units": r.get("qualifying_units"), "units": r.get("units"),
                          "seeds": r.get("seeds"), "comparisons": c, "n_patches": r.get("n_patches")}
    noise = _read(root, "results/v10/noise/noise_study.json")
    ver = {k: _compact(_read(root, f"results/v10/verification/{k}.json")) for k in
           ("B_kernel_geometry", "C_zz_phase", "D_parameters_and_readout", "G_noise", "timing")}
    math = {k: _compact(_read(root, f"results/v10/math_validation/{k}.json")) for k in
            ("parameter_accounting", "readout_and_trainability", "expressibility", "noise_validation",
             "hardware_readiness", "feature_maps_synthetic")}
    runs = []
    rdir = root / "results" / "v10" / "runs"
    if rdir.exists():
        for d in sorted(rdir.iterdir(), reverse=True)[:50]:
            s = _read(root, f"results/v10/runs/{d.name}/summary.json")
            if s:
                runs.append({"run": d.name, **{k: s.get(k) for k in ("status", "started_utc", "seconds", "artifact")}})
    hybrid_ablation = []
    if "radiomics" in nested:
        for m in nested["radiomics"]["models"]:
            if m["key"].startswith("hybrid_") or m["key"] in ("late_fusion", "vqc", "logreg_pca4"):
                hybrid_ablation.append(m)
    return {"nested": nested, "hybrid_ablation": hybrid_ablation, "noise_study": noise, "stage2_verification": ver,
            "mathematical_validation": math, "runs": runs, "quantum_models": quantum_models(state),
            "execution_note": "All quantum results are exact or noisy classical simulations (custom NumPy simulator). "
                              "No quantum hardware was used."}
