"""Comparison service: validated uploads, dataset samples with neighbouring slices, runs, history, exports, reports
and the benchmarking view (recorded evaluation metrics + measured latencies)."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import statistics
from pathlib import Path

from app.core.apikit import UploadedFile
from app.core.config import DISCLAIMER, PROJECT_ROOT
from app.core.errors import ApiError
from app.core.security import iso, new_id, utcnow
from app.db.database import loads

log = logging.getLogger("lti.compare")
EXT = {"png": "png", "jpeg": "jpg", "dicom": "dcm", "nifti": "nii"}
CONTEXT_KEYS = ("prev2", "prev", "next", "next2")
FAMILIES = ("classical", "deep_learning", "quantum", "hybrid")
FAMILY_LABEL = {"classical": "Classical ML", "deep_learning": "Deep Learning", "quantum": "Quantum ML",
                "hybrid": "Hybrid ML"}


def _detect(data: bytes, name: str | None) -> str:
    from src.preprocessing import formats
    try:
        fmt = formats.detect_format(data, name)
        formats.decode(data, name)
    except formats.UnsupportedFormat as e:
        raise ApiError(415, "unsupported_media_type", str(e))
    except formats.CorruptImage as e:
        raise ApiError(422, "corrupt_image", f"The file could not be decoded: {e}")
    if fmt not in EXT:
        raise ApiError(415, "unsupported_media_type", f"unsupported image format {fmt!r}")
    return fmt


class ComparisonService:
    def __init__(self, settings, db, storage, model_service, manager):
        self.settings, self.db, self.storage = settings, db, storage
        self.models, self.manager = model_service, manager

    # ------------------------------------------------------------------ samples (real slices + neighbours)
    def _sample_index(self) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        base = PROJECT_ROOT / "dataset" / "sample_data" / "lits_sample"
        if (base / "labels" / "slices.csv").is_file():
            rows = list(csv.DictReader(open(base / "labels" / "slices.csv")))
            for r in rows:
                idx[r["file_name"]] = {"id": r["file_name"], "path": base / r["image_path"], "source": "lits_sample",
                                       "category": "real_ct", "label_tumour": int(r["label_tumour"]),
                                       "liver_px_512": int(r["liver_px_512"]), "tumour_px_512": int(r["tumour_px_512"]),
                                       "scan_id": r["scan_id"], "split": r["split"], "index": int(r["upstream_index"])}
        full = PROJECT_ROOT / "dataset" / "liver_ct_lits_verified_v8"
        lab = PROJECT_ROOT / "data" / "labels" / "verified_slice_labels.csv"
        if full.is_dir() and lab.is_file():
            for r in csv.DictReader(open(lab)):
                p = full / r["path"]
                if p.is_file() and r["file_name"] not in idx:
                    idx[r["file_name"]] = {"id": r["file_name"], "path": p, "source": "liver_ct_lits_verified_v8",
                                           "category": "real_ct", "label_tumour": int(r["label_tumour"]),
                                           "liver_px_512": int(r["liver_px_512"] or 0),
                                           "tumour_px_512": int(r["tumour_px_512"] or 0), "scan_id": r["scan_id"],
                                           "split": r["split"], "index": int(r["upstream_index"])}
        syn = PROJECT_ROOT / "dataset" / "sample_data" / "synthetic"
        if (syn / "manifest.csv").is_file():
            for r in csv.DictReader(open(syn / "manifest.csv")):
                p = syn / r["image_path"]
                idx[p.name] = {"id": p.name, "path": p, "source": "synthetic", "category": "synthetic_phantom",
                               "label_tumour": None, "scan_id": r["scan_id"], "split": None, "index": None,
                               "note": "software test phantom, NOT medical data: expected to be rejected as OOD"}
        by_scan: dict[str, dict[int, str]] = {}
        for s in idx.values():
            if s["category"] == "real_ct":
                by_scan.setdefault(s["scan_id"], {})[s["index"]] = s["id"]
        for s in idx.values():
            ctx = {}
            if s["category"] == "real_ct":
                nb = by_scan[s["scan_id"]]
                for key, off in (("prev2", -2), ("prev", -1), ("next", 1), ("next2", 2)):
                    if s["index"] + off in nb:
                        ctx[key] = nb[s["index"] + off]
                if not ("prev" in ctx and "next" in ctx and "prev2" in ctx and "next2" in ctx):
                    ctx.pop("prev2", None)
                    ctx.pop("next2", None)
                if not ("prev" in ctx and "next" in ctx):
                    ctx = {}                 # a lone neighbour is useless: models read both sides
            s["context_ids"] = ctx
        return idx

    def samples(self) -> list[dict]:
        out = []
        for s in self._sample_index().values():
            c = s["context_ids"]
            out.append({k: s.get(k) for k in ("id", "source", "category", "label_tumour", "liver_px_512",
                                               "tumour_px_512", "scan_id", "split", "note")} |
                       {"context": {"adjacent": "prev" in c and "next" in c,
                                    "five_slice": all(k in c for k in CONTEXT_KEYS), "ids": c}})
        return sorted(out, key=lambda s: (s["category"] != "real_ct", s["source"], s["id"]))

    # ------------------------------------------------------------------ uploads
    def create_upload(self, user: dict, file: UploadedFile | None, context_files: dict, modality: str,
                      sample_id: str | None = None) -> dict:
        if user["role"] == "viewer":
            raise ApiError(403, "forbidden", "Viewers cannot run comparisons")
        modality = (modality or "CT").upper()
        if modality not in ("CT", "MRI"):
            raise ApiError(422, "validation_error", "modality must be CT or MRI")
        source = "upload"
        if sample_id:
            s = self._sample_index().get(sample_id)
            if s is None:
                raise ApiError(404, "not_found", "Sample not found")
            data, name, source = s["path"].read_bytes(), s["id"], s["source"]
            ctx = {k: self._sample_index()[v]["path"].read_bytes() for k, v in s["context_ids"].items()}
        else:
            if file is None:
                raise ApiError(422, "validation_error", "Upload a file or choose a sample")
            data, name = file.read_bytes(), file.filename
            ctx = {k: f.read_bytes() for k, f in context_files.items() if f is not None}
        fmt = _detect(data, name)
        for k, b in ctx.items():
            _detect(b, None)
        if ("prev2" in ctx or "next2" in ctx) and not ("prev" in ctx and "next" in ctx):
            raise ApiError(422, "validation_error", "Outer slices (prev2/next2) need the previous and next slices too")
        uid = new_id("upl_")
        base = f"compare/{user['id']}/{uid}"
        self.storage.put_bytes(f"{base}/input.{EXT[fmt]}", data)
        stored = {"input": f"{base}/input.{EXT[fmt]}", "context": {}}
        for k, b in ctx.items():
            key = f"{base}/{k}.bin"
            self.storage.put_bytes(key, b)
            stored["context"][k] = key
        analysis = self.analyse(data, name, ctx, modality)
        import hashlib
        now = utcnow()
        exp = iso(now + dt.timedelta(days=self.settings.retention_days)) if self.settings.retention_days > 0 else None
        row = {"id": uid, "user_id": user["id"], "input_name": name if self.settings.store_original_filenames else None,
               "input_sha256": hashlib.sha256(data).hexdigest(), "input_format": fmt, "input_bytes": len(data),
               "modality": modality, "source": source, "sample_id": sample_id,
               "context": {"slices": sorted(ctx), "storage": stored}, "analysis": analysis, "created_at": iso(now),
               "expires_at": exp}
        with self.db.session() as c:
            c.run("INSERT INTO comparison_uploads (id, user_id, input_name, input_sha256, input_format, input_bytes, "
                  "modality, source, sample_id, context, analysis, created_at, expires_at) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(row.values()))
        return self._upload_out(row)

    def analyse(self, data: bytes, name: str | None, ctx: dict, modality: str) -> dict:
        from src.v10.engine.compare import ComparisonRunner
        from src.v10.engine.interface import ModelUnavailable
        eng = self.models.engine()
        runner = ComparisonRunner(eng)
        try:
            prep = runner.prepare(data, name, ctx.get("prev"), ctx.get("next"), ctx.get("prev2"), ctx.get("next2"))
        except ModelUnavailable as e:
            raise ApiError(503, "engine_unavailable", str(e))
        plan = runner.plan(prep)
        summ = prep.summary()
        summ["modality"] = modality
        if modality == "MRI":
            summ["modality_note"] = ("every model was trained on CT slices (LiTS); MRI input is out of scope and is "
                                     "expected to be rejected by the out-of-distribution check")
        summ["compatibility"] = [{k: p.get(k) for k in ("model_id", "name", "family", "eligible", "skip_status",
                                                          "skip_reason", "context_slices_each_side")} for p in plan]
        summ["eligible_by_family"] = {f: sum(1 for p in plan if p["family"] == f and p["eligible"]) for f in FAMILIES}
        import base64
        summ["preview_png_b64"] = base64.b64encode(prep.fe.preview_png).decode() if prep.fe.preview_png else None
        return summ

    @staticmethod
    def _upload_out(r: dict) -> dict:
        ctx = loads(r["context"], {}) if isinstance(r["context"], str) else r["context"]
        an = loads(r["analysis"], {}) if isinstance(r["analysis"], str) else r["analysis"]
        return {"id": r["id"], "input_name": r["input_name"], "input_format": r["input_format"],
                "input_bytes": r["input_bytes"], "input_sha256": r["input_sha256"], "modality": r["modality"],
                "source": r["source"], "sample_id": r["sample_id"], "context_slices": ctx.get("slices", []),
                "analysis": an, "created_at": r["created_at"]}

    def get_upload(self, user: dict, uid: str, internal: bool = False) -> dict:
        with self.db.session() as c:
            r = c.one("SELECT * FROM comparison_uploads WHERE id = ?", (uid,))
        if r is None or (r["user_id"] != user["id"] and user["role"] != "admin"):
            raise ApiError(404, "not_found", "Upload not found")
        out = self._upload_out(r)
        if internal:
            out["storage"] = loads(r["context"], {}).get("storage")
        return out

    # ------------------------------------------------------------------ runs
    def start(self, user: dict, upload_id: str, mode: str, model_ids: list[str] | None) -> dict:
        if user["role"] == "viewer":
            raise ApiError(403, "forbidden", "Viewers cannot run comparisons")
        up = self.get_upload(user, upload_id, internal=True)
        if mode == "run_selected" and not model_ids:
            raise ApiError(422, "validation_error", "Select at least one model")
        return self.manager.start(user, up, mode, model_ids)

    def _run_row(self, user: dict, rid: str) -> dict:
        with self.db.session() as c:
            r = c.one("SELECT * FROM comparison_runs WHERE id = ?", (rid,))
        if r is None or (r["user_id"] != user["id"] and user["role"] != "admin"):
            raise ApiError(404, "not_found", "Comparison run not found")
        return r

    def get_run(self, user: dict, rid: str) -> dict:
        r = self._run_row(user, rid)
        st = self.manager.state(rid)
        out = {k: r[k] for k in ("id", "upload_id", "mode", "status", "n_models", "n_completed", "n_failed", "wall_ms",
                                 "fastest_family", "slowest_family", "majority_label", "agreement_pct", "error",
                                 "created_at", "started_at", "finished_at")}
        out["requested_models"] = loads(r["requested_models"], [])
        out["hardware"] = loads(r["hardware"], {})
        out["result"] = loads(r["result"], {}) or None
        if st is not None and st.status not in ("completed", "cancelled", "failed"):
            out["status"] = st.status
            out["live_events"] = len(st.events)
        return out

    def history(self, user: dict, limit: int = 50, offset: int = 0) -> dict:
        with self.db.session() as c:
            where, params = ("", ()) if user["role"] == "admin" else ("WHERE r.user_id = ?", (user["id"],))
            rows = c.all(f"SELECT r.id, r.upload_id, r.mode, r.status, r.n_models, r.n_completed, r.n_failed, r.wall_ms, "
                         f"r.fastest_family, r.slowest_family, r.majority_label, r.agreement_pct, r.created_at, "
                         f"r.finished_at, r.summary, u.input_name, u.modality, u.source, u.sample_id "
                         f"FROM comparison_runs r JOIN comparison_uploads u ON u.id = r.upload_id {where} "
                         f"ORDER BY r.created_at DESC LIMIT ? OFFSET ?", (*params, limit, offset))
            total = c.scalar(f"SELECT COUNT(*) FROM comparison_runs r {where}", params)
        for r in rows:
            s = loads(r.pop("summary"), {})
            r["families"] = {f: {k: (s.get("families") or {}).get(f, {}).get(k) for k in
                                 ("n_completed", "inference_ms_mean", "wall_ms", "combined_label")} for f in FAMILIES}
            r["stopped"] = s.get("stopped")
        return {"items": rows, "total": total}

    def compare_runs(self, user: dict, a: str, b: str) -> dict:
        ra, rb = self.get_run(user, a), self.get_run(user, b)
        ma = {m["model_id"]: m for m in (ra["result"] or {}).get("models", [])}
        mb = {m["model_id"]: m for m in (rb["result"] or {}).get("models", [])}
        rows = []
        for mid in sorted(set(ma) | set(mb)):
            x, y = ma.get(mid, {}), mb.get(mid, {})
            ix, iy = (x.get("timings_ms") or {}).get("inference"), (y.get("timings_ms") or {}).get("inference")
            rows.append({"model_id": mid, "family": x.get("family") or y.get("family"),
                         "a": {"status": x.get("status"), "label": x.get("label"), "probability": x.get("probability"),
                               "inference_ms": ix},
                         "b": {"status": y.get("status"), "label": y.get("label"), "probability": y.get("probability"),
                               "inference_ms": iy},
                         "label_changed": x.get("label") != y.get("label"),
                         "probability_delta": None if x.get("probability") is None or y.get("probability") is None
                         else round(y["probability"] - x["probability"], 4),
                         "inference_ms_delta": None if ix is None or iy is None else round(iy - ix, 2)})
        fam = {f: {"a": ((ra["result"] or {}).get("families") or {}).get(f), "b": ((rb["result"] or {}).get("families")
                                                                                 or {}).get(f)} for f in FAMILIES}
        return {"a": {k: ra[k] for k in ("id", "upload_id", "status", "created_at", "wall_ms", "majority_label")},
                "b": {k: rb[k] for k in ("id", "upload_id", "status", "created_at", "wall_ms", "majority_label")},
                "same_input": ra["upload_id"] == rb["upload_id"], "models": rows, "families": fam}

    # ------------------------------------------------------------------ exports
    def export_csv(self, user: dict, rid: str) -> str:
        run = self.get_run(user, rid)
        res = run["result"] or {}
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["run_id", "model_id", "name", "family", "version", "status", "label", "probability", "threshold",
                    "confidence_margin", "saturated", "queue_ms", "features_ms", "inference_ms", "total_ms",
                    "cold_start", "execution_mode", "qubits", "reason"])
        for m in res.get("models", []):
            t = m.get("timings_ms") or {}
            w.writerow([rid, m["model_id"], m.get("name"), m["family"], m.get("version"), m["status"], m.get("label"),
                        m.get("probability"), m.get("threshold"), m.get("confidence"), m.get("saturated"),
                        t.get("queue"), t.get("features"), t.get("inference"), t.get("total"), m.get("cold_start"),
                        m.get("execution_mode"), (m.get("quantum") or {}).get("qubits"), m.get("reason")])
        w.writerow([])
        w.writerow(["family", "n_completed", "n_attempted", "n_failed", "success_rate", "inference_ms_mean",
                    "inference_ms_sum", "wall_ms", "fastest_model", "slowest_model", "combined_label"])
        for f, s in (res.get("families") or {}).items():
            w.writerow([f, s.get("n_completed"), s.get("n_attempted"), s.get("n_failed"), s.get("success_rate"),
                        s.get("inference_ms_mean"), s.get("inference_ms_sum"), s.get("wall_ms"),
                        s.get("fastest_model"), s.get("slowest_model"), s.get("combined_label")])
        return buf.getvalue()

    def report_pdf(self, user: dict, rid: str) -> bytes:
        from app.execution.report import comparison_pdf
        run = self.get_run(user, rid)
        if not run["result"]:
            raise ApiError(409, "not_ready", "The run has no results yet")
        up = self.get_upload(user, run["upload_id"])
        return comparison_pdf(run, up, DISCLAIMER)

    # ------------------------------------------------------------------ benchmarks
    def benchmarks(self, user: dict) -> dict:
        eng = self.models.engine()
        evals = []
        for d in eng.list(include_legacy=False):
            if d.get("family") == "segmentation" or d.get("role") == "software_test":
                continue
            ev = ((d.get("metrics") or {}).get("evaluations") or [None])[0]
            m = (ev or {}).get("metrics") or {}
            src = "v10_nested" if d.get("source") == "v10" else "v9_fixed_split"
            evals.append({"model_id": d["id"], "name": d.get("name"), "family": eng.compare_family(d),
                          "availability": d.get("availability"), "servable": d.get("servable"),
                          "protocol": src, "evaluation": (ev or {}).get("name"),
                          "evaluation_unit": (ev or {}).get("evaluation_unit"),
                          "metrics": {"accuracy": m.get("accuracy"), "precision": m.get("precision"),
                                      "recall": m.get("sensitivity"), "specificity": m.get("specificity"),
                                      "f1": m.get("f1"), "roc_auc": m.get("roc_auc", m.get("auc")),
                                      "pr_auc": m.get("pr_auc"), "ece": m.get("ece"), "brier": m.get("brier"),
                                      "balanced_accuracy": m.get("balanced_accuracy"),
                                      "primary_mean_within_unit_auc": m.get("primary_mean_within_unit_auc")},
                          "confusion": m.get("confusion"), "caveats": (ev or {}).get("caveats", [])})
        lat = self._latency_stats()
        protocols = {
            "v10_nested": "V10 nested leave-one-reconstructed-scan-unit-out evaluation (8 units, 5 seeds for stochastic "
                          "models); threshold and calibration chosen on inner folds only; patch level. Primary metric: "
                          "mean within-unit ROC AUC.",
            "v9_fixed_split": "V9 fixed scan-group split: train G03-G06, validation G02/G09/G10 (thresholds and "
                              "calibration), test G01/G07/G08 scored once; patch level. 531 of 622 test patches come "
                              "from G01."}
        fam = {}
        for f in FAMILIES:
            rows = [e for e in evals if e["family"] == f]
            fam[f] = {"label": FAMILY_LABEL[f], "n_models": len(rows),
                      "n_servable": sum(1 for e in rows if e["servable"]),
                      "best_by_protocol": {p: _best(rows, p) for p in protocols}, "latency": lat["families"].get(f)}
        return {"evaluations": evals, "families": fam, "protocols": protocols, "latency": lat,
                "notes": ["Accuracy metrics come from recorded evaluations on ground-truth-labelled, scan-group-held-out "
                          "data, never from single uploads.",
                          "Metrics are comparable only within the same protocol (V9 fixed split vs V10 nested).",
                          "'Best' is the highest primary AUC within a protocol (V10: mean within-unit AUC; V9: test ROC AUC); it is not a claim of clinical superiority.",
                          "Latencies are measured from comparison runs on this server (warm = model already used in "
                          "this process)."]}

    def _latency_stats(self) -> dict:
        with self.db.session() as c:
            rows = c.all("SELECT model_id, family, inference_ms, features_ms, total_ms, cold_start FROM "
                         "comparison_model_results WHERE status = 'ok' AND inference_ms IS NOT NULL")
        models: dict[str, dict] = {}
        for r in rows:
            m = models.setdefault(r["model_id"], {"model_id": r["model_id"], "family": r["family"], "warm": [],
                                                  "cold": [], "features": []})
            (m["cold"] if r["cold_start"] else m["warm"]).append(float(r["inference_ms"]))
            if r["features_ms"] is not None:
                m["features"].append(float(r["features_ms"]))
        out_m = []
        for m in models.values():
            w, cld = m.pop("warm"), m.pop("cold")
            ft = m.pop("features")
            m.update(n_warm=len(w), n_cold=len(cld), warm_median_ms=_med(w), warm_p95_ms=_p95(w),
                     cold_median_ms=_med(cld), features_median_ms=_med(ft))
            out_m.append(m)
        fams = {}
        for f in FAMILIES:
            xs = [m["warm_median_ms"] for m in out_m if m["family"] == f and m["warm_median_ms"] is not None]
            fams[f] = {"n_models_measured": len(xs), "median_of_model_warm_medians_ms": _med(xs)}
        return {"models": sorted(out_m, key=lambda m: (m["warm_median_ms"] is None, m["warm_median_ms"] or 0)),
                "families": fams, "n_measurements": len(rows)}

    def benchmarks_csv(self, user: dict) -> str:
        b = self.benchmarks(user)
        buf = io.StringIO()
        w = csv.writer(buf)
        keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "ece",
                "primary_mean_within_unit_auc"]
        lat = {m["model_id"]: m for m in b["latency"]["models"]}
        w.writerow(["model_id", "family", "protocol", "availability"] + keys + ["warm_median_inference_ms",
                                                                                "cold_median_inference_ms"])
        for e in b["evaluations"]:
            L = lat.get(e["model_id"], {})
            w.writerow([e["model_id"], e["family"], e["protocol"], e["availability"]] +
                       [e["metrics"].get(k) for k in keys] + [L.get("warm_median_ms"), L.get("cold_median_ms")])
        return buf.getvalue()

    def last_success(self) -> dict[str, str]:
        with self.db.session() as c:
            rows = c.all("SELECT model_id, MAX(created_at) AS t FROM comparison_model_results WHERE status = 'ok' "
                         "GROUP BY model_id")
            rows2 = c.all("SELECT model_id, MAX(finished_at) AS t FROM predictions WHERE status = 'completed' "
                          "GROUP BY model_id")
        out: dict[str, str] = {}
        for r in rows + rows2:
            if r["t"] and (r["model_id"] not in out or r["t"] > out[r["model_id"]]):
                out[r["model_id"]] = r["t"]
        return out

    def stats(self, user: dict) -> dict:
        with self.db.session() as c:
            where, p = ("", ()) if user["role"] == "admin" else (" AND user_id = ?", (user["id"],))
            n = c.scalar(f"SELECT COUNT(*) FROM comparison_runs WHERE status = 'completed'{where}", p)
            avg = c.scalar("SELECT AVG(inference_ms) FROM comparison_model_results WHERE status = 'ok'")
        return {"completed_runs": n or 0, "avg_inference_ms": round(float(avg), 2) if avg is not None else None}


def _med(xs):
    return round(statistics.median(xs), 2) if xs else None


def _p95(xs):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], 2)


def _best(rows: list[dict], protocol: str):
    key = "primary_mean_within_unit_auc" if protocol == "v10_nested" else "roc_auc"
    cand = [e for e in rows if e["protocol"] == protocol and e["metrics"].get(key) is not None]
    if not cand:
        return None
    b = max(cand, key=lambda e: e["metrics"][key])
    return {"model_id": b["model_id"], "metric": key, "value": b["metrics"][key]}
