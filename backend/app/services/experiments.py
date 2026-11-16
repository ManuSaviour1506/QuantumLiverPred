"""Experiments: user-launched runs (background jobs) and recorded results imported from saved result files.

Kinds
  quantum_diagnostics  circuit resources (qubits, depth, gate counts), dead parameters by commutation, Jacobian rank,
                       gradient variance and expressibility of a V10 VQC configuration (exact simulation)
  noise_study          noisy density-matrix simulation of a deployed V10 VQC (depolarising / readout / shots) on real
                       patch features when the dataset is registered, otherwise circuit-level on random inputs
  nested_evaluation    unit-grouped nested cross-validation on a registered dataset (patch arrays when available,
                       otherwise whole-image radiomics), with seeds, calibration and per-unit metrics
Recorded experiments (source = 'recorded'): V9 comparison, V10 nested evaluations, Stage 2 verification and the V10
mathematical validation — read from results/ and never edited.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from app.core.errors import ApiError, forbidden, not_found
from app.core.security import iso, new_id, utcnow
from app.db.database import loads
from app.services import audit

log = logging.getLogger("lti.experiments")

RECORDED = [
    ("exp_rec_v10_nested_radiomics", "V10 nested evaluation — radiomics (confirmatory family)", "nested_evaluation",
     "results/v10/nested/radiomics/results.json", "results/v10/nested/radiomics/comparisons.json"),
    ("exp_rec_v10_nested_imagenet", "V10 nested evaluation — frozen ImageNet features (secondary family)",
     "nested_evaluation", "results/v10/nested/imagenet_gap/results.json",
     "results/v10/nested/imagenet_gap/comparisons.json"),
    ("exp_rec_v10_noise", "V10 noise study (deployed VQC, real patch features)", "noise_study",
     "results/v10/noise/noise_study.json", None),
    ("exp_rec_v9_comparison", "V9 model comparison (fixed split, recorded)", "v9_comparison",
     "results/v9/metrics_v9.csv", None),
    ("exp_rec_stage2", "Stage 2 — verification of V9 audit findings A–G", "verification",
     "results/v10/verification/statistics.json", None),
    ("exp_rec_math", "V10 mathematical and quantum validation (synthetic)", "quantum_diagnostics",
     "results/v10/math_validation/parameter_accounting.json", None),
]


def _exp_out(r: dict) -> dict:
    res = loads(r["results"], {})
    return {"id": r["id"], "name": r["name"], "kind": r["kind"], "source": r["source"], "status": r["status"],
            "owner_id": r["owner_id"], "dataset_version_id": r["dataset_version_id"], "config": loads(r["config"], {}),
            "error": r["error"], "created_at": r["created_at"], "started_at": r["started_at"],
            "finished_at": r["finished_at"], "has_results": bool(res)}


class ExperimentService:
    def __init__(self, settings, db, jobs, dataset_service):
        self.settings, self.db, self.jobs, self.datasets = settings, db, jobs, dataset_service
        self.root = Path(settings.project_root)

    # ------------------------------------------------------------------ recorded results
    def import_recorded(self) -> int:
        n = 0
        now = iso(utcnow())
        with self.db.session() as c:
            for eid, name, kind, path, comp in RECORDED:
                p = self.root / path
                if not p.exists():
                    continue
                if p.suffix == ".csv":
                    import pandas as pd
                    df = pd.read_csv(p)
                    res = {"summary": df.where(df.notna(), None).to_dict(orient="records"), "file": path}
                else:
                    res = json.loads(p.read_text())
                    if comp and (self.root / comp).exists():
                        res = {"results": res, "comparisons": json.loads((self.root / comp).read_text())}
                    res["file"] = path
                c.run("INSERT INTO experiments (id, name, kind, owner_id, source, status, config, results, created_at, "
                      "finished_at) VALUES (?, ?, ?, NULL, 'recorded', 'succeeded', ?, ?, ?, ?) "
                      "ON CONFLICT (id) DO UPDATE SET results = excluded.results, finished_at = excluded.finished_at",
                      (eid, name, kind, {"file": path}, res, now, now))
                n += 1
        return n

    # ------------------------------------------------------------------ CRUD
    def list(self, user: dict) -> list[dict]:
        with self.db.session() as c:
            if user["role"] == "admin":
                rows = c.all("SELECT * FROM experiments ORDER BY source DESC, created_at DESC")
            else:
                rows = c.all("SELECT * FROM experiments WHERE source = 'recorded' OR owner_id = ? "
                             "ORDER BY source DESC, created_at DESC", (user["id"],))
        return [_exp_out(r) for r in rows]

    def _row(self, c, user: dict, eid: str) -> dict:
        r = c.one("SELECT * FROM experiments WHERE id = ?", (eid,))
        if r is None or (r["source"] != "recorded" and user["role"] != "admin" and r["owner_id"] != user["id"]):
            raise not_found("Experiment")
        return r

    def get(self, user: dict, eid: str) -> dict:
        with self.db.session() as c:
            return _exp_out(self._row(c, user, eid))

    def results(self, user: dict, eid: str) -> dict:
        with self.db.session() as c:
            r = self._row(c, user, eid)
        return {"id": r["id"], "status": r["status"], "results": loads(r["results"], {})}

    def full(self, user: dict, eid: str) -> dict:
        with self.db.session() as c:
            r = self._row(c, user, eid)
        out = _exp_out(r)
        out["results"] = loads(r["results"], {})
        return out

    def create(self, user: dict, name: str, kind: str, dataset_id: str | None, config: dict, ip: str | None) -> dict:
        if user["role"] == "viewer":
            raise forbidden("Viewers cannot run experiments")
        cfg = self._validate_config(kind, dict(config))
        dvid = None
        if kind == "nested_evaluation":
            if not dataset_id:
                raise ApiError(422, "dataset_required", "A nested evaluation needs a dataset")
            d = self.datasets.get(user, dataset_id)
            v = d["latest_version"]
            if v is None or v["status"] not in ("validated", "missing_files", "validation_failed"):
                raise ApiError(409, "dataset_not_validated", "Validate the dataset before running an evaluation")
            dvid = v["id"]
            cfg["dataset_id"] = dataset_id
        eid = new_id("exp_")
        with self.db.session() as c:
            with c.transaction():
                c.run("INSERT INTO experiments (id, name, kind, owner_id, source, status, config, dataset_version_id, "
                      "created_at) VALUES (?, ?, ?, ?, 'run', 'queued', ?, ?, ?)",
                      (eid, name, kind, user["id"], cfg, dvid, iso(utcnow())))
                self.jobs.enqueue(c, "experiment", eid)
                audit.log(c, "experiment.created", user["id"], "experiment", eid, ip, {"kind": kind},
                          secret=self.settings.jwt_secret)
        self.jobs.wake()
        return self.get(user, eid)

    @staticmethod
    def _validate_config(kind: str, cfg: dict) -> dict:
        def num(key, default, lo, hi, cast=int):
            v = cast(cfg.get(key, default))
            if not lo <= v <= hi:
                raise ApiError(422, "invalid_config", f"{key} must be between {lo} and {hi}")
            return v
        if kind == "quantum_diagnostics":
            ent = cfg.get("entangler", "brick")
            if ent not in ("brick", "ring", "linear", "full"):
                raise ApiError(422, "invalid_config", "entangler must be brick, ring, linear or full")
            return {"n_qubits": num("n_qubits", 4, 2, 8), "layers": num("layers", 2, 1, 4), "entangler": ent,
                    "n_init": num("n_init", 60, 10, 300)}
        if kind == "noise_study":
            levels = cfg.get("p2_levels", [0.0, 0.002, 0.01, 0.03])
            if not isinstance(levels, list) or not 1 <= len(levels) <= 8 or not all(0 <= float(x) <= 0.2 for x in levels):
                raise ApiError(422, "invalid_config", "p2_levels: 1-8 values in [0, 0.2]")
            return {"model_id": str(cfg.get("model_id", "v10_vqc_radiomics")), "p2_levels": [float(x) for x in levels],
                    "readout_error": float(cfg.get("readout_error", 0.0)), "shots": num("shots", 0, 0, 100000),
                    "n_samples": num("n_samples", 300, 50, 1000)}
        if kind == "nested_evaluation":
            allowed = {"logreg", "random_forest", "svm_rbf", "knn", "hist_gb", "decision_tree", "extra_trees",
                       "logreg_pca4", "svm_rbf_pca4", "vqc", "qksvm", "hybrid_no_fusion", "hybrid_twin"}
            models = cfg.get("models", ["logreg", "logreg_pca4", "vqc", "qksvm"])
            if not isinstance(models, list) or not models or not set(models) <= allowed:
                raise ApiError(422, "invalid_config", f"models must be a non-empty subset of {sorted(allowed)}")
            seeds = cfg.get("seeds", [0])
            if not isinstance(seeds, list) or not 1 <= len(seeds) <= 5:
                raise ApiError(422, "invalid_config", "seeds: 1-5 integers")
            return {"models": models, "seeds": [int(s) for s in seeds], "features": "radiomics",
                    "inner_k": num("inner_k", 3, 2, 5), "max_patches": num("max_patches", 3000, 50, 10000)}
        raise ApiError(422, "invalid_config", f"unknown experiment kind {kind!r}")

    # ------------------------------------------------------------------ execution (background job)
    def execute(self, eid: str) -> None:
        with self.db.session() as c:
            r = c.one("SELECT * FROM experiments WHERE id = ?", (eid,))
            if r is None or r["status"] not in ("queued", "running"):
                return
            c.run("UPDATE experiments SET status = 'running', started_at = ? WHERE id = ?", (iso(utcnow()), eid))
        cfg = loads(r["config"], {})
        t0 = time.time()
        try:
            fn = {"quantum_diagnostics": self._run_diagnostics, "noise_study": self._run_noise,
                  "nested_evaluation": self._run_nested}[r["kind"]]
            res = fn(cfg, r)
            res["seconds"] = round(time.time() - t0, 1)
            status, err = "succeeded", None
        except ApiError as e:
            res, status, err = {}, "failed", e.message
        except Exception as e:
            log.error("experiment failed", extra={"fields": {"experiment_id": eid}}, exc_info=True)
            res, status, err = {}, "failed", f"{type(e).__name__}: {str(e)[:300]}"
        with self.db.session() as c:
            c.run("UPDATE experiments SET status = ?, results = ?, error = ?, finished_at = ? WHERE id = ?",
                  (status, res, err, iso(utcnow()), eid))

    def _run_diagnostics(self, cfg: dict, row: dict) -> dict:
        from src.v10.quantum.circuits import build_v10_vqc, dead_by_commutation, gate_counts
        from src.v10.quantum.diagnostics import expressibility_kl, gradient_variance, jacobian_rank
        n, L, ent = cfg["n_qubits"], cfg["layers"], cfg["entangler"]
        circ = build_v10_vqc(n, L, ent)
        masks = [(q,) for q in range(n)]
        dead = dead_by_commutation(circ, masks)
        jr = jacobian_rank(circ, masks, n_inputs=24, n_points=2)
        gv = gradient_variance(lambda nn, d: build_v10_vqc(nn, d, ent), [n], [L],
                               lambda nn: [tuple(range(nn))], n_init=cfg["n_init"])
        ex = expressibility_kl(lambda w: circ.state(np.full((1, n), 0.5), w)[0], n, circ.n_weights, n_pairs=300)
        return {"execution_mode": "custom_numpy_simulator (exact state vectors; no shots, no hardware)",
                "circuit": {"qubits": n, "layers": L, "entangler": ent, "trainable_parameters": int(circ.n_weights),
                            **gate_counts(circ)},
                "dead_parameters": {"count": len(dead.get("dead_weights", [])), "detail": dead},
                "jacobian_rank": jr, "gradient_variance": gv, "expressibility": ex,
                "summary": [{"metric": "trainable parameters", "value": int(circ.n_weights)},
                            {"metric": "dead parameters", "value": len(dead.get("dead_weights", []))},
                            {"metric": "Jacobian rank", "value": jr.get("rank") if isinstance(jr, dict) else jr},
                            {"metric": "expressibility KL (lower = closer to Haar)",
                             "value": ex.get("kl_to_haar")},
                            {"metric": "gradient variance (global parity observable)",
                             "value": gv[0]["mean_var_live"] if gv else None}]}

    def _run_noise(self, cfg: dict, row: dict) -> dict:
        from src.v10.quantum import noise_study
        d = self.root / "artifacts" / "v10" / cfg["model_id"]
        if not (d / "model_card.json").exists():
            raise ApiError(422, "model_unavailable", f"noise study needs a deployed V10 VQC artifact "
                                                     f"({cfg['model_id']} not found)")
        X = y = None
        feats = sorted((self.root / "results" / "v10" / "features").glob("radiomics_*.npz"))
        ds_root = next((Path(p) for p in self.settings.dataset_roots
                        if (Path(p) / "patches" / "patches_v8.npz").exists()), None)
        if feats and ds_root is not None:
            from src.v10.data.lits_dataset import load_patch_dataset
            X, y = np.load(feats[0])["X"], load_patch_dataset(ds_root)["y"]
        try:
            return noise_study.run(d, X, y, cfg["p2_levels"], cfg["readout_error"], cfg["shots"], cfg["n_samples"])
        except ValueError as e:
            raise ApiError(422, "invalid_config", str(e))

    def _run_nested(self, cfg: dict, row: dict) -> dict:
        from src.v10.evaluation import specs as S
        from src.v10.evaluation.nested import compare, run_nested
        ds_id = cfg["dataset_id"]
        root = self.datasets.patch_dataset_root(ds_id)
        if root is not None:
            from src.v10.data.lits_dataset import load_patch_dataset
            from src.v10.features.extract import cached
            ds = load_patch_dataset(root)
            X = cached("radiomics", ds["patches2d"], self.root / "results" / "v10" / "features")
            y, units = ds["y"], ds["units"]
            data_note = f"{ds['dataset_version']}: liver patches, radiomics features"
        else:
            X, y, units, data_note = self._whole_image_features(row["dataset_version_id"])
        if len(y) > cfg["max_patches"]:
            idx = np.sort(np.random.default_rng(0).choice(len(y), cfg["max_patches"], replace=False))
            X, y, units = X[idx], y[idx], units[idx]
        if len(np.unique(units)) < 3:
            raise ApiError(422, "too_few_units", "A grouped evaluation needs at least 3 groups (scan or patient ids)")
        specs = [s for s in S.all_specs() if s.key in set(cfg["models"])]
        for s in specs:
            s.fixed_from = None
        res = run_nested(X, y, units, specs, cfg["seeds"], cfg["inner_k"], log=lambda m: None)
        res.pop("_oof", None)
        comp = compare(res, [p for p in S.CONFIRMATORY if p[0] in cfg["models"] and p[1] in cfg["models"]],
                       "user-run (exploratory)")
        summary = [{"model": k, "family": v["family"], "mean_within_unit_auc": v["primary_mean_within_unit_auc"],
                    "pooled_auc": v["pooled_oof"].get("roc_auc"), "seed_sd": v["mean_seed_sd_within_unit"],
                    "fit_s": v["fit_seconds_mean"]} for k, v in res["models"].items()]
        return {"data": data_note, "summary": summary, "results": res, "comparisons": comp,
                "status_note": "user-launched run: exploratory; not part of the pre-registered analysis"}

    def _whole_image_features(self, version_id: str):
        from src.v10.data.ingest import read_image
        from src.v10.features.extract import radiomics
        from PIL import Image
        with self.db.session() as c:
            v = c.one("SELECT * FROM dataset_versions WHERE id = ?", (version_id,))
            rows = c.all("SELECT image_path, label, patient_id, scan_id, file_status FROM dataset_files "
                         "WHERE dataset_version_id = ? ORDER BY row_index", (version_id,))
            root = self.datasets._root_of(c, v)
        rows = [r for r in rows if r["file_status"] in ("ok", "warning") and r["label"]]
        labels = sorted({r["label"] for r in rows})
        if len(labels) != 2:
            raise ApiError(422, "binary_labels_required", f"nested evaluation needs exactly two labels, found {labels}")
        imgs, y, units = [], [], []
        for r in rows:
            img = read_image(Path(root) / r["image_path"])
            g = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(np.asarray(Image.fromarray(g).resize((96, 96), Image.BILINEAR)))
            y.append(labels.index(r["label"]))
            units.append(r["patient_id"] or r["scan_id"] or "")
        if any(u == "" for u in units):
            raise ApiError(422, "groups_required", "every row needs a patient_id or scan_id for a grouped evaluation")
        X = radiomics(np.stack(imgs))
        return X, np.array(y), np.array(units), (f"whole images resized to 96x96, radiomics; positive class = "
                                                 f"'{labels[1]}'")

    def report_pdf(self, user: dict, eid: str) -> bytes:
        from app.services.reports import experiment_report
        return experiment_report(self.full(user, eid))
