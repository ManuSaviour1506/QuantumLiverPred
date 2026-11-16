"""V10-trained models served through the common interface.

Kinds (model_card.json "kind"):
  v10_patch_sklearn     scikit-learn pipeline on patch features (radiomics / ImageNet GAP)
  v10_patch_vqc         VQC on fold-fitted PCA angles
  v10_patch_qksvm       quantum-kernel SVM on fold-fitted PCA angles
  v10_patch_hybrid      hybrid (and its classical twin / ablations)
  v10_patch_late_fusion LR + VQC late fusion
  synthetic_image_logreg  SOFTWARE-TEST model trained on clearly labelled synthetic phantoms; never served unless
                        ALLOW_SYNTHETIC_MODELS=true and never presented as a tumour classifier

Patch models share the V9 front end (decode, QC, OOD, liver gate) and V9's slice-level rule (largest liver-patch
probability compared with the training-chosen threshold +/- 0.10).
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.v10.engine import artifacts as A
from src.v10.engine import explain
from src.v10.engine.interface import (BaseModel, InputRejected, ModelInput, ModelUnavailable, PredictionOutput,
                                      slice_call)
from src.v10.evaluation.grouped_cv import FoldPreprocessor

PATCH_KINDS = {"v10_patch_sklearn", "v10_patch_vqc", "v10_patch_qksvm", "v10_patch_hybrid", "v10_patch_late_fusion",
               "v10_patch_numpy_cnn", "v10_patch_backbone_ft"}
INPUT_KIND = {"v10_patch_numpy_cnn": "stacks", "v10_patch_backbone_ft": "patches"}


# ------------------------------------------------------------------ preprocessor persistence
def pp_to_arrays(pp: FoldPreprocessor, prefix: str = "pp_") -> dict:
    out = {f"{prefix}mu": pp.mu, f"{prefix}sd": pp.sd, f"{prefix}a": pp.a, f"{prefix}b": pp.b,
           f"{prefix}cfg": np.array([pp.pca or 0, -1.0 if pp.angle_max is None else pp.angle_max, int(pp.clip)], float),
           f"{prefix}scaler": np.array(list(pp.scaler.encode()), np.uint8)}
    if pp.pca:
        out[f"{prefix}pmean"], out[f"{prefix}comp"] = pp.pmean, pp.comp
    return out


def pp_from_arrays(a: dict, prefix: str = "pp_") -> FoldPreprocessor:
    pca, amax, clip = a[f"{prefix}cfg"]
    pp = FoldPreprocessor(bytes(a[f"{prefix}scaler"].tolist()).decode(), int(pca) or None,
                          None if amax < 0 else float(amax), bool(clip))
    pp.mu, pp.sd, pp.a, pp.b = a[f"{prefix}mu"], a[f"{prefix}sd"], a[f"{prefix}a"], a[f"{prefix}b"]
    if pp.pca:
        pp.pmean, pp.comp = a[f"{prefix}pmean"], a[f"{prefix}comp"]
    return pp


def late_fusion_to_arrays(m) -> dict:
    out = {"lf_coef": m.lr_model.coef_, "lf_intercept": m.lr_model.intercept_, "lf_scale": np.array([m.sc, m.sq])}
    out.update(pp_to_arrays(m.pp, "lfpp_"))
    out.update(m.vqc.to_arrays())
    return out


class _LateFusionRuntime:
    def __init__(self, a):
        from src.v10.quantum.vqc import VQCClassifier
        self.coef, self.intercept = a["lf_coef"].ravel(), float(a["lf_intercept"].ravel()[0])
        self.sc, self.sq = (float(v) for v in a["lf_scale"])
        self.pp = pp_from_arrays(a, "lfpp_")
        self.vqc = VQCClassifier.from_arrays(a)

    def decision_function(self, X):
        X = np.asarray(X, float)
        return 0.5 * (X @ self.coef + self.intercept) / self.sc + \
            0.5 * self.vqc.decision_function(self.pp.transform(X)) / self.sq


def _platt(s, ab):
    return 1 / (1 + np.exp(-np.clip(ab[0] * np.asarray(s, float) + ab[1], -50, 50)))


class V10PatchModel(BaseModel):
    def __init__(self, engine, directory: Path, card: dict):
        self.engine, self.dir, self.card = engine, Path(directory), card
        self.model_id, self.version = card["model_id"], card.get("version", "v10.0")
        self.model = None
        self.pp = None

    # ------------------------------------------------------------------ loading
    def load_model(self) -> None:
        if self.model is not None:
            return
        c = self.card
        if c.get("training_data") != "real" and not self.engine.allow_synthetic:
            raise ModelUnavailable("trained on synthetic data (software test only); not served")
        A.verify(self.dir, c)
        expected = A.preprocessing_signature(c["feature_set"], c["preprocessing"])
        if c.get("preprocessing_signature") != expected:
            raise ModelUnavailable("preprocessing signature of the artifact does not match the current code")
        arr = A.load_arrays(self.dir)
        kind = c["kind"]
        self.input_kind = INPUT_KIND.get(kind, "features")
        self.pp = pp_from_arrays(arr) if self.input_kind == "features" else None
        self.platt = tuple(float(v) for v in arr["platt"])
        # V11: models with piecewise-constant scores (decision tree) may carry an isotonic calibration fitted on the same
        # leave-one-unit-out out-of-fold scores as their original Platt parameters (scripts/v11_recalibrate_decision_tree.py)
        self.iso = (np.asarray(arr["iso_x"], float), np.asarray(arr["iso_y"], float)) if "iso_x" in arr else None
        if kind == "v10_patch_numpy_cnn":
            import json as _json
            from src.classical.numpy_cnn import CNNConfig, NumpyCNN
            cfg = _json.loads(bytes(arr["cnn_config_json"].tolist()).decode())
            net = NumpyCNN(CNNConfig(**cfg))
            net.set_state({k[5:]: v for k, v in arr.items() if k.startswith("cnn__")})
            self.model = net
        elif kind == "v10_patch_backbone_ft":
            from src.classical.cnn_finetune import TailClassifier
            m = TailClassifier(c["arch"])
            m.load_state({k[6:]: v for k, v in arr.items() if k.startswith("tail__")})
            self.model = m
        elif kind == "v10_patch_sklearn":
            self.model = A.load_sklearn(self.dir)
        elif kind == "v10_patch_vqc":
            from src.v10.quantum.vqc import VQCClassifier
            self.model = VQCClassifier.from_arrays(arr)
        elif kind == "v10_patch_qksvm":
            from src.v10.quantum.qksvm import QuantumKernelSVM
            self.model = QuantumKernelSVM.from_arrays(arr)
        elif kind == "v10_patch_hybrid":
            from src.v10.hybrid.models import HybridClassifier
            self.model = HybridClassifier.from_arrays(arr)
        elif kind == "v10_patch_late_fusion":
            self.model = _LateFusionRuntime(arr)
        else:
            raise ModelUnavailable(f"unknown model kind {kind!r}")
        # self-test: finite output of the right shape on a fixed input
        x = {"stacks": np.zeros((2, 3, 64, 64), np.float32), "patches": np.zeros((2, 96, 96), np.uint8)}.get(
            self.input_kind, np.zeros((2, int(c.get("feature_dim") or 1))))
        s = self._scores(x)
        if s.shape != (2,) or not np.all(np.isfinite(s)):
            raise ModelUnavailable("self-test failed (non-finite or wrong-shaped output)")

    # ------------------------------------------------------------------ inference
    def _scores(self, F: np.ndarray) -> np.ndarray:
        if self.input_kind == "stacks":
            p = np.asarray(self.model.predict_proba(F), float)
            p = p[:, 1] if p.ndim == 2 else p
            return np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
        if self.input_kind == "patches":
            return np.asarray(self.model.predict_images(np.asarray(F, np.uint8)), float).ravel()
        Z = self.pp.transform(F)
        if self.card["kind"] == "v10_patch_sklearn":
            from src.v10.classical.models import decision
            return decision(self.model, Z)
        return np.asarray(self.model.decision_function(Z), float)

    def calibrate(self, scores) -> np.ndarray:
        """Score -> probability with the model's own calibration (isotonic when the card says so, else Platt)."""
        if getattr(self, "iso", None) is not None:
            return np.interp(np.asarray(scores, float), self.iso[0], self.iso[1])
        return _platt(scores, self.platt)

    @property
    def calibration_note(self) -> str:
        if getattr(self, "iso", None) is not None:
            return "isotonic calibration on leave-one-unit-out out-of-fold scores (V11 recalibration, see model card)"
        return "Platt-calibrated on inner out-of-fold predictions of training units (V10 nested evaluation)"

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        self.load_model()
        return self.calibrate(self._scores(np.asarray(F, float)))

    def features(self, patches2d: np.ndarray) -> np.ndarray:
        from src.v10.features import extract
        return extract.extract(self.card["feature_set"], patches2d)

    def preprocess(self, inp: ModelInput):
        from src.v10.engine import patch_pipeline
        return patch_pipeline.run(self.engine.v9, inp.data, inp.filename, inp.prev, inp.next)

    def predict(self, inp: ModelInput, explain_req: bool = False) -> PredictionOutput:
        self.load_model()
        t0 = time.perf_counter()
        fe = self.preprocess(inp)
        md = self.get_metadata()
        po = PredictionOutput(self.model_id, self.version, "withheld", None, None, self.card["threshold"],
                              md["execution_mode"], md["dataset_version"],
                              probability_note=self.calibration_note + "; not a clinically calibrated confidence")
        po.details = {"image": fe.meta, "quality": fe.quality, "ood": fe.ood, "liver_gate": fe.gate,
                      "liver_check": fe.liver_check}
        if fe.preview_png:
            po.images["preview"] = fe.preview_png
        po.timings_ms.update(fe.timings_ms)
        if fe.stop_reason:
            po.label, po.reason = "Uncertain / unable to classify", fe.stop_reason
            po.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
            return po
        t1 = time.perf_counter()
        if self.input_kind == "stacks":
            if fe.stacks is None:
                po.status, po.label = "needs_context", "Uncertain / unable to classify"
                po.reason = "this model reads adjacent slices; upload the previous and next slices to run it"
                return po
            from src.inference.registry import _down64
            F = _down64(fe.stacks).astype(np.float32) / 255.0
        elif self.input_kind == "patches":
            F = fe.patches
        else:
            F = self.features(fe.patches)
        po.timings_ms["features"] = round((time.perf_counter() - t1) * 1000, 1)
        t2 = time.perf_counter()
        p = self.calibrate(self._scores(F))
        po.timings_ms["model"] = round((time.perf_counter() - t2) * 1000, 1)
        k = int(np.argmax(p))
        thr = float(self.card["threshold"])
        pos = [fe.positions[i] for i in fe.liver_idx]
        po.status, po.probability, po.label = "ok", round(float(p[k]), 4), slice_call(float(p[k]), thr)
        po.details.update({"n_liver_patches": int(len(p)), "band": 0.10,
                           "top_patch": {"y": int(pos[k][0]), "x": int(pos[k][1])},
                           "patch_probabilities": [{"y": int(y), "x": int(x), "p": round(float(v), 4)}
                                                   for (y, x), v in zip(pos, p)]})
        po.images["patch_map"] = explain.patch_map_png(fe.image, pos, list(p), threshold=thr)
        po.explanation["patch_map"] = {"method": "patch probabilities of this model on the 96-px grid",
                                       "caution": explain.CAUTION}
        if explain_req and self.input_kind == "features":
            po.explanation.update(self._explain(F[k]))
        po.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return po

    def _explain(self, f: np.ndarray) -> dict:
        names = self.card.get("feature_names") or []
        kind = self.card["kind"]
        try:
            if kind == "v10_patch_sklearn":
                clf = self.model.named_steps["clf"]
                if hasattr(clf, "coef_"):
                    z = self.model.named_steps["scale"].transform(self.pp.transform(f[None]))[0]
                    return {"feature_contributions": {"method": "standardised feature x coefficient (top patch)",
                                                      "items": explain.linear_contributions(clf.coef_, z, names)}}
                return {"feature_importance": {"method": "training-time importance", "items":
                                               self.card.get("feature_importance", [])}}
            fn = self._scores
            sens = explain.numeric_sensitivity(fn, f.astype(float), eps=1e-3 * (np.abs(f).mean() + 1e-6))
            order = np.argsort(-np.abs(sens))[:8]
            return {"input_sensitivity": {"method": "finite-difference d logit / d feature at the top patch "
                                                    "(input perturbation analysis)",
                                          "items": [{"feature": names[i] if i < len(names) else f"f{i}",
                                                     "sensitivity": float(sens[i])} for i in order]}}
        except Exception as e:  # explanation failure never changes the prediction
            return {"explanation_error": type(e).__name__}

    # ------------------------------------------------------------------ description
    def get_metadata(self) -> dict:
        c = self.card
        return {"id": self.model_id, "name": c["name"], "family": c["family"], "architecture": c["architecture"],
                "architecture_detail": c.get("architecture_detail"), "task": c["task"], "version": self.version,
                "source": "v10", "role": c.get("role"), "training_status": "TRAINED" if c.get("training_data") == "real"
                else "SYNTHETIC (software test only)", "training_date": c.get("saved_utc"),
                "dataset_version": c.get("dataset_version"), "training_data": c.get("training_data"),
                "execution_mode": c.get("execution_mode"), "threshold": c.get("threshold"), "calibrated": True,
                "artifact_path": f"artifacts/v10/{self.model_id}", "artifact_sha256": next(iter(c["files"].values()), None),
                "size_bytes": A.directory_size(self.dir), "parameters": c.get("n_parameters"),
                "input_contract": {"kind": "ct_slice", "formats": ["png", "jpeg", "dicom", "nifti"],
                                   "context_slices_each_side": 1 if c.get("requires_context") else 0,
                                   "requires_context": bool(c.get("requires_context")), "patch_px": 96,
                                   "stride_px": 64, "features": c["feature_set"]},
                "circuit": c.get("circuit"), "configuration": c.get("selected_config"),
                "preprocessing": c.get("preprocessing"), "seeds": c.get("seeds")}

    def get_metrics(self) -> dict:
        return {"evaluations": self.card.get("evaluations", [])}

    def attainable_probability_range(self) -> tuple[float, float] | None:
        """For models whose score is the logit of a clipped predict_proba (trees, forests, k-NN, boosting, MLP), the
        Platt-calibrated probability is bounded; returns (min, max) or None when the score is unbounded."""
        if self.card["kind"] != "v10_patch_sklearn" or self.model is None:
            return None
        clf = self.model.named_steps["clf"]
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        if hasattr(clf, "decision_function") and not isinstance(clf, (MLPClassifier, HistGradientBoostingClassifier)):
            return None
        if getattr(self, "iso", None) is not None:
            return float(np.min(self.iso[1])), float(np.max(self.iso[1]))
        L = float(np.log((1 - 1e-6) / 1e-6))
        return float(_platt(-L, self.platt)), float(_platt(L, self.platt))

    def deployability(self, band: float = 0.10) -> str | None:
        """Reason why the model can never make a confident call under the +/- band rule, or None if it can."""
        r = self.attainable_probability_range()
        if r is None:
            return None
        thr = float(self.card["threshold"])
        lo, hi = r
        if hi < thr + band:
            return (f"non-deployable: its calibrated probability is bounded at {hi:.3f}, below threshold + band "
                    f"({thr:.3f} + {band:.2f}), so it can never return 'Tumour detected' (audit finding 3; the "
                    "threshold was not changed because no validation data is available to re-select it)")
        return None


class SyntheticImageModel(BaseModel):
    """Logistic regression on 32x32 down-sampled synthetic phantoms. SOFTWARE TEST ONLY."""

    def __init__(self, engine, directory: Path, card: dict):
        self.engine, self.dir, self.card = engine, Path(directory), card
        self.model_id, self.version = card["model_id"], card.get("version", "v10.0-test")
        self.a = None

    def load_model(self) -> None:
        if not self.engine.allow_synthetic:
            raise ModelUnavailable("synthetic software-test model; enable ALLOW_SYNTHETIC_MODELS only in tests")
        A.verify(self.dir, self.card)
        self.a = A.load_arrays(self.dir)

    def preprocess(self, inp: ModelInput):
        from src.v10.engine.patch_pipeline import decode
        _, can, qc = decode(inp.data, inp.filename)
        small = np.asarray(Image.fromarray(can.image).resize((32, 32), Image.BILINEAR), float).ravel() / 255.0
        return can.image, small, qc

    def predict_proba(self, X):
        z = (np.asarray(X, float) - self.a["mu"]) / self.a["sd"] @ self.a["coef"] + float(self.a["b"][0])
        return 1 / (1 + np.exp(-z))

    def predict(self, inp: ModelInput, explain_req: bool = False) -> PredictionOutput:
        self.load_model()
        t0 = time.perf_counter()
        img, x, qc = self.preprocess(inp)
        p = float(self.predict_proba(x[None])[0])
        thr = float(self.card["threshold"])
        buf = io.BytesIO()
        Image.fromarray(img).resize((384, 384)).save(buf, format="PNG")
        label = "Synthetic class B" if p >= thr else "Synthetic class A"
        return PredictionOutput(self.model_id, self.version, "ok", label, round(p, 4), thr, "classical_cpu_numpy",
                                self.card.get("dataset_version"),
                                probability_note="software-test model trained on synthetic phantoms; meaningless "
                                                 "for medical images",
                                details={"quality": {"status": qc["status"]}},
                                timings_ms={"total": round((time.perf_counter() - t0) * 1000, 1)},
                                images={"preview": buf.getvalue()},
                                notes=["SYNTHETIC SOFTWARE-TEST MODEL — not a tumour classifier"])

    def get_metadata(self) -> dict:
        c = self.card
        return {"id": self.model_id, "name": c["name"], "family": "classical", "architecture": "Logistic Regression",
                "task": c["task"], "version": self.version, "source": "v10", "role": "software_test",
                "training_status": "SYNTHETIC (software test only)", "training_date": c.get("saved_utc"),
                "dataset_version": c.get("dataset_version"), "training_data": "synthetic",
                "execution_mode": "classical_cpu_numpy", "threshold": c.get("threshold"), "calibrated": False,
                "artifact_path": f"artifacts/v10/{self.model_id}", "artifact_sha256": next(iter(c["files"].values())),
                "size_bytes": A.directory_size(self.dir), "parameters": 32 * 32 + 1,
                "input_contract": {"kind": "any_image", "formats": ["png", "jpeg", "dicom", "nifti"],
                                   "requires_context": False}}

    def get_metrics(self) -> dict:
        return {"evaluations": self.card.get("evaluations", [])}
