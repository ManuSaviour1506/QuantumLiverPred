"""Adapter that serves the trained V9 models through the V10 model interface.

The V9 code is vendored byte-for-byte (src/classical, src/inference, ... — SHA-256 identical to the V9 repository) and
is called, never modified. Each V9 model keeps its V9 identity, threshold, calibration and recorded metrics; the
recorded metrics are from V9's fixed split (train G03–G06, validation G02/G09/G10, test G01/G07/G08), patch level, and
carry the Stage 2 caveats.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np

from src.v10.engine import explain
from src.v10.engine.interface import BaseModel, InputRejected, ModelInput, ModelUnavailable, PredictionOutput

V9_SPLIT = {"train": ["G03", "G04", "G05", "G06"], "validation": ["G02", "G09", "G10"], "test": ["G01", "G07", "G08"]}
V9_CAVEATS = [
    "V9 fixed split; the test groups were scored once, but the validation set is saturated (Stage 2 finding F).",
    "Patch-level metrics on liver-tissue patches of held-out reconstructed scan groups, not patient-level.",
    "531 of the 622 test patches come from one group (G01).",
    "Models on V9 CNN embeddings: the CNN was trained on the V9 training groups; their comparison is exploratory.",
]
QUANTUM_KINDS = {"v9_vqc", "v9_qksvm", "v9_hybrid", "vqc", "quantum_kernel", "hybrid"}


def _family(meta: dict, fam_dir: str) -> str:
    k = str(meta.get("kind") or "")
    if k in ("v9_vqc", "vqc", "v9_qksvm", "quantum_kernel"):
        return "quantum"
    if k in ("v9_hybrid", "hybrid"):
        return "hybrid"
    if k in ("numpy_cnn", "cnn_finetune"):
        return "deep_learning"
    return "classical" if fam_dir == "classical" else fam_dir


def _architecture_label(mid: str, meta: dict) -> str:
    k = meta.get("kind")
    if k == "cnn_finetune":
        return {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "efficientnet_b0": "EfficientNet-B0"}.get(
            meta.get("arch"), str(meta.get("arch")))
    if k == "numpy_cnn":
        return "CNN (NumPy)"
    if k in ("v9_vqc", "vqc"):
        return "VQC"
    if k in ("v9_qksvm", "quantum_kernel"):
        return "Quantum kernel SVM"
    if k in ("v9_hybrid", "hybrid"):
        return "Hybrid quantum-classical"
    if "logreg" in mid or "lr_control" in mid:
        return "Logistic Regression"
    if "random_forest" in mid:
        return "Random Forest"
    if "svm" in mid or "rbf" in mid:
        return "SVM (RBF)"
    if "twin" in mid:
        return "Classical twin (MLP)"
    return k or "model"


class V9Model(BaseModel):
    def __init__(self, engine, model_id: str, metrics_row: dict | None):
        self.engine, self.model_id, self.version = engine, model_id, "v9"
        self.metrics_row = metrics_row or {}

    @property
    def lm(self):
        return self.engine.v9.models.get(self.model_id)

    def load_model(self) -> None:
        st = self.engine.v9.status.get(self.model_id, {})
        if st.get("status") != "loaded":
            raise ModelUnavailable(st.get("detail") or "not loaded")

    def preprocess(self, inp: ModelInput):
        from src.v10.engine.patch_pipeline import decode
        return decode(inp.data, inp.filename)

    def predict(self, inp: ModelInput, explain_req: bool = False) -> PredictionOutput:
        from src.inference.pipeline import InputError, analyse
        self.load_model()
        t0 = time.perf_counter()
        try:
            out = analyse(self.engine.v9, inp.data, inp.filename, model_ids=[self.model_id], prev=inp.prev,
                          nxt=inp.next, prev2=inp.prev2, nxt2=inp.next2)
        except InputError as e:
            raise InputRejected(e.status, e.message, e.detail)
        card = out["models"][0]
        liver_check = None
        gate = out.get("liver_gate") or {}
        if gate.get("n_liver_patches"):
            liver_check = self._liver_check(inp, gate)
            if liver_check.get("status") == "rejected":
                card = dict(card, status="withheld", label="Uncertain / unable to classify", probability=None,
                            patch_probabilities=None, top_patch=None,
                            reason=(f"liver region not confirmed: the patch gate kept {gate['n_liver_patches']} patch(es) "
                                    f"but the U-Net liver segmentation found {liver_check['unet_liver_px']} px "
                                    f"(< {liver_check['min_px']} px); the models only analyse liver tissue"))
        meta = self.get_metadata()
        po = PredictionOutput(
            model_id=self.model_id, model_version="v9", status=card.get("status", "withheld"), label=card.get("label"),
            probability=card.get("probability"), threshold=card.get("threshold", self.lm.threshold if self.lm else None),
            execution_mode=meta["execution_mode"], dataset_version=meta["dataset_version"],
            probability_note=card.get("probability_note") or "", reason=card.get("reason"),
            details={"image": out.get("image"), "quality": out.get("quality"), "ood": out.get("ood"),
                     "liver_gate": {k: v for k, v in (out.get("liver_gate") or {}).items() if k != "p_liver"},
                     "n_liver_patches": card.get("n_liver_patches"), "top_patch": card.get("top_patch"),
                     "patch_probabilities": card.get("patch_probabilities"), "band": card.get("band"),
                     "context": card.get("context"), "liver_check": liver_check},
            timings_ms={"total": round((time.perf_counter() - t0) * 1000, 1),
                        "model": card.get("processing_ms") or 0.0})
        if out.get("preview_png"):
            po.images["preview"] = base64.b64decode(out["preview_png"])
        pp = card.get("patch_probabilities") or []
        if pp:
            from src.v10.engine.patch_pipeline import decode
            _, can, _ = decode(inp.data, inp.filename)
            po.images["patch_map"] = explain.patch_map_png(can.image, [(p["y"], p["x"]) for p in pp],
                                                           [p["p"] for p in pp], threshold=po.threshold)
            po.explanation["patch_map"] = {"method": "patch probabilities of this model on the 96-px grid",
                                           "caution": explain.CAUTION}
        if explain_req and po.status == "ok" and self.lm is not None and self.lm.kind == "numpy_cnn":
            po.explanation.update(self._gradcam(inp))
        if self.engine.v9.status.get(self.model_id, {}).get("status") == "loaded" and self.lm.kind == "numpy_cnn":
            po.notes.append("NumPy CNN trained from scratch on the V9 training groups")
        return po

    def _liver_check(self, inp: ModelInput, gate: dict) -> dict:
        from src.v10.engine.patch_pipeline import decode, liver_validation
        _, can, _ = decode(inp.data, inp.filename)
        pi = ni = None
        if inp.prev is not None and inp.next is not None:
            try:
                pi, ni = decode(inp.prev, None)[1].image, decode(inp.next, None)[1].image
            except InputRejected:
                pi = ni = None
        lc = liver_validation(self.engine.v9, can.image, pi, ni)
        lc.pop("mask64", None)
        return lc

    def _gradcam(self, inp: ModelInput) -> dict:
        from src.inference.v9_services import ServiceUnavailable, gradcam_slice
        from src.inference.pipeline import InputError
        try:
            g = gradcam_slice(self.engine.v9, inp.data, inp.filename, inp.prev, inp.next, model_id=self.model_id)
        except (ServiceUnavailable, InputError) as e:
            return {"gradcam": {"status": "unavailable", "reason": getattr(e, "message", str(e))}}
        if g.get("status") == "no_liver":
            return {"gradcam": {"status": "unavailable", "reason": g.get("detail")}}
        return {"gradcam": {"status": "ok", "method": g["method"], "caution": g["caution"],
                            "liver_source": g["liver_source"], "png_b64": g["overlay_png"]}}

    def predict_proba(self, patches_or_ctx):
        from src.inference.registry import PatchContext
        self.load_model()
        ctx = patches_or_ctx
        if isinstance(patches_or_ctx, np.ndarray):
            p = patches_or_ctx.astype(np.uint8)
            stacks = np.repeat(p[:, None], 3, 1) if p.ndim == 3 else p
            emb = self.engine.v9.models[self.engine.v9.selected_cnn].m if self.engine.v9.selected_cnn else None
            ctx = PatchContext(stacks[:, 1] if stacks.ndim == 4 else p, stacks, self.engine.v9.ft_model(), None, emb)
        _, prob = self.lm.probabilities(ctx)
        return np.asarray(prob, float)

    def get_metadata(self) -> dict:
        lm = self.lm
        st = self.engine.v9.status.get(self.model_id, {})
        meta = lm.meta if lm else {}
        fam_dir = st.get("family") or "classical"
        family = _family(meta, fam_dir)
        quantum = family in ("quantum", "hybrid")
        from src.inference.registry import training_label
        art_dir = Path(self.engine.root) / "artifacts" / fam_dir / self.model_id
        size = int(sum(p.stat().st_size for p in art_dir.glob("*") if p.is_file())) if art_dir.exists() else None
        ctx_slices = getattr(lm, "context_slices", 0) if lm else 0
        return {
            "id": self.model_id, "name": meta.get("name") or self.model_id, "family": family,
            "architecture": _architecture_label(self.model_id, meta), "architecture_detail": meta.get("architecture"),
            "task": "tumour vs tumour-free liver tissue (96x96 patches) aggregated to a slice-level call",
            "version": "v9", "source": "v9", "role": meta.get("role") or self.engine.v9.role(self.model_id),
            "training_status": training_label(meta, lm is not None), "training_date": meta.get("saved_utc"),
            "dataset_version": "lits_verified_v8 (V9 fixed split)", "training_data": "real",
            "execution_mode": "custom_numpy_simulator" if quantum else "classical_cpu_numpy",
            "quantum_backend": meta.get("quantum_backend"), "threshold": meta.get("threshold"),
            "calibrated": bool(lm is not None and lm.calibrator is not None),
            "artifact_path": f"artifacts/{fam_dir}/{self.model_id}", "artifact_sha256": meta.get("npz_sha256"),
            "size_bytes": size, "parameters": meta.get("trainable_parameters") or meta.get("parameters"),
            "input_contract": {"kind": "ct_slice", "formats": ["png", "jpeg", "dicom", "nifti"],
                               "context_slices_each_side": ctx_slices,
                               "requires_context": bool(ctx_slices),
                               "patch_px": 96, "stride_px": 64},
            "circuit": meta.get("circuit") if quantum else None,
            "configuration": meta.get("configuration") if quantum else None,
        }

    def get_metrics(self) -> dict:
        lm = self.lm
        meta = lm.meta if lm else {}
        row = self.metrics_row
        test = dict(meta.get("test_summary") or {})
        for k_csv, k in (("balanced_accuracy", "balanced_accuracy"), ("precision", "precision"), ("brier", "brier"),
                         ("train_seconds", "train_seconds"), ("inference_ms_per_patch", "inference_ms_per_patch"),
                         ("parameters", "parameters")):
            v = row.get(k_csv)
            if v is not None and v == v:
                test[k] = v
        if {"tp", "fn", "fp", "tn"} <= set(row) and all(row[k] == row[k] for k in ("tp", "fn", "fp", "tn")):
            tp, fn, fp, tn = (float(row[k]) for k in ("tp", "fn", "fp", "tn"))
            den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            test["mcc"] = float((tp * tn - fp * fn) / den) if den else None
            test["confusion"] = {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}
        return {"evaluations": [{
            "name": "V9 held-out test (recorded)", "source": "results/v9/metrics_v9.csv and model.json",
            "split": V9_SPLIT, "evaluation_unit": "patch level on held-out reconstructed scan groups",
            "test_patches": 622, "status": "recorded V9 result",
            "metrics": test, "validation": {"roc_auc": meta.get("validation_auc"), "ece": meta.get("validation_ece")},
            "per_group_auc": {k[4:]: v for k, v in row.items() if k.startswith("auc_G") and v == v},
            "caveats": V9_CAVEATS}]}
