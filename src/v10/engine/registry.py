"""Model engine: discovers, integrity-checks and serves every available model through the common interface.

Sources
  V9   artifacts/{classical,quantum,hybrid}/<id>/model.{json,npz}  loaded by the vendored V9 registry (46 models);
       V9-version models are offered; V8-era models stay loadable but are listed as legacy and not offered.
  V10  artifacts/v10/<id>/model_card.json (+ model.npz / model.joblib)

A model is servable only if it loaded and passed its checks AND every runtime dependency is present; otherwise it is
listed with status and reason and prediction requests for it are refused.

Availability categories (descriptor "availability"), V10 audit finding 7:
  available             loads, passes integrity + self-test, every dependency present
  requires_context      available, but needs adjacent slices (2.5-D / 3-D / 5-slice) for every prediction
  missing_dependency    registered and loadable, but a runtime dependency is missing (ImageNet backbone weights, an
                        upstream model such as the fine-tuned EfficientNet-B0)
  missing_weights       the model's own weight file is absent
  legacy                V8-era model kept for reproducibility; not offered
  not_trained           never trained in this environment
  failed_integrity      artifact present but failed integrity / self-test
  non_deployable        loads, but cannot make meaningful predictions (e.g. decision tree bounded below its band)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.v10.engine import artifacts as A
from src.v10.engine import weights as W
from src.v10.engine.interface import ModelInput, ModelUnavailable, PredictionOutput
from src.v10.engine.v9_adapter import V9Model
from src.v10.engine.v10_models import PATCH_KINDS, SyntheticImageModel, V10PatchModel

log = logging.getLogger("lti.engine")


class _Entry:
    def __init__(self, model, descriptor: dict):
        self.model, self.descriptor = model, descriptor


class ModelEngine:
    def __init__(self, project_root: Path, allow_synthetic: bool = False):
        self.root = Path(project_root)
        self.allow_synthetic = allow_synthetic
        self.v9 = None
        self.entries: dict[str, _Entry] = {}
        self.load_seconds = None
        self.v9_error = None
        self.weights: dict = {}
        self.model_load_ms: dict[str, float] = {}

    # ------------------------------------------------------------------ loading
    def load(self) -> "ModelEngine":
        t0 = time.time()
        self.weights = W.check()
        self._load_v9()
        self._load_v10()
        self._apply_runtime_checks()
        self.load_seconds = round(time.time() - t0, 2)
        return self

    # ------------------------------------------------------------------ runtime dependency checks
    @staticmethod
    def compare_family(d: dict) -> str:
        """Family used by the four-paradigm comparison: classical | deep_learning | quantum | hybrid."""
        fam = d.get("family") or "classical"
        arch = str(d.get("architecture") or "")
        mid = str(d.get("id") or "")
        backbone_id = any(mid.startswith(p) for p in ("resnet", "densenet", "efficientnet", "convnext", "cnn"))
        if fam == "classical" and (arch in ("ResNet-50", "DenseNet-121", "EfficientNet-B0") or backbone_id) \
                or arch.startswith("CNN"):
            return "deep_learning"
        return fam if fam in ("classical", "deep_learning", "quantum", "hybrid") else "classical"

    def _dependencies(self, mid: str, d: dict) -> list[dict]:
        """Runtime dependencies of a patch model (every patch model uses the EfficientNet-B0 front end)."""
        deps = []
        if d.get("family") == "segmentation" or d.get("role") == "software_test":
            return deps
        eff = self.weights.get("efficientnet_b0", {})
        deps.append({"name": "EfficientNet-B0 ImageNet weights (OOD detector + liver gate front end)",
                     "ok": bool(eff.get("ok")), "detail": eff.get("reason") or eff.get("file")})
        arch = str(d.get("architecture") or "")
        for key, label in (("densenet121", "DenseNet-121"), ("resnet50", "ResNet-50")):
            if arch == label or key in mid:
                w = self.weights.get(key, {})
                deps.append({"name": f"{label} ImageNet weights", "ok": bool(w.get("ok")),
                             "detail": w.get("reason") or w.get("file")})
        if self.v9 is not None:
            deps.append({"name": "U-Net liver validation (unet25d_v9)", "ok": self.v9.unet is not None,
                         "detail": "optional: without it only the patch gate checks for liver", "optional": True})
        return deps

    def _apply_runtime_checks(self) -> None:
        for mid, e in self.entries.items():
            d = e.descriptor
            d["compare_family"] = self.compare_family(d)
            d["load_ms"] = self.model_load_ms.get(mid)
            deps = self._dependencies(mid, d)
            d["dependencies"] = deps
            ic = d.get("input_contract") or {}
            d["context_slices_each_side"] = int(ic.get("context_slices_each_side") or 0)
            missing = [x for x in deps if not x["ok"] and not x.get("optional")]
            st, reason = d.get("status"), str(d.get("reason") or "")
            if d.get("servable") and missing:
                d.update(servable=False, status="missing_dependency", deployment_status="not_deployed",
                         reason="; ".join(f"{x['name']}: {x['detail']}" for x in missing))
            elif d.get("servable") and isinstance(e.model, V10PatchModel):
                why = e.model.deployability()
                if why:
                    d.update(servable=False, status="non_deployable", deployment_status="not_deployed", reason=why)
            st, reason = d.get("status"), str(d.get("reason") or "")
            if d.get("servable"):
                av = "requires_context" if d["context_slices_each_side"] else "available"
            elif st == "legacy":
                av = "legacy"
            elif st == "not_trained":
                av = "not_trained"
            elif st in ("missing_dependency", "non_deployable"):
                av = st
            elif "model.npz missing" in reason or ("No such file" in reason and "artifacts/weights" not in reason):
                av = "missing_weights"
            elif "artifacts/weights" in reason or "which is not loaded" in reason:
                av = "missing_dependency"
            else:
                av = "failed_integrity"
            d["availability"] = av

    def _v9_metrics_rows(self) -> dict:
        p = self.root / "results" / "v9" / "metrics_v9.csv"
        if not p.exists():
            return {}
        import pandas as pd
        df = pd.read_csv(p)
        return {str(r["model_id"]): r.to_dict() for _, r in df.iterrows()}

    def _load_v9(self) -> None:
        if not (self.root / "artifacts" / "classical").exists():
            self.v9_error = "V9 artifacts not imported (run scripts/v10_import_v9_artifacts.py)"
            return
        import src.inference.registry as R9
        from src.inference.registry import Registry
        # V11: per-model load time (artifact read + integrity + build + self-test), measured by wrapping two V9
        # functions for the duration of the load only; the vendored code itself is unchanged.
        orig_load, orig_test = R9.load_model, R9.LoadedModel.self_test
        starts: dict[str, float] = {}

        def timed_load(fam, mid):
            starts[mid] = time.perf_counter()
            return orig_load(fam, mid)

        def timed_test(lm):
            try:
                return orig_test(lm)
            finally:
                if lm.model_id in starts:
                    self.model_load_ms[lm.model_id] = round((time.perf_counter() - starts.pop(lm.model_id)) * 1000, 1)
        R9.load_model, R9.LoadedModel.self_test = timed_load, timed_test
        try:
            reg = Registry()
            reg.load()
        except Exception as e:
            self.v9_error = f"{type(e).__name__}: {e}"
            log.error("V9 registry failed to load", exc_info=True)
            return
        finally:
            R9.load_model, R9.LoadedModel.self_test = orig_load, orig_test
        self.v9 = reg
        rows = self._v9_metrics_rows()
        for mid, st in reg.status.items():
            if st.get("family") == "segmentation":
                continue
            m = V9Model(self, mid, rows.get(mid) or rows.get(f"v8:{mid}"))
            is_v9 = st.get("version") == "v9" or mid.endswith("_v9") or "_v9_" in mid
            loaded = st.get("status") == "loaded"
            if loaded:
                d = m.get_metadata()
                d["metrics"] = m.get_metrics()
            else:
                d = {"id": mid, "name": mid, "family": st.get("family") or "classical", "architecture": "",
                     "task": "", "version": "v9" if is_v9 else "v8", "source": "v9",
                     "training_status": "NOT TRAINED" if st.get("status") == "not_trained" else "FAILED",
                     "execution_mode": None, "metrics": {}, "input_contract": {}}
                # V11: describe a model that could not load from its own model.json (name, architecture, threshold,
                # artifact folder) so the registry shows WHAT is missing, not just an id.
                mj = self.root / "artifacts" / (st.get("family") or "classical") / mid / "model.json"
                if mj.is_file():
                    try:
                        meta = json.loads(mj.read_text())
                        d.update(name=meta.get("name") or mid, architecture=str(meta.get("arch") or "").replace(
                                     "_", "-").upper() if meta.get("arch") else "",
                                 architecture_detail=meta.get("architecture"), task=meta.get("task") or "",
                                 threshold=meta.get("threshold"), artifact_path=str(mj.parent.relative_to(self.root)),
                                 expected_sha256=meta.get("npz_sha256"),
                                 training_status=("TRAINED (weights not in this installation)"
                                                  if meta.get("status") == "trained" else d["training_status"]))
                    except (OSError, ValueError):
                        pass
            if loaded and is_v9 and d["family"] != "segmentation":
                status, servable, reason = "available", True, None
            elif loaded:
                status, servable, reason = "legacy", False, "V8-era model kept for reproducibility; not offered"
            else:
                status = "not_trained" if st.get("status") == "not_trained" else "failed"
                servable, reason = False, st.get("detail")
            d.update(status=status, servable=servable, reason=reason,
                     deployment_status="deployed" if servable else "not_deployed")
            self.entries[mid] = _Entry(m if loaded else None, d)

    def _load_v10(self) -> None:
        base = self.root / "artifacts" / "v10"
        if not base.exists():
            return
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            try:
                card = A.load_card(d)
            except Exception as e:
                self.entries[d.name] = _Entry(None, {"id": d.name, "name": d.name, "family": "unknown",
                                                     "architecture": "", "task": "", "version": "v10", "source": "v10",
                                                     "status": "failed", "servable": False, "reason": str(e),
                                                     "deployment_status": "not_deployed", "metrics": {}})
                continue
            kind = card.get("kind")
            m = SyntheticImageModel(self, d, card) if kind == "synthetic_image_logreg" else (
                V10PatchModel(self, d, card) if kind in PATCH_KINDS else None)
            if m is None:
                continue
            desc = m.get_metadata()
            desc["metrics"] = m.get_metrics()
            try:
                if isinstance(m, V10PatchModel) and self.v9 is None:
                    raise ModelUnavailable("needs the V9 front end (OOD detector, liver gate), which is not loaded")
                t_load = time.perf_counter()
                m.load_model()
                self.model_load_ms[card["model_id"]] = round((time.perf_counter() - t_load) * 1000, 1)
                desc.update(status="available", servable=True, reason=None, deployment_status="deployed")
            except (ModelUnavailable, A.ArtifactError) as e:
                synth = card.get("training_data") != "real"
                desc.update(status="synthetic_only" if synth else "failed", servable=False, reason=str(e),
                            deployment_status="not_deployed")
            except Exception as e:
                desc.update(status="failed", servable=False, reason=f"{type(e).__name__}: {e}",
                            deployment_status="not_deployed")
            self.entries[card["model_id"]] = _Entry(m, desc)

    # ------------------------------------------------------------------ queries
    def list(self, include_legacy: bool = False) -> list[dict]:
        out = [e.descriptor for e in self.entries.values() if include_legacy or e.descriptor.get("status") != "legacy"]
        order = {"available": 0, "synthetic_only": 1, "not_trained": 2, "failed": 3, "legacy": 4}
        return sorted(out, key=lambda d: (order.get(d.get("status"), 9), d.get("source") != "v10", d.get("family", ""),
                                          d.get("id")))

    def describe(self, model_id: str) -> dict:
        e = self.entries.get(model_id)
        if e is None:
            raise KeyError(model_id)
        return e.descriptor

    def get(self, model_id: str):
        e = self.entries.get(model_id)
        if e is None:
            raise KeyError(model_id)
        if not e.descriptor.get("servable"):
            raise ModelUnavailable(e.descriptor.get("reason") or e.descriptor.get("status") or "not servable")
        return e.model

    def predict(self, model_id: str, inp: ModelInput, explain: bool = False) -> PredictionOutput:
        return self.get(model_id).predict(inp, explain)

    def summary(self) -> dict:
        ds = [e.descriptor for e in self.entries.values()]
        return {"models_total": len(ds), "models_servable": sum(1 for d in ds if d.get("servable")),
                "weights": {k: {"ok": v["ok"], "reason": v["reason"]} for k, v in self.weights.items()},
                "unet_liver_validation": bool(self.v9 is not None and self.v9.unet is not None),
                "v9_engine": "loaded" if self.v9 is not None else ("error: " + self.v9_error if self.v9_error else
                                                                    "not loaded"),
                "ood_detector": bool(self.v9 is not None and self.v9.ood is not None and self.weights.get("efficientnet_b0", {}).get("ok")),
                "liver_gate": bool(self.v9 is not None and self.v9.gate is not None and self.weights.get("efficientnet_b0", {}).get("ok")),
                "allow_synthetic": self.allow_synthetic, "engine_load_seconds": self.load_seconds}
