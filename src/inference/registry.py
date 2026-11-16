"""Model registry: loads every saved model with integrity checks and exposes one predictor interface.

A predictor maps a batch of 96x96 liver patches (plus a per-request feature cache) to calibrated tumour probabilities.
Models whose artifacts are missing are reported as NOT TRAINED; models that fail integrity checks are reported as
FAILED and never produce predictions (they are not replaced by an untrained or random model).
"""
from __future__ import annotations

import json
import time
import traceback

import numpy as np
from PIL import Image

from src.classical import backbones as B
from src.classical import baselines as BL
from src.classical.cnn_finetune import TailClassifier
from src.classical.numpy_cnn import CNNConfig, NumpyCNN
from src.classical.nn import Tensor
from src.evaluation.guards import SafeCalibrator
from src.hybrid.hybrid_model import ClassicalTwin, HybridQuantum
from src.inference.artifacts import ModelIntegrityError, list_models, load_model, model_dir
from src.inference.features import backbone
from src.preprocessing.ct import to_rgb
from src.quantum.models import PCAEncoder, QKernelSVM, VQC
from src.quantum.v9 import HybridV9, QKernelV9, TwinV9, VQCv9

EXPECTED = {
    "classical": ["efficientnet_b0_ft", "densenet121_ft", "resnet50_ft", "cnn25d_scratch", "cnn2d_scratch",
                  "svm_rbf_radiomics", "random_forest_radiomics", "logreg_radiomics",
                  "pca8_logreg_ft", "pca8_svm_rbf_ft", "hybrid_classical_twin_ft",
                  "pca8_logreg", "pca8_svm_rbf", "hybrid_classical_twin"],
    "quantum": ["vqc_8q_ft", "qkernel_svm_8q_ft", "vqc_8q", "qkernel_svm_8q"],
    "hybrid": ["hybrid_cnn_vqc_8q_ft", "hybrid_cnn_vqc_8q"],
}
# primary = deployed model of its family; baseline = classical ML reference; control = classical model on exactly the
# same inputs as a quantum / hybrid model; ablation = 2-D version of the 2.5-D CNN; experiment = quantum / hybrid model
# on ImageNet features (feature extractor never trained on CT), kept for the comparison, lower validation AUC.
ROLE = {"densenet121_ft": "primary", "resnet50_ft": "primary", "efficientnet_b0_ft": "primary",
        "cnn25d_scratch": "primary", "cnn2d_scratch": "ablation", "svm_rbf_radiomics": "baseline",
        "random_forest_radiomics": "baseline", "logreg_radiomics": "baseline",
        "pca8_logreg": "control", "pca8_svm_rbf": "control", "hybrid_classical_twin": "control",
        "pca8_logreg_ft": "control", "pca8_svm_rbf_ft": "control", "hybrid_classical_twin_ft": "control",
        "vqc_8q_ft": "primary", "qkernel_svm_8q_ft": "primary", "hybrid_cnn_vqc_8q_ft": "primary",
        "vqc_8q": "experiment", "qkernel_svm_8q": "experiment", "hybrid_cnn_vqc_8q": "experiment"}


# V9 models are discovered from artifacts/*/<id>/model.json with "version": "v9"; their role is written into the
# metadata by scripts/v9_evaluate.py (primary / ablation / baseline / control / experiment). Models that V9 could not
# train in this environment are listed explicitly so the dashboard shows them as NOT TRAINED with the reason.
NOT_TRAINED_V9 = {
    "convnext_tiny_ft_v9": ("classical", "ConvNeXt-Tiny: ImageNet weights not available offline (no network access)"),
    "efficientnet_b2_ft_v9": ("classical", "EfficientNet-B2: ImageNet weights not available offline (no network access)"),
    "unet3d_v9": ("segmentation", "3-D U-Net: needs contiguous labelled volumes with voxel spacing and far more compute "
                                  "than 2 CPU cores; the 2.5-D U-Net is the trained segmentation model"),
    "nnunet_v9": ("segmentation", "nnU-Net: requires PyTorch, which cannot be installed offline"),
}


class PatchContext:
    """Per-request lazy cache of features shared between models (frozen backbone outputs, ImageNet GAP, radiomics)."""

    def __init__(self, patches: np.ndarray, stacks: np.ndarray | None, ft=None, stacks5: np.ndarray | None = None,
                 emb_net=None):
        self.patches = patches          # (N, 96, 96) uint8 current slice
        self.stacks = stacks            # (N, 3, 96, 96) uint8 previous/current/next or None
        self.stacks5 = stacks5          # (N, 5, 96, 96) uint8 slices i-2..i+2 or None (V9 5-slice / 3-D models)
        self.ft = ft                    # fine-tuned EfficientNet-B0 (TailClassifier) for the *_ft feature models
        self.emb_net = emb_net          # selected V9 2.5-D CNN whose embedding feeds the V9 quantum models
        self._frozen, self._gap, self._ftgap, self._rad, self._emb = {}, None, None, None, None

    def cnn_embedding(self):
        if self._emb is None:
            if self.emb_net is None:
                raise ModelIntegrityError("needs the selected V9 2.5-D CNN, which is not loaded")
            self._emb = self.emb_net.embed(_down64(self.stacks).astype(np.float32) / 255.0)
        return self._emb

    def frozen(self, arch):
        if arch not in self._frozen:
            net = backbone(arch)
            self._frozen[arch] = net.frozen(B.Backbone.preprocess(to_rgb(self.patches))) if len(self.patches) else None
        return self._frozen[arch]

    def imagenet_gap(self):
        if self._gap is None:
            h = self.frozen("efficientnet_b0")
            self._gap = backbone("efficientnet_b0").tail(Tensor(h))["final"].data.mean((1, 2)).astype(np.float64)
        return self._gap

    def finetuned_gap(self):
        if self._ftgap is None:
            if self.ft is None:
                raise ModelIntegrityError("needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded")
            self._ftgap = self.ft.gap_from_frozen(self.frozen("efficientnet_b0"))
        return self._ftgap

    def radiomics(self):
        if self._rad is None:
            self._rad = BL.features(self.patches)
        return self._rad


def training_label(meta: dict | None, loaded: bool) -> str:
    """TRAINED / PARTIALLY TRAINED / NOT TRAINED.

    PARTIALLY TRAINED: the model's own weights are trained on CT, but its input passes through an ImageNet backbone that
    is frozen or only tail-fine-tuned on CT (fine-tuned backbones, and heads on ImageNet / fine-tuned EfficientNet
    features). TRAINED: every learned stage was trained on the CT training scans (CNNs from scratch, radiomics, U-Net,
    and heads on V9 CNN embeddings)."""
    if not loaded:
        return "NOT TRAINED"
    m = meta or {}
    arch = str(m.get("architecture") or "").lower()
    if "frozen" in arch or m.get("features") == "effnet_ft_v8" or m.get("feature_source") in ("imagenet", "finetuned"):
        return "PARTIALLY TRAINED"
    return "TRAINED"


def _down64(stack):
    return np.stack([[np.asarray(Image.fromarray(c).resize((64, 64), Image.LANCZOS)) for c in s] for s in stack])


class LoadedModel:
    def __init__(self, family, model_id, arrays, meta, ft=None, emb_net=None):
        self.family, self.model_id, self.meta, self.arrays = family, model_id, meta, arrays
        self.ft, self.emb_net = ft, emb_net
        self.kind = meta.get("kind")
        ch = meta.get("config", {}).get("in_channels")
        self.context_slices = 0
        if self.kind == "numpy_cnn" and ch in (3, 5):
            self.context_slices = ch // 2
        if str(self.kind).startswith("v9_") and meta.get("features") == "cnn25d_v9":
            self.context_slices = 1
        self.needs_context = self.context_slices > 0
        cal = meta.get("calibration")
        self.calibrator = SafeCalibrator.from_dict(cal) if cal else None
        self.threshold = float(meta.get("threshold", 0.5))
        self._build()

    def _build(self):
        a, k = self.arrays, self.kind
        if k == "cnn_finetune":
            self.m = TailClassifier(self.meta["arch"])
            self.m.load_state(a)
        elif k == "numpy_cnn":
            self.m = NumpyCNN(CNNConfig(**self.meta["config"]))
            self.m.set_state(a)
        elif k in ("pca8_control", "vqc", "quantum_kernel"):
            self.enc = PCAEncoder.from_arrays(a)
            if k == "vqc":
                self.m = VQC.from_arrays(a)
            elif k == "quantum_kernel":
                self.m = QKernelSVM.from_arrays(a)
        elif k == "hybrid":
            self.m = HybridQuantum.from_arrays(a)
        elif k == "classical_twin":
            self.m = ClassicalTwin.from_arrays(a)
        elif k in ("v9_vqc", "v9_qksvm", "v9_lr_control", "v9_rbf_control"):
            self.enc = PCAEncoder.from_arrays(a)
            if k == "v9_vqc":
                self.m = VQCv9.from_arrays(a)
            elif k == "v9_qksvm":
                self.m = QKernelV9.from_arrays(a)
        elif k == "v9_hybrid":
            self.m = HybridV9.from_arrays(a)
        elif k == "v9_twin":
            self.m = TwinV9.from_arrays(a)
        elif k not in ("radiomics_baseline", "radiomics_v9"):
            raise ModelIntegrityError(f"unknown model kind {k}")

    def scores(self, ctx: PatchContext) -> np.ndarray:
        k, a = self.kind, self.arrays
        if k == "cnn_finetune":
            return self.m.predict_from_frozen(ctx.frozen(self.meta["arch"]))
        if k == "numpy_cnn":
            if self.context_slices == 2:
                if ctx.stacks5 is None:
                    raise ModelIntegrityError("needs two slices above and two below")
                return self.m.decision_function(_down64(ctx.stacks5).astype(np.float32) / 255.0)
            X = _down64(ctx.stacks).astype(np.float32) / 255.0
            if not self.needs_context:
                X = X[:, 1:2]
            return self.m.decision_function(X)
        if k.startswith("v9_"):
            F = ctx.cnn_embedding() if self.meta.get("features") == "cnn25d_v9" else ctx.finetuned_gap()
            if k in ("v9_hybrid", "v9_twin"):
                return self.m.logits(F)
            A = self.enc.transform(F)
            if k == "v9_vqc":
                return self.m.logits(A)
            if k == "v9_qksvm":
                return self.m.decision(A)
            if "coef" in a:
                return A @ a["coef"] + float(np.ravel(a["intercept"])[0])
            d2 = (A ** 2).sum(1)[:, None] + (a["sv"] ** 2).sum(1)[None] - 2 * A @ a["sv"].T
            return np.exp(-float(a["gamma"]) * np.maximum(d2, 0)) @ a["dual"] + float(np.ravel(a["intercept"])[0])
        if k == "radiomics_baseline":
            return BL.predict_arrays(a, ctx.radiomics())
        if k == "radiomics_v9":
            return BL.predict_arrays(a, BL.features(BL.normalise_patches(ctx.patches, self.meta.get("normalization", "none"),
                                                                         float(self.meta.get("smooth", 0.0)))))
        feats = (lambda: ctx.finetuned_gap()) if self.meta.get("feature_source") == "finetuned" else (lambda: ctx.imagenet_gap())
        if k in ("pca8_control", "vqc", "quantum_kernel"):
            A = self.enc.transform(feats())
            if k == "vqc":
                return self.m.logits(A)
            if k == "quantum_kernel":
                return self.m.decision(A)
            if "coef" in a:
                return A @ a["coef"] + float(a["intercept"][0])
            d2 = (A ** 2).sum(1)[:, None] + (a["sv"] ** 2).sum(1)[None] - 2 * A @ a["sv"].T
            return np.exp(-float(a["gamma"]) * np.maximum(d2, 0)) @ a["dual"] + float(a["intercept"][0])
        if k in ("hybrid", "classical_twin"):
            return self.m.logits(feats())
        raise ModelIntegrityError(k)

    def probabilities(self, ctx):
        s = self.scores(ctx)
        return s, (self.calibrator(s) if self.calibrator else 1 / (1 + np.exp(-s)))

    def self_test(self):
        """Deterministic smoke test on a fixed synthetic patch set; must give finite, non-constant scores."""
        rng = np.random.default_rng(0)
        base = rng.integers(60, 200, (4, 96, 96)).astype(np.uint8)
        base[1, 30:60, 30:60] = 40
        base[2] = np.clip(base[2] * 0.5, 0, 255).astype(np.uint8)
        ctx = PatchContext(base, np.repeat(base[:, None], 3, 1), self.ft, np.repeat(base[:, None], 5, 1), self.emb_net)
        s = self.scores(ctx)
        if not np.all(np.isfinite(s)):
            raise ModelIntegrityError(f"{self.model_id}: non-finite self-test scores")
        if np.ptp(s) < 1e-9:
            raise ModelIntegrityError(f"{self.model_id}: constant self-test scores")


class Registry:
    def __init__(self):
        self.models: dict[str, LoadedModel] = {}
        self.status: dict[str, dict] = {}
        self.gate = None
        self.ood = None
        self.load_seconds = None
        self.unet = None              # V9 2.5-D U-Net (segmentation)
        self.v9_ids: dict[str, list[str]] = {"classical": [], "quantum": [], "hybrid": []}
        self.selected_cnn = None      # model id of the selected V9 2.5-D CNN

    def load(self):
        t0 = time.time()
        present = dict((mid, fam) for fam, mid in list_models())
        for fam, ids in EXPECTED.items():
            for mid in ids:
                if mid not in present:
                    self.status[mid] = {"family": fam, "status": "not_trained", "detail": "no artifact found"}
                    continue
                try:
                    arrays, meta = load_model(fam, mid)
                    if meta.get("status") != "trained":
                        self.status[mid] = {"family": fam, "status": "not_trained", "detail": meta.get("status")}
                        continue
                    ft = self.models["efficientnet_b0_ft"].m if "efficientnet_b0_ft" in self.models else None
                    lm = LoadedModel(fam, mid, arrays, meta, ft)
                    lm.self_test()
                    self.models[mid] = lm
                    self.status[mid] = {"family": fam, "status": "loaded", "detail": "integrity and self-test passed"}
                except Exception as e:  # never substitute another model
                    self.status[mid] = {"family": fam, "status": "failed", "detail": f"{type(e).__name__}: {e}",
                                        "trace": traceback.format_exc(limit=2)}
        self._load_v9(present)
        from src.inference.gate_ood import load_gate, load_ood
        self.gate, self.ood = load_gate(), load_ood()
        self.load_seconds = round(time.time() - t0, 1)
        return self

    def _load_v9(self, present):
        import json as _json

        from src.common.paths import ARTIFACTS, RESULTS
        # V10 change (audit finding 6): the deployable copy lives with the artifacts; results/ is only a fallback.
        sel_p = ARTIFACTS / "manifests" / "v9_cnn_selection.json"
        if not sel_p.exists():
            sel_p = RESULTS / "v9" / "cnn_selection.json"
        self.selected_cnn = _json.loads(sel_p.read_text()).get("selected_25d") if sel_p.exists() else None
        v9 = []
        for mid, fam in present.items():
            if fam not in ("classical", "quantum", "hybrid") or mid in self.status:
                continue
            try:
                # V11 fix: read the metadata only. Previously a V9 model whose model.npz was missing raised here and
                # was silently skipped, so it vanished from the registry instead of being reported as missing.
                meta = json.loads((model_dir(fam, mid) / "model.json").read_text())
            except Exception:
                continue
            if meta.get("version") == "v9":
                v9.append((meta.get("role_order", 99), mid, fam))
        # the embedding network must load first: V9 quantum models on CNN features depend on it
        v9.sort(key=lambda t: (t[1] != self.selected_cnn, t[0], t[1]))
        for _, mid, fam in v9:
            try:
                arrays, meta = load_model(fam, mid)
                emb = self.models[self.selected_cnn].m if self.selected_cnn in self.models else None
                lm = LoadedModel(fam, mid, arrays, meta, self.ft_model(), emb)
                lm.self_test()
                self.models[mid] = lm
                self.status[mid] = {"family": fam, "status": "loaded", "version": "v9", "detail": "integrity and self-test passed"}
                self.v9_ids[fam].append(mid)
            except Exception as e:
                self.status[mid] = {"family": fam, "status": "failed", "version": "v9", "detail": f"{type(e).__name__}: {e}"}
        for mid, (fam, why) in NOT_TRAINED_V9.items():
            if mid not in self.status:
                self.status[mid] = {"family": fam, "status": "not_trained", "version": "v9", "detail": why}
        try:
            from src.segmentation.unet import UNet25D
            p = ARTIFACTS / "segmentation" / "unet25d_v9" / "model.npz"
            if p.exists():
                self.unet = UNet25D.load(p)
                self.status["unet25d_v9"] = {"family": "segmentation", "status": "loaded", "version": "v9",
                                             "detail": "checkpoint loaded"}
            else:
                self.status["unet25d_v9"] = {"family": "segmentation", "status": "not_trained", "version": "v9",
                                             "detail": "no checkpoint"}
        except Exception as e:
            self.status["unet25d_v9"] = {"family": "segmentation", "status": "failed", "version": "v9", "detail": str(e)}

    def role(self, mid):
        lm = self.models.get(mid)
        if lm is not None and lm.meta.get("version") == "v9":
            return lm.meta.get("role")
        return ROLE.get(mid)

    def version(self, mid):
        return self.status.get(mid, {}).get("version", "v8")

    def ft_model(self):
        return self.models["efficientnet_b0_ft"].m if "efficientnet_b0_ft" in self.models else None

    def by_family(self, family, version: str = "v9"):
        if family == "all":
            return [m for f in ("classical", "quantum", "hybrid") for m in self.by_family(f, version)]
        v8 = list(EXPECTED[family])
        v9 = list(self.v9_ids.get(family, [])) + [m for m, (f, _) in NOT_TRAINED_V9.items() if f == family]
        return {"v8": v8, "v9": v9, "all": v9 + v8}[version]
