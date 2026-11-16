"""Deep-learning training without PyTorch, on the V9 NumPy implementations (real architectures, forward and backward
passes, losses and optimisers):

* 2.5-D residual squeeze-excitation CNN (V9 `NumpyCNN`): Adam-type optimiser with warm-up and cosine decay, focal or
  class-weighted cross-entropy, CT augmentation on training data only (shift, rotation, intensity, gamma, noise),
  early stopping on validation AUC, best checkpoint kept.
* ImageNet backbones (ResNet-50, EfficientNet-B0, DenseNet-121) with the timm ImageNet weights shipped in
  artifacts/weights (V9 NumPy re-implementation of the networks): the last block and a new linear head are fine-tuned
  (`TailClassifier`, AdamW, BCE / focal loss, augmented views of the training patches only, early stopping).

Both write V10 artifacts (model card + npz) that the engine serves through the common interface.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.v10.engine import artifacts as A
from src.v10.evaluation.metrics import full_metrics, roc_auc, youden_threshold
from src.v10.evaluation.nested import apply_platt, platt

BACKBONES = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121"}


def split_by_units(units: np.ndarray, val_units: list[str], test_units: list[str]):
    units = np.asarray(units)
    te = np.where(np.isin(units, test_units))[0]
    va = np.where(np.isin(units, val_units))[0]
    tr = np.where(~np.isin(units, list(val_units) + list(test_units)))[0]
    if not len(tr) or not len(va):
        raise ValueError("empty training or validation partition")
    return tr, va, te


def a4_config(project_root: Path) -> dict:
    p = Path(project_root) / "artifacts" / "classical" / "cnn25d_v9_a4_res_se_focal" / "model.json"
    cfg = json.loads(p.read_text())["config"] if p.exists() else {}
    return cfg


def train_numpy_cnn(root: Path, ds: dict, model_id: str, val_units: list[str], test_units: list[str],
                    max_epochs: int | None = None, subset: int | None = None, seed: int = 42, log=print) -> Path:
    from src.classical.numpy_cnn import CNNConfig, NumpyCNN
    from src.inference.registry import _down64
    cfg = a4_config(root)
    if not cfg:
        raise RuntimeError("V9 a4 recipe not found (artifacts/classical/cnn25d_v9_a4_res_se_focal/model.json)")
    cfg = dict(cfg, seed=seed)
    if max_epochs:
        cfg["max_epochs"] = max_epochs
    X = _down64(ds["stacks25d"]).astype(np.float32) / 255.0
    y, units = ds["y"], ds["units"]
    tr, va, te = split_by_units(units, val_units, test_units)
    if subset:
        rng = np.random.default_rng(seed)
        tr = np.sort(rng.choice(tr, min(subset, len(tr)), replace=False))
        va = np.sort(rng.choice(va, min(max(subset // 3, 20), len(va)), replace=False))
    net = NumpyCNN(CNNConfig(**cfg))
    t0 = time.time()
    net.fit(X[tr], y[tr], X[va], y[va], log=log)
    secs = time.time() - t0
    sv = net.predict_proba(X[va])
    sv = np.asarray(sv)[:, 1] if np.ndim(sv) == 2 else np.asarray(sv)
    z_va = np.log(np.clip(sv, 1e-6, 1 - 1e-6) / np.clip(1 - sv, 1e-6, 1))
    ab = platt(z_va, y[va])
    thr = youden_threshold(y[va], apply_platt(z_va, ab))
    evals = [{"name": "validation units (early stopping, calibration, threshold)", "status": "completed",
              "evaluation_unit": "patches of the validation units", "units": val_units,
              "metrics": full_metrics(y[va], z_va, apply_platt(z_va, ab), thr)}]
    if len(te):
        st = net.predict_proba(X[te])
        st = np.asarray(st)[:, 1] if np.ndim(st) == 2 else np.asarray(st)
        z_te = np.log(np.clip(st, 1e-6, 1 - 1e-6) / np.clip(1 - st, 1e-6, 1))
        evals.append({"name": "held-out test units (scored once)", "status": "completed (real data)",
                      "evaluation_unit": "patches of held-out reconstructed scan-group units", "units": test_units,
                      "metrics": full_metrics(y[te], z_te, apply_platt(z_te, ab), thr),
                      "per_unit_auc": {u: roc_auc(y[te][units[te] == u], z_te[units[te] == u])
                                       for u in np.unique(units[te]) if len(np.unique(y[te][units[te] == u])) == 2}})
    tmp = Path(root) / "artifacts" / "v10" / model_id
    tmp.mkdir(parents=True, exist_ok=True)
    net.save(tmp / "cnn_state.npz")
    state = dict(np.load(tmp / "cnn_state.npz", allow_pickle=False))
    (tmp / "cnn_state.npz").unlink()
    card = {"name": "2.5-D residual-SE CNN (V10 retrain, V9 a4 recipe)", "family": "deep_learning",
            "kind": "v10_patch_numpy_cnn", "architecture": "CNN (NumPy, 2.5-D residual SE)",
            "architecture_detail": json.dumps(cfg), "task": "tumour vs tumour-free liver tissue (2.5-D patches)",
            "role": "comparison", "version": "v10.0", "feature_set": "stacks25d_64", "feature_dim": 3 * 64 * 64,
            "preprocessing": {"input": "3 x 64 x 64 (previous / current / next slice), scaled to [0, 1]"},
            "training_data": ds.get("training_data", "real") if not subset else
            f"{ds.get('training_data', 'real')} (subset; smoke run)", "dataset_version": ds["dataset_version"],
            "data_split": {"train_units": sorted(set(units[tr])), "val_units": val_units, "test_units": test_units},
            "threshold": float(thr), "platt": list(ab), "seeds": [seed], "execution_mode": "classical_cpu_numpy",
            "training_seconds": round(secs, 1), "training_history": getattr(net, "history", None),
            "n_parameters": int(net.n_parameters()), "evaluations": evals, "requires_context": True}
    card["preprocessing_signature"] = A.preprocessing_signature(card["feature_set"], card["preprocessing"])
    arrays = {f"cnn__{k}": v for k, v in state.items() if k != "meta"}
    arrays["cnn_config_json"] = np.frombuffer(json.dumps(cfg).encode(), np.uint8)
    arrays["platt"] = np.array(ab, float)
    return A.save(Path(root) / "artifacts" / "v10", model_id, card, arrays)


def train_backbone(root: Path, ds: dict, arch: str, model_id: str, val_units: list[str], test_units: list[str],
                   max_epochs: int = 12, subset: int | None = None, seed: int = 42, n_views: int = 2,
                   log=print) -> Path:
    from src.classical import backbones as B
    from src.classical.cnn_finetune import train_tail
    from src.inference.features import backbone
    from src.preprocessing.ct import augment_patch, to_rgb
    if arch not in BACKBONES:
        raise ValueError(f"arch must be one of {list(BACKBONES)}")
    P, y, units = ds["patches2d"], ds["y"], ds["units"]
    tr, va, te = split_by_units(units, val_units, test_units)
    rng = np.random.default_rng(seed)
    if subset:
        tr = np.sort(rng.choice(tr, min(subset, len(tr)), replace=False))
        va = np.sort(rng.choice(va, min(max(subset // 3, 20), len(va)), replace=False))
        te = np.sort(rng.choice(te, min(max(subset // 3, 20), len(te)), replace=False)) if len(te) else te
    net = backbone(arch)

    def frozen(imgs):
        return np.concatenate([net.frozen(B.Backbone.preprocess(to_rgb(imgs[s:s + 32]))) for s in
                               range(0, len(imgs), 32)])
    log(f"[{arch}] frozen features for {len(tr)} training patches x {n_views} views")
    views = [frozen(P[tr])]
    for v in range(1, n_views):
        aug = np.stack([augment_patch(p, rng) for p in P[tr]])      # augmentation on training data only
        views.append(frozen(aug))
    h_va = frozen(P[va])
    t0 = time.time()
    m, info = train_tail(arch, views, y[tr], h_va, y[va], max_epochs=max_epochs, seed=seed, log=log)
    secs = time.time() - t0
    z_va = m.predict_from_frozen(h_va)
    ab = platt(z_va, y[va])
    thr = youden_threshold(y[va], apply_platt(z_va, ab))
    evals = [{"name": "validation units (early stopping, calibration, threshold)", "status": "completed",
              "evaluation_unit": "patches of the validation units", "units": val_units,
              "metrics": full_metrics(y[va], z_va, apply_platt(z_va, ab), thr)}]
    if len(te):
        z_te = m.predict_from_frozen(frozen(P[te]))
        evals.append({"name": "held-out test units (scored once)", "status": "completed (real data)",
                      "evaluation_unit": "patches of held-out reconstructed scan-group units", "units": test_units,
                      "metrics": full_metrics(y[te], z_te, apply_platt(z_te, ab), thr)})
    st = m.state()
    arrays = {f"tail__{k}": np.asarray(v) for k, v in st.items()}
    arrays["platt"] = np.array(ab, float)
    card = {"name": f"{BACKBONES[arch]} (ImageNet-pretrained, last block fine-tuned) — V10",
            "family": "deep_learning", "kind": "v10_patch_backbone_ft", "arch": arch,
            "architecture": BACKBONES[arch], "architecture_detail": f"timm ImageNet weights ({B.FILES[arch]}); last "
                                                                    "block + linear head fine-tuned; earlier layers frozen",
            "task": "tumour vs tumour-free liver tissue (96x96 patches)", "role": "comparison", "version": "v10.0",
            "feature_set": f"imagenet_input_{arch}", "feature_dim": 96 * 96,
            "preprocessing": {"input": "96 x 96 grey patch -> RGB -> ImageNet normalisation"},
            "training_data": ds.get("training_data", "real") if not subset else
            f"{ds.get('training_data', 'real')} (subset; smoke run)", "dataset_version": ds["dataset_version"],
            "data_split": {"train_units": sorted(set(units[tr])), "val_units": val_units, "test_units": test_units},
            "threshold": float(thr), "platt": list(ab), "seeds": [seed], "execution_mode": "classical_cpu_numpy",
            "training_seconds": round(secs, 1), "training_history": info.get("history"), "pretrained": True,
            "pretrained_weights": B.FILES[arch], "evaluations": evals}
    card["preprocessing_signature"] = A.preprocessing_signature(card["feature_set"], card["preprocessing"])
    return A.save(Path(root) / "artifacts" / "v10", model_id, card, arrays)
