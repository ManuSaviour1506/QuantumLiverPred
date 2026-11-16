#!/usr/bin/env python3
"""Train the 2.5-D residual-SE CNN (V9 a4 recipe) on the real patch dataset with unit-grouped splits.

    python scripts/train_cnn.py --dataset DIR [--backend numpy|torch] [--val-units G02,G08+G09,G10]
                                [--test-units G01] [--epochs N] [--subset N]

numpy backend (default; no PyTorch needed): V9 NumpyCNN, early stopping on validation AUC, CT augmentation on
training data only, focal loss, warm-up + cosine schedule; saves artifacts/v10/v10_cnn25d_numpy.
torch backend: src/v10/deep_learning/torch_training.py (only where PyTorch is installed; untested in the build
environment). One full numpy fit took about 27 minutes on 2 CPU cores in V9.
"""
import argparse
import sys

from _common import ROOT, dataset_arg, require_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    dataset_arg(ap)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--val-units", default="G02,G08+G09,G10")
    ap.add_argument("--test-units", default="G01")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None, help="smoke run on N training patches (not a result)")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from src.v10.data.lits_dataset import load_patch_dataset
    ds = load_patch_dataset(require_dataset(a.dataset))
    val, test = a.val_units.split(","), [u for u in a.test_units.split(",") if u]
    if a.backend == "torch":
        from src.v10.deep_learning import torch_training as T
        if not T.available():
            sys.exit("PyTorch / torchvision are not installed; use --backend numpy")
        from src.inference.registry import _down64
        from src.v10.deep_learning.numpy_training import split_by_units
        X = _down64(ds["stacks25d"]).astype("float32") / 255.0
        tr, va, te = split_by_units(ds["units"], val, test)
        info = T.train("cnn25d", X[tr], ds["y"][tr], X[va], ds["y"][va], ROOT / "artifacts" / "v10_torch" / "cnn25d",
                       epochs=a.epochs or 30, focal_gamma=2.0, seed=a.seed)
        print(info["best_val_auc"])
    else:
        from src.v10.deep_learning.numpy_training import train_numpy_cnn
        d = train_numpy_cnn(ROOT, ds, a.model_id or ("v10_cnn25d_numpy" + ("_smoke" if a.subset else "")), val, test,
                            a.epochs, a.subset, a.seed)
        print(f"saved {d.relative_to(ROOT)}")
