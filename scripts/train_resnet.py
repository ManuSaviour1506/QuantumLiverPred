#!/usr/bin/env python3
"""Fine-tune resnet50 (ImageNet-pretrained) on the real patch dataset with unit-grouped splits.

    python scripts/train_resnet.py --dataset DIR [--backend numpy|torch] [--epochs 12] [--subset N]

numpy backend (default): the V9 NumPy resnet50 with the timm ImageNet weights in artifacts/weights (real pretrained
weights, no download needed); last block + linear head fine-tuned with AdamW, augmented training views only, early
stopping on validation AUC; saves artifacts/v10/v10_resnet50_ft.
torch backend: torchvision resnet50 with ImageNet weights when downloadable (otherwise trained from scratch and recorded
as NOT pretrained); only where PyTorch is installed — untested in the build environment.
"""
import argparse
import sys

from _common import ROOT, dataset_arg, require_dataset

ARCH = "resnet50"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    dataset_arg(ap)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--val-units", default="G02,G08+G09,G10")
    ap.add_argument("--test-units", default="G01")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--subset", type=int, default=None, help="smoke run on N training patches (not a result)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from src.v10.data.lits_dataset import load_patch_dataset
    ds = load_patch_dataset(require_dataset(a.dataset))
    val, test = a.val_units.split(","), [u for u in a.test_units.split(",") if u]
    if a.backend == "torch":
        from src.v10.deep_learning import torch_training as T
        if not T.available():
            sys.exit("PyTorch / torchvision are not installed; use --backend numpy")
        from src.v10.deep_learning.numpy_training import split_by_units
        X = ds["patches2d"][:, None].astype("float32") / 255.0
        tr, va, te = split_by_units(ds["units"], val, test)
        info = T.train(ARCH, X[tr], ds["y"][tr], X[va], ds["y"][va], ROOT / "artifacts" / "v10_torch" / ARCH,
                       epochs=a.epochs, seed=a.seed)
        print(info["best_val_auc"], "pretrained:", info["pretrained"])
    else:
        from src.v10.deep_learning.numpy_training import train_backbone
        d = train_backbone(ROOT, ds, ARCH, f"v10_{ARCH}_ft" + ("_smoke" if a.subset else ""), val, test, a.epochs,
                           a.subset, a.seed)
        print(f"saved {d.relative_to(ROOT)}")
