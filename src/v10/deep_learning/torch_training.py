"""PyTorch training path for machines where PyTorch / torchvision can be installed (CPU or GPU).

NOT EXECUTED IN THE BUILD ENVIRONMENT: PyTorch could not be installed there (no package installs). The code is
written against the stable torch / torchvision APIs and is exercised only when `import torch` succeeds; the test
suite skips it otherwise and says so. Without PyTorch, use the NumPy path (`numpy_training.py`), which trains the same
families with real forward / backward passes.

Models
  cnn25d           2.5-D residual squeeze-excitation CNN (3 x 64 x 64 input), same design as the V9 a4 recipe
  resnet50 / efficientnet_b0 / densenet121   torchvision architectures; ImageNet weights when they can be downloaded
                   (otherwise trained from scratch and recorded as NOT pretrained — never claimed as pretrained)
Training: AdamW, cosine schedule with warm-up, class-weighted BCE or focal loss, augmentation on training data only
(random affine, intensity jitter), early stopping on validation AUC, best checkpoint saved, full history recorded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def build(arch: str, pretrained: bool = True):
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    class SE(nn.Module):
        def __init__(self, c, r=4):
            super().__init__()
            self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, c // r), nn.ReLU(),
                                   nn.Linear(c // r, c), nn.Sigmoid())

        def forward(self, x):
            return x * self.f(x)[:, :, None, None]

    class Res(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.a = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                   nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), SE(cout))
            self.s = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            return torch.relu(self.a(x) + self.s(x))

    used_pretrained = False
    if arch == "cnn25d":
        net = nn.Sequential(Res(3, 16), nn.MaxPool2d(2), Res(16, 32), nn.MaxPool2d(2), Res(32, 64), nn.MaxPool2d(2),
                            Res(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
        return net, used_pretrained
    ctor = {"resnet50": (tvm.resnet50, "ResNet50_Weights"), "efficientnet_b0": (tvm.efficientnet_b0,
            "EfficientNet_B0_Weights"), "densenet121": (tvm.densenet121, "DenseNet121_Weights")}[arch]
    weights = None
    if pretrained:
        try:
            weights = getattr(tvm, ctor[1]).IMAGENET1K_V1
            net = ctor[0](weights=weights)
            used_pretrained = True
        except Exception:
            net = ctor[0](weights=None)
    else:
        net = ctor[0](weights=None)
    if arch == "resnet50":
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    else:
        net.classifier = nn.Linear(net.classifier.in_features, 1)
    return net, used_pretrained


def train(arch: str, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, out_dir: Path,
          epochs: int = 30, batch: int = 32, lr: float = 1e-3, patience: int = 6, focal_gamma: float = 0.0,
          seed: int = 42, pretrained: bool = True, log=print) -> dict:
    """X: (N, C, H, W) float32 in [0, 1] (C = 3 for cnn25d stacks; grey patches are repeated to 3 channels)."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, used_pretrained = build(arch, pretrained)
    net = net.to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]

    def prep(xb, train=False):
        xb = torch.as_tensor(xb, device=dev)
        if xb.shape[1] == 1:
            xb = xb.repeat(1, 3, 1, 1)
        if train:   # augmentation on training batches only
            theta = torch.zeros(len(xb), 2, 3, device=dev)
            ang = (torch.rand(len(xb), device=dev) - 0.5) * (np.pi / 15)
            theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang), -torch.sin(ang)
            theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang), torch.cos(ang)
            theta[:, :, 2] = (torch.rand(len(xb), 2, device=dev) - 0.5) * 0.12
            grid = F.affine_grid(theta, xb.shape, align_corners=False)
            xb = F.grid_sample(xb, grid, padding_mode="border", align_corners=False)
            xb = xb * (1 + (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5) * 0.2)
        if arch != "cnn25d":
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
            xb = (xb - mean) / std
        return xb

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(X_tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    pos_w = torch.tensor(float((y_tr == 0).sum() / max(1, (y_tr == 1).sum())), device=dev)
    best, best_state, wait, hist = -np.inf, None, 0, []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(X_tr))
        tl = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            xb, yb = prep(X_tr[idx], True), torch.as_tensor(y_tr[idx], dtype=torch.float32, device=dev)
            z = net(xb).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(z, yb, pos_weight=pos_w, reduction="none")
            if focal_gamma:
                p = torch.sigmoid(z)
                pt = torch.where(yb == 1, p, 1 - p)
                loss = loss * (1 - pt) ** focal_gamma
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            zv = torch.cat([net(prep(X_va[s:s + 128])).squeeze(1).cpu() for s in range(0, len(X_va), 128)]).numpy()
        auc = float(roc_auc_score(y_va, zv)) if len(np.unique(y_va)) == 2 else float("nan")
        hist.append({"epoch": ep + 1, "train_loss": tl / len(X_tr), "val_auc": auc, "seconds": round(time.time() - t0, 1)})
        log(f"[torch {arch}] epoch {ep + 1}: loss {tl / len(X_tr):.4f} val AUC {auc:.4f}")
        if auc > best + 1e-4:
            best, wait = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(best_state, out_dir / "checkpoint_best.pt")
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")
    info = {"history": hist, "best_val_auc": best, "pretrained": used_pretrained, "device": str(dev),
            "torch_version": torch.__version__, "train_seconds": round(time.time() - t0, 1)}
    (out_dir / "training.json").write_text(json.dumps(info, indent=1))
    return info
