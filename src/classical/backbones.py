"""ImageNet-pretrained ResNet-50, EfficientNet-B0 and DenseNet-121 in NumPy, loading timm weights.

Weights (timm GitHub releases; ImageNet-1k):
  resnet50_a1_0-14fe96d1.pth       (timm resnet50.a1_in1k)
  efficientnet_b0_ra-3dd342df.pth  (timm efficientnet_b0.ra_in1k)
  densenet121_ra-50efcf5c.pth      (timm densenet121.ra_in1k)

Every network is split into a FROZEN part (`frozen(x)`, inference only) and a TAIL (`tail(h)`, autodiff-capable)
so the final stage can be fine-tuned from cached frozen outputs. `tail_split` chooses where the split is:
  resnet50:        "layer4.2"  -> only the last bottleneck block is trainable, or "none" (all frozen)
  efficientnet_b0: "blocks.6"  -> the last MBConv block + conv_head + bn2 are trainable
  densenet121:     "denselayer13" -> dense layers 13-16 of block 4 + norm5 are trainable
Feature taps: `taps` gives the global-average-pooled output of the last two stages (final and intermediate).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import nn
from .nn import Tensor
from .torchload import load_state_dict

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "artifacts" / "weights"
FILES = {"resnet50": "resnet50_a1_0-14fe96d1.pth", "efficientnet_b0": "efficientnet_b0_ra-3dd342df.pth",
         "densenet121": "densenet121_ra-50efcf5c.pth"}
URLS = {"resnet50": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
        "efficientnet_b0": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth",
        "densenet121": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/densenet121_ra-50efcf5c.pth"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def weights_sha(arch: str) -> str:
    h = hashlib.sha256()
    with open(WEIGHTS / FILES[arch], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Backbone:
    arch = ""
    tail_split = "none"

    def __init__(self, sd: dict | None = None, tail_split: str | None = None):
        sd = load_state_dict(str(WEIGHTS / FILES[self.arch])) if sd is None else sd
        self.sd = {k: v for k, v in sd.items() if hasattr(v, "shape") and not k.endswith("num_batches_tracked")}
        if tail_split is not None:
            self.tail_split = tail_split
        self.P: dict[str, Tensor] = {}
        for k, v in self.sd.items():
            self.P[k] = nn.Tensor(v, requires_grad=False, name=k)

    # --- helpers
    def conv(self, x, name, stride=1, pad=None, depthwise=False):
        w = self.P[name + ".weight"]
        b = self.P.get(name + ".bias")
        k = w.data.shape[-1]
        pad = (k // 2 if stride == 1 else ((stride - 1) + (k - 1)) // 2) if pad is None else pad
        if depthwise:
            return nn.depthwise_conv2d(x, w, stride, pad)
        return nn.conv2d(x, w, b, stride, pad)

    def bn(self, x, name, eps=1e-5):
        return nn.batchnorm(x, self.P[name + ".weight"], self.P[name + ".bias"], self.sd[name + ".running_mean"],
                            self.sd[name + ".running_var"], eps)

    def tail_params(self) -> list[Tensor]:
        return [self.P[k] for k in self.tail_keys()]

    def tail_keys(self) -> list[str]:
        raise NotImplementedError

    def set_trainable(self, on: bool):
        for k in self.P:
            self.P[k].requires_grad = False
        if on:
            for k in self.tail_keys():
                if not k.endswith(("running_mean", "running_var")):
                    self.P[k].requires_grad = True

    def n_params(self) -> dict:
        tot = sum(v.size for k, v in self.sd.items() if not k.endswith(("running_mean", "running_var")) and not k.startswith(("fc.", "classifier.")))
        tr = sum(self.P[k].data.size for k in self.tail_keys() if not k.endswith(("running_mean", "running_var")))
        return {"backbone_params": int(tot), "tail_trainable_params": int(tr)}

    @staticmethod
    def preprocess(img_uint8: np.ndarray) -> np.ndarray:
        """uint8 HWC RGB (already resized) -> float32 NHWC ImageNet-normalised (fixed constants, not data-derived)."""
        x = img_uint8.astype(np.float32) / 255.0
        return ((x - MEAN) / STD)[None] if x.ndim == 3 else (x - MEAN) / STD

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = self.frozen(x)
        f = self.tail(Tensor(h))["final"]
        w, b = self.classifier()
        return nn.linear(nn.gap(f), w, b).data

    def taps(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """GAP features of the final stage and of the intermediate stage (inference)."""
        h = self.frozen(x)
        out = self.tail(Tensor(h))
        return {"final": out["final"].data.mean((1, 2)), "mid": self._mid}


# ======================================================================================================= ResNet-50
class ResNet50(Backbone):
    arch = "resnet50"
    tail_split = "layer4.2"
    LAYERS = (3, 4, 6, 3)

    def classifier(self):
        return self.P["fc.weight"], self.P["fc.bias"]

    def block(self, x, p, stride):
        sc = x
        y = nn.relu(self.bn(self.conv(x, p + ".conv1", 1, 0), p + ".bn1"))
        y = nn.relu(self.bn(self.conv(y, p + ".conv2", stride, 1), p + ".bn2"))
        y = self.bn(self.conv(y, p + ".conv3", 1, 0), p + ".bn3")
        if p + ".downsample.0.weight" in self.P:
            sc = self.bn(self.conv(x, p + ".downsample.0", stride, 0), p + ".downsample.1")
        return nn.relu(nn.add(y, sc))

    def _blocks(self):
        for li, n in enumerate(self.LAYERS):
            for bi in range(n):
                yield f"layer{li + 1}.{bi}", (2 if (bi == 0 and li > 0) else 1)

    def frozen(self, x: np.ndarray) -> np.ndarray:
        t = Tensor(x)
        t = nn.relu(self.bn(self.conv(t, "conv1", 2, 3), "bn1"))
        t = Tensor(nn.maxpool(t.data))
        self._mid = None
        for name, s in self._blocks():
            if name == self.tail_split:
                break
            t = self.block(t, name, s)
            if name == "layer3.5":
                self._mid = t.data.mean((1, 2))
        return t.data

    def tail(self, h: Tensor) -> dict:
        started = self.tail_split == "none"
        t = h
        for name, s in self._blocks():
            if name == self.tail_split:
                started = True
            if started and self.tail_split != "none":
                t = self.block(t, name, s)
        return {"final": t}

    def tail_keys(self):
        if self.tail_split == "none":
            return []
        names, started = [], False
        for name, _ in self._blocks():
            started = started or name == self.tail_split
            if started:
                names.append(name + ".")
        return [k for k in self.P if any(k.startswith(n) for n in names)]


# ================================================================================================= EfficientNet-B0
class EfficientNetB0(Backbone):
    arch = "efficientnet_b0"
    tail_split = "blocks.6"
    # (stage, number of blocks, stride of first block); kernel sizes are read from the weights
    STAGES = ((0, 1, 1), (1, 2, 2), (2, 2, 2), (3, 3, 2), (4, 3, 1), (5, 4, 2), (6, 1, 1))

    def classifier(self):
        return self.P["classifier.weight"], self.P["classifier.bias"]

    def se(self, x, p):
        s = nn.gap(x, keepdims=True)
        s = nn.silu(self.conv(s, p + ".conv_reduce", 1, 0))
        s = nn.sigmoid(self.conv(s, p + ".conv_expand", 1, 0))
        return nn.mul_channel(x, s)

    def block(self, x, p, stride, stage):
        cin = x.data.shape[-1]
        if stage == 0:  # depthwise-separable block
            y = nn.silu(self.bn(self.conv(x, p + ".conv_dw", stride, depthwise=True), p + ".bn1"))
            y = self.se(y, p + ".se")
            y = self.bn(self.conv(y, p + ".conv_pw", 1, 0), p + ".bn2")
        else:  # inverted residual (MBConv)
            y = nn.silu(self.bn(self.conv(x, p + ".conv_pw", 1, 0), p + ".bn1"))
            y = nn.silu(self.bn(self.conv(y, p + ".conv_dw", stride, depthwise=True), p + ".bn2"))
            y = self.se(y, p + ".se")
            y = self.bn(self.conv(y, p + ".conv_pwl", 1, 0), p + ".bn3")
        if stride == 1 and y.data.shape[-1] == cin:
            y = nn.add(y, x)
        return y

    def _blocks(self):
        for st, n, s in self.STAGES:
            for bi in range(n):
                yield f"blocks.{st}.{bi}", (s if bi == 0 else 1), st

    def frozen(self, x):
        t = nn.silu(self.bn(self.conv(Tensor(x), "conv_stem", 2), "bn1"))
        self._mid = None
        for name, s, st in self._blocks():
            if self.tail_split != "none" and name.startswith(self.tail_split + ".") or name == self.tail_split:
                break
            t = self.block(t, name, s, st)
            if name == "blocks.5.3":
                self._mid = t.data.mean((1, 2))
        if self.tail_split == "none":
            t = nn.silu(self.bn(self.conv(t, "conv_head", 1, 0), "bn2"))
        return t.data

    def tail(self, h):
        if self.tail_split == "none":
            return {"final": h}
        t, started = h, False
        for name, s, st in self._blocks():
            started = started or name.startswith(self.tail_split + ".") or name == self.tail_split
            if started:
                t = self.block(t, name, s, st)
        t = nn.silu(self.bn(self.conv(t, "conv_head", 1, 0), "bn2"))
        return {"final": t}

    def tail_keys(self):
        if self.tail_split == "none":
            return []
        return [k for k in self.P if k.startswith(self.tail_split + ".") or k.startswith(("conv_head.", "bn2."))]


# =================================================================================================== DenseNet-121
class DenseNet121(Backbone):
    arch = "densenet121"
    tail_split = "denselayer13"
    BLOCKS = (6, 12, 24, 16)

    def classifier(self):
        return self.P["classifier.weight"], self.P["classifier.bias"]

    def layer(self, x, p):
        y = nn.relu(self.bn(x, p + ".norm1"))
        y = self.conv(y, p + ".conv1", 1, 0)
        y = nn.relu(self.bn(y, p + ".norm2"))
        y = self.conv(y, p + ".conv2", 1, 1)
        return nn.concat([x, y])

    def frozen(self, x):
        t = Tensor(x)
        t = nn.relu(self.bn(self.conv(t, "features.conv0", 2, 3), "features.norm0"))
        t = Tensor(nn.maxpool(t.data))
        self._mid = None
        for bi, n in enumerate(self.BLOCKS):
            for li in range(n):
                name = f"features.denseblock{bi + 1}.denselayer{li + 1}"
                if bi == 3 and self.tail_split != "none" and li + 1 >= int(self.tail_split.replace("denselayer", "")):
                    return t.data
                t = self.layer(t, name)
            if bi < 3:
                p = f"features.transition{bi + 1}"
                if bi == 2:
                    self._mid = t.data.mean((1, 2))
                t = nn.relu(self.bn(t, p + ".norm"))
                t = self.conv(t, p + ".conv", 1, 0)
                t = Tensor(nn.avgpool2(t.data))
        if self.tail_split == "none":
            t = nn.relu(self.bn(t, "features.norm5"))
        return t.data

    def tail(self, h):
        if self.tail_split == "none":
            return {"final": h}
        t = h
        for li in range(int(self.tail_split.replace("denselayer", "")), self.BLOCKS[3] + 1):
            t = self.layer(t, f"features.denseblock4.denselayer{li}")
        return {"final": nn.relu(self.bn(t, "features.norm5"))}

    def tail_keys(self):
        if self.tail_split == "none":
            return []
        first = int(self.tail_split.replace("denselayer", ""))
        pre = tuple(f"features.denseblock4.denselayer{i}." for i in range(first, self.BLOCKS[3] + 1)) + ("features.norm5.",)
        return [k for k in self.P if k.startswith(pre)]


ARCHS = {"resnet50": ResNet50, "efficientnet_b0": EfficientNetB0, "densenet121": DenseNet121}


def build(arch: str, tail_split: str | None = None) -> Backbone:
    return ARCHS[arch](tail_split=tail_split)
