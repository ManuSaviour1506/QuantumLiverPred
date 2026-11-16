# Deep learning

8 of 17 registered models are servable on the installation that generated this file (`python scripts/v11_build_model_index.py`).

Model code lives in `src/`, trained artifacts in `artifacts/`; this folder is the index. Full machine-readable details: [`../model_index.json`](../model_index.json).

| model | architecture | version | availability | threshold | artifact | implementation |
|---|---|---|---|---|---|---|
| `cnn25d_v9_a1_aug` | CNN (NumPy) | v9 | requires_context | 0.4790 | `artifacts/classical/cnn25d_v9_a1_aug` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `cnn25d_v9_a2_aug_focal` | CNN (NumPy) | v9 | requires_context | 0.2776 | `artifacts/classical/cnn25d_v9_a2_aug_focal` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `cnn25d_v9_a3_res_se` | CNN (NumPy) | v9 | requires_context | 0.9693 | `artifacts/classical/cnn25d_v9_a3_res_se` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `cnn25d_v9_a4_res_se_focal` | CNN (NumPy) | v9 | requires_context | 0.7001 | `artifacts/classical/cnn25d_v9_a4_res_se_focal` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `cnn2d_v9` | CNN (NumPy) | v9 | available | 0.3232 | `artifacts/classical/cnn2d_v9` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `cnn3d_v9` | CNN (NumPy) | v9 | requires_context | 0.7706 | `artifacts/classical/cnn3d_v9` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `cnn5s_v9` | CNN (NumPy) | v9 | requires_context | 0.4578 | `artifacts/classical/cnn5s_v9` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `densenet121_ft_v9` | DenseNet-121 | v9 | available | 0.7358 | `artifacts/classical/densenet121_ft_v9` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
| `convnext_tiny_ft_v9` |  | v9 | not_trained — ConvNeXt-Tiny: ImageNet weights not available offline (no network access) | — | `—` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
| `efficientnet_b2_ft_v9` |  | v9 | not_trained — EfficientNet-B2: ImageNet weights not available offline (no network access) | — | `—` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
| `efficientnet_b0_ft` | EFFICIENTNET-B0 | v8 | missing_weights — ModelIntegrityError: efficientnet_b0_ft: model.npz missing | 0.6565 | `artifacts/classical/efficientnet_b0_ft` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
| `efficientnet_b0_ft_v9` | EFFICIENTNET-B0 | v9 | missing_weights — ModelIntegrityError: efficientnet_b0_ft_v9: model.npz missing | 0.6683 | `artifacts/classical/efficientnet_b0_ft_v9` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
| `resnet50_ft` | RESNET50 | v8 | missing_weights — ModelIntegrityError: resnet50_ft: model.npz missing | 0.6171 | `artifacts/classical/resnet50_ft` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
| `resnet50_ft_v9` | RESNET50 | v9 | missing_weights — ModelIntegrityError: resnet50_ft_v9: model.npz missing | 0.7230 | `artifacts/classical/resnet50_ft_v9` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
| `cnn25d_scratch` | CNN (NumPy) | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.6463 | `artifacts/classical/cnn25d_scratch` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `cnn2d_scratch` | CNN (NumPy) | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.4640 | `artifacts/classical/cnn2d_scratch` | `src/classical/numpy_cnn.py`, `src/classical/nn.py` |
| `densenet121_ft` | DenseNet-121 | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.7334 | `artifacts/classical/densenet121_ft` | `src/classical/backbones.py`, `src/classical/cnn_finetune.py` |
