# Model index (V11)

The requested `models/{classical,deep_learning,quantum,hybrid}` layout, mapped onto the existing project without moving code or artifacts:

| family | index | implementation | trained artifacts |
|---|---|---|---|
| classical | [classical/](classical/README.md) | `src/classical/`, `src/v10/classical/` | `artifacts/classical/*` (radiomics, CNN-embedding heads), `artifacts/v10/v10_*` |
| deep learning | [deep_learning/](deep_learning/README.md) | `src/classical/numpy_cnn.py`, `src/classical/backbones.py`, `src/v10/deep_learning/` | `artifacts/classical/cnn*_v9`, `artifacts/classical/*_ft*`, `artifacts/weights/` (ImageNet backbones) |
| quantum | [quantum/](quantum/README.md) | `src/quantum/`, `src/v10/quantum/` | `artifacts/quantum/*`, `artifacts/v10/v10_qksvm_radiomics`, `artifacts/v10/v10_vqc_radiomics` |
| hybrid | [hybrid/](hybrid/README.md) | `src/hybrid/`, `src/v10/hybrid/` | `artifacts/hybrid/*`, `artifacts/v10/v10_*fusion*` |

Shared front end (every family): OOD detector `artifacts/ood/`, liver gate `artifacts/classical/liver_gate`, U-Net liver validation `artifacts/segmentation/unet25d_v9`.

Engine on the generating installation: 33 of 63 models servable. Counts by availability:

| family | available | requires_context | missing_dependency | missing_weights | legacy | not_trained | failed_integrity | non_deployable |
|---|---|---|---|---|---|---|---|---|
| classical | 14 | 3 | 6 | 0 | 6 | 0 | 0 | 0 |
| deep_learning | 2 | 6 | 0 | 4 | 3 | 2 | 0 | 0 |
| quantum | 2 | 2 | 4 | 0 | 2 | 0 | 0 | 0 |
| hybrid | 3 | 1 | 2 | 0 | 1 | 0 | 0 | 0 |

Models reported as `missing_weights` / `missing_dependency` need V9 files that are not in this package — see [`../MISSING_ARTIFACTS.md`](../MISSING_ARTIFACTS.md).
