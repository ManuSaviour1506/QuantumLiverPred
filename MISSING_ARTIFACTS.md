# Missing artifacts — what this package does not contain, and how to restore it

Nothing below was fabricated, replaced by a placeholder or substituted with another model. Every model that needs a
missing file is listed by the running platform as unavailable, with the reason (Model Registry page,
`GET /api/v1/models`, `models/model_index.json`).

## Why these files are missing

This package was built from the V10 project folder on your Mac (`~/Downloads/liver_tumour_intelligence_v10`). That
folder is the **lean V10 release**, which never contained the seven large V9 files below or the full dataset. Those
live in the V9 project folder (`liver_tumour_project_v9`) and in the full V10 copy. During this build I did not have
access to your `~/Downloads` folder outside the V10 project, so I could not copy them in.

Everything else is included. That covers all three ImageNet backbone weights (156 MB, SHA-256 verified), every other
trained model artifact, the OOD detector, the liver gate, the U-Net, the 8-slice real CT sample and all
manifests/labels.

## 1. Seven V9 model files (≈ 72 MB)

| file | bytes | SHA-256 | needed by |
|---|---:|---|---|
| `artifacts/classical/efficientnet_b0_ft/model.npz` | 4,561,382 | `7accb9dc20f346828275d71f6daec5e39b5bfdb42f43194af4f7b89a3e75c5dd` | `efficientnet_b0_ft` and the 12 models built on its features (below) |
| `artifacts/classical/efficientnet_b0_ft_v9/model.npz` | 4,561,382 | `da3a25dc2fe0e25caeb374897b2cae3e428dba16b8199ae3f1e2b666acebd95a` | `efficientnet_b0_ft_v9` |
| `artifacts/classical/resnet50_ft/model.npz` | 17,887,950 | `1e08ef03e0a5f967ade1b305025864f585408f1abb6e8cc6d56a646f4bbf32c1` | `resnet50_ft` |
| `artifacts/classical/resnet50_ft_v9/model.npz` | 17,887,950 | `295a03c00083ea02b8999b49ecef1793d169c981765f74db1d729c156851b694` | `resnet50_ft_v9` |
| `artifacts/classical/densenet121_ft/checkpoint_best.npz` | 2,627,488 | `edeab770c75ff55602e674659d85d81bbb881e03cb0b97dfd04f525e3d07d741` | training checkpoint only (serving never reads it) |
| `artifacts/classical/efficientnet_b0_ft/checkpoint_best.npz` | 4,561,382 | `7accb9dc20f346828275d71f6daec5e39b5bfdb42f43194af4f7b89a3e75c5dd` | training checkpoint only |
| `artifacts/classical/resnet50_ft/checkpoint_best.npz` | 17,887,950 | `1e08ef03e0a5f967ade1b305025864f585408f1abb6e8cc6d56a646f4bbf32c1` | training checkpoint only |

**16 of the 63 registered models cannot load without them.**

| | V9 comparison models (become servable when restored) | V8-era models (return to the *legacy* reproducibility list) |
|---|---|---|
| missing weights | `efficientnet_b0_ft_v9`, `resnet50_ft_v9` | `efficientnet_b0_ft`, `resnet50_ft` |
| missing dependency (`efficientnet_b0_ft` features) | `lr_control_v9_eff`, `rbf_control_v9_eff`, `twin_v9_eff`, `qksvm_v9_eff`, `vqc_v9_eff`, `hybrid_v9_eff` | `hybrid_classical_twin_ft`, `pca8_logreg_ft`, `pca8_svm_rbf_ft`, `qkernel_svm_8q_ft`, `vqc_8q_ft`, `hybrid_cnn_vqc_8q_ft` |

Before V11, the two `_ft_v9` backbones vanished from the registry silently. V11 now lists them as `missing_weights`
(regression test: `tests/test_regression_cases.py::test_models_with_missing_weight_files_are_reported_not_hidden`).

**Restore** (every file is verified by SHA-256; V9 is only read):

```bash
python scripts/v10_import_v9_artifacts.py --v9 /path/to/liver_tumour_project_v9
# or, as part of installation:
python install.py --v9 /path/to/liver_tumour_project_v9        # Windows: install.bat --v9 C:\path\to\liver_tumour_project_v9
./setup.sh --v9 /path/to/liver_tumour_project_v9               # macOS / Linux, original script
```

Restart the platform afterwards. The registry re-checks every artifact at start-up, so the models above become
available on their own.

## 2. Full CT dataset `liver_ct_lits_verified_v8` (1,757 files)

1,745 axial CT slices (476 with tumour), 2,449 training patches, expert masks, labels and verification records. This
package includes the slice manifest (`dataset/manifests/lits_verified_v8_manifest.csv`), every slice label
(`data/labels/verified_slice_labels.csv`), the SHA-256 of all 1,757 files
(`dataset/manifests/lits_verified_v8_SHA256SUMS.txt`) and the 8-slice real sample.

The 64 × 64 mask maps derived from its expert masks (`data/masks/*.npz`, from the V9 project) are left out for the
same reason. Serving never reads them; only the 7 dataset-integrity tests in
`tests/v10_research/test_v10_data_integrity.py` need them, and they skip with that reason when the maps are absent.

**Without it:** the bundled sample still serves live comparisons, benchmarking and tests. You cannot retrain models or
run full-dataset benchmarks.

**Restore:**

```bash
cp -R /path/to/liver_tumour_intelligence_v10_full/dataset/liver_ct_lits_verified_v8 dataset/
python install.py --check                     # verifies all 1,757 files against the SHA-256 list
```

The *New evaluation* benchmark then offers the full dataset automatically. To keep it elsewhere, set
`LTI_DATASET_DIR=/path/to/liver_ct_lits_verified_v8`. Source and licence: `DATASET_GUIDE.md` (LiTS-derived,
CC BY-NC-SA 4.0).

## 3. MRI data

No MRI data and no MRI-trained model exist in any version of this project, so none are included and none were
invented. An upload marked MRI is flagged as out of scope, because every model was trained on CT, and the
out-of-distribution check is expected to withhold it.
