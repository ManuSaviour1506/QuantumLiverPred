# dataset/

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

| Folder | Contents |
|---|---|
| `liver_ct_lits_verified_v8/` | the full CT dataset (1,745 slices, 2,449 patches; CC BY-NC-SA 4.0) — present in the full project folder / full ZIP, absent from the lean ZIP; used by default by all scripts and registered by `setup.sh` |
| `manifests/lits_verified_v8_manifest.csv` | all 1,745 slices of `liver_ct_lits_verified_v8`: relative path, label, reconstructed scan group (`scan_id`), slice index, modality, split, source; empty `patient_id` / `volume_id` (not available) |
| `manifests/lits_verified_v8_SHA256SUMS.txt` | SHA-256 of all 1,757 files of the dataset folder, for verifying a copy |
| `sample_data/lits_sample/` | 8 real CT slices from that dataset (CC BY-NC-SA 4.0, attribution in its README) — used by the end-to-end tests and for trying the workspace |
| `sample_data/synthetic/` | 24 generated phantoms (not medical images) for software tests only |
| `processed/` | empty in the release; `scripts/prepare_dataset.py` writes feature caches to `results/v10/features/` |

How to obtain, verify and use the full dataset, and how to upload your own data: [../DATASET_GUIDE.md](../DATASET_GUIDE.md).
Patient identifiers do not exist in this source and are never invented; units are reconstructed scan groups.
