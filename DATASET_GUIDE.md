# Dataset guide

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

## 1. The dataset the V10 models were trained on

**`liver_ct_lits_verified_v8`** — 1,745 axial CT slices (512 × 512 PNG, windowed, grey in all three RGB channels),
matched pixel-for-pixel to the Hugging Face dataset `seongun/liver-tumor-classification`, which is derived from the
LiTS benchmark. Slice labels were re-derived from the expert segmentation masks (1 = any tumour pixel).

| Item | Value |
|---|---|
| Licence | **CC BY-NC-SA 4.0** — non-commercial, attribution, share-alike (inherited from the source) |
| Citation | P. Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)", *Medical Image Analysis* 84, 102680 (2023), doi:10.1016/j.media.2022.102680 |
| Slices | 1,745 (476 with tumour) in 10 reconstructed scan groups G01–G10 |
| Patches (training unit of the V10 models) | 2,449 liver-tissue patches of 96 × 96 px (1,381 tumour, 1,068 tumour-free) in `patches/patches_v8.npz`, row-aligned with `manifests/patches_task_a.csv` |
| Version string recorded in every model card and prediction | `lits_verified_v8@<first 16 hex of the content hash>` |

### What the units are — and are not

The source has **no patient or volume identifiers**. Scan groups were *reconstructed* from slice adjacency; they are
the unit of grouping in every split and every cross-validation fold, so no group is ever in both training and
evaluation data. They are **not patients**: two groups may come from the same patient, and one patient may span
groups. Results are therefore reported per reconstructed scan group, never as patient-level performance. Groups
G06+G07 and G08+G09 are merged into one unit each for evaluation because they are adjacent in the source ordering
(8 evaluation units in total).

### Patch labelling rules (from the dataset README)

* tumour patch: centred on tumour, ≥ 5 % tumour and ≥ 40 % liver-or-tumour pixels;
* tumour-free patch: ≥ 50 % liver, no tumour inside the patch or within an 8-px margin;
* patches with 0–5 % tumour are discarded as ambiguous.

## 2. Where the dataset lives

The project keeps its copy in **`dataset/liver_ct_lits_verified_v8/`** (the folder exactly as obtained: `images/`,
`labels/`, `manifests/`, `patches/`, `masks/`, `verification/`, `README.md`). Every script uses it by default
(`--dataset` or `LTI_DATASET_DIR` override it), `setup.sh` verifies it against
`dataset/manifests/lits_verified_v8_SHA256SUMS.txt` and registers it in the platform, and Docker Compose mounts it
read-only at `/data/liver_ct_lits_verified_v8`. Keep it a real folder, not a symlink (dataset registration only
accepts folders inside `DATASET_ROOTS`, default `dataset/`).

## 3. Redistribution

The dataset is licensed **CC BY-NC-SA 4.0**: you may copy and share it for **non-commercial** purposes, with
attribution (the citation above) and under the same licence. Accordingly:

* the **full** project folder / `liver_tumour_intelligence_v10_full.zip` (`python scripts/package_release.py --full`)
  includes the dataset — for your own use, backups and non-commercial academic sharing with attribution;
* the **lean** `liver_tumour_intelligence_v10.zip` (`python scripts/package_release.py`) leaves it out (and the 64 × 64
  mask maps derived from it); only the 8-slice sample in `dataset/sample_data/lits_sample` is kept, with attribution.

`dataset/manifests/lits_verified_v8_manifest.csv` lists the 1,745 slices (path, label, reconstructed group, slice
index, split, source) and `dataset/manifests/lits_verified_v8_SHA256SUMS.txt` the SHA-256 of all 1,757 files of the
dataset folder, so any copy can be verified file by file:
`cd dataset/liver_ct_lits_verified_v8 && shasum -a 256 -c ../manifests/lits_verified_v8_SHA256SUMS.txt`
(Linux: `sha256sum -c`).

`dataset/sample_data/synthetic` holds 24 **synthetic phantoms** (generated shapes, not medical images). They exist only
for software tests; models trained on them are labelled synthetic, refused by the engine unless
`ALLOW_SYNTHETIC_MODELS=true`, and that setting is rejected in production.

## 4. Using the dataset

```bash
source .venv/bin/activate          # after ./setup.sh
# 1. validate (decodes every image, checks hashes, labels, duplicates, groups and leakage)
python scripts/validate_dataset.py
# 2. write a manifest (path, label, group, slice index, split, source) — also done by the platform on registration
python scripts/generate_manifest.py
# 3. extract and cache the feature sets used by the models (radiomics, ImageNet GAP embeddings)
python scripts/prepare_dataset.py
# 4. nested evaluation and final training (see V10_REPRODUCIBILITY.md)
python scripts/run_nested_evaluation.py --features radiomics
```

Without arguments the scripts use `dataset/liver_ct_lits_verified_v8`; pass `--dataset /other/folder` or set `LTI_DATASET_DIR` to use another copy.

## 5. Uploading or registering datasets in the platform

*Datasets → Upload* (researchers and admins) accepts a ZIP archive with images and a manifest CSV
(`file_name` or `image_path`, `label`, optional `group_id` / `scan_id` / `patient_id` / `split`). Admins can register
a server-side folder under one of the `DATASET_ROOTS` without uploading (*Datasets → Register server folder*).

Ingestion is defensive: archive paths are checked (no absolute paths, `..`, symlinks; size and file-count limits),
every image is decoded and checked for size, channels and constant content, labels are validated against the allowed
set, duplicates are found by pixel hash, and leakage is checked (the same group, scan, patient or pixel hash in more
than one split). The validation report lists every problem; nothing is silently fixed. Uploaded datasets are **not**
used for training unless an administrator starts a training run on them explicitly.

PNG and JPEG files carry no DICOM / NIfTI metadata: the platform records them as such and never invents spacing,
windowing, orientation or patient fields. DICOM (uncompressed transfer syntaxes) and NIfTI (`.nii`, `.nii.gz`) are read
by built-in decoders; install `pydicom` for compressed DICOM.

## 6. Adding new data

Do not mix datasets without documenting provenance: register each source as its own dataset (name, source, licence,
version), keep its manifest, and record the dataset version in any model trained on it. Do not train on labels that
were not produced by a documented rule or by expert annotation, and do not assign labels at random.
