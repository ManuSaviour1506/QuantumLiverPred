# Real-data sample (8 CT slices) — test fixture

Eight slices copied unchanged from the verified LiTS-derived dataset (`liver_ct_lits_verified_v8`), with their rows of
`labels/slices.csv`. They let the end-to-end tests and the demo run real inference without the full dataset.

* `images/test/ct_lits_4277-4279.png`: three consecutive tumour slices of reconstructed group G01 (test split)
* `images/test/ct_lits_4353-4354.png`: two G01 slices without tumour (outside the liver in this source)
* `images/val/ct_lits_4649-4651.png`: three consecutive tumour slices of group G02 (validation split)

Licence: CC BY-NC-SA 4.0 (non-commercial use, attribution, share-alike), inherited from the source
`seongun/liver-tumor-classification` (Hugging Face), which is derived from the LiTS benchmark.
Cite: P. Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)", Medical Image Analysis 84, 102680 (2023),
doi:10.1016/j.media.2022.102680.

Research and teaching use only; not for clinical diagnosis.
