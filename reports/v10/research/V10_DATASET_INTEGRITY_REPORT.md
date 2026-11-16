# V10 Dataset Integrity Report (Stage 3)

Repository: `liver_tumour_project_v9` · Run: 12 Nov 2026 · Read-only audit of the V9 data files.

Reproduce with `python scripts/v10_audit_dataset.py` (a few seconds). Outputs are in `results/v10/dataset/`
(`integrity.json` plus CSV tables and `run_manifest.json` with input hashes). Automated checks:
`tests/test_v10_data_integrity.py`. No file under `data/` was modified.

## Bottom line

1. **The Task A labels are genuine expert annotations.** They reproduce exactly from the LiTS expert masks and show no
   relationship to file order, index or any modulo rule. Task A is tumour patch against tumour-free liver patch, both
   taken from tumour-bearing slices.
2. **There are no patient or volume identifiers.** The ten groups are reconstructed from runs of tumour slices.
   Patient-level evaluation is not possible with this data, and V10 results must be reported at
   reconstructed-scan-group level.
3. **Three new data findings.**
   - A labelling defect in the V9 liver-gate data: 43 "no liver" patches contain 35–53% liver.
   - Strong evidence that test group G01 joins two different series.
   - Direct evidence that G06/G07 and G08/G09 may share scans.
4. **`vol_008` does not exist in this repository.** No group has 387 slices.
5. **Image-level checks cannot run.** No CT image files are in the repository.

## 1. What the repository contains

| Item | Present |
|---|---|
| CT images (`data/raw/scans`, 5,986 files referenced by the manifest) | **No** (0 of 5,986) |
| Processed patch arrays (`data/processed/*.npz`) | **No** |
| Feature caches (`artifacts/cache`) | **No** |
| Image manifest with per-file hashes, sizes, QC verdicts and slice mean/SD intensity | Yes |
| Verified slice labels, group and split tables, patch tables | Yes |
| 64 × 64 expert maps for the 476 tumour slices (8 × 8-pixel cells: background / liver / tumour) | Yes |
| Label-verification evidence (upstream label string, pixel-hash sequence, mask transfer chunks) | Yes |

Everything below is therefore computed from manifests, masks and verification files. Pixel-level duplicate detection,
image similarity across groups and merged-volume detection from images could not be run.

## 2. Dataset structure

| Set | Count | Use |
|---|---|---|
| Images in the source folder | 5,986 | |
| `ct_lits` slices, verified against LiTS-derived upstream (512 × 512, grey) | 1,745 | Supervised data |
| — with tumour | 476 | Task A patches come only from these |
| — without tumour | 1,269 | Liver-gate negatives only |
| `ct_slice` images (undocumented labels, no patient IDs) | 4,236 | Excluded; 200 used as out-of-distribution controls |
| Synthetic drawings | 5 | Excluded |

| Split | Groups | Task A tumour | Task A tumour-free liver | Gate liver | Gate other |
|---|---|---|---|---|---|
| Train | G03, G04, G05, G06 | 831 | 712 | 1,543 | 872 |
| Validation | G02, G09, G10 | 174 | 110 | 284 | 775 |
| Test | G01, G07, G08 | 376 | 246 | 622 | 565 |

Overall Task A prevalence: 1,381 tumour of 2,449 patches (56%). About 5.1 patches per slice.

## 3. Labels and annotations

| Check | Result |
|---|---|
| SHA-256 of the 3,986-character upstream label string equals the recorded value | Pass |
| All 1,745 slice labels equal the upstream string | Pass |
| File number to upstream index mapping is monotonic (55 failed downloads skipped) | Pass |
| Set of mask indices equals the set of tumour slices (476) | Pass |
| Mask transfer chunks decode to the stored masks | Pass |
| Missing values; label values | None; {0, 1} only |
| Folder names (normal / steatosis / …) not used as labels | Pass |
| Label vs upstream index, file number and row position, modulo 2–10 | χ² ≤ 0.93 everywhere; residue-class tumour rates within 0.017 of the overall 0.273 |
| Label changes along index order | 27 in 1,744 steps: long contiguous runs |

**On the modulo concern.** The row-number rule recorded in the project's history applies to the excluded `ct_slice`
images, whose folder was assigned by row number in an earlier build script. It does not apply to any label used for
training or evaluation.

**What the labels mean.** Each upstream label is "tumour present" exactly when the expert mask has tumour pixels. Every
no-tumour slice has an empty mask, with no liver annotated at all. A slice-level classifier would learn liver presence,
which is why V9 (correctly) defines Task A at patch level inside tumour-bearing slices. Task A results support claims
about distinguishing tumour tissue from tumour-free liver tissue, not about detecting tumour in a patient or a scan.

## 4. Masks

| Check | Result |
|---|---|
| Shape and values | 476 × 64 × 64; values {0, 1, 2} |
| Tumour cells per tumour slice (min / median / max) | 29 / 128 / 242; none empty |
| Liver cells per tumour slice | 124 / 443 / 940 |
| Agreement with recorded 512 × 512 pixel counts | r = 0.999 (tumour), 1.000 (liver) |
| Tumour cells with no liver or tumour neighbour | 0 |
| No-tumour slices with any liver annotation | 0 |

The masks are a lossy 64 × 64 reduction (argmax per 8 × 8 cell) of the original 512 × 512 masks. Full-resolution masks
are not in the repository.

## 5. Patches

### 5.1 Task A (all compared models): Pass

All 2,449 patches reproduce from the masks: tumour and liver fractions match to 10⁻⁴, every patch satisfies its labelling
rule (tumour: tumour fraction ≥ 0.05 and liver ≥ 0.40; tumour-free: no tumour within a one-cell margin and liver ≥ 0.50),
every centre cell has the right class, there are no duplicate coordinates, and group, split and scan fields agree with the
slice table.

### 5.2 Liver gate (inference only): **defect found**

The liver-tissue gate is a supporting model that finds liver patches at inference. Its "other" (no-liver) class contains
**43 patches that are 35–53% liver tissue**: 42 in training (all from G06) and 1 in validation (G10). All 43 lie at the
left image edge (x = 0).

**Cause.** `src/data/patches.py` line 96 tests `m[cy − half − 1 : cy + half + 1, cx − half − 1 : cx + half + 1]` for
liver. For candidates at the image edge (cx = half, so x = 0) the slice starts at −1, which Python reads as the last
column; the slice is empty and the liver test passes vacuously. Of the 44 left-edge negatives on tumour slices, 43 contain
liver. No candidate at y = 0 was selected.

**Impact.** Task A, and therefore every classical, quantum and hybrid comparison, is unaffected. The deployed gate was
trained with 42 mislabelled negatives among 872 (4.8%), which can bias it against liver near the image edge.

**Correction for V10.** Clip slice bounds at zero (as V9's own Task A code already does) and rebuild the gate set. The
affected patches are listed in `results/v10/dataset/gate_defect_patches.csv`.

## 6. Duplicates

- Exact duplicates (decoded-pixel hash): 0 among 5,986 images; all 1,745 verified slices have distinct pixels.
- Near-duplicates (32 × 32 thumbnail correlation ≥ 0.999, computed by V9's `prepare_data.py`): 14,723 pairs in total,
  226 between verified slices. All 226 are index-adjacent slices of the same group with the same label; none crosses a
  group or split boundary; none links a verified slice to an excluded one.
- Adjacent slices are near-copies and each slice yields about five patches, so patches are strongly dependent. Patch
  counts must not be used as sample sizes for inference.

## 7. Patient, scan and volume identity

**No identifiers exist.** The upstream dataset provides no patient, volume or slice-position field. `src/data/scans.py`
reconstructs groups from contiguous runs of tumour slices (14 runs), merges runs at most 10 no-tumour slices apart
(10 groups), and attaches no-tumour slices to the nearest group by a midpoint rule.

**The data are not complete volumes.** None of the 1,269 no-tumour slices has liver annotated, and several tumour runs end
at large liver cross-sections (for example 923, 655 and 550 liver cells of 4,096 at the last slice of G08, G07 and G04).
In a complete volume the next slices would show tumour-free liver. So liver-only slices are absent from the upstream
sample, and index adjacency does not imply anatomical adjacency at run ends.

### 7.1 Evidence about group boundaries

| Boundary | V9 splits | Gap between tumour runs (slices) | Liver-mask IoU across the gap | Identical padding run crosses it |
|---|---|---|---|---|
| G01 / G02 | test / val | 316 | 0.23 | No |
| G02 / G03 | val / train | 287 | 0.36 | No |
| G03 / G04 | train / train | 32 | 0.52 | Not audited |
| G04 / G05 | train / train | 15 | 0.20 | Not audited |
| G05 / G06 | train / train | 70 | 0.00 | Not audited |
| **G06 / G07** | **train / test** | 53 | 0.09 | **Yes** |
| G07 / G08 | test / test | 133 | 0.03 | Not audited |
| **G08 / G09** | **test / val** | 30 | 0.53 | **Yes** |
| G09 / G10 | val / val | 154 | 0.00 | Not audited |

"Padding run" means a run of identical zero-padding counts in the upstream raw slices, recorded in
`data/labels/verification/scan_signature_audit.json`. The project audited only boundaries between different V9 splits, and
the per-slice counts are not in the repository. A shared padding value is weak evidence of a shared scan: the same value
recurs in 13 separate index runs. For comparison, consecutive tumour slices inside a run have liver IoU median 0.98
(1st percentile 0.80).

### 7.2 Evidence inside groups: G01 probably joins two series

Among all 462 steps between index-adjacent tumour slices, one is extreme on both measures:

| Step | Group | Change in slice mean intensity | Liver-mask IoU | Liver area ratio |
|---|---|---|---|---|
| 202 → 203 | **G01** | −15.0 (median step 0.27; next largest 6.5) | **0.33** (median 0.98; next lowest 0.72) | 0.33 |
| 959 → 960 | G05 | +4.8 | 0.74 | 1.28 |

At 202 → 203 the liver area falls from 589 to 197 cells and the liver centroid shifts, while every neighbouring step
changes by under 1%. This indicates that G01's single "scan" G01-R1 joins two series. The second part (slices 203–215)
contributes 41 of G01's 531 patches. The project's padding audit found one padding value across slices 100–233, which
is consistent with one scan, but that value is not unique. Whether the two parts belong to one patient cannot be decided
without the original volumes. Both parts are in the test split, so there is no leakage. V10 keeps G01 as one evaluation
unit, because merging is the safe direction.

### 7.3 `vol_008`

No file in the repository contains `vol_008` or any identifier of the form `vol_NNN` (checked over the whole folder in
Stage 1, and again over 180 text files here). No group has 387 slices; group sizes are 37–305 slices (median 156.5), and
there are ten groups. The observation in the master prompt appears to describe a different dataset or project version.
The closest V9 analogue is the G01 join above.

## 8. Groups, prevalence and intensity

| Group | Split | Slices | Tumour slices | Tumour runs | Task A patches | Tumour share | Median slice intensity, tumour slices |
|---|---|---|---|---|---|---|---|
| G01 | test | 249 | 116 | 1 | 531 | 0.635 | 43.1 |
| G02 | val | 305 | 33 | 1 | 171 | 0.579 | 54.9 |
| G03 | train | 173 | 14 | 1 | 43 | 0.907 | 38.4 |
| G04 | train | 37 | 13 | 1 | 91 | 0.429 | 75.6 |
| G05 | train | 140 | 79 | 4 | 278 | 0.820 | 50.2 |
| G06 | train | 246 | 178 | 2 | 1,131 | 0.464 | 64.2 |
| G07 | test | 101 | 8 | 1 | 56 | 0.429 | 67.0 |
| G08 | test | 86 | 5 | 1 | 35 | 0.429 | 95.5 |
| G09 | val | 104 | 12 | 1 | 61 | 0.590 | 44.8 |
| G10 | val | 304 | 18 | 1 | 52 | 0.750 | 50.5 |

- **Group identity predicts the label share**: tumour share ranges from 0.43 to 0.91 (χ² = 173 on 9 degrees of freedom,
  descriptive only because patches are dependent). Groups never straddle splits, so a model cannot use group identity
  directly, but pooled metrics and fixed thresholds are sensitive to this prevalence shift.
- **Intensity differs between groups.** Median whole-slice intensity of tumour slices ranges from 38 to 95. G01 is among
  the darkest, consistent with V9's G01 analysis. Whole-slice means include body and background, so this is a coarse
  indicator only.
- **Concentration.** G06 supplies 46% of all Task A patches; G03 has only 4 tumour-free patches and G10 only 13, so their
  per-group AUCs rest on very few pairs.

## 9. G01, G07 and G08

| | G01 | G07 | G08 |
|---|---|---|---|
| Test patches (tumour / tumour-free) | 531 (337 / 194) | 56 (24 / 32) | 35 (15 / 20) |
| Share of test patches | 85% | 9% | 6% |
| Tumour slices | 116 (one run) | 8 | 5 |
| AUC of all six CNN-feature quantum models and controls | 0.966–0.999 | 1.000 | 1.000 |

From Stage 2: every difference between CNN-feature models comes from G01, and an exact group-level test cannot go below
p = 1 for those comparisons. G01 is also the darkest and probably two joined series. The V9 test metric is effectively a
G01 metric.

## 10. Leakage risks

| Risk | Status | Consequence for V10 |
|---|---|---|
| Same group in two splits | None (all 10 groups in one split; slice and patch tables agree) | Keep group-disjoint splits |
| Related groups across splits | Possible at G06/G07 (train/test) and G08/G09 (test/val); unknown at five other boundaries | Merge G06+G07 and G08+G09 into single units |
| Near-duplicate slices across groups | None | No action |
| Embedding CNN trained on test groups | No (trained on G03–G06) | Embeddings still exploratory because validation was reused and the test groups were examined |
| Validation reused for many choices | Yes | Grouped inner cross-validation |
| Test set examined during V9 design | Yes (G01 analysis) | G01, G07, G08 results in V10 are not confirmatory |
| Liver-gate negatives containing liver | 43 patches | Gate only; fix before rebuilding the gate |

## 11. Evaluation units proposed for V10

Merging groups with direct evidence of a shared scan gives eight units:

| Unit | Task A patches | Tumour / tumour-free | Share of patches |
|---|---|---|---|
| G01 | 531 | 337 / 194 | 21.7% |
| G02 | 171 | 99 / 72 | 7.0% |
| G03 | 43 | 39 / 4 | 1.8% |
| G04 | 91 | 39 / 52 | 3.7% |
| G05 | 278 | 228 / 50 | 11.4% |
| **G06+G07** | **1,187** | 549 / 638 | **48.5%** |
| **G08+G09** | 96 | 51 / 45 | 3.9% |
| G10 | 52 | 39 / 13 | 2.1% |

- The units are very unequal. Holding out G06+G07 removes 48.5% of the data from training. This must shape the V10
  protocol: report fold-wise results, and do not treat the pooled out-of-fold AUC as the only summary.
- A sensitivity analysis with the original ten groups is planned. Merging further on weaker evidence (for example G03/G04,
  liver IoU 0.52 across a 32-slice gap) would leave too few units to evaluate.
- The merge decision uses only metadata, never results.

## 12. Is patient-level evaluation possible?

**No.** The dataset has no patient or volume identifiers, the reconstructed groups are not verified patients, and at
least one group probably joins two series. V10 will evaluate at the level of reconstructed scan-group units and will
state in every result that **patient-level generalisation cannot be established without patient identifiers**.

Recovering identifiers needs the original LiTS segmentation volumes; `scripts/v9_recover_volume_ids.py` would then map
every tumour slice to a volume by matching expert maps.

## 13. Data limitations

1. No images, processed arrays or feature caches in the repository: no image-level check and no retraining is possible.
2. No patient or volume identity; eight usable evaluation units after merging, one holding 48.5% of the data.
3. Masks only at 64 × 64 resolution.
4. The upstream sample omits liver-only slices, so the data are not complete volumes and slice adjacency is unreliable at
   run ends.
5. Labels define tumour-versus-tumour-free tissue within tumour slices only; no tumour-free liver slices exist, so
   slice- or scan-level specificity cannot be measured.
6. Strong between-group differences in prevalence (0.43–0.91) and intensity.
7. The test groups were examined in V9; none of them can serve as an untouched confirmatory set in V10.
8. CC BY-NC-SA 4.0 licence (non-commercial research use with attribution).

## 14. What would change with more data

| If provided | What becomes possible |
|---|---|
| The `scans` folder (or `data/processed` and `artifacts/cache`) | Fold-specific feature extraction, all Stage 2 items marked "not executable", pixel-level duplicate and similarity checks, the corrected liver gate |
| The original LiTS segmentation volumes | Volume identity for every tumour slice, a check of the G01 join and of the G06/G07 and G08/G09 relations, possibly more evaluation units |
| Per-slice padding counts from the upstream raw images | Relation evidence at the five boundaries that were never audited |
