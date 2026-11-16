# V9 Source Audit (Stage 1)

Repository: `liver_tumour_project_v9` · Audit date: 2026-11-11 · Stage 1 of the V10 master prompt

Revision 2 (11 Nov 2026, 15:45 UTC). Updated for the revised master prompt: sections 4.6 and 4.7 (row-index label
concern; G01, G07, G08 and `vol_008`), a re-check of the environment in section 3, a finding-by-finding
verifiability map (section 12), updated blockers (section 13), a prioritized implementation plan (section 14) and
the decisions that need approval (section 15).

This audit is read-only. No V9 source file, checkpoint, prediction or result was changed. The only file added is this
report (`reports/V9_SOURCE_AUDIT.md`). One side effect of the earlier quantum audit is disclosed in section 11.

Every statement below is labelled by how it was established:

- **[verified]** recomputed or re-derived during this audit from files in the repository
- **[recorded]** quoted from the repository's own result files or documentation, not recomputed
- **[inference]** a conclusion that follows from the evidence but could not be tested here

---

## 1. Bottom line

1. **The code is careful and the saved results are internally consistent.** All 45 prediction files reproduce the
   reported metrics exactly, 46 of 47 model checkpoints match their recorded SHA-256, and slice and patch labels
   reproduce from the expert masks with zero mismatches. [verified]
2. **Labels are trustworthy; patient identity is not.** Labels come from LiTS expert masks and are not derived from
   file names, folders or indices. But the dataset has no patient or volume identifiers. The ten "scan groups" are
   reconstructed from runs of tumour slices and cannot be confirmed as independent patients. [verified]
3. **There is no train-to-test leakage through the CNN, but validation is heavily reused and the test set has
   informed design.** The CNN that produces the embeddings was fitted on training groups only. The same three
   validation groups are then used for early stopping, recipe selection, quantum model selection, calibration and
   thresholds. V9 recipes were motivated by an analysis of test scan G01. [verified]
4. **The quantum simulator is mathematically correct, with four structural design problems**: dead parameters,
   redundant parameters, parity-type readouts, and a ZZ feature map whose pairwise term is negligible at the selected
   bandwidth. [verified]
5. **The evidence base is too small for the claims V10 wants to make.** One group supplies 85 % of test patches and
   one supplies 73 % of training patches. Every model uses a single seed. [verified]
6. **Most of the V10 experimental plan cannot run from this folder as delivered.** The CT images, the processed patch
   arrays and the cached features are not included, there are no genuine patient IDs, and PyTorch, PennyLane and
   Qiskit are not installed and cannot be installed in this session. Section 13 lists exactly what is blocked and
   what is needed.
7. **Two concerns raised in the master prompt do not apply to the V9 data as delivered.** Labels show no
   relationship to row index under any modulus from 2 to 10. No identifier `vol_008`, and no group of 387 slices,
   exists anywhere in this repository; V9 has ten reconstructed groups, not nine. [verified]

---

## 2. Repository architecture

No version control (`.git` absent). 275 MB in total: `artifacts/` 235 MB, `results/` 34 MB, `data/` 4.7 MB.
90 Python files; 11,226 lines in `src/`, `scripts/`, `tests/` and `backend/`. [verified]

| Path | Contents |
|---|---|
| `run_v9.py`, `run_v8.py` | Command dispatchers (prepare, train, evaluate, reports, serve, test) |
| `src/common/` | Paths, JSON and array I/O, seeding |
| `src/data/` | Verified labels (`labels.py`), scan-group reconstruction (`scans.py`), split search (`split.py`), patch sampling from masks (`patches.py`) |
| `src/preprocessing/`, `src/quality_control/` | CT windowing and formats (PNG, DICOM, NIfTI), image QC, out-of-distribution detector |
| `src/classical/` | NumPy CNN (`numpy_cnn.py`), NumPy autodiff (`nn.py`), ImageNet backbones in NumPy (`backbones.py`, `torchload.py`), tail fine-tuning (`cnn_finetune.py`), radiomics baselines (`baselines.py`) |
| `src/quantum/` | State-vector simulator (`simulator.py`), V8 VQC and kernel SVM (`models.py`), V9 models and noise (`v9.py`), optional PennyLane cross-check (`pennylane_backend.py`) |
| `src/hybrid/` | V8 hybrid and classical twin |
| `src/evaluation/` | Metrics and cluster bootstrap (`metrics.py`), calibration guards (`guards.py`), post-training finalisation (`finalize.py`), V9 operating points (`v9.py`) |
| `src/segmentation/`, `src/volume/`, `src/explain/` | 2.5-D U-Net, whole-volume pipeline, Grad-CAM |
| `src/inference/` | Model registry with integrity checks, inference pipeline, services |
| `scripts/` | 24 scripts: data preparation, label verification, training (V8 and V9), quantum sweeps, evaluation, reports, ID recovery |
| `tests/`, `backend/tests/` | 49 + 10 tests |
| `backend/`, `frontend/` | Starlette API and React dashboard (built bundle in `frontend/dist`) |
| `data/` | Manifests, verified labels, verification evidence, 64 × 64 expert masks. **No images.** |
| `artifacts/` | 47 saved models, 3 ImageNet weight files, OOD detector |
| `results/` | Prediction CSVs, metrics, comparison JSON, logs, figures, three PDF reports, one Excel workbook |

**No PyTorch anywhere.** All networks, including ResNet-50, EfficientNet-B0 and DenseNet-121 inference and tail
fine-tuning, are implemented in NumPy. The master prompt's assumption of a PyTorch code base does not hold.

### Model inventory (47 saved models) [verified]

| Family | Count | Models |
|---|---|---|
| Classical | 34 | 9 NumPy CNNs (V8 2-D and 2.5-D; V9 ablations a1–a4, 2-D, 5-slice, 3-D), 6 fine-tuned backbones, 6 radiomics models, 12 controls (PCA logistic regression, RBF-SVM, classical twins), liver gate |
| Quantum | 8 | V8: VQC and kernel SVM on ImageNet and fine-tuned features. V9: VQC and kernel SVM on CNN and EfficientNet features |
| Hybrid | 4 | V8 on ImageNet and fine-tuned features; V9 on CNN and EfficientNet features |
| Segmentation | 1 | 2.5-D U-Net |

### Data flow for the quantum experiments

`scans` → `prepare_data.py` (manifest, labels, groups, split, patches) → `v9_train_cnn.py` (CNN on training
patches) → `v9_evaluate.py --select` (CNN chosen on validation AUC) → `v9_quantum.py` (CNN embedding or
EfficientNet features → PCA on training rows → quantum models and controls, selected on validation AUC) →
`finalize()` (Platt calibration and threshold on validation; test scores recorded) → `v9_evaluate.py` (metrics,
bootstrap, paired tests) → `v9_reports.py`.

---

## 3. Environment and dependencies [verified]

| | Shell on the user's computer | Cloud workspace |
|---|---|---|
| Can see the project folder | Yes | Only files copied in |
| Python | 3.10.12, 4 cores, 3.9 GB RAM | 3.13.18, 2 cores, 8 GB RAM |
| Present | numpy 2.2.6, pandas, Pillow, matplotlib, openpyxl, reportlab | numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, scikit-image, pandas, pytest 9.1.1, starlette, uvicorn, OpenCV |
| Missing | scipy, scikit-learn, scikit-image, pytest, starlette, uvicorn | nibabel, pydicom |
| Missing in both | PyTorch, torchvision, PennyLane, Qiskit, Qiskit Aer | |
| Package installs | Refused by the network proxy (HTTP 403) | Disabled for this account |

Consequences:

- The project's training and test code needs scikit-learn, so it can only run in the cloud workspace, on a copy.
- PennyLane and Qiskit validation (master prompt Part 9) cannot be done in this session.
- `requirements.txt` is incomplete: OpenCV (`cv2`) is imported by `src/classical/numpy_cnn.py`,
  `src/classical/baselines.py`, `src/segmentation/unet.py` and `scripts/v9_train_backbones.py`; `reportlab` and
  `openpyxl` by `scripts/v9_reports.py`; `playwright` by `scripts/e2e_frontend.py`. None is listed.
- Versions are lower bounds only (`>=`); there is no lock file. The Dockerfile targets V8 and states it was never built.

Re-check at 15:42 UTC on 11 Nov 2026, as the revised prompt requires: PyTorch, PennyLane, Qiskit, Qiskit Aer and
XGBoost are still absent from both environments, a test download of PennyLane is still refused in both, and neither
environment has a GPU. The PennyLane equivalence test therefore remains unexecuted.

Scope of this statement: both environments are sandboxes. Packages installed in the Python environment on the
user's own macOS system are not visible from either, so nothing here says what that machine can run.

---

## 4. Dataset, labels, masks and identifiers

### 4.1 What is in the folder

| Item | Present | Notes |
|---|---|---|
| CT images (`data/raw/scans`) | **No** | Only `data/raw/README.md`. The packaging script excludes them deliberately |
| Processed patch arrays (`data/processed/`) | **No** | `patches_v8.npz`, `v9_task_a_5s.npz`, `v9_seg.npz` are needed by every training script |
| Cached backbone features (`artifacts/cache/`) | **No** | Needed for EfficientNet features and tail fine-tuning |
| Manifests, labels, verification evidence | Yes | 8 manifest files, 2 label files, 7 verification files |
| Expert masks | Yes | 476 maps at 64 × 64, three classes |
| Per-patch predictions | Yes | Validation and test rows only; no training-row scores |

### 4.2 Label integrity [verified]

| Check | Result |
|---|---|
| Slice label equals the upstream label string for all 1,745 verified slices | Pass |
| SHA-256 of the upstream label string equals the recorded value | Pass |
| Set of mask indices equals the set of tumour slices (476) | Pass |
| Every tumour slice has tumour cells in its mask (minimum 29, median 128) | Pass |
| Masks decoded from the transfer chunks equal the stored masks | Pass |
| Tumour pixel counts at 512 × 512 agree with mask cell counts (r = 0.999) | Pass |
| Label independent of file-number parity (633 / 636 negative, 239 / 237 positive) | Pass |
| Folder names not used as labels (code inspection and `tests/test_data_labels_split.py`) | Pass |
| All 2,449 patch rows: tumour and liver fractions reproduce from the masks | 0 mismatches |
| All 2,449 patch rows obey the labelling rule and centre-cell class | 0 violations |
| Duplicate patch coordinates | 0 |

Labels are not derived from indices, file names or modulo rules. The 4,236 `ct_slice` images with undocumented
labels and the 5 synthetic drawings are excluded from all supervised work. [verified]

### 4.3 What the task actually is

- 1,745 verified slices: 476 with tumour, 1,269 without. All 1,269 no-tumour slices have **empty** upstream masks.
- A slice-level tumour classifier would therefore learn liver presence. The project correctly avoids this: Task A is
  **tumour patch versus tumour-free liver patch, both taken from tumour slices only**.
- Task A has 2,449 patches from 476 slices (5.1 per slice).

| Split | Groups | Tumour patches | Tumour-free liver patches | Slices |
|---|---|---|---|---|
| Train | G03, G04, G05, G06 | 831 | 712 | 284 |
| Validation | G02, G09, G10 | 174 | 110 | 63 |
| Test | G01, G07, G08 | 376 | 246 | 129 |

Concentration: G06 supplies 1,131 of 1,543 training patches (73 %); G01 supplies 531 of 622 test patches (85 %);
G02 supplies 171 of 284 validation patches (60 %). G03 contributes 4 tumour-free patches. [verified]

### 4.4 Patient and volume identifiers: the central limitation

- The upstream dataset has no patient, volume or slice-position field (`src/data/scans.py` states this). [verified]
- Groups are built by `assign_groups`: contiguous runs of tumour slices, merged when at most 10 no-tumour slices
  apart; no-tumour slices attached by a midpoint rule. This gives 14 runs in 10 groups. [verified]
- The split was fixed before training by an exhaustive search over group assignments for a 60 / 15 / 25 share of
  tumour slices, with no reference to model results. No group appears in two splits. [verified]

Independent evidence on the group boundaries, computed from the masks [verified]:

| Boundary | Splits | Gap (slices) | Liver-mask IoU across the gap |
|---|---|---|---|
| G01 → G02 | test → validation | 316 | 0.233 |
| G02 → G03 | validation → train | 287 | 0.356 |
| G06 → G07 | train → test | 53 | 0.092 |
| G08 → G09 | test → validation | 30 | 0.529 |
| Within runs, consecutive slices (n = 462) | | 0 | median 0.980 |

- The low IoU at boundaries is consistent with a change of scan, but it would also occur within one patient after a
  gap of tens of slices. It does not prove independence. [inference]
- The project's own padding-signature audit [recorded] found runs of identical padding crossing the G06/G07
  (train/test) and G08/G09 (test/validation) boundaries. The project notes that no tumour slices lie in those runs
  and that the signature is not unique to a patient.
- The claim that no-tumour slices "lie outside the liver" is an inference from the empty masks. Several tumour runs
  end at large liver cross-sections (923, 655 and 553 liver cells of 4,096 at run edges), which is hard to reconcile
  with the next slice being outside the liver if index order is contiguous. Either the upstream dataset dropped
  liver-only slices or some "Normal" slices contain unannotated liver. This does not affect Task A labels. It does
  affect the liver-gate labels and weakens step 1 of the grouping argument. It cannot be resolved without the
  images or the original volumes. [inference]

**Conclusion.** Under the master prompt's rule 3.2, the groups must not be treated as independent patients. Any V10
result on this data must be labelled "reconstructed scan-group level" unless genuine LiTS volume IDs are recovered.
`scripts/v9_recover_volume_ids.py` exists for this and is tested on synthetic volumes only; it needs the original
LiTS segmentation volumes.

### 4.5 Duplicates and dependence [verified]

- 5,986 images in the manifest; 0 exact duplicates; all pass QC.
- 14,723 near-duplicate pairs (thumbnail correlation ≥ 0.999). 226 are between verified slices; all 226 are adjacent
  slices, and none crosses a group or split boundary.
- Adjacent slices are near-copies and each slice yields about five patches, so the 2,449 patches are far from
  independent. Patch-level counts must not be used as the sample size for inference.

### 4.6 The row-index / modulo label concern [verified]

The master prompt records a historical concern that labels may have been generated by row-index modulo rules.

- For the 1,745 verified slices, the label is independent of upstream index, file number and row position modulo
  every k from 2 to 10: chi-square statistics are between 0.0 and 0.9 on 1 to 9 degrees of freedom, and the tumour
  rate in any residue class differs from the overall rate (0.273) by at most 0.017.
- Labels change only 27 times along the 1,744 index transitions. They form long contiguous runs, as anatomy would
  produce; a modulo rule would produce hundreds of changes.
- The rule the concern describes does exist in the project's history, but for a different set of images: the 4,236
  `ct_slice` files had their folder assigned by row number in an earlier build script. Those files are excluded from
  all supervised training and evaluation (`data/labels/excluded_unverified.csv`). Their only use is as 200
  out-of-distribution control images in `scripts/train_gate_ood.py`, where no label is involved.
- One benign artefact for anyone repeating this check: in `patches_task_a.csv` the label is strongly related to
  row position modulo 7, because each slice writes its tumour patches first and its tumour-free patches after.
  This is file order only. Patch labels come from the masks (section 4.2) and row position is never a model input.

**Conclusion.** The V9 Task A labels are real annotations derived from expert masks. The limitation is what they
label: patches of tumour tissue against tumour-free liver tissue within tumour-bearing slices. They do not support a
claim about detecting tumour in a patient or a scan.

### 4.7 G01, G07, G08 and `vol_008` [verified]

| Group | Split | Slices | Tumour slices | Patches | Tumour share of patches |
|---|---|---|---|---|---|
| G01 | test | 249 | 116 | 531 | 0.635 |
| G02 | validation | 305 | 33 | 171 | 0.579 |
| G03 | train | 173 | 14 | 43 | 0.907 |
| G04 | train | 37 | 13 | 91 | 0.429 |
| G05 | train | 140 | 79 | 278 | 0.820 |
| G06 | train | 246 | 178 | 1,131 | 0.464 |
| G07 | test | 101 | 8 | 56 | 0.429 |
| G08 | test | 86 | 5 | 35 | 0.429 |
| G09 | validation | 104 | 12 | 61 | 0.590 |
| G10 | validation | 304 | 18 | 52 | 0.750 |

- The test set is 622 patches from three groups; G01 supplies 531 of them and 337 of the 376 tumour patches.
- G07 and G08 score AUC 1.000 for all six CNN-feature quantum models and controls, so every difference between
  those models comes from G01.
- Group identity carries label information: the tumour share of patches ranges from 0.429 to 0.907 across groups.
  Groups never straddle splits, so a model cannot use group identity directly, but pooled metrics and fixed
  thresholds are sensitive to this prevalence shift.
- **`vol_008`.** No file in the repository contains the string `vol_008` or any identifier of the form `vol_NNN`.
  No group has 387 slices; group sizes are 37 to 305 slices with a median of 156.5. There are ten groups, not nine.
  The reported observation (a 387-slice volume against a median of 82, possibly merged) cannot be verified in V9
  and appears to describe a different dataset or an earlier project version. It is not carried forward as a
  finding about V9.
- Whether any V9 group merges several real volumes, or splits one, cannot be decided from the manifests. G05
  contains four separate tumour runs and G06 two, so merged volumes inside a group are plausible. Merging is the
  safe direction for leakage; splitting is not (section 4.4).

---

## 5. Training and evaluation pipeline

### 5.1 What is sound [verified by code inspection]

- CNN normalisation statistics, PCA, feature scaling and class weights are fitted on training rows only.
- Early stopping, recipe selection, hyper-parameters, Platt calibration and thresholds use validation rows only.
- Test scores are written once by `finalize()` and never enter a choice in code.
- Every model is scored on the same 622 test rows; row order, labels and split are identical across all 45
  prediction files.
- The selected CNN (`cnn25d_v9_a4_res_se_focal`, 325,501 parameters, best epoch 22 of 28) was trained on groups
  G03–G06 only. **The concern that the embedding CNN saw test patients is not supported by the code.**

### 5.2 Problems

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| E1 | High | **Validation reuse.** The same three validation groups serve CNN early stopping, CNN recipe selection, selection among 30 quantum and control configurations per feature set, calibration and threshold choice | `v9_train_cnn.py`, `v9_evaluate.py --select`, `v9_quantum.py`, `finalize.py`. Validation AUC is 0.999 to 1.000 for most CNN-feature models, so it no longer discriminates |
| E2 | High | **Test-informed design.** V9 recipes were motivated by an analysis of test scan G01 (intensity, noise, feature shift, V8 errors, a montage of missed patches) | Docstrings of `v9_train_radiomics.py` and `v9_train_backbones.py`; `README.md`. The specific options were then chosen on validation, but the option space came from the test scan. V9-versus-V8 test comparisons are therefore exploratory, and G01, G07 and G08 are no longer an untouched test set for V10 |
| E3 | High | **Three test clusters, one dominant.** G07 and G08 score AUC 1.000 for every CNN-feature quantum model and control, so all differences come from G01 | Recomputed per-scan AUCs from the prediction files |
| E4 | Medium | **Bootstrap p-values are weak.** `p = 2·min(P(d ≤ 0), P(d ≥ 0))` over a two-stage bootstrap with three scan clusters. This is not a permutation test, and with three clusters the between-scan variance is essentially unestimated | `src/evaluation/metrics.py::paired_auc_difference` |
| E5 | Medium | **No multiplicity control.** 30 paired comparisons are reported, 15 of them involving quantum or hybrid models, all uncorrected. The JSON records "3 planned comparisons" per feature set | `results/v9/comparison_v9.json` |
| E6 | Medium | **Non-significance is presented beside ΔAUC without an equivalence margin.** "Hybrid vs twin, p = 1.000" is not evidence of equivalence | `results/v9/RESULTS_V9.md` |
| E7 | Medium | **Single seed (42)** for every CNN, VQC, hybrid and split-dependent sampler. Seed variance is unknown | All training scripts |
| E8 | Low | Early stopping picks the maximum of a noisy validation curve (0.9947 to 0.9996 over the last 12 epochs), which biases validation AUC upward | Training history in the selected CNN's `model.json` |
| E9 | Low | `v9_evaluate.py` writes test summaries into each `model.json` (`update_meta`). Not used for selection, but it mixes evaluation output into model artifacts | `scripts/v9_evaluate.py` |

---

## 6. Quantum and hybrid implementation

### 6.1 Structure

- `simulator.py`: batched exact state-vector simulation; gates RX, RY, RZ, H, CNOT, CZ; `<Z_q>` expectations; adjoint
  gradients with respect to weights and data angles; parameter-shift gradients; ZZ feature map; fidelity kernel.
- `vqc_circuit(n, L)`: per layer, `RY(x_q)` on every qubit (data re-uploading), then `RY(w)`, `RZ(w)` on every qubit,
  then a CNOT ring `0→1, 1→2, …, n−1→0`.
- `VQCv9`: optional trainable per-qubit scale `a` and offset `o` on the inputs; readout `<Z_0>` or a trainable linear
  combination of all `<Z_q>`.
- `ZZFeatureMap`: per repetition, H on every qubit, `RZ(2 x_q)`, and on neighbouring pairs
  `CNOT · RZ(2(π − x_a)(π − x_b)) · CNOT`. Two repetitions, linear chain.
- `HybridV9`: features → linear projection → tanh → angles → circuit → `<Z_q>` → linear readout, with optional fusion
  that concatenates the tanh projection to the readout input. `TwinV9` replaces the circuit with a dense tanh layer.
- `noisy_expect_z`: Pauli-trajectory depolarising noise after every gate plus binomial shot sampling.
  `noisy_kernel`: global depolarising approximation plus binomial sampling of kernel entries.

### 6.2 Verified mathematical facts

All checks below were run during this audit on a copy of the source. Tolerances are machine precision unless stated.

1. **State vector and gradients are correct.** The project's own tests pass: state vector equals dense-matrix
   evolution, adjoint equals parameter shift, data gradients equal finite differences, kernel is symmetric with unit
   diagonal and non-negative eigenvalues.
2. **Dead parameters.** The last layer's RZ gates act immediately before the CNOT ring and a Z-basis measurement. RZ
   is diagonal and the ring only permutes basis states, so these phases cannot change any measurement probability.
   Over 20 random draws of inputs and weights, the gradient of every `<Z_q>` with respect to these `n` weights is
   exactly zero by both adjoint and parameter shift (checked for 4 qubits × 2 layers, 8 × 2 and 8 × 3). Every other
   weight has a maximum gradient of at least 0.27 over the same draws, so this is structural, not a coincidence of
   the settings.
3. **One further dead parameter with `<Z_0>` readout**: the last-layer RY on qubit 0 (see point 5).
4. **Redundant parameters.** With the trainable feature map, `RY(a·x + o)` is followed directly by `RY(w)` on the
   same qubit, so the offset `o_q` only shifts every layer's RY weight on that qubit. The gradient with respect to
   `o_q` equals the sum of the RY-weight gradients across layers, and shifting `o_q` by `d` while shifting those
   weights by `−d` leaves the output unchanged. These `n` parameters have non-zero gradient but add no capacity.
   For the selected VQC (8 qubits, 2 layers, trainable map, all-Z readout) the reported 57 parameters comprise
   8 dead, 8 redundant and 41 independent.
5. **Effective readout observables.** Propagating Z through the CNOT ring gives, before the ring:
   `<Z_0>` is the parity `Z_1 Z_2 ⋯ Z_{n−1}`, and for q ≥ 1 `<Z_q>` is the parity `Z_0 Z_1 ⋯ Z_q`. Confirmed on
   random states. So `<Z_0>` is an (n−1)-qubit parity that does not involve qubit 0, and the "all-Z" readout is a
   set of parities of weight 2 to n, not a set of local observables.
6. **One-layer model.** With one layer, `<Z_0>` equals the product of `cos(x_q + w_q)` over q = 1 … n−1 exactly, and
   does not depend on `x_0` at all. A product of seven cosines is small in magnitude for generic inputs, which is
   consistent with the collapse of the 8-qubit one-layer model (validation AUC 0.716, test 0.364 [recorded]).
7. **ZZ pair phase.** Expanding, `2(π − x_a)(π − x_b) = 2π² − 2π(x_a + x_b) + 2 x_a x_b`. At the selected bandwidth
   0.05 (inputs at most 0.157 rad) the phase spans 1.93 rad over the input range. Of that, the linear term spans
   1.97 rad and the bilinear term 0.05 rad. On the stored support vectors the pair phase spans 1.46 rad (CNN
   features) and 1.04 rad (EfficientNet features), with bilinear spans of 0.03 and 0.02 rad. So the entangling gates
   **do** depend on the data, but almost entirely through a term linear in `x_a + x_b`; the genuinely pairwise
   interaction is close to absent. The single-qubit phase `2x` spans only 0.31 rad, so most data dependence at this
   bandwidth sits in the ZZ gates.
8. **Kernel concentration on the stored support vectors.** Off-diagonal fidelity among the 38 support vectors of the
   CNN-feature kernel has mean 0.993 and standard deviation 0.008; among the 131 support vectors of the
   EfficientNet-feature kernel, mean 0.909 and standard deviation 0.070. The full training kernel cannot be
   recomputed because training features are not in the folder.
9. **Hybrid versus twin.** On CNN features the two models make identical decisions on all 622 test patches and have
   the same AUC, but their outputs are not identical: rank correlation 0.981, mean absolute probability difference
   0.013, maximum 0.218, log loss 0.140 versus 0.144. On EfficientNet features decisions differ on 21 patches.

### 6.3 Issues

| ID | Severity | Finding | Where |
|---|---|---|---|
| Q1 | High | Kernel bandwidth is selected at the lower edge of the grid (0.05) in all eight kernel runs, where kernel values are concentrated near 1 | `scripts/v9_quantum.py` grid `(0.05, 0.1, 0.25, 0.5, 1.0)` |
| Q2 | High | Hybrid with fusion gives the classical projection a direct path to the readout; without fusion the hybrid is below its twin in all four pairs [recorded] | `src/quantum/v9.py::_HBase._feat` |
| Q3 | Medium | Dead and redundant parameters are counted as trainable (6.2 points 2–4) | `simulator.vqc_circuit`, `VQCv9.n_params` |
| Q4 | Medium | Readouts are high-weight parities (6.2 point 5) | `simulator.vqc_circuit`, `VQCv9.logits_from_expect` |
| Q5 | Medium | ZZ map convention makes the pairwise term negligible at small bandwidth (6.2 point 7) | `simulator.ZZFeatureMap.states` |
| Q6 | Medium | Kernel-target alignment is **uncentred**: `<K, yyᵀ> / (‖K‖‖yyᵀ‖)`. With a concentrated kernel and unequal classes this mostly measures concentration, not class structure. The low value at bandwidth 0.05 should not be read as poor class alignment | `v9.kernel_target_alignment` |
| Q7 | Medium | Controls are unevenly tuned: RBF-SVM searches 3 gamma values; the quantum kernel searches 5 bandwidths; the VQC has 57 parameters against 9 for its control | `scripts/v9_quantum.py` |
| Q8 | Medium | Noise study: one error rate for one- and two-qubit gates; no readout error, relaxation or dephasing; 16 trajectories; one random stream; no repeats or intervals; kernel gate noise is an affine map that cannot change AUC; noise applied to the test kernel only | `v9.noisy_expect_z`, `v9.noisy_kernel` |
| Q9 | Medium | Encoder min–max scales PCA components on training rows and clips to [0, π]; out-of-range test features saturate. A plausible cause of the VQC validation-to-test gaps, not tested | `models.PCAEncoder` |
| Q10 | Low | V9 quantum training histories are not saved: the `fit` return value is discarded and logging is disabled during the sweep. Optimiser settings exist only as code defaults | `scripts/v9_quantum.py` |
| Q11 | Low | PennyLane backend covers the fixed-map VQC only and has never been executed; no Qiskit path; no transpilation | `pennylane_backend.py` |
| Q12 | Low | Test gaps: no test for dead parameters, kernel alignment, noisy kernel properties, hybrid gradients, or the ZZ map against a reference | `tests/` |

---

## 7. Saved artifacts and results [verified]

| Check | Result |
|---|---|
| `model.npz` SHA-256 equals the hash recorded in `model.json` | 46 of 47 match; the U-Net's `model.json` records no hash |
| Prediction files | 45 in `results/predictions` (25 V9, 20 V8), each with 284 validation and 622 test rows |
| V8 baseline predictions preserved | 20 files in `results/v8_baseline/predictions`, byte-identical to their copies |
| Row order, labels, split and scan IDs agree with `patches_task_a.csv` in all 45 files | Pass |
| Validation AUC, test AUC, accuracy and confusion matrix recomputed from predictions and stored thresholds, against `metrics_v9.csv` | 45 of 45 identical (largest difference 2 × 10⁻¹⁶) |
| Every prediction file has a model artifact | Pass |

Minor: `results/v9/G01_ANALYSIS.md` is named in `v9_g01_analysis.py` but is not in the folder. A source comment in
`src/data/patches.py` says the patch is 16 cells wide; it is 12 (96 px / 8).

---

## 8. Tests

- Recorded by the project on 2026-11-06: 58 passed, 1 skipped (PennyLane), over `tests/` and `backend/tests/`.
- Re-run during this audit, `tests/` only, on a copy without model artifacts or scans: **42 passed, 7 skipped**.
  The skips: 3 need model artifacts (not copied for this run; their hashes were checked separately), 3 need the
  scans folder (absent from the repository), 1 needs PennyLane.
- `backend/tests/` (10 tests) was not run in this audit.
- The tests have meaningful assertions. Coverage gaps for the quantum code are listed under Q12.

---

## 9. Reproducibility limitations

1. No images, patch arrays or feature caches: no model can be retrained and no feature can be re-extracted.
2. No version control and no dependency lock; four imported packages missing from `requirements.txt`.
3. V9 quantum optimiser settings and training curves are not persisted with the models.
4. Single seed; no record of seed sensitivity.
5. Label verification against the upstream source was performed in a browser session and transferred as text with
   checksums. The offline re-check passes; the online check could not be repeated here (no network).
6. The split and the test set are fixed and have been examined (E2), so they cannot serve as a confirmatory test.

---

## 10. Corrections to my earlier "Quantum Audit and Improvement Report"

Re-examination for this audit found statements in that report that were wrong or stronger than the evidence.

| Earlier statement | Status | Correct statement |
|---|---|---|
| F2: the ZZ pair phase "is close to the constant 2π², so the entangling gates barely depend on the data" | **Wrong** | The phase varies by about 1.9 rad over the input range at bandwidth 0.05. The variation is almost entirely linear in `x_a + x_b`; only the bilinear part (0.05 rad) is negligible (6.2 point 7) |
| F2: kernel values "mean 0.993, standard deviation 0.008" | Needs scope | These are statistics over the 38 stored support vectors, not the training set |
| F2: "Validation AUC and alignment disagree" | Unreliable | The alignment formula is uncentred (Q6); the comparison must be redone with centred alignment |
| F2: the gain over RBF-SVM "reflects the control's coarse gamma grid" | Untested | A hypothesis. It needs a properly tuned RBF control on the same features |
| F5: `<Z_0>` "is a parity over many qubits"; "magnitude about 0.01" | Imprecise | It is exactly the parity of qubits 1 … n−1. Its typical magnitude depends on the weights (0.07 in a second random check) |
| B1: "Default to all-Z readout" | Too strong | All-Z readouts are parities of weight 2 to n. Whether they are better must be measured |
| F5: "8 of the 32 circuit weights have exactly zero gradient" | Confirmed, and extended | Also 8 redundant feature-map offsets (6.2 point 4) |

The other findings in that report (saturated benchmark, validation-driven VQC selection, noise-study limits,
statistics, tooling) are confirmed by this audit.

---

## 11. Side effects of auditing

The earlier quantum audit imported `src.quantum.simulator` on the user's computer, which created
`src/__pycache__/` and `src/quantum/__pycache__/` (3 bytecode files, 11 Nov 2026). They are covered by
`.gitignore`, contain no source, and do not affect behaviour. All later commands ran with bytecode writing disabled.
No other file in the repository was created or changed before this report.

---

## 12. Which original findings can be verified from existing artifacts

The revised master prompt lists findings A to G, plus items on the test groups (its section 3.4), timing (its
section 9) and statistics. This table says what the files in this folder already settle and what they cannot.
"Verified" means recomputed or re-derived in this audit; "recorded" means present in the project's result files.

| Finding | Claim to check | Status from existing artifacts | What still needs data or Stage 2 work |
|---|---|---|---|
| **A. CNN feature saturation** | Logistic regression reaches test AUC about 0.998 with 5 to 9 parameters | **Verified.** On 4, 6, 8 and 10 PCA components (5 to 11 parameters) the control scores test AUC 0.9983 each time. Feature pipeline verified by code: the CNN was fitted on training groups only | Feature distributions, correlations, redundancy and separability diagnostics need the embeddings. The CNN checkpoint is present, so they can be regenerated once the patch arrays or scans are supplied |
| **B. Kernel saturation** | Bandwidth 0.05 in all eight runs; support-vector kernel mean 0.993, SD 0.008; alignment 0.04 and 0.46; Gaussian correlation 0.94 | **Verified in part.** Bandwidth: all eight V9 runs, and both saved V8 kernel models also store 0.05. Kernel statistics: recomputed on the 38 stored support vectors. Alignment values: recorded for the 4-qubit CNN-feature run, with an uncentred formula. Gaussian correlation: recomputed as 0.94 on the support vectors (best of six widths); this is a correlation, not equivalence | Complete kernel matrix, spectrum, numerical and effective rank, conditioning, class-wise similarity. Only support-vector subsets are stored (38 to 393 of 1,543 training points, class known from the sign of the dual coefficient), and a support-vector subset is biased towards the margin |
| **C. ZZ feature-map mathematics** | Phase expansion; the first "nearly constant" explanation was wrong | **Verified.** Conventions read from source: `RZ(θ) = diag(e^{−iθ/2}, e^{iθ/2})` and `CNOT·RZ_b(φ)·CNOT = exp(−i·φ/2·Z_a Z_b)`. Phase contributions quantified in 6.2 point 7. The incorrect explanation is not repeated | Fidelity and sensitivity over the real input distribution need features. Alternative maps can be built and tested on synthetic inputs now |
| **D. Dead parameters and readout** | Zero gradients from final RZ, CNOT ring and Z measurement | **Verified as structural** over 20 random draws by adjoint and parameter shift (6.2 points 2 to 4). Heisenberg-picture observables derived and confirmed numerically (6.2 point 5): `<Z_0>` corresponds to `Z_1⋯Z_{n−1}`, not a parity of all qubits | Gradient variance across qubits, depths, initialisations and observables, and alternative readouts, can be done now on synthetic inputs. Their effect on classification needs features |
| **E. Hybrid versus classical twin** | Identical test AUC to about 10 decimals; rank correlation about 0.98; no-fusion hybrid below twin | **Verified from predictions.** AUC 0.9985729112610275 for both; rank correlation 0.981; identical decisions on 622 of 622 test patches; probabilities differ (mean 0.013, maximum 0.218); log loss 0.140 against 0.144. No-fusion: recorded test AUCs are below the twin in all four pairs | All eight required ablations need the input features, and several need retraining. Stored weights allow only inspection of the readout weights |
| **F. VQC validation instability** | Four validation and test AUC pairs | **Verified** against the sweep file: all four pairs match. Predictions were saved only for the selected configuration of each model type, so per-group behaviour of the other configurations cannot be recomputed | Seed sensitivity, scaling variants, clipping and shift analysis need features |
| **G. Noise simulation** | Seven limitations | **Verified by code reading.** One probability for all gates; no readout error, T1 or T2; `trajectories=16`; one generator (`default_rng(7)`) shared by every setting; kernel model `λK + (1−λ)/2ⁿ`; noise applied to the test kernel only. Non-monotone rows are recorded. Two clarifications below | A new noise module can be written and tested now. Its effect on classification needs features |
| **Test groups** | 622 patches, three groups; G07 and G08 at AUC 1.000; G01 with 531 patches; `vol_008` | First three **verified**. `vol_008` **not found** in this repository (section 4.7) | Image similarity and merged-volume checks need the images; true volume boundaries need the LiTS volumes |
| **Timing** | 0.46 ms against 0.0002 ms per patch; about 2,000 times; 53 s against under 0.1 s | **Recorded**, not re-measured: 0.4567 and 0.0002 ms per patch, 53.4 s and 0.0 s. The logistic-regression time is stored to one significant digit, so the ratio is only known to lie between roughly 1,800 and 3,000 | A re-benchmark on synthetic inputs of the same shape is possible now. End-to-end timing needs features |
| **Statistics** | Three planned comparisons | **Verified.** The sweep files record three planned comparisons per feature set; the comparison file reports 30 paired tests, 15 involving quantum or hybrid models, none corrected | A correction family must be defined before new tests (section 15) |
| **Label generation** | Possible row-index modulo labels | **Verified absent** for the V9 labels (section 4.6) | Nothing |

Clarifications on finding G:

- **Shot noise.** The earlier figure of about 0.003 is the Bernoulli standard error `sqrt(p(1−p)/N)` at `p ≈ 0.993`
  with 1,024 shots. It applies only to kernel entries near that value. The general bound at 1,024 shots is 0.0156,
  reached at `p = 0.5`.
- **Affine kernel noise.** In V9 the SVM is trained on the clean kernel and the map `K → λK + c` is applied to the
  test rows only. The test scores then become `λ·s + constant`, so AUC cannot change; that is why the no-shot rows
  are identical. This does not generalise. If the SVM is trained on the noisy kernel, the constant offset drops out
  through the constraint `Σ αᵢyᵢ = 0`, but scaling by λ is equivalent to replacing C by λC, so the fitted model
  changes unless C is re-tuned. This is a derivation, to be confirmed numerically in Stage 2.

---

## 13. What blocks the V10 plan, and what is needed

| Master-prompt item | Status in this folder and session | Needed to unblock |
|---|---|---|
| Stage 2: verify findings A to G | Possible now for C and D in full, for every code-level claim in G, and for A, B, E and F in part (section 12) | Features for the remaining parts |
| Stage 3: dataset integrity | Labels, masks, patches, groups and manifest-level duplicates can be audited now. Pixel-level similarity and merged-volume checks cannot | The scans folder |
| Patient-level splitting (3.2) | Not possible: no patient or volume identifiers exist | Original LiTS segmentation volumes, for `scripts/v9_recover_volume_ids.py` |
| Fold-specific feature extraction (3.3) | Not possible: no images or patch arrays | Scans folder, or `data/processed` from the V9 build machine |
| Classical baselines (4.1) | Logistic regression, linear and RBF SVM, random forest, extra trees, histogram gradient boosting, KNN, decision tree and MLP are available through scikit-learn. XGBoost is not installed and will not be reported | Features to fit them on |
| Deep-learning baselines (4.2) | NumPy implementations and V9 checkpoints exist. No PyTorch and no GPU. Retraining inside folds needs images and hours of CPU: one V9 CNN fit took 1,645 s on 2 cores | Scans folder; compute time |
| Quantum models, hybrids, diagnostics, noise (4.3, 7, G) | Code, unit tests and synthetic-input experiments are possible now | Features for any statement about classification |
| Feature representations (5) | None can be computed. Radiomics, ImageNet and CNN features all start from the image patches | Scans folder, or `data/processed` and `artifacts/cache` |
| Learning curves at 20, 50, 100, 200 patients (5) | Not possible at any of these sizes: the data has ten reconstructed groups | A feasible alternative is proposed in section 15 |
| Nested patient-level evaluation (6) | Blocked by identity and by features | As above |
| PennyLane or Qiskit cross-check (8) | Not possible in either sandbox | A machine with PennyLane installed; the test and the exact command will be provided |
| Quantum hardware (8) | No access | A cloud quantum account; otherwise a hardware-readiness report only |
| Timing benchmark (9) | Possible now on synthetic inputs | Features for end-to-end timing |

Network package installs are a setting on the account. A change takes effect in a new chat, not this one.

---

## 14. Prioritized implementation plan

Order follows the master prompt's stages. Nothing below has been started. Each stage ends with its named
deliverable and a list of what was and was not executed.

| Priority | Stage | Work | Runs now? | Deliverable |
|---|---|---|---|---|
| 1 | 2 | Verify A to G: derivations for C and D; gradient tables over seeds and inputs; kernel analysis on the stored support-vector subsets, labelled as such; hybrid and twin prediction comparison; noise-code review with the affine derivation tested numerically; timing re-benchmark on synthetic shapes | Yes, with the limits in section 12 | `reports/V9_CRITICAL_VERIFICATION.md`, verification scripts and raw outputs under `results/v10/verification/` |
| 2 | 3 | Dataset integrity: manifest consistency, label and mask checks as tests, group structure, prevalence by group, duplicate analysis from the manifest; a clear statement on patient-level evaluation | Yes at manifest level; image-level checks need scans | `reports/V10_DATASET_INTEGRITY_REPORT.md`, `scripts/v10_audit_dataset.py`, `tests/test_v10_data_integrity.py` |
| 3 | 4 | V10 code in a separate tree: corrected circuits with explicit parameter accounting; feature maps (existing ZZ, scaled pairwise, standard ZZ, re-uploading); readouts (single Z, local Z before entangling, two-qubit correlators, parities, probabilities); fidelity and projected kernels with centred alignment and spectrum tools; no-fusion and fusion hybrids with ablation switches; classical baselines with matched search budgets; grouped nested cross-validation; metrics including MCC, log loss and calibration; group-level bootstrap, grouped permutation tests, Holm correction and equivalence tests; noise module with separate one-qubit, two-qubit, readout, amplitude-damping and phase-damping channels (T1 and T2 expressed through them, with idle errors under stated gate durations) and shot sampling | Yes; tested on synthetic data | Code under `src/v10/`, `scripts/v10_*.py`, `tests/test_v10_*.py`, `configs/v10/` |
| 4 | 5 | Mathematical and simulator validation: gate matrices, unitarity, known one- and two-qubit states, CNOT ordering, fidelity, kernel properties, gradients against finite differences, noise channels against closed forms | Yes. Comparison with PennyLane or Qiskit: no | `reports/V10_MATHEMATICAL_AND_QUANTUM_AUDIT.md` |
| 5 | 6 | Pilot: full pipeline on synthetic data; statistics module on the existing V9 prediction files; one real fold once data is available | Partly | Pilot log; not reported as findings |
| 6 | 7 | Full nested benchmark on real features | **No.** Needs the scans, and ideally the LiTS volumes | Predictions, metrics, configurations, logs, figures under `results/v10/` |
| 7 | 8 | Noise and resource experiments: circuit-level results on synthetic inputs; classification impact on real features | Circuit level: yes. Classification impact: no | Raw noise results with seeds, convergence analysis; hardware-readiness report |
| 8 | 9, 10 | Final analysis and reports, stating which of the twelve research questions were answered and which were not | After the above | `V10_EXPERIMENTAL_RESULTS.md`, `V10_RESEARCH_REPORT.md`, `V10_REPRODUCIBILITY.md` |

If the data never arrives, stages 2 to 5 and the synthetic parts of 6 and 8 can still be completed. The reports
would then answer the questions about circuit correctness, trainability and noise behaviour, and would state
plainly that questions about predictive contribution, low-data behaviour and survival under grouped evaluation
remain unanswered.

---

## 15. Decisions that need approval before Stage 2

1. **Where V10 lives.** The layout suggested in the master prompt places new files at paths that V9 already uses,
   for example `src/quantum/simulator.py`. Proposal: put all new code under `src/v10/`, `scripts/v10_*.py`,
   `tests/test_v10_*.py`, `configs/v10/`, `results/v10/` and `artifacts/v10/`, add `run_v10.py`,
   `requirements-v10.txt` and `V10_REPRODUCIBILITY.md`, and leave every existing file untouched.
2. **Unit of evaluation.** Without the LiTS volumes there are no patients, only ten reconstructed scan groups.
   Proposal: evaluate at scan-group level, label every result that way, and make no patient-level claim.
3. **Evaluation protocol.** Proposal: leave-one-group-out over the ten groups as the outer loop, grouped
   cross-validation over the remaining groups as the inner loop. Results on G01, G07 and G08 are flagged as
   previously examined.
4. **A simplification that must be explicit.** Tuning the CNN inside every inner fold is not affordable on two CPU
   cores. Proposal: fix the CNN recipe in advance (the V9 recipe a4), train one CNN per outer fold on that fold's
   training groups, and tune only the downstream models in the inner loop. This is weaker than the fully nested
   design the prompt describes and would be reported as such.
5. **Pre-specified analysis.** Proposal: primary metric is pooled out-of-fold patch-level ROC-AUC with group-level
   resampling, with per-group AUC as the main secondary; five seeds for the downstream models, reported separately
   from fold variability; an equivalence margin of ±0.02 AUC; one Holm family consisting of each quantum model
   against its matched control within each feature representation, with every other comparison labelled
   exploratory.
6. **Learning curves.** Patient counts of 20 to 200 do not exist. Proposal: curves over the number of training
   groups (1 to 9) and over group-stratified slice counts (20, 50, 100, 200, all).
7. **Data.** Whether the scans folder (or `data/processed` and `artifacts/cache`) and the LiTS volumes can be
   provided. Without them the plan stops after the synthetic parts of stage 6.

None of these changes has been made. This report is the only output of Stage 1.
