# V10 Implementation Plan (for review)

Date: 12 Nov 2026 · Status: **proposal; nothing in this plan has been run on real data.** It incorporates the five
clarifications given with the Stage 2–5 approval and the findings of `V9_CRITICAL_VERIFICATION.md`,
`V10_DATASET_INTEGRITY_REPORT.md` and `V10_MATHEMATICAL_AND_QUANTUM_AUDIT.md`.

## 1. What exists now

All new code is separate from V9 (clarification 1). No V9 source file, checkpoint or result was modified.

| Path | Contents | Tests |
|---|---|---|
| `src/v10/quantum/simulator.py` | Batched state-vector simulator; gates RX, RY, RZ, RZZ, H, X, Y, Z, CNOT, CZ; trainable, data, trainable-scale, pair and negated angles; adjoint and parameter-shift gradients; dense unitaries; inverse and composed circuits | `test_v10_simulator.py` |
| `src/v10/quantum/circuits.py` | V9 topology (to load and audit V9 weights); V10 ansatz (final RY layer, scale-only map, ring / linear / brickwork entanglers); Heisenberg propagation of Z strings; commutation proof of dead weights; gate counts | `test_v10_circuits.py` |
| `src/v10/quantum/feature_maps.py` | Diagonal Z feature-map family `φ = c0π² − c1π(u_a + u_b) + c2 u_a u_b` (V9 / Qiskit ZZ, its parts, scaled pairwise, product state), diagonal and gate forms, compute–uncompute circuit | `test_v10_feature_maps.py` |
| `src/v10/quantum/kernels.py` | Fidelity, projected, RBF, polynomial and linear kernels; centring; centred and uncentred alignment; spectrum, effective rank, conditioning; class similarity; kernel agreement; finite-shot estimation; PSD repair with a record | `test_v10_kernels.py` |
| `src/v10/quantum/diagnostics.py` | Parameter tables, Jacobian rank, gradient variance, expressibility (KL to Haar), Meyer–Wallach entanglement, feature-map sensitivity | `test_v10_circuits.py` |
| `src/v10/quantum/noise.py` | Density-matrix simulator with separate one- and two-qubit depolarizing, amplitude and phase damping, T1/T2 with idle periods, readout error; shot sampling; Pauli trajectories with independent seeds; noisy compute–uncompute kernels | `test_v10_noise.py` |
| `src/v10/evaluation/metrics.py` | ROC-AUC, PR-AUC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, MCC, log loss, Brier, ECE, pair counts | `test_v10_evaluation.py` |
| `src/v10/evaluation/statistics.py` | Group bootstrap, exact group sign-flip test, Holm, TOST from intervals, DeLong (patch-level reference only) | `test_v10_evaluation.py` |
| `src/v10/evaluation/grouped_cv.py` | Merging of related groups, leave-one-unit-out outer folds, grouped inner folds, leakage assertions, fold-only preprocessing (standard, robust, min–max, tanh, arctan) | `test_v10_evaluation.py` |
| `src/v10/data/validation.py` | Read-only dataset integrity checks | `test_v10_data_integrity.py` |
| `scripts/v10_verify_v9_findings.py`, `v10_audit_dataset.py`, `v10_math_validation.py` | Stage 2, 3 and 5 | outputs in `results/v10/` |

Test status in this environment: 85 V10 tests pass. On the same copy the V9 suite gives 42 passed and 7 skipped. The
skips are PennyLane not installed, trained models absent and the scans folder absent, the same as before V10 was added.

## 2. Evaluation units (clarification 2)

- **Primary units: eight reconstructed scan-group units**: G01, G02, G03, G04, G05, G06+G07, G08+G09, G10. G06/G07 and
  G08/G09 are merged because a run of identical upstream padding signatures crosses each boundary. That is the only
  available evidence of a shared scan. Merging is the leakage-safe direction.
- **G01 stays one unit** even though it probably joins two series: splitting it could separate two parts of one patient.
- **Sensitivity analysis:** the original ten groups as units.
- **Wording:** every result is labelled "reconstructed scan-group units". None is described as patient-level, and every
  report states that **patient-level generalisation cannot be established without patient identifiers**.
- **Recovering identifiers** needs the original LiTS segmentation volumes (`scripts/v9_recover_volume_ids.py`). If they
  become available the unit definition is replaced before any benchmark is run, never after.

**Consequence of unit sizes.** G06+G07 holds 48.5% of patches and 80% of all tumour/tumour-free pairs; G03 has only 4
tumour-free patches. This drives the choice of primary metric in section 4.

## 3. Features and the CNN (clarification 3)

Each feature set is labelled by its leakage status. Only the first two can support confirmatory claims.

| Feature set | Training inside the outer fold? | Leakage status | Role |
|---|---|---|---|
| Radiomics (44 handcrafted features, V9 `patch_features`, imported read-only) | No training | Leakage-free; scaling and PCA fitted on outer-training rows only | **Confirmatory** |
| Frozen ImageNet EfficientNet-B0 global-average-pooled features (weights in `artifacts/weights`) | No (externally pretrained, frozen) | Leakage-free | **Confirmatory** |
| CNN retrained inside each outer fold (V9 recipe a4, fixed in advance, no recipe search) | Yes, on outer-training units only; early stopping on one inner unit | Leakage-free with respect to the outer test unit | Secondary, if compute allows |
| Existing V9 CNN embeddings | Trained on G03–G06 once; validation reused; test groups examined | **Exploratory only** | Exploratory |
| Existing V9 fine-tuned EfficientNet features | Same split, same reuse | **Exploratory only** | Exploratory |

**Why a fixed recipe is not enough on its own.** Fixing the CNN recipe removes recipe selection from the test data, but a
single CNN trained once still sees the groups that later serve as outer test units in leave-one-unit-out. Hence either
one CNN per outer fold, or a frozen external extractor.

**Compute.** One V9 CNN fit took 1,645 s on two cores, so eight outer folds take about 3.7 hours. That is feasible
overnight on your own machine. If it is not run, the fold-trained CNN features are omitted, not replaced by the V9
embeddings.

**PCA for quantum inputs** is fitted inside each fold. Scalers that tame heavy tails (robust, tanh, arctan) are compared
with V9's min–max scaler in the inner loop.

## 4. Statistical analysis plan (clarification 4)

To be frozen in `configs/v10/analysis_plan.json` (with its SHA-256 recorded) **before** any outer-test predictions exist.

**Primary estimand.** Within-unit ROC-AUC: for each held-out unit, the AUC of the model fitted without that unit,
computed only from positive–negative pairs inside the unit. Summary: the unweighted mean over units with at least 10
patches of each class, a rule based on metadata only (7 of 8 units qualify; G03 does not).

Why not pooled out-of-fold AUC as primary:

- Pooling compares scores produced by eight different fitted models, so cross-fold differences in score scale enter
  the metric.
- Units differ in prevalence (0.43–0.91) and intensity.
- Pair weighting would make the result 80% a G06+G07 metric.

Pooled out-of-fold AUC (after per-fold calibration on inner data) and the pair-weighted within-unit AUC are reported
as secondary metrics, together with every unit's own result.

**Other metrics** (all reported, none primary): PR-AUC, accuracy, balanced accuracy, sensitivity, specificity,
precision, F1, MCC, log loss, Brier, ECE, confusion matrices. Thresholds are chosen by Youden's J on inner data only.
Everything is patch-level within units; nothing is described as patient-level.

**Test.** An exact sign-flip test on per-unit differences (seed-averaged within each unit), two-sided. With 7 qualifying
units there are only 2^7 = 128 sign patterns, so the smallest attainable p is 2/128 = 0.0156, reached only when all 7
units favour the same model.

**Confirmatory family size is limited by that floor.** Holm's first threshold is 0.05/m. With m = 6 it is 0.0083, below
the floor, so a 6-comparison family could never produce a significant result. The family therefore has **3 comparisons,
on one pre-specified feature set (radiomics)**:

1. VQC against logistic regression on the same inputs
2. Quantum-kernel SVM against an RBF-SVM with an equal search budget
3. No-fusion hybrid against its classical twin

Holm thresholds are then 0.0167, 0.025 and 0.05. In practice a confirmatory difference needs all 7 units to agree in
direction. This is stated before the analysis, not after.

The same three comparisons on frozen ImageNet features are reported as a pre-specified **secondary** family (own Holm
correction, labelled secondary). Every other comparison is exploratory and labelled so: CNN-feature results, alternative
feature maps, readouts, fusion hybrids, learning curves, noise.

Alternative, if you prefer both feature sets confirmatory: include G03 (8 units, floor 2/256 = 0.0078). That allows
6 comparisons (first threshold 0.0083), but every one then needs all 8 units to agree. G03's within-unit AUC rests on 4
tumour-free patches (156 pairs), so its sign would be close to a coin flip.

**Seeds.** Five per stochastic configuration. Seed variability (SD across seeds within a unit) and unit variability
(spread across units) are reported separately; seeds are never counted as independent samples.

**Equivalence margin: to be justified, not assumed.** Proposed rule, applied before any outer-test result is seen: the
margin is the larger of 0.02 and twice the seed-to-seed SD of the control model's within-unit AUC, estimated in the pilot
on inner folds only. Rationale: a difference smaller than what reseeding alone produces is not practically meaningful.
Equivalence is declared only if the 90% interval lies inside ±margin. With 7 units that may be unattainable, in which
case the result is reported as inconclusive. A non-significant difference is never reported as equivalence.

## 5. Components still to build (Stage 4 remainder)

| Module | Content |
|---|---|
| `src/v10/features/` | Radiomics, frozen ImageNet and fold-trained CNN extractors (wrapping V9 NumPy code read-only); per-fold caching keyed by fold and input hash |
| `src/v10/classical/baselines.py` | Logistic regression, linear SVM, RBF SVM, polynomial SVM, random forest, extra trees, histogram gradient boosting, KNN, decision tree, regularised MLP; equal search budgets on log grids; capacity-controlled MLP with a parameter count near the VQC's. XGBoost only if installed and actually run |
| `src/v10/quantum/vqc.py` | Training (Adam, class-balanced loss, early stopping on an inner unit), V10 ansatz (ring vs brickwork), readouts (local Z, correlators, parity, basis probabilities), declared / dead / redundant / independent parameter counts saved per model |
| `src/v10/quantum/qksvm.py` | Kernel SVM with joint bandwidth, pair-coefficient and C search on log grids (edge selections flagged); fidelity and projected kernels; per-fold kernel diagnostics saved |
| `src/v10/hybrid/` | No-fusion hybrid (primary), fusion hybrid, classical twin; ablations: zeroed quantum branch, zeroed classical branch, random quantum features, frozen and trainable fusion |
| `src/v10/evaluation/nested.py` | Leave-one-unit-out outer loop, grouped inner loop, saving every configuration's predictions, configuration and log; leakage assertions on every fold |
| `scripts/v10_run_benchmark.py`, `v10_noise_experiments.py`, `v10_statistical_analysis.py`, `v10_resource_benchmark.py` | Runners; the statistics script reads the frozen analysis plan and refuses to run if its hash changed |
| `tests/test_v10_*` additions | Nested-loop isolation (no outer row reaches any inner fit), baselines' search budgets, hybrid ablation switches, PennyLane equivalence (skipped with the stated reason when not installed) |
| Optional | Corrected liver-gate patch set (fixes the edge-slicing defect; Task A unaffected) |

## 6. Experiments, in order, with stop points

| Step | What | Needs real data | Stop for review |
|---|---|---|---|
| 1 | Build section 5 modules; unit tests; full pipeline on synthetic data (runtime, memory, leakage assertions, output integrity) | No | **Yes** |
| 2 | Pilot on inner folds only: runtimes per configuration, seed SD of the control (sets the equivalence margin), freeze the analysis plan | Yes | **Yes** |
| 3 | Full nested benchmark on confirmatory feature sets (radiomics, frozen ImageNet) | Yes | No |
| 4 | Exploratory sets (fold-trained CNN if run; V9 embeddings), learning curves, feature-map and readout comparisons | Yes | No |
| 5 | Noise experiments on the selected models (noisy training, validation and test kernels; separate mechanisms; ≥ 200 trajectories or exact density matrices; convergence) | Yes (for classification impact) | No |
| 6 | Statistical analysis from the frozen plan; resource benchmark; reports | — | **Yes** (draft reports) |

**Learning curves.** Patient counts of 20 to 200 do not exist. Curves will use the number of training units (1 to 6 of
the 7 non-test units in each outer fold) and unit-stratified subsamples of training slices (20, 50, 100, 200, all). They
are exploratory.

**Compute budget (two CPU cores, estimated from V9 timings; to be confirmed in step 2).**

| Item | Estimate |
|---|---|
| Inner tuning: 8 outer × 7 inner folds × about 12 VQC configurations × 1 seed (4–6 qubits) | 3–5 hours |
| Outer refits with 5 seeds for each selected model | under 1 hour |
| Kernels and classical baselines | under 1 hour |
| Fold-trained CNN (optional) | about 3.7 hours |
| Noise (exact density matrices up to 8 qubits) | about 1 hour |

## 7. Data required (clarification 5)

Until the following arrive, work is limited to source verification, manifest inspection, mathematical verification,
synthetic tests and pipeline validation. No real-data classification result will be claimed.

| Needed | For |
|---|---|
| `scans` folder (1,745 `ct_lits_*.png` in their category folders), or V9's `data/processed/*.npz` | Every feature set; steps 2–6 |
| V9's `artifacts/cache` (optional) | Faster frozen ImageNet features |
| Original LiTS segmentation volumes (optional) | Volume identity; possibly more and better-defined units |

## 8. Decisions requested

1. **Primary estimand:** unweighted mean within-unit AUC over units with at least 10 patches per class (section 4),
   instead of pooled out-of-fold AUC.
2. **Confirmatory family:** 3 comparisons on radiomics (7 units), with frozen ImageNet as a secondary family. The
   alternative is 6 comparisons over 8 units including G03 (section 4). Under either option, significance requires every
   unit to favour the same model.
3. **Equivalence margin rule:** larger of 0.02 and twice the control's seed SD, fixed in the pilot before outer testing.
4. **Fold-trained CNN:** run it (about 3.7 hours on two cores, ideally on your machine), or omit it and keep the V9
   embeddings as exploratory only.
5. **Brickwork entangler** as the default V10 VQC design (shallower, nearest-neighbour, larger gradients at shallow depth
   in the 2–8 qubit range), with the ring kept as the V9-comparable variant.

## 9. Reproducibility

```bash
pip install -r requirements-v10.txt          # numpy, scipy, scikit-learn, pandas, pytest (tested versions listed)
python -m pytest -q -p no:cacheprovider tests/test_v10_*.py
python scripts/v10_verify_v9_findings.py      # Stage 2  -> results/v10/verification/
python scripts/v10_audit_dataset.py           # Stage 3  -> results/v10/dataset/
python scripts/v10_math_validation.py         # Stage 5  -> results/v10/math_validation/
```

Set `PYTHONDONTWRITEBYTECODE=1` when running inside the V9 folder so that no bytecode caches are written next to V9
sources. Every script writes a `run_manifest.json` with the environment, package versions, seeds and SHA-256 hashes of the
input files it read.
