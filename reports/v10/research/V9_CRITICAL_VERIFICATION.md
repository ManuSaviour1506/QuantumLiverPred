# V9 Critical Verification (Stage 2)

Repository: `liver_tumour_project_v9` · Run: 11–12 Nov 2026 · Environment: Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, 2 CPU cores, Linux (cloud workspace, on a copy of the repository) · Classical simulation only.

Reproduce with `python scripts/v10_verify_v9_findings.py` (about 2.5 minutes). Raw outputs are in
`results/v10/verification/` (one JSON per finding plus `run_manifest.json` with input-file hashes). No V9 file was modified.

## Verdicts at a glance

| Finding | Verdict | One-line result |
|---|---|---|
| A. CNN feature saturation | **Verified** (score level); feature-level diagnostics not executable | A 5-parameter logistic regression on 4 PCA components reaches test AUC 0.9983; 80% of its logit range comes from one component |
| B. Quantum-kernel saturation | **Partially verified**; one statistic is subset-specific | At the selected bandwidth the kernel is close to rank one (largest eigenvalue 83–96% of the trace), but "mean 0.993, SD 0.008" holds only for one model's 38 support vectors |
| C. ZZ phase mathematics | **Verified and extended** | The entangling gates matter, but at the selected bandwidth almost entirely through the constant and linear parts; removing the bilinear term changes the centred kernel by 3–5% |
| D. Dead VQC parameters and readout | **Verified as structural**; plus redundant parameters | 8 of the selected VQC's 48 circuit and map weights are provably dead and 8 more are redundant; the dead weights never moved during training |
| E. Hybrid versus classical twin | **Verified, with a correction** | The equal test AUC is exact (identical pair counts), but the predictions differ: 9 pair swaps each way, different log loss |
| F. VQC validation instability | **Verified** | Across 10 CNN-feature VQC configurations, validation and test AUC have rank correlation 0.20; validation is saturated |
| G. Noise-model limitations | **Verified** (all code-level claims) | 16 trajectories give seed-to-seed logit noise of 0.27 against an input spread of 2.34; the kernel's noise-free AUC is invariant by construction |
| Timing | **Verified** (order of magnitude) | VQC head 0.35 ms per patch against 0.00016 ms for logistic regression (about 2,200 times) in this environment |
| Statistics | **Verified**; group-level inference impossible here | With 3 test groups, only one of which differs, an exact group-level test cannot go below p = 1; patch-level tests report p < 0.001 for the same comparisons |

"Not executable" always means the same thing: the repository holds no feature vectors, because the CT images
(`data/raw/scans`), the processed patch arrays (`data/processed`) and the feature caches (`artifacts/cache`) are absent.

### How real data was used without features

Kernel models store their support vectors, which are real training points in the encoded [0, π] space. Each quantum
kernel and its RBF control share an identical encoder (checked byte-for-byte), and so do the V8 VQCs and V8 kernels.
Pooling the two support-vector sets gives four sets of real, labelled training points. Labels come from the sign of
the SVM dual coefficient; this convention was confirmed on a fitted scikit-learn SVC.

| Set | Features | Qubits | Points | Tumour / tumour-free |
|---|---|---|---|---|
| `v9_cnn_4q` | V9 CNN embedding, PCA-4 | 4 | 40 | 22 / 18 |
| `v9_eff_4q` | Fine-tuned EfficientNet, PCA-4 | 4 | 389 | 209 / 180 |
| `v8_imagenet_8q` | Frozen ImageNet EfficientNet, PCA-8 | 8 | 437 | 230 / 207 |
| `v8_finetuned_8q` | Fine-tuned EfficientNet, PCA-8 | 8 | 136 | 70 / 66 |

These points lie near decision boundaries, so every statistic computed on them is **margin-biased**. They are used to
describe kernel geometry, never to estimate classification performance.

---

## A. CNN feature saturation — Verified (score level)

**Claim.** Logistic regression reaches test AUC about 0.998 with 5–9 parameters on the V9 CNN embeddings, so little
information is left for a quantum model.

**Sources.** `scripts/v9_quantum.py` lines 117–145 (VQC sweep and logistic-regression controls);
`results/v9/quantum/quantum_experiments_cnn.json`; `artifacts/classical/lr_control_v9_cnn`.

**Evidence.**

| PCA components | Parameters | Val AUC | Test AUC |
|---|---|---|---|
| 4 | 5 | 0.9994 | 0.9983 |
| 6 | 7 | 0.9990 | 0.9983 |
| 8 | 9 | 0.9990 | 0.9983 |
| 10 | 11 | 0.9990 | 0.9983 |

- The signal is nearly one-dimensional: 80% of the control's logit range comes from the first principal component
  (77% on EfficientNet features). On the real support-vector points the first component separates the classes with
  Cohen's d = −2.24 even though these are the hardest points.
- The quantum kernel ranks the test patches almost exactly as the CNN does (Spearman 0.999). Hybrid and twin: 0.967
  and 0.969. VQC: 0.562. RBF control: 0.591.
- Of 92,496 tumour/tumour-free test pairs, the CNN misorders 121. No model built on its embedding repairs more of those
  than it breaks:

| Model on CNN embedding | Pairs the CNN misorders that the model gets right | Pairs the CNN gets right that the model misorders |
|---|---|---|
| Logistic regression | 26 | 59 |
| Quantum kernel SVM | 18 | 106 |
| Hybrid (fusion) | 23 | 34 |
| Classical twin | 21 | 32 |
| VQC | 14 | 2,377 |

**Provenance of the embedding.** The CNN (`cnn25d_v9_a4_res_se_focal`) was trained on groups G03–G06 only. Validation
groups were used for its early stopping (best epoch 22 of 28) and for choosing it among four recipes, and again for
choosing the quantum models. The test groups were examined in V9's G01 analysis. Under your rule 3, these embeddings
are **exploratory** features for V10: not trained on test groups, but not leakage-free in the nested sense.

**Not executable.** Feature distributions, correlations, redundancy and separability diagnostics on the full embeddings.

**Required correction (V10).** Evaluate quantum models on feature sets with headroom. Re-derive any CNN embedding inside
each outer fold, or use a frozen external extractor. Report pair discordance against the strongest classical model.

**Verification test.** `results/v10/verification/A_feature_saturation.json`.

---

## B. Quantum-kernel saturation — Partially verified

**Claim.** Bandwidth 0.05 was selected in all eight runs; support-vector kernel mean about 0.993 and SD about 0.008;
alignment about 0.04 at 0.05 and 0.46 at 0.25; correlation with a Gaussian kernel about 0.94.

**Sources.** Bandwidth grid `scripts/v9_quantum.py` line 153; RBF gamma grid line 169; alignment formula
`src/quantum/v9.py` lines 331–334; feature map `src/quantum/simulator.py` lines 201–225.

**Evidence.**

1. **Selection at the grid edge: confirmed.** All eight V9 kernel runs and both saved V8 kernels use 0.05, the smallest
   value in the grid `(0.05, 0.1, 0.25, 0.5, 1.0)`.
2. **The 0.993 / 0.008 figure is subset-specific.** It reproduces exactly on the 38 support vectors of
   `qksvm_v9_cnn` (mean 0.9926, SD 0.0077). Adding the two other real training points that the RBF control kept
   gives mean 0.946, SD 0.141, minimum 0.416. Those two points sit at the encoded extremes: one has its third component
   at π, where all 38 kernel support vectors lie below 0.1; the other has its first two components at 0.
3. **Near rank one at the selected bandwidth: confirmed on every set.**

| Set (points) | Off-diagonal mean | Largest-eigenvalue share | Effective rank | Within − between class similarity |
|---|---|---|---|---|
| `v9_cnn_4q` (40) | 0.946 | 0.958 | 1.25 | −0.0009 |
| `v9_eff_4q` (389) | 0.904 | 0.906 | 1.53 | +0.0085 |
| `v8_imagenet_8q` (437) | 0.852 | 0.855 | 2.05 | +0.0008 |
| `v8_finetuned_8q` (136) | 0.822 | 0.826 | 2.28 | +0.0007 |

   Values at bandwidth 0.05. At 0.0025 the share is at least 0.999 on every set.

4. **Alignment: confirmed in direction, but V9's formula is uncentred.** V9's `kernel_target_alignment` equals the
   uncentred alignment (checked numerically). With a kernel whose entries are all near 1, that value is dominated by the
   constant part. With the centred formula `<HKH, HyyᵀH> / (‖HKH‖ ‖HyyᵀH‖)` on the `v9_cnn_4q` points, alignment rises
   from 0.124 at bandwidth 0.05 to 0.51–0.56 between 0.35 and 1.5. The recorded uncentred values on the full training
   set (0.041 at 0.05, 0.458 at 0.25) show the same direction.
5. **A classical kernel aligns better on the same points.** The best RBF exceeds the best quantum bandwidth on every
   set: 0.732 against 0.557 (`v9_cnn_4q`), 0.252 against 0.134, 0.058 against 0.051, 0.119 against 0.098.
6. **Gaussian correlation: reproduced on the 38 points, not general.** The best Pearson correlation between
   off-diagonal entries of the bandwidth-0.05 quantum kernel and an RBF kernel is 0.938 on the 38 SVs. It is 0.983,
   0.847, 0.808 and 0.858 on the four pooled sets. Centred alignment between the quantum and the best-matching RBF
   kernel is 0.82–0.93: similar geometry, not equivalence.

**Bandwidth grid used here.** 0.0025 to 3.0, log-spaced. Upper end: inputs are min–max encoded to [0, π] and
`exp(−i u Z)` has period π in u, so bandwidth 1 spans one full period and larger values wrap distinct inputs onto equal
phases. Lower end: where off-diagonal entries exceed 0.999.

**Not executable.** The complete training kernel (1,543 points) and a re-selection of bandwidth on validation.

**Required correction (V10).** Log-spaced bandwidth grid that extends past both current edges, with the selected value
flagged when it lands on an edge; centred alignment; full spectrum, effective rank and conditioning on the training
kernel inside each fold; an RBF control tuned over a log-spaced gamma grid of equal size.

**Verification tests.** `tests/test_v10_kernels.py` (validity, centring removes a constant offset, rank-one and identity
spectra, concentration grows as bandwidth shrinks); `results/v10/verification/B_kernel_geometry.json`.

---

## C. ZZ feature-map mathematics — Verified and extended

**Claim.** `φ = 2(π − x_a)(π − x_b) = 2π² − 2π(x_a + x_b) + 2 x_a x_b`. The earlier "nearly constant" explanation was
wrong.

**Sources.** `src/quantum/simulator.py` line 223 (`RZ(2(π − x_a)(π − x_b))` inside a CNOT pair) and lines 201–225.

**Implemented unitary (derived from the code).**

```latex
U(x) = \left[ D(x)\, H^{\otimes n} \right]^{2},\qquad
D(x) = \exp\!\Big(-i \Big[\sum_q u_q Z_q + \sum_{(a,b)} (\pi - u_a)(\pi - u_b)\, Z_a Z_b\Big]\Big),\qquad u = \text{bandwidth}\cdot x
```

Pairs are the linear chain (q, q+1). Conventions confirmed numerically: `RZ(θ) = diag(e^{−iθ/2}, e^{iθ/2})` and
`CNOT · (I ⊗ RZ(2φ)) · CNOT = exp(−iφ Z⊗Z)`. This is the gate sequence of Qiskit's `ZZFeatureMap` with its default
`φ(x, y) = (π − x)(π − y)` ([Qiskit documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/dev/qiskit.circuit.library.zz_feature_map)),
except that Qiskit's default entanglement is `full` and V9 uses `linear`. The V10 implementation reproduces V9's states
exactly.

**Mathematical facts that the earlier explanations missed.**

- **Periodicity.** `exp(−iφ Z_a Z_b)` changes only by a global phase when φ → φ + π, and so does `exp(−iu Z)` when
  u → u + π. The constant π² therefore acts as a fixed ZZ rotation with φ = π² mod π = 0.4448 rad: a fixed entangling
  gate, not an identity.
- **All three parts are two-qubit terms.** Each multiplies `Z_a Z_b`. The "linear" part is an entangling rotation whose
  angle depends additively on the data. Only the bilinear part couples the two features multiplicatively.

**Evidence on real points, at the selected bandwidth 0.05.**

| Set | Pair-phase span (rad) | of which linear | of which bilinear | Centred kernel change: remove bilinear | remove constant | remove all pair terms |
|---|---|---|---|---|---|---|
| `v9_cnn_4q` | 0.96 | 0.99 | 0.025 | 5% | 29% | 95% |
| `v9_eff_4q` | 0.62 | 0.63 | 0.011 | 3% | 38% | 95% |
| `v8_imagenet_8q` | 0.79 | 0.81 | 0.021 | 3% | 52% | 94% |
| `v8_finetuned_8q` | 0.71 | 0.72 | 0.017 | 3% | 45% | 95% |

"Centred kernel change" is `‖HK_full H − HK_variant H‖_F / ‖HK_full H‖_F`. The centred alignment between the full kernel
and the kernel without its bilinear term is at least 0.9996 on every set.

- **Sensitivity.** The diagonal Fubini–Study metric (state-change per unit input change) is 0.027–0.062 per feature
  with pair terms and 0.0025 without them: the ZZ terms supply 10–24 times more data sensitivity.
- **Entanglement.** Mean Meyer–Wallach entanglement is 0.57–0.62 at bandwidth 0.01, produced by the constant term even
  when the data barely vary. It falls to 0.08–0.28 at 0.05.

**Alternatives (descriptive, margin-biased points).** At bandwidth 0.05 the product-state map with no pair terms aligns
better than the V9 map on all four sets (for example `v9_cnn_4q` 0.248 against 0.124). The scaled pairwise map `2α u_a u_b`
behaves like the product-state map unless α is large, because u ≤ 0.157 rad at this bandwidth. That is the risk the
master prompt anticipated, so α must be tuned jointly with bandwidth. The highest quantum alignment observed on
`v9_cnn_4q` is the scaled pairwise map with α = 16 at bandwidth 0.25 (0.701). That is still below the best RBF (0.732).

**Required correction (V10).** Treat the map as a family `φ = c0 π² − c1 π (u_a + u_b) + c2 u_a u_b` with bandwidth and
coefficients tuned inside folds; report the contribution of each part; include the product-state map as a no-entanglement
control.

**Verification tests.** `tests/test_v10_feature_maps.py`: diagonal form equals gate form for every preset and
entanglement pattern; equals V9's map for n = 4, 6, 8; the CNOT–RZ–CNOT identity; period π; phase decomposition sums to
the total. `results/v10/verification/C_zz_phase.json`.

---

## D. Dead VQC parameters and readout — Verified as structural

**Claim.** Final RZ rotations before the CNOT ring and Z measurement have exactly zero gradient.

**Sources.** `src/quantum/simulator.py` lines 187–198 (`vqc_circuit`: RZ at line 195, ring at line 197);
`src/quantum/v9.py` lines 62–69 (trainable map `A·a + o`, readouts), line 119 (`n_params`).

**Heisenberg-picture readouts.** The V9 ring applies CNOT(0,1), CNOT(1,2), …, CNOT(n−1,0). Conjugating Z strings through
it (CNOT: Z_t → Z_c Z_t) gives, before the ring:

| Measured | Equivalent observable before the ring |
|---|---|
| ⟨Z_0⟩ | Z_1 Z_2 ⋯ Z_{n−1} (does not involve qubit 0) |
| ⟨Z_q⟩, q ≥ 1 | Z_0 Z_1 ⋯ Z_q |

Confirmed on random states for n = 3–6. So "all-Z readout" is a set of parities of weight 2 to n, not local observables.

**Why parameters die.** A Z string commutes with RZ, so every last-layer RZ cancels. With ⟨Z_0⟩ readout, qubit 0 is
outside the propagated string, so the last-layer RY on qubit 0 also cancels.

**Evidence (proof by commutation and numerical gradients agree in every case).**

| Configuration | Circuit + map weights | Proved dead | Redundant directions | Independent (Jacobian rank) |
|---|---|---|---|---|
| 8 qubits, 2 layers, trainable map, all-Z (selected `vqc_v9_cnn`) | 48 | 8 (last-layer RZ) | 8 (map offsets) | 32 |
| 8 qubits, 3 layers, ⟨Z_0⟩ (V8 `vqc_8q`) | 48 | 9 (RZ + last RY on qubit 0) | 0 | 39 |
| 4 qubits, 3 layers, trainable map, ⟨Z_0⟩ (`vqc_v9_eff`) | 32 | 5 | 4 (map offsets) | 23 |
| 8 qubits, 2 layers, all-Z | 32 | 8 | 0 | 24 |

- **Redundancy.** With the trainable map, `RY(a·x + o)` is followed directly by `RY(w)` on the same qubit, so each offset
  `o_q` only shifts that qubit's RY weights. Shifting `o_q` by d and those weights by −d leaves every output unchanged.
- **Trained models.** In all four saved VQCs the dead weights moved by at most 5.8 × 10⁻¹⁰ during training, against a
  median of 0.26–0.48 rad for live weights. They are effectively at their initial values. The change is not bitwise
  zero only because adjoint gradients of dead weights are about 10⁻¹⁷ rounding noise and Adam divides by `√v + 10⁻⁸`.
- **At trained weights on real inputs** (V8 `vqc_8q`, 437 real training points): ⟨Z_0⟩ has median magnitude 0.11 and range
  −0.43 to 0.33. A readout scale of 10.2 compensates.

**A bug found and fixed in my own V10 code.** The first version of the V10 commutation prover reported the qubit-0 map
scale and offset as dead for the 4-qubit ⟨Z_0⟩ model. Those weights are shared across layers and are active in earlier
layers. The rule now requires every gate that uses a weight to be dead, and a regression test covers it.

**Required correction (V10).** The V10 ansatz ends with a trainable RY layer after the last entangler and uses a scale-only
map. Proved dead weights: 0. Jacobian rank: equal to the declared weight count in every tested configuration.
Parameter counts are reported as declared, dead, redundant and independent.

**Verification tests.** `tests/test_v10_circuits.py` (V9 topology reproduction, Heisenberg readouts, dead weights proved and
numerical, one-layer closed form, offset redundancy, shared-weight regression, V10 ansatz with no dead or redundant
weights). `results/v10/verification/D_parameters_and_readout.json`.

---

## E. Hybrid versus classical twin — Verified, with a correction

**Claim.** The fusion hybrid and its twin have identical test AUC to about 10 decimals, rank correlation about 0.98, and
the no-fusion hybrid is below the twin.

**Sources.** Fusion path `src/quantum/v9.py` lines 151–152 (`_feat` concatenates the classical projection);
`results/predictions/hybrid_v9_*.csv`, `twin_v9_*.csv`.

**Evidence, CNN features, test set (622 patches).**

| | Hybrid | Twin |
|---|---|---|
| Correctly ordered / wrongly ordered pairs (of 92,496) | 92,364 / 132 | 92,364 / 132 |
| ROC-AUC | 0.9985729113 | 0.9985729113 |
| Log loss | 0.1398 | 0.1441 |
| Brier | 0.0393 | 0.0438 |
| ECE | 0.045 | 0.048 |
| Sensitivity / specificity / MCC | 0.992 / 0.947 / 0.946 | 0.992 / 0.947 / 0.946 |

- The AUCs are **exactly** equal: identical integer pair counts, not a 10-decimal coincidence.
- The predictions are **not** identical. The hybrid orders 9 pairs correctly that the twin misorders, and the twin 9 the
  other way. Spearman 0.981, Pearson 0.986, Kendall 0.907. Mean absolute probability difference 0.013 (maximum 0.218).
  Thresholded decisions agree on all 622 patches.
- On validation the AUCs differ by one pair (0.99911 against 0.99916).
- **EfficientNet features:** hybrid 0.9135 against twin 0.9097 (1,374 against 1,023 pair wins, log loss 0.471 against
  0.499, decisions differ on 21 patches). Per-group differences are small and all favour the hybrid (see Statistics).

**Recorded sweep.**

| Features | Qubits | Fusion | Hybrid − twin (test AUC) |
|---|---|---|---|
| CNN | 4 | no | −0.024 |
| CNN | 8 | no | −0.016 |
| CNN | 4 | yes | −0.0005 |
| CNN | 8 | yes | 0.0000 |
| EfficientNet | 4 | no | −0.0045 |
| EfficientNet | 8 | no | −0.016 |
| EfficientNet | 4 | yes | −0.0038 |
| EfficientNet | 8 | yes | +0.0038 |

Without fusion the hybrid is below its twin in all four pairs. The V8 hybrids are also below their twins (−0.016 and
−0.064).

**Not executable.** All eight ablations in the master prompt; they need the input features and retraining.

**Required correction (V10).** No-fusion hybrid as the primary test of the quantum branch; fusion hybrid only with branch
ablations (zeroed branches, random quantum features, frozen and trainable fusion); compare raw predictions, pair
discordance, log loss and calibration, not only AUC.

**Verification test.** `results/v10/verification/E_hybrid_vs_twin.json`.

---

## F. VQC validation instability — Verified

**Claim.** Four quoted validation/test pairs show validation AUC does not predict test AUC.

**Sources.** `scripts/v9_quantum.py` lines 117–147 (sweep), 192–193 (selection by validation AUC); encoder
`src/quantum/models.py` line 38 (min–max to [0, π], clipped).

**Evidence.**

- All four quoted pairs match the recorded values to four decimals.
- Across the 10 CNN-feature VQC configurations the rank correlation between validation and test AUC is 0.20. The
  selected configuration scores 0.973 on test; the best scores 0.997 and the median 0.974. On EfficientNet features the
  correlation is 0.76.
- **Validation is saturated.** The selected VQC scores 1.000, 1.000 and 0.996 on the three validation groups, exactly as
  the logistic-regression control does. On test it scores 0.966 on G01 (control 0.998) and 1.000 on G07 and G08.
- **The one-layer collapse is structural.** With one layer, ⟨Z_0⟩ equals `∏_{q≥1} cos(x_q + w_q)` exactly and does not
  depend on x_0 at all. A product of seven cosines is small and unstable in sign, which matches the recorded
  validation 0.716 and test 0.364.
- **Min–max encoding compresses the data.** The training range of the CNN-embedding PCA components reaches 51 units on
  the standardised scale, so a few extreme points set the [0, π] range. On the real support-vector points the central 90% of values
  occupy only 2.5–22% of [0, π] per component (37–69% on EfficientNet features; margin-biased estimate).

**Not executable.** Seed sensitivity, alternative scalers and the fraction of test inputs clipped. Predictions were saved
only for the selected configuration of each model type, so per-group behaviour of the others is unknown.

**Required correction (V10).** At least five seeds per configuration; scalers fitted on training rows only (standard,
robust, min–max, tanh, arctan; implemented in `src/v10/evaluation/grouped_cv.py`); grouped inner cross-validation instead
of one saturated validation split; save predictions for every configuration.

**Verification tests.** `tests/test_v10_circuits.py::test_v9_one_layer_z0_closed_form`;
`tests/test_v10_evaluation.py::test_fold_preprocessor_uses_training_rows_only`; `results/v10/verification/F_vqc_selection.json`.

---

## G. Noise-model limitations — Verified

**Claim.** One depolarizing rate for all gates; no readout error, T1 or T2; 16 trajectories; one random stream;
non-monotone results; an affine kernel model; noisy test kernels against clean training.

**Code evidence.**

| Claim | Location | Confirmed |
|---|---|---|
| One rate p after every gate on each touched qubit (a CNOT gets two independent one-qubit errors) | `src/quantum/v9.py` line 305 | Yes |
| No readout error, amplitude or phase damping | `src/quantum/v9.py` lines 274, 319 | Yes |
| 16 trajectories | `scripts/v9_quantum.py` lines 220, 229 | Yes |
| One generator shared by every setting | `scripts/v9_quantum.py` line 214 | Yes |
| Kernel noise `λK + (1 − λ)/2ⁿ` with `λ = (1 − p)^(2·gates)` | `src/quantum/v9.py` line 324 | Yes |
| Noise only on the test kernel | `scripts/v9_quantum.py` lines 234, 239 | Yes |

**Monte-Carlo variability of V9's own routine.** At the trained `vqc_v9_cnn` weights, 100 synthetic inputs, p = 0.005:

| Trajectories | Seed-to-seed SD of logits | Clean logit spread across inputs | Rank correlation between seeds |
|---|---|---|---|
| 16 | 0.268 | 2.34 | 0.968–0.979 |
| 256 | 0.060 | 2.34 | 0.998 |

How much this moves test AUC cannot be computed without the test features.

**Non-monotone results.** Adding 1,024 shots *raised* AUC in five CNN-feature settings and three EfficientNet settings
(for example the CNN VQC at p = 0.005: 0.9559 → 0.9646). One case also rises with p (EfficientNet hybrid at 1,024 shots,
p 0.005 → 0.01). Single-stream Monte-Carlo error with 16 trajectories, plus independent shot noise that can move AUC in
either direction, explain this; AUC is not monotone in noise.

**The affine kernel model, derived and checked on real points.**

- *Clean-trained SVM, noisy test kernel (what V9 did).* Scores become `λ(s − b) + b`, an increasing transform of the clean
  scores, so AUC cannot change. That is why the four noise-free kernel rows in V9 are identical. Thresholded metrics
  would change, but V9 did not report them.
- *SVM trained on the noisy kernel.* The constant offset `c·11ᵀ` drops out exactly through the constraint `Σ αᵢ yᵢ = 0`.
  Scaling by λ is equivalent to training on the clean kernel with C replaced by λC. Confirmed on the `v9_eff_4q` points
  for p = 0.001, 0.005 and 0.01. So the result changes only through the effective C, and a fixed C grid can select
  differently.

**Shot noise.** The Bernoulli standard error `√(p(1 − p)/N)` is at most 0.0156 at 1,024 shots (p = 0.5). The figure of
about 0.003 used in my earlier report applies only near p ≈ 0.99 (0.0026 at p = 0.993).

**Required correction (V10).** Implemented in `src/v10/quantum/noise.py`: separate one-qubit and two-qubit depolarizing,
amplitude and phase damping, T1/T2 relaxation with idle periods, readout assignment error, multinomial shots, independent
seeds per trajectory, and noisy training, validation and test kernels. All values used will be labelled hypothetical.

**Verification tests.** `tests/test_v10_noise.py` (zero-noise equivalence, trace preservation and positivity, depolarizing
closed forms and separation of one- and two-qubit noise, damping, T1/T2 and idle relaxation, readout formula, trajectory
convergence and seed reproducibility, shot standard error); `tests/test_v10_kernels.py::test_svm_affine_kernel_maps`;
`results/v10/verification/G_noise.json`.

---

## Timing — Verified (order of magnitude, this environment only)

**Claim.** VQC inference about 0.46 ms per patch against 0.0002 ms for logistic regression (about 2,000 times);
training 53 s against under 0.1 s.

| Quantity | Recorded by V9 | Re-measured here |
|---|---|---|
| VQC head (encoding + simulation + readout), ms per patch | 0.4567 | 0.35 |
| state-vector evolution (timed separately) | — | 0.36 |
| readout / encoding (timed separately) | — | 0.0007 / 0.0002 |
| Logistic-regression head, ms per patch | 0.0002 | 0.00016 |
| Ratio | about 2,300 | about 2,200 |
| Peak memory, VQC head / LR head | — | 10.3 MB / 0.7 MB |
| VQC training | 53.4 s | 2.76 s per epoch on 1,543 synthetic samples, so 53.4 s is about 19 epochs |
| Feature extraction (recorded) | 10.1 ms per patch | not re-measurable |

Best of seven runs after a warm-up, trained V9 weights, synthetic inputs of the V9 shapes. Components were timed
separately, so within timing noise they can add up to slightly more than the total. The ratio compares two NumPy
implementations on two CPU cores. It is not a quantum-versus-classical speed comparison, and feature extraction costs
30 times more than either head.

**Verification test.** `results/v10/verification/timing.json`.

---

## Statistics — Verified; group-level inference is impossible with this test set

**Facts.** `comparison_v9.json` reports 30 paired tests, 15 involving quantum or hybrid models. The sweep files record
"3 planned comparisons" per feature set. No multiplicity correction was applied. V9's p-value
(`src/evaluation/metrics.py` line 120) is `2·min(P(d ≤ 0), P(d ≥ 0))` over a two-stage bootstrap with three test scans.

**Re-analysis of the six quantum-versus-control comparisons** (`src/v10/evaluation/statistics.py`):

| Comparison | ΔAUC | Groups that differ | Exact group sign-flip p | Smallest attainable p | Patch-level DeLong p | V9 reported p |
|---|---|---|---|---|---|---|
| VQC vs LR (CNN) | −0.025 | 1 (G01) | 1.00 | 1.00 | < 0.001 | 0.53 |
| Kernel vs RBF (CNN) | +0.016 | 1 (G01) | 1.00 | 1.00 | 0.0003 | 0.51 |
| Hybrid vs twin (CNN) | 0.000 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| VQC vs LR (EfficientNet) | −0.003 | 1 (G01) | 1.00 | 1.00 | 0.052 | 0.42 |
| Kernel vs RBF (EfficientNet) | −0.015 | 3 | 0.75 | 0.25 | < 0.001 | 0.41 |
| Hybrid vs twin (EfficientNet) | +0.004 | 3 (all favour hybrid) | 0.25 | 0.25 | 0.22 | 0.24 |

- With three test groups an exact group-level test cannot go below p = 0.25. When G07 and G08 score 1.000 for both
  models, only G01 differs and the floor is p = 1.
- The patch-level DeLong test, which treats 622 dependent patches as independent, reports p < 0.001 for three of the same
  comparisons. That is pseudo-replication; it is included only to show the size of the error.
- A group bootstrap over three groups has at most 27 distinct resamples.

**Required correction (V10).** Group-level resampling and exact sign-flip tests; a pre-declared Holm family; equivalence
only through a pre-justified margin; and a prominent statement that inference needs more independent groups than V9's
test split provides. Leave-one-group-out over all ten groups raises the unit count from 3 to 10 (or 8 after merging
possibly related groups; see the Stage 3 report).

**Verification tests.** `tests/test_v10_evaluation.py` (metrics against scikit-learn, sign-flip floor with three groups,
bootstrap, TOST, Holm, DeLong); `results/v10/verification/statistics.json`.

---

## Corrections to earlier statements (mine and the original audit's)

| Earlier statement | Correction |
|---|---|
| Kernel mean 0.993, SD 0.008 (support vectors) | True only for `qksvm_v9_cnn`'s 38 support vectors; 0.946 and 0.141 with two more real training points |
| Gaussian-kernel correlation about 0.94 | Reproduces on those 38 points; 0.81–0.98 on the pooled sets; a correlation, not equivalence |
| Hybrid and twin AUC equal "to 10 decimals" | Exactly equal (identical pair counts), yet the predictions differ |
| Dead weights have "exactly zero gradient" | Structurally zero; in training they moved by at most 6 × 10⁻¹⁰ through rounding noise |
| ⟨Z_0⟩ "is a parity over many qubits" | It is exactly Z_1 ⋯ Z_{n−1}; it excludes qubit 0 |
| Non-monotone noise results (unspecified) | Mostly shot noise raising AUC at fixed p; one rise with p |
| Shot standard error about 0.003 | Only near p ≈ 0.99; the maximum at 1,024 shots is 0.0156 |
| (My V10 code) commutation prover | Over-reported shared map weights as dead; fixed, with a regression test |

## What Stage 2 could not do

Every item marked "not executable" above needs feature vectors. With the CT images, or the V9 `data/processed` and
`artifacts/cache` folders, the same script can be extended to the full training kernel, the hybrid ablations, seed
sensitivity, scaler variants and the feature-level saturation diagnostics.
