# V10 Mathematical and Quantum Audit (Stage 5, synthetic data)

Run: 12 Nov 2026 · Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, 2 CPU cores · **Classical simulation
only; no quantum hardware; no real data in this stage.** Every input below is drawn from a seeded generator, and every
noise value is a hypothetical simulation setting.

Reproduce with `python scripts/v10_math_validation.py` (about 3.5 minutes) and
`python -m pytest -p no:cacheprovider tests/test_v10_*.py` (85 tests). Raw outputs: `results/v10/math_validation/`.
The Stage 2 report (`V9_CRITICAL_VERIFICATION.md`) covers the same mathematics on V9's real stored points.

## Summary

1. **The V10 simulator is correct to machine precision** against independent references (dense matrices, finite
   differences, the parameter-shift rule, V9's simulator, compute–uncompute circuits).
2. **The V10 ansatz has no dead or redundant parameters.** All 104 tested configurations (2–8 qubits, 1–4 layers, ring,
   linear and brickwork entanglers, with and without a trainable scale) have zero provably dead weights and full Jacobian
   rank.
3. **Readout design changes trainability less than the entangler does.** With V9's CNOT ring, every observable's
   gradient variance falls by roughly half per added qubit, local readouts included. The ring is a sequential chain, so
   its two-qubit depth equals the qubit count. A brickwork entangler (two-qubit depth 2 per layer) keeps 2–2.6 times
   larger gradients at shallow depth. These are statements about 2–8 qubits only.
4. **The noise module reproduces every closed form exactly**, and its Monte-Carlo trajectories converge to the exact
   density matrix at the expected rate (error ∝ N^−0.49).
5. **No external cross-check was possible.** PennyLane and Qiskit are not installed and cannot be installed in this
   session. No claim of external equivalence is made.

## 1. Simulator correctness

| Check | Maximum error |
|---|---|
| Unitarity of RX, RY, RZ, RZZ, H, X, Y, Z, CNOT, CZ (random angles) | 2.2 × 10⁻¹⁶ |
| State normalisation (4 qubits, 3 layers, 16 inputs) | 8.9 × 10⁻¹⁶ |
| U†U = I for a full circuit; inverse circuit equals U† | 6.7 × 10⁻¹⁶; 4.4 × 10⁻¹⁶ |
| Adjoint gradient against the parameter-shift rule | 3.3 × 10⁻¹⁵ |
| Adjoint gradient against central finite differences (step 10⁻⁶) | 1.8 × 10⁻⁹ (finite-difference truncation) |
| V10 against V9 simulator, V9 VQC (6 qubits, 3 layers) | 0 |
| V10 against V9 ZZ feature map (fidelity of states) | 1 − 10⁻¹⁶ |
| Diagonal feature-map form against explicit gate form | 1 − 10⁻¹⁶ |
| Compute–uncompute P(0…0) against the fidelity kernel | 2.1 × 10⁻¹⁵ |
| Bell state from H, CNOT | 0 |

Conventions (identical to V9): qubit 0 is the most significant bit; `RX/RY/RZ(t) = exp(−i t P / 2)`;
`RZZ(t) = exp(−i t Z⊗Z / 2)`; CNOT flips the target when the control is 1. Unit tests check every gate against
`scipy.linalg.expm`, known one- and two-qubit states (|1⟩ from X, the Bell and GHZ states), CNOT ordering in both
directions, Z-string expectations against explicit operators, and gradients for every parameter type (trainable, data,
trainable data scale, feature-map pair angle, negated angle).

## 2. ZZ feature map

Derived in full in the Stage 2 report, section C. In brief:

```latex
U(x) = \left[ D(x)\, H^{\otimes n} \right]^{\text{reps}},\qquad
D(x) = \exp\!\Big(-i \Big[\sum_q u_q Z_q + \sum_{(a,b)} \varphi(u_a,u_b)\, Z_a Z_b\Big]\Big)
```

with `φ = c0 π² − c1 π (u_a + u_b) + c2 u_a u_b` and u = bandwidth · x. V9 and Qiskit's `ZZFeatureMap` use c0 = c1 = c2 = 1.
Each ZZ term is periodic in φ with period π up to a global phase, so the constant π² is a fixed entangling rotation
(φ = 0.4448 rad). All three parts are two-qubit interactions; only the bilinear part couples features multiplicatively.
V10 implements the whole family (`src/v10/quantum/feature_maps.py`), with presets for each part, a scaled pairwise map
`α u_a u_b`, a product-state map (no pair terms) and linear, ring or full entanglement.

**Synthetic illustration.** On inputs uniform in [0, π]⁴, the off-diagonal kernel mean at bandwidth 0.05 is 0.81. On
V9's real stored points it is 0.90–0.95 (Stage 2). The real data are more concentrated than uniform inputs because
min–max encoding squeezes most points into a small part of [0, π]. Synthetic results therefore understate the
concentration problem.

## 3. Variational circuits

### 3.1 V9 topology

Per layer: data RY on every qubit, trainable RY and RZ on every qubit, then CNOT(0,1), CNOT(1,2), …, CNOT(n−1,0).
Heisenberg propagation through that ring maps ⟨Z_0⟩ to `Z_1 ⋯ Z_{n−1}` and ⟨Z_q⟩ (q ≥ 1) to `Z_0 ⋯ Z_q`. So the
last-layer RZ gates (and, for ⟨Z_0⟩, the last RY on qubit 0) commute with every measured observable and are dead. The
trainable-map offset duplicates the RY weight that follows it. Reference accounting:

| V9 configuration | Declared circuit + map weights | Dead | Redundant | Independent |
|---|---|---|---|---|
| 8 qubits, 2 layers, trainable map, all-Z | 48 | 8 | 8 | 32 |
| 8 qubits, 3 layers, ⟨Z_0⟩ | 48 | 9 | 0 | 39 |
| 4 qubits, 3 layers, trainable map, ⟨Z_0⟩ | 32 | 5 | 4 | 23 |

### 3.2 V10 ansatz

Per layer: data RY (re-uploaded; optional trainable scale shared across layers, no offset), trainable RY and RZ,
entangler (ring, linear or brickwork). Then a **final trainable RY layer**, so no trainable gate sits directly before a
measurement it commutes with.

| Result over 104 configurations | Value |
|---|---|
| Configurations with a provably dead weight (all-qubit readout) | 0 |
| Configurations with Jacobian rank below the declared weight count | 0 |

Dead weights are detected two independent ways: a commutation proof (`dead_by_commutation`, which now requires every gate
using a shared weight to be dead) and zero gradients over many random draws. Redundancy is measured by the numerical rank
of the Jacobian of all observables with respect to all weights over many inputs.

**One caveat.** A single local observable sees only part of the circuit. With ⟨Z_0⟩ alone, 2n − 3 V10 weights have zero
gradient variance because they lie outside that observable's backward light cone. This is inherent to a single local
readout, not a defect. V10 therefore reads all qubits (or correlators) and combines them.

### 3.3 Readout design and trainability

Variance over 200 random weight initialisations of `∂C/∂θ_k`, averaged over live weights, where C is the mean of the
listed observables for one random input. Ratio per added qubit from a log-linear fit over n = 2, 4, 6, 8.

| Design / observable | Var at 8 qubits, 2 layers | Var at 8 qubits, 6 layers | Ratio per added qubit, 2 layers | Ratio per added qubit, 6 layers |
|---|---|---|---|---|
| V9, ⟨Z_0⟩ (parity of n − 1 qubits) | 2.65 × 10⁻³ | 1.95 × 10⁻³ | 0.50 | 0.51 |
| V9, mean of all ⟨Z_q⟩ (parities) | 5.57 × 10⁻⁴ | 2.34 × 10⁻⁴ | 0.45 | 0.40 |
| V10 ring, mean local ⟨Z_q⟩ | 7.25 × 10⁻⁴ | 1.93 × 10⁻⁴ | 0.50 | 0.40 |
| V10 ring, mean neighbour correlator ⟨Z_q Z_{q+1}⟩ | 6.09 × 10⁻⁴ | 2.34 × 10⁻⁴ | 0.42 | 0.36 |
| V10 ring, global parity | 2.61 × 10⁻³ | 1.99 × 10⁻³ | 0.53 | 0.51 |
| **V10 brickwork, mean local ⟨Z_q⟩** | **1.44 × 10⁻³** | 2.44 × 10⁻⁴ | **0.56** | 0.41 |
| V10 brickwork, mean neighbour correlator | 1.02 × 10⁻³ | 2.28 × 10⁻⁴ | 0.46 | 0.36 |

Interpretation:

- **The entangler, not the readout label, sets the light cone.** The ring CNOT chain has two-qubit depth n per layer
  (16 for 8 qubits and 2 layers), so even one layer spreads every qubit's information across the register and a "local"
  readout behaves like a global one. The brickwork entangler has two-qubit depth 2 per layer (4 for 8 qubits and
  2 layers) and keeps light cones local in shallow circuits. That is where it retains larger gradients. The advantage
  disappears by 6 layers.
- **Averages over many observables have smaller per-weight variance than single parities.** This is partly a
  normalisation effect: averaging n terms divides each gradient by n. Variance should therefore be compared within one
  observable type, as in the ratio columns.
- **No barren-plateau claim is made.** Every design shows variance falling with n over 2–8 qubits, but four qubit counts
  cannot establish an asymptotic scaling law.
- **Readout effect on classification is untested** (no features). The plan is to compare local Z, correlators, parities
  and basis probabilities inside the nested evaluation.

### 3.4 Expressibility and entanglement (4 qubits, data fixed at 0)

| Layers | V9: KL to Haar | V10: KL to Haar | V9: Meyer–Wallach | V10: Meyer–Wallach |
|---|---|---|---|---|
| 1 | 0.354 | 0.031 | 0.79 | 0.79 |
| 2 | 0.011 | 0.012 | 0.83 | 0.84 |
| 3 | 0.011 | 0.016 | 0.83 | 0.84 |
| 4 | 0.013 | 0.014 | 0.83 | 0.82 |

KL divergence of the histogram of fidelities between random parameter pairs (1,500 pairs, 75 bins) from the Haar
distribution `P(F) = (N − 1)(1 − F)^(N − 2)`. One V9 layer is far from Haar because its last-layer RZ gates are dead;
from two layers on the designs are indistinguishable. Expressibility is a property of the circuit family and is **not**
evidence of better classification.

## 4. Quantum-kernel mathematics

- **Fidelity kernel** `K_ij = |⟨ψ(x_i)|ψ(x_j)⟩|²`: symmetric, unit diagonal, entries in [0, 1], positive semi-definite
  (tested).
- **Projected kernel** `K_ij = exp(−γ Σ_q ‖ρ_q(x_i) − ρ_q(x_j)‖²_F)` from one-qubit reduced states;
  `‖ρ − ρ'‖²_F = |r − r'|² / 2` for Bloch vectors. Bloch vectors are checked against explicit Pauli operators.
- **Centred alignment** `A(K, y) = ⟨HKH, HyyᵀH⟩_F / (‖HKH‖_F ‖HyyᵀH‖_F)`, `H = I − 11ᵀ/N`, labels in {−1, +1}. Centring
  removes the component of the kernel along the all-ones direction, so a kernel whose entries are all close to 1 is not
  credited for its constant part. A unit test confirms that adding `c·11ᵀ` leaves the centred value unchanged and changes
  the uncentred (V9) value.
- **Spectrum.** Eigenvalues; numerical rank (tolerance N · ε · λ_max); entropy effective rank `exp(−Σ p_i log p_i)` with
  `p = λ / trace`; participation ratio; largest-eigenvalue share (1 means rank one); condition number of K and of the
  centred kernel on its non-null space (centring always removes one dimension).
- **Finite shots.** A compute–uncompute entry estimated from N shots has standard error `√(K(1 − K)/N)`: at most 0.0156
  at 1,024 shots. Square training kernels are estimated once per unordered pair and mirrored. Under noise the diagonal is
  measured, because it is below 1.
- **Affine kernel maps and SVMs** (unit-tested and confirmed on real points in Stage 2). Adding `c·11ᵀ` to training and
  test kernels leaves an SVM unchanged, through the constraint `Σ αᵢ yᵢ = 0`. Scaling by λ equals replacing C with λC.
  Scoring a clean-trained SVM on a scaled test kernel applies an increasing affine map to its scores, which leaves AUC
  unchanged but moves fixed thresholds.

## 5. Noise model

### 5.1 Assumptions (implemented in `src/v10/quantum/noise.py`)

| Mechanism | Model | Applied |
|---|---|---|
| One-qubit depolarizing (p1) | `(1 − p1)ρ + p1/3 (XρX + YρY + ZρZ)` | After every one-qubit gate, on its wire |
| Two-qubit depolarizing (p2) | `(1 − p2)ρ + p2/15 Σ_{P≠II} PρP` | After every two-qubit gate, on its pair |
| Amplitude damping (γ) | Kraus `[[1,0],[0,√(1−γ)]]`, `[[0,√γ],[0,0]]` | Per gate, or from T1 |
| Phase damping (λ) | Kraus `[[1,0],[0,√(1−λ)]]`, `[[0,0],[0,√λ]]` | Per gate, or from T2 |
| Thermal relaxation | `γ = 1 − e^{−t/T1}`, `λ = 1 − e^{−t(2/T2 − 1/T1)}` (requires T2 ≤ 2 T1) | Gate duration, and idle time when idle=True |
| Readout assignment | `P(1|0) = e0`, `P(0|1) = e1` per qubit; `⟨Z⟩_meas = (1 − e0 − e1)⟨Z⟩ + (e1 − e0)` | On the measured distribution |
| Shots | Multinomial sampling of bit strings | Optional |

The one-qubit depolarizing parameter follows the "probability that a Pauli error occurs" convention used by V9. The
"replace with I/2" convention with parameter λ corresponds to p1 = 3λ/4.

### 5.2 Validation

| Closed form | Error |
|---|---|
| One-qubit depolarizing on |0⟩: ⟨Z⟩ = 1 − 4p/3 | 1 × 10⁻¹⁶ |
| Two-qubit depolarizing on |00⟩: ⟨Z_0⟩ = 1 − 16p/15 | 0 |
| Amplitude damping on |1⟩: P(1) = 1 − γ | 0 |
| Phase damping on |+⟩: coherence √(1 − λ) | 2 × 10⁻¹⁶ |
| T1: P(1) = e^{−t/T1}; T2: coherence = e^{−t/T2} | 0; 2 × 10⁻¹⁶ |

Unit tests also check trace preservation, Hermiticity and positivity after every channel; that p1 does not act on
two-qubit gates and p2 does not act on one-qubit gates; idle relaxation; and the readout formula.

**Trajectory convergence** (Pauli channels, V10 VQC with 4 qubits and 2 layers, p1 = 0.002, p2 = 0.02, 5 independent seed
sets per row):

| Trajectories | 16 | 50 | 100 | 200 | 400 | 800 | 1,600 |
|---|---|---|---|---|---|---|---|
| Mean absolute error of ⟨Z⟩ against the exact density matrix | 0.020 | 0.011 | 0.0072 | 0.0060 | 0.0044 | 0.0027 | 0.0020 |

Log–log slope −0.49 (expected −0.5). V9 used 16 trajectories. For a target error of about 0.005 per expectation, roughly
300 trajectories are needed at these rates; the exact density matrix is preferable up to 8 qubits.

**Separate mechanisms** (hypothetical values, labelled as such in the output: p1 = 10⁻³, p2 = 10⁻², readout 0.02 / 0.03,
T1 = 100 µs, T2 = 80 µs, gates 50 ns / 300 ns; V10 VQC, 4 qubits, 2 layers, 32 inputs):

| Mechanism alone | Mean shrinkage of |⟨Z⟩| |
|---|---|
| One-qubit depolarizing | 2.2% |
| Two-qubit depolarizing | 6.8% |
| Readout error | 5.5% |
| T1/T2 during gates | 4.1% |
| T1/T2 including idle periods | 6.7% |
| All combined | 19.8% |

**Noisy kernels** (3 qubits, 12 points, the same hypothetical model):

- Not symmetric: `max |K − Kᵀ| = 0.014`, because amplitude damping and readout error are not time-symmetric.
- Diagonal 0.73–0.76 instead of 1, which is why V10 measures it.
- The exact noisy kernel, symmetrised, is positive definite (smallest eigenvalue 0.008). Finite-shot estimates lose that:
  over 5 repeats there were on average 0.6 negative eigenvalues per matrix at both 1,024 and 8,192 shots. V10 records
  this and repairs it only explicitly (`clip_to_psd`, which reports what it changed).

## 6. Hardware readiness (logical circuits, no transpilation)

| Circuit | Qubits | One-qubit gates | Two-qubit gates | Depth | Two-qubit depth | Needs non-neighbour coupling on a line |
|---|---|---|---|---|---|---|
| V9 VQC, 2 layers, ring | 8 | 48 | 16 | 22 | 16 | Yes |
| V10 VQC, 2 layers, ring | 8 | 56 | 16 | 23 | 16 | Yes |
| V10 VQC, 2 layers, linear | 8 | 56 | 14 | 16 | 9 | No |
| V10 VQC, 2 layers, brickwork | 8 | 56 | 14 | 11 | 4 | No |
| V9 ZZ map, 2 repetitions | 4 | 22 | 12 | 19 | 10 | No |
| Compute–uncompute kernel circuit | 4 | 44 | 24 | 38 | 20 | No |
| Compute–uncompute kernel circuit | 8 | 92 | 56 | 62 | 36 | No |

These are counts before compilation. No transpilation was possible (no Qiskit). The ring's closing CNOT(n−1, 0) needs a
ring of couplers or SWAP insertion on line or heavy-hex layouts. The brickwork entangler has the lowest two-qubit depth
and needs only nearest-neighbour coupling on a line. No hardware was used, and none is planned without an account and
calibration data.

## 7. External framework comparison

Not performed. PennyLane and Qiskit are absent from both environments, and package downloads are refused in this session.
V9's `tests/test_quantum.py::test_pennylane_backend_matches_numpy_simulator` remains skipped. The V10 simulator is
validated only against analytic results and internal references. To run the comparison, install
`pennylane>=0.36` (and optionally `qiskit`) on any machine and add tests that compare V10 states, expectations and
parameter-shift gradients with `default.qubit`, and the feature map with `zz_feature_map(..., entanglement="linear")`.

## 8. Limitations of this stage

- Synthetic inputs only; no statement here concerns classification performance on liver data.
- Gradient-variance and expressibility results cover 2–8 qubits and the stated designs only.
- Noise values are hypothetical and not tied to any device.
- No external cross-check.
