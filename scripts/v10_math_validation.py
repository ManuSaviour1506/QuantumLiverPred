"""Stage 5 (synthetic data only): mathematical and simulator validation of the V10 quantum code.

  python scripts/v10_math_validation.py [--quick]

Outputs results/v10/math_validation/*.json. No real data is used; every input is drawn from a seeded generator and
every noise parameter is a hypothetical simulation setting.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v10.quantum import kernels as KK  # noqa: E402
from src.v10.quantum.circuits import build_v9_vqc, build_v10_vqc, dead_by_commutation, effective_readouts, gate_counts  # noqa: E402
from src.v10.quantum.diagnostics import expressibility_kl, gradient_variance, jacobian_rank, meyer_wallach  # noqa: E402
from src.v10.quantum.feature_maps import ZFeatureMap  # noqa: E402
from src.v10.quantum.noise import DensityMatrixSimulator, NoiseModel, TrajectorySimulator, noisy_fidelity_kernel  # noqa: E402
from src.v10.quantum.simulator import Circuit, gate_matrix  # noqa: E402
from src.v10.utils import run_manifest, write_json  # noqa: E402

HYPOTHETICAL = NoiseModel(p1=1e-3, p2=1e-2, readout_e0=0.02, readout_e1=0.03, t1=100.0, t2=80.0, dur_1q=0.05, dur_2q=0.3,
                          idle=True, calibration="hypothetical: values of a plausible order of magnitude, not from any device "
                                                 "(times in microseconds)")


def simulator_checks(seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    out["gate_unitarity_max_error"] = max(float(np.abs(gate_matrix(n, t).conj().T @ gate_matrix(n, t) - np.eye(len(gate_matrix(n, t)))).max())
                                          for n in ("RX", "RY", "RZ", "RZZ", "H", "X", "Y", "Z", "CNOT", "CZ") for t in rng.uniform(-np.pi, np.pi, 5))
    c = build_v10_vqc(4, 3, trainable_scale=True)
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (16, 4))
    psi = c.state(X, w)
    out["normalisation_max_error"] = float(np.abs(np.linalg.norm(psi, axis=1) - 1).max())
    U = c.unitary(X[0], w)
    out["unitary_max_error"] = float(np.abs(U.conj().T @ U - np.eye(16)).max())
    out["inverse_circuit_max_error"] = float(np.abs(c.inverse().unitary(X[0], w) - U.conj().T).max())
    masks = [(0,), (1, 2), (0, 1, 2, 3)]
    dL = rng.normal(size=(16, 3))
    gw, gx = c.adjoint_grad(X, w, dL, masks)
    ps = c.parameter_shift_grad(X, w, dL, masks)
    f = lambda X_, w_: np.sum(dL * c.expect(c.state(X_, w_), masks))  # noqa: E731
    eps = 1e-6
    fd = np.array([(f(X, w + eps * e) - f(X, w - eps * e)) / (2 * eps) for e in np.eye(len(w))])
    out["adjoint_vs_parameter_shift_max_error"] = float(np.abs(gw - ps).max())
    out["adjoint_vs_finite_difference_max_error"] = float(np.abs(gw - fd).max())
    from src.quantum.simulator import ZZFeatureMap, vqc_circuit
    v9 = vqc_circuit(6, 3)
    w9 = rng.normal(size=v9.n_weights)
    X6 = rng.uniform(0, np.pi, (10, 6))
    out["v10_vs_v9_vqc_state_max_error"] = float(np.abs(v9.state(X6, w9) - build_v9_vqc(6, 3).state(X6, w9)).max())
    fm = ZFeatureMap(6, 2, "v9", entanglement="linear", bandwidth=0.4)
    a, b = ZZFeatureMap(6, 2).states(0.4 * X6), fm.states(X6)
    out["v10_vs_v9_feature_map_fidelity_min"] = float((np.abs(np.sum(np.conj(a) * b, 1)) ** 2).min())
    g = fm.gate_circuit().state(X6)
    out["feature_map_diagonal_vs_gate_fidelity_min"] = float((np.abs(np.sum(np.conj(g) * b, 1)) ** 2).min())
    fm4 = ZFeatureMap(3, 2, "v9", bandwidth=0.7)
    XA, XB = rng.uniform(0, np.pi, (5, 3)), rng.uniform(0, np.pi, (4, 3))
    out["compute_uncompute_vs_fidelity_max_error"] = float(np.abs(noisy_fidelity_kernel(fm4, XA, XB, NoiseModel())
                                                                  - KK.fidelity_kernel(fm4.states(XA), fm4.states(XB))).max())
    bell = Circuit(2, [("H", (0,), None), ("CNOT", (0, 1), None)]).state(np.zeros((1, 0)))[0]
    out["bell_state_max_error"] = float(np.abs(bell - np.array([1, 0, 0, 1]) / np.sqrt(2)).max())
    return out


def readout_and_trainability(quick=False) -> dict:
    n_list = [2, 4, 6, 8]
    L_list = [1, 2, 4] if quick else [1, 2, 3, 4, 6]
    n_init = 60 if quick else 200
    obs = {
        "v9_z0 (parity of qubits 1..n-1)": (lambda n, L: build_v9_vqc(n, L), lambda n: [(0,)]),
        "v9_allz_mean (parities)": (lambda n, L: build_v9_vqc(n, L), lambda n: [(q,) for q in range(n)]),
        "v10_local_z0": (lambda n, L: build_v10_vqc(n, L), lambda n: [(0,)]),
        "v10_local_mean_z": (lambda n, L: build_v10_vqc(n, L), lambda n: [(q,) for q in range(n)]),
        "v10_nn_correlator_mean": (lambda n, L: build_v10_vqc(n, L), lambda n: [(q, q + 1) for q in range(n - 1)]),
        "v10_global_parity": (lambda n, L: build_v10_vqc(n, L), lambda n: [tuple(range(n))]),
        "v10_brick_local_mean_z": (lambda n, L: build_v10_vqc(n, L, entangler="brick"), lambda n: [(q,) for q in range(n)]),
        "v10_brick_nn_correlator_mean": (lambda n, L: build_v10_vqc(n, L, entangler="brick"), lambda n: [(q, q + 1) for q in range(n - 1)]),
    }
    res = {}
    for name, (builder, masks_fn) in obs.items():
        rows = gradient_variance(builder, n_list, L_list, masks_fn, n_init=n_init, seed=11)
        fits = {}
        for L in L_list:
            r = [x for x in rows if x["layers"] == L and x["mean_var_live"] > 0]
            if len(r) >= 3:
                slope = float(np.polyfit([x["qubits"] for x in r], np.log([x["mean_var_live"] for x in r]), 1)[0])
                fits[f"L{L}"] = {"slope_log_var_per_qubit": slope, "factor_per_added_qubit": float(np.exp(slope))}
        res[name] = {"rows": rows, "log_variance_slope_vs_qubits": fits}
    eff = {}
    for n in (4, 8):
        c9, c10 = build_v9_vqc(n, 2), build_v10_vqc(n, 2)
        eff[f"n{n}"] = {"v9_measured_Zq_equivalent_before_ring": [list(m) for m in effective_readouts(c9, [(q,) for q in range(n)])],
                        "v10_measured_Zq": "local: the final trainable RY layer follows the entangler, so Z_q is not a parity of the inputs to it"}
    return {"n_init": n_init, "qubits": n_list, "layers": L_list, "observables": res, "effective_readouts": eff,
            "caution": ("gradient variance at 2-8 qubits cannot establish or exclude a barren plateau; the fitted per-qubit factors "
                        "describe this range only")}


def v10_parameter_accounting(quick=False) -> dict:
    rows = []
    for n in (2, 3, 4, 6, 8):
        for L in ((1, 2) if quick else (1, 2, 3, 4)):
            for scale in (False, True):
                for ent in ("ring", "linear", "brick"):
                    if n == 2 and ent in ("linear", "brick"):
                        continue
                    c = build_v10_vqc(n, L, entangler=ent, trainable_scale=scale)
                    masks = [(q,) for q in range(n)]
                    dead = dead_by_commutation(c, masks)["dead_weights"]
                    need = int(np.ceil((c.n_weights + 4) / n))
                    jr = jacobian_rank(c, masks, n_inputs=max(need, 8), n_points=2, seed=n * 100 + L)
                    rows.append({"qubits": n, "layers": L, "trainable_scale": scale, "entangler": ent, "declared": c.n_weights,
                                 "proved_dead": len(dead), "jacobian_rank": jr["rank_max"], "full_rank": jr["rank_max"] == c.n_weights,
                                 **{k: v for k, v in gate_counts(c).items() if k in ("one_qubit_gates", "two_qubit_gates", "depth", "two_qubit_depth")}})
    v9rows = []
    for n, L, tm, ro in ((8, 2, True, "allz"), (8, 3, False, "z0"), (4, 3, True, "z0")):
        c = build_v9_vqc(n, L, trainable_map=tm)
        masks = [(q,) for q in range(n)] if ro == "allz" else [(0,)]
        dead = dead_by_commutation(c, masks)["dead_weights"]
        jr = jacobian_rank(c, masks, n_inputs=48, n_points=2, seed=5)
        v9rows.append({"config": [n, L, tm, ro], "declared": c.n_weights, "proved_dead": len(dead), "jacobian_rank": jr["rank_max"],
                       "redundant": c.n_weights - len(dead) - jr["rank_max"]})
    return {"v10": rows, "all_v10_full_rank": all(r["full_rank"] for r in rows), "all_v10_zero_dead": all(r["proved_dead"] == 0 for r in rows),
            "v9_reference": v9rows}


def expressibility(quick=False) -> dict:
    out = []
    pairs = 400 if quick else 1500
    for n in (2, 4):
        for L in (1, 2, 3, 4):
            for design in ("v9", "v10"):
                c = build_v9_vqc(n, L) if design == "v9" else build_v10_vqc(n, L)
                x0 = np.zeros((1, n))
                e = expressibility_kl(lambda w: c.state(x0, w)[0], n, c.n_weights, n_pairs=pairs, seed=n * 10 + L)
                W = np.random.default_rng(1).uniform(-np.pi, np.pi, (200, c.n_weights))
                S = np.stack([c.state(x0, w)[0] for w in W])
                out.append({"design": design, "qubits": n, "layers": L, **e, "meyer_wallach_mean": float(meyer_wallach(S, n).mean())})
    return {"rows": out, "definition": "KL divergence of the fidelity histogram of random parameter pairs from the Haar distribution "
                                       "(75 bins); data inputs fixed to 0; lower is closer to Haar. Not a measure of classification quality."}


def feature_maps_synthetic() -> dict:
    rng = np.random.default_rng(3)
    out = []
    for n in (4, 8):
        X = rng.uniform(0, np.pi, (120, n))
        y = (X[:, 0] + X[:, 1] > np.pi).astype(int)
        for bw in (0.01, 0.05, 0.25, 1.0):
            for pair, alpha in (("v9", 1), ("none", 1), ("product", 1), ("product", 16)):
                S = ZFeatureMap(n, 2, pair, alpha=alpha, bandwidth=bw).states(X)
                K = KK.fidelity_kernel(S, S)
                sp = KK.spectrum(K)
                out.append({"qubits": n, "bandwidth": bw, "map": pair if pair != "product" else f"product_alpha{alpha}",
                            "offdiag_mean": KK.offdiag_stats(K)["mean"], "top_share": sp["top_eigenvalue_share"],
                            "effective_rank": sp["effective_rank_entropy"], "alignment_centred_toy_label": KK.alignment(K, y)})
    return {"rows": out, "inputs": "uniform on [0, pi]^n (synthetic); toy label y = [x0 + x1 > pi] for alignment only"}


def noise_validation(quick=False) -> dict:
    out = {"model_hypothetical": HYPOTHETICAL.as_dict()}
    E = np.zeros((1, 0))
    p = 0.12
    one = Circuit(1, [("RZ", (0,), ("c", 0.0))])
    two = Circuit(2, [("CNOT", (0, 1), None)])
    flip = Circuit(1, [("X", (0,), None)])
    plus = Circuit(1, [("H", (0,), None)])
    out["closed_forms_max_error"] = {
        "depolarizing_1q_<Z>=1-4p/3": float(abs(DensityMatrixSimulator(one, NoiseModel(p1=p)).expect(E, None, [(0,)])[0, 0] - (1 - 4 * p / 3))),
        "depolarizing_2q_<Z0>=1-16p/15": float(abs(DensityMatrixSimulator(two, NoiseModel(p2=p)).expect(E, None, [(0,)])[0, 0] - (1 - 16 * p / 15))),
        "amplitude_damping_P1=1-g": float(abs(DensityMatrixSimulator(flip, NoiseModel(amp_damp=0.2)).probabilities(E, None)[0][1] - 0.8)),
        "phase_damping_coherence=sqrt(1-l)": float(abs(2 * DensityMatrixSimulator(plus, NoiseModel(phase_damp=0.3)).run(E)[0][0, 1].real - np.sqrt(0.7))),
        "T1_P1=exp(-t/T1)": float(abs(DensityMatrixSimulator(flip, NoiseModel(t1=50, t2=70, dur_1q=10)).probabilities(E, None)[0][1] - np.exp(-0.2))),
        "T2_coherence=exp(-t/T2)": float(abs(2 * abs(DensityMatrixSimulator(plus, NoiseModel(t1=50, t2=70, dur_1q=10)).run(E)[0][0, 1]) - np.exp(-10 / 70)))}
    # separate mechanisms on a V10 circuit
    rng = np.random.default_rng(4)
    c = build_v10_vqc(4, 2)
    w = rng.normal(size=c.n_weights)
    X = rng.uniform(0, np.pi, (32, 4))
    masks = [(q,) for q in range(4)]
    clean = c.forward(X, w, masks)
    H = HYPOTHETICAL
    settings = {"one_qubit_depolarizing": NoiseModel(p1=H.p1), "two_qubit_depolarizing": NoiseModel(p2=H.p2),
                "readout_error": NoiseModel(readout_e0=H.readout_e0, readout_e1=H.readout_e1),
                "relaxation_T1_T2_gates_only": NoiseModel(t1=H.t1, t2=H.t2, dur_1q=H.dur_1q, dur_2q=H.dur_2q),
                "relaxation_T1_T2_with_idle": NoiseModel(t1=H.t1, t2=H.t2, dur_1q=H.dur_1q, dur_2q=H.dur_2q, idle=True),
                "combined": H}
    sep = {}
    for k, m in settings.items():
        z = DensityMatrixSimulator(c, m).expect(X, w, masks)
        sep[k] = {"mean_abs_change_<Z>": float(np.abs(z - clean).mean()), "max_abs_change_<Z>": float(np.abs(z - clean).max()),
                  "mean_shrinkage_|<Z>|": float(1 - np.abs(z).sum() / np.abs(clean).sum())}
    out["separate_mechanisms_v10_4q_2L"] = sep
    # trajectory convergence to the exact density matrix (Pauli channels)
    m = NoiseModel(p1=0.002, p2=0.02)
    exact = DensityMatrixSimulator(c, m).expect(X[:8], w, masks)
    conv = []
    for N in ((16, 64, 256) if quick else (16, 50, 100, 200, 400, 800, 1600)):
        errs = []
        for s in range(5):
            tr = TrajectorySimulator(c, m.p1, m.p2).run(X[:8], w, masks, trajectories=N, seed=1000 * N + s)
            errs.append(np.abs(tr.mean(0) - exact).mean())
        conv.append({"trajectories": N, "mean_abs_error_vs_exact": float(np.mean(errs)), "sd_over_5_seeds": float(np.std(errs))})
    if len(conv) >= 3:
        slope = float(np.polyfit(np.log([r["trajectories"] for r in conv]), np.log([r["mean_abs_error_vs_exact"] for r in conv]), 1)[0])
    else:
        slope = None
    out["trajectory_convergence"] = {"rows": conv, "log_log_slope": slope, "expected_slope": -0.5}
    # noisy kernels: symmetry, diagonal, PSD, finite shots
    fm = ZFeatureMap(3, 2, "v9", bandwidth=0.25)
    Xk = rng.uniform(0, np.pi, (12, 3))
    K0 = KK.fidelity_kernel(fm.states(Xk), fm.states(Xk))
    km = NoiseModel(p1=H.p1, p2=H.p2, readout_e0=H.readout_e0, readout_e1=H.readout_e1, t1=H.t1, t2=H.t2,
                    dur_1q=H.dur_1q, dur_2q=H.dur_2q)
    Kn = noisy_fidelity_kernel(fm, Xk, Xk, km, symmetric=False)
    shots = {}
    for N in (1024, 8192):
        ests = [noisy_fidelity_kernel(fm, Xk, Xk, km, symmetric=True, shots=N, rng=np.random.default_rng(N + s)) for s in range(5)]
        shots[f"shots_{N}"] = {"min_eigenvalue_mean": float(np.mean([np.linalg.eigvalsh(e).min() for e in ests])),
                               "negative_eigenvalues_mean": float(np.mean([(np.linalg.eigvalsh(e) < -1e-12).sum() for e in ests])),
                               "mean_abs_error_vs_exact_noisy": float(np.mean([np.abs(e - Kn).mean() for e in ests]))}
    out["noisy_kernel_3q"] = {"asymmetry_max_|K-K^T|": float(np.abs(Kn - Kn.T).max()), "diagonal_min_max": [float(np.diag(Kn).min()), float(np.diag(Kn).max())],
                              "min_eigenvalue_exact_noisy": float(np.linalg.eigvalsh(0.5 * (Kn + Kn.T)).min()),
                              "mean_abs_change_vs_clean": float(np.abs(Kn - K0).mean()),
                              "corr_offdiag_clean_vs_noisy": KK.kernel_agreement(K0, Kn)["pearson_offdiag"], "finite_shots": shots}
    return out


def hardware_readiness() -> dict:
    rows = []
    for name, c in (("V9 VQC 8q 2L (ring)", build_v9_vqc(8, 2, True)), ("V10 VQC 8q 2L ring", build_v10_vqc(8, 2)),
                    ("V10 VQC 8q 2L linear", build_v10_vqc(8, 2, entangler="linear")),
                    ("V10 VQC 8q 2L brick", build_v10_vqc(8, 2, entangler="brick")),
                    ("V9 ZZ map 4q reps 2 (linear)", ZFeatureMap(4, 2, "v9").gate_circuit()),
                    ("compute-uncompute kernel circuit 4q", ZFeatureMap(4, 2, "v9").compute_uncompute_circuit()),
                    ("compute-uncompute kernel circuit 8q", ZFeatureMap(8, 2, "v9").compute_uncompute_circuit())):
        g = gate_counts(c)
        ring = any(op[0] == "CNOT" and abs(op[1][0] - op[1][1]) > 1 for op in c.ops)
        rows.append({"circuit": name, **{k: g[k] for k in ("qubits", "one_qubit_gates", "two_qubit_gates", "depth", "two_qubit_depth")},
                     "needs_non_nearest_neighbour_coupling_on_a_line": ring})
    return {"rows": rows, "transpiled": False,
            "note": ("logical counts before compilation; no transpilation was done (Qiskit not installed). A ring entangler needs a "
                     "ring of couplers or SWAP insertion on line or heavy-hex layouts; no hardware was used")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    out = ROOT / "results" / "v10" / "math_validation"
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    steps = {"simulator_checks": simulator_checks, "readout_and_trainability": lambda: readout_and_trainability(a.quick),
             "parameter_accounting": lambda: v10_parameter_accounting(a.quick), "expressibility": lambda: expressibility(a.quick),
             "feature_maps_synthetic": feature_maps_synthetic, "noise_validation": lambda: noise_validation(a.quick),
             "hardware_readiness": hardware_readiness}
    for k, fn in steps.items():
        t0 = time.time()
        try:
            write_json(out / f"{k}.json", fn())
            status[k] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
        except Exception as e:
            import traceback
            status[k] = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()}
        print(k, status[k]["status"], status[k].get("seconds"), flush=True)
    write_json(out / "run_manifest.json", run_manifest(ROOT, [], {"all": "fixed seeds in code"},
                                                       {"steps": status, "data": "synthetic only",
                                                        "command": "python scripts/v10_math_validation.py" + (" --quick" if a.quick else "")}))
    if any(s["status"] != "ok" for s in status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
