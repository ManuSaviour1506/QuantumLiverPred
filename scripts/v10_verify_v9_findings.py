"""Stage 2: independent verification of the original V9 audit findings A-G (plus timing and statistics) from the files that
exist in the repository. Read-only with respect to V9: V9 modules are imported, never modified; outputs go to
results/v10/verification/.

  python scripts/v10_verify_v9_findings.py [--root .] [--quick]

What cannot be done without the CT images (no feature vectors exist in the repository) is recorded as "not executable"
with the reason, never estimated.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v10.evaluation import metrics as M  # noqa: E402
from src.v10.evaluation import statistics as ST  # noqa: E402
from src.v10.quantum import kernels as KK  # noqa: E402
from src.v10.quantum.circuits import build_v9_vqc, dead_by_commutation, effective_readouts, gate_counts  # noqa: E402
from src.v10.quantum.diagnostics import fubini_study_diag, jacobian_rank, meyer_wallach, parameter_table  # noqa: E402
from src.v10.quantum.feature_maps import ZFeatureMap, v9_phase_parts  # noqa: E402
from src.v10.utils import run_manifest, write_json  # noqa: E402

BW_GRID = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
GAMMA_GRID = [10 ** e for e in np.linspace(-3, 2, 11)]
NOT_EXECUTABLE = "not executable: the repository holds no feature vectors (CT images, data/processed and artifacts/cache are absent)"


def load_npz(root, fam, mid):
    with np.load(root / "artifacts" / fam / mid / "model.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_json(root, rel):
    return json.loads((root / rel).read_text())


def preds(root, mid, split="test"):
    p = root / "results" / "predictions" / f"{mid}.csv"
    d = pd.read_csv(p)
    return d[d.split == split].sort_values("row").reset_index(drop=True)


def meta(root, fam, mid):
    return json.loads((root / "artifacts" / fam / mid / "model.json").read_text())


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------------------------------ point sets
def svc_label_convention() -> dict:
    """Confirm that sign(dual_coef_) of a fitted sklearn SVC gives the class (+ -> classes_[1]) of each support vector."""
    from sklearn.svm import SVC
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=80) > 0).astype(int)
    m = SVC(kernel="rbf", C=1.0).fit(X, y)
    agree = float(np.mean((m.dual_coef_.ravel() > 0).astype(int) == y[m.support_]))
    return {"sign_of_dual_coef_equals_class_of_support_vector": agree == 1.0, "agreement": agree}


def point_set(root, kernel, control, kfam="quantum", cfam="classical"):
    q, c = load_npz(root, kfam, kernel), load_npz(root, cfam, control)
    enc_same = all(np.array_equal(q[k], c[k]) for k in q if k.startswith("enc_"))
    A = np.vstack([q["sv_inputs"], c["sv"]])
    y = np.r_[(q["dual"] > 0).astype(int), (c["dual"] > 0).astype(int)]
    key = [tuple(np.round(r, 12)) for r in A]
    first, conflicts = {}, 0
    for i, k in enumerate(key):
        if k in first:
            conflicts += int(y[first[k]] != y[i])
        else:
            first[k] = i
    idx = np.array(sorted(first.values()))
    return {"A": A[idx], "y": y[idx], "n_kernel_sv": int(len(q["sv_inputs"])), "n_control_sv": int(len(c["sv"])),
            "n_unique": int(len(idx)), "label_conflicts_between_duplicates": conflicts, "encoder_identical": bool(enc_same),
            "selected_bandwidth": float(q["hyper"][2]), "qubits": int(q["hyper"][0]), "reps": int(q["hyper"][1]),
            "C": float(q["hyper"][3]), "encoder_lo": q["enc_lo"].tolist(), "encoder_hi": q["enc_hi"].tolist(),
            "class_counts": {"1": int(y[idx].sum()), "0": int((y[idx] == 0).sum())}}


POINT_SETS = {"v9_cnn_4q": ("qksvm_v9_cnn", "rbf_control_v9_cnn"), "v9_eff_4q": ("qksvm_v9_eff", "rbf_control_v9_eff"),
              "v8_imagenet_8q": ("qkernel_svm_8q", "pca8_svm_rbf"), "v8_finetuned_8q": ("qkernel_svm_8q_ft", "pca8_svm_rbf_ft")}


# ------------------------------------------------------------------------------------------------ finding A
def finding_a(root) -> dict:
    out = {"claim": "CNN embeddings are already almost linearly separable; logistic regression reaches test AUC ~0.998 with 5-9 parameters"}
    rows = []
    for tag in ("cnn", "eff"):
        Q = load_json(root, f"results/v9/quantum/quantum_experiments_{tag}.json")
        for r in Q["configurations"]:
            if r["kind"] == "lr_control":
                rows.append({"features": tag, "pca_components": r["qubits"], "parameters": r["parameters"], "C": r["C"],
                             "val_auc": r["val_auc"], "test_auc": r["test_auc"]})
    out["lr_control_sweep_recorded"] = rows
    cnn_meta = meta(root, "classical", "cnn25d_v9_a4_res_se_focal")
    out["embedding_provenance"] = {"cnn": "cnn25d_v9_a4_res_se_focal", "split_file": cnn_meta.get("split_file"),
                                   "training_status": cnn_meta.get("training_status"),
                                   "validation_used_for": "early stopping (val ROC-AUC) and recipe selection among a1-a4",
                                   "best_epoch": cnn_meta["training"]["best_epoch"],
                                   "epochs_run": len(cnn_meta["training"]["history"]),
                                   "classification_for_v10": "exploratory: CNN fitted on training groups only, but validation groups reused for CNN and quantum selection, and the test groups were examined in V9's G01 analysis"}
    # rank agreement of test and validation scores between the CNN and the models built on its embedding
    fams = {"cnn25d_v9_a4_res_se_focal": "CNN itself", "lr_control_v9_cnn": "logistic regression on PCA-8 of embedding",
            "vqc_v9_cnn": "VQC", "qksvm_v9_cnn": "quantum kernel SVM", "rbf_control_v9_cnn": "RBF-SVM",
            "hybrid_v9_cnn": "hybrid", "twin_v9_cnn": "classical twin"}
    agree = {}
    for split in ("test", "val"):
        S = {m: preds(root, m, split) for m in fams}
        y = S["cnn25d_v9_a4_res_se_focal"].y.to_numpy()
        names = list(fams)
        mat = {a: {b: spearman(S[a].score.to_numpy(), S[b].score.to_numpy()) for b in names} for a in names}
        agree[split] = {"spearman": mat, "auc": {m: M.roc_auc(y, S[m].score.to_numpy()) for m in names}}
    out["score_rank_agreement"] = agree
    # pairwise discordance against the CNN on the test set
    S = {m: preds(root, m) for m in fams}
    y = S["cnn25d_v9_a4_res_se_focal"].y.to_numpy()
    pos, neg = y == 1, y == 0
    base = S["cnn25d_v9_a4_res_se_focal"].score.to_numpy()
    cb = base[pos][:, None] > base[neg][None, :]
    disc = {}
    for m in fams:
        if m == "cnn25d_v9_a4_res_se_focal":
            continue
        s = S[m].score.to_numpy()
        cm = s[pos][:, None] > s[neg][None, :]
        disc[m] = {"pairs": int(cb.size), "cnn_wrong_model_right": int((~cb & cm).sum()), "cnn_right_model_wrong": int((cb & ~cm).sum()),
                   "both_wrong": int((~cb & ~cm).sum())}
    out["test_pair_discordance_vs_cnn"] = disc
    # how concentrated is the logistic-regression control on the PCA components?
    coef = {}
    for mid in ("lr_control_v9_cnn", "lr_control_v9_eff", "pca8_logreg", "pca8_logreg_ft"):
        a = load_npz(root, "classical", mid)
        cf = a["coef"].ravel()
        span = a["enc_hi"] - a["enc_lo"]
        eff = np.abs(cf) * np.pi            # change of the logit across each component's full encoded range [0, pi]
        coef[mid] = {"abs_coef": np.abs(cf).tolist(), "logit_range_per_component": eff.tolist(),
                     "share_of_first_component": float(eff[0] / eff.sum()), "pca_component_range_training": span.tolist()}
    out["lr_control_coefficients"] = coef
    # class separation of the stored real training points (support vectors, margin-biased)
    sep = {}
    for name, (k, c) in POINT_SETS.items():
        ps = point_set(root, k, c)
        A, yy = ps["A"], ps["y"]
        d = (A[yy == 1].mean(0) - A[yy == 0].mean(0)) / np.sqrt(0.5 * (A[yy == 1].var(0) + A[yy == 0].var(0)) + 1e-12)
        sep[name] = {"n": ps["n_unique"], "cohens_d_per_component": d.tolist(),
                     "note": "support vectors lie near the decision boundary, so these effect sizes understate separability"}
    out["support_vector_class_separation"] = sep
    out["not_executable"] = {"feature_distributions_correlations_redundancy": NOT_EXECUTABLE,
                             "separability_diagnostics_on_full_embeddings": NOT_EXECUTABLE}
    return out


# ------------------------------------------------------------------------------------------------ finding B
def kernel_states(A, n, bw, pair="v9"):
    return ZFeatureMap(n, 2, pair, entanglement="linear", bandwidth=bw).states(A)


def finding_b(root, quick=False) -> dict:
    from src.quantum.simulator import ZZFeatureMap as V9Map
    from src.quantum.simulator import fidelity_kernel as v9_fid
    from src.quantum.v9 import kernel_target_alignment as v9_kta
    out = {"claim": "selected bandwidth 0.05 everywhere; support-vector kernel mean ~0.993, SD ~0.008; alignment 0.04 at 0.05 vs 0.46 at 0.25; Gaussian correlation ~0.94",
           "svc_label_convention": svc_label_convention(), "bandwidth_grid": BW_GRID,
           "bandwidth_grid_justification": ("inputs are min-max encoded to [0, pi]; u = bandwidth * x; exp(-i u Z) has period pi in u, so "
                                            "bandwidth 1 spans one full period and bandwidths above 1 wrap distinct inputs onto equal "
                                            "single-qubit phases; the lower end is where off-diagonal entries exceed 0.999"),
           "point_sets_note": ("only support vectors are stored; each set is the union of the quantum kernel's and its RBF control's "
                               "support vectors (identical encoders), labels from the sign of the dual coefficients. These are real "
                               "training points but biased towards the decision boundary; full training kernels cannot be built")}
    rec = {}
    for tag in ("cnn", "eff"):
        Q = load_json(root, f"results/v9/quantum/quantum_experiments_{tag}.json")
        rec[tag] = [{"qubits": r["qubits"], "bandwidth": r["bandwidth"], "C": r["C"], "val_auc": r["val_auc"], "test_auc": r["test_auc"],
                     "kta_uncentred_full_training_set": r.get("kernel_target_alignment_train")} for r in Q["configurations"] if r["kind"] == "qksvm"]
    v8 = [{"model": m, "hyper": load_npz(root, "quantum", m)["hyper"].tolist()} for m in ("qkernel_svm_8q", "qkernel_svm_8q_ft")]
    out["recorded_selections"] = {"v9": rec, "v8": v8,
                                  "all_selected_at_0.05": all(r["bandwidth"] == 0.05 for t in rec.values() for r in t) and all(v["hyper"][2] == 0.05 for v in v8),
                                  "0.05_is_lowest_grid_value_in_v9": True}
    # V9 KTA formula = uncentred alignment
    rng = np.random.default_rng(0)
    Kt = np.exp(-rng.uniform(0, 1, (30, 30)))
    Kt = 0.5 * (Kt + Kt.T)
    yt = rng.integers(0, 2, 30)
    out["v9_kta_formula_is_uncentred_alignment"] = bool(np.isclose(v9_kta(Kt, yt), KK.alignment(Kt, yt, centred=False)))
    sets = {}
    for name, (k, c) in POINT_SETS.items():
        ps = point_set(root, k, c)
        A, y, n = ps["A"], ps["y"], ps["qubits"]
        S_v9 = V9Map(n, 2).states(ps["selected_bandwidth"] * A)
        S_v10 = kernel_states(A, n, ps["selected_bandwidth"])
        same = float(np.abs(v9_fid(S_v9, S_v9) - KK.fidelity_kernel(S_v10, S_v10)).max())
        grid = BW_GRID if not quick else [0.01, 0.05, 0.25, 1.0]
        per_bw = []
        for bw in grid:
            S = kernel_states(A, n, bw)
            K = KK.fidelity_kernel(S, S)
            rep = KK.full_report(K, y)
            rbf = []
            for g in GAMMA_GRID:
                Kr = KK.rbf_kernel(A, A, g)
                ag = KK.kernel_agreement(K, Kr)
                rbf.append({"gamma": g, "label_alignment_centred": KK.alignment(Kr, y), **ag})
            best_match = max(rbf, key=lambda r: r["centred_alignment"])
            per_bw.append({"bandwidth": bw, **rep, "rbf_best_match_to_quantum": best_match,
                           "rbf_best_label_alignment": max(r["label_alignment_centred"] for r in rbf)})
        cls = {"linear": KK.full_report(KK.linear_kernel(A, A), y),
               "poly2": KK.full_report(KK.polynomial_kernel(A, A, 2, 1.0 / A.shape[1], 1.0), y),
               "poly3": KK.full_report(KK.polynomial_kernel(A, A, 3, 1.0 / A.shape[1], 1.0), y),
               "rbf_by_gamma": [{"gamma": g, **KK.full_report(KK.rbf_kernel(A, A, g), y)} for g in GAMMA_GRID]}
        # feature-map variants and projected kernels on the same points
        variants = {}
        for bw in (ps["selected_bandwidth"], 0.25, 1.0):
            row = {}
            for pair, alpha in (("v9", 1.0), ("v9_without_bilinear", 1.0), ("v9_bilinear", 1.0), ("v9_constant", 1.0),
                                ("v9_linear", 1.0), ("none", 1.0), ("product", 1.0), ("product", 4.0), ("product", 16.0)):
                S = ZFeatureMap(n, 2, pair, alpha=alpha, entanglement="linear", bandwidth=bw).states(A)
                K = KK.fidelity_kernel(S, S)
                key = pair if pair != "product" else f"product_alpha{alpha:g}"
                row[key] = {"offdiag": KK.offdiag_stats(K), "alignment_centred": KK.alignment(K, y),
                            "effective_rank": KK.spectrum(K)["effective_rank_entropy"], "top_share": KK.spectrum(K)["top_eigenvalue_share"]}
            R = KK.bloch_vectors(kernel_states(A, n, bw), n)
            for g in (0.1, 1.0, 10.0):
                Kp = KK.projected_kernel(R, R, g)
                row[f"projected_gamma{g:g}"] = {"offdiag": KK.offdiag_stats(Kp), "alignment_centred": KK.alignment(Kp, y),
                                                "effective_rank": KK.spectrum(Kp)["effective_rank_entropy"],
                                                "top_share": KK.spectrum(Kp)["top_eigenvalue_share"]}
            variants[f"bw_{bw:g}"] = row
        meta_ps = {k2: v for k2, v in ps.items() if k2 not in ("A", "y")}
        sets[name] = {**meta_ps, "v10_kernel_equals_v9_kernel_max_abs_diff": same, "per_bandwidth": per_bw,
                      "classical_kernels": cls, "feature_map_variants": variants}
    out["point_sets"] = sets
    out["not_executable"] = {"full_training_kernel_and_bandwidth_reselection": NOT_EXECUTABLE}
    return out


# ------------------------------------------------------------------------------------------------ finding C
def finding_c(root) -> dict:
    from src.quantum.simulator import ZZFeatureMap as V9Map
    from src.v10.quantum.simulator import gate_matrix
    phi = 0.83
    cn = gate_matrix("CNOT")
    sandwich = cn @ np.kron(np.eye(2), gate_matrix("RZ", 2 * phi)) @ cn
    zz = np.diag(np.exp(-1j * phi * np.array([1, -1, -1, 1])))
    out = {"claim": "phi = 2(pi - x_a)(pi - x_b) = 2 pi^2 - 2 pi (x_a + x_b) + 2 x_a x_b; the 'nearly constant' explanation was wrong",
           "implemented_unitary": ("U(x) = [ D(x) H^{(x)n} ]^reps,  D(x) = exp(-i [ sum_q u_q Z_q + sum_(a,b) (pi - u_a)(pi - u_b) Z_a Z_b ]),  "
                                   "u = bandwidth * x, linear chain pairs (q, q+1), reps = 2 (src/quantum/simulator.py ZZFeatureMap.states)"),
           "gate_identity_CNOT_RZ2phi_CNOT_equals_exp_minus_i_phi_ZZ": bool(np.allclose(sandwich, zz)),
           "matches_qiskit_ZZFeatureMap_gate_sequence": "yes, apart from entanglement: Qiskit's default is 'full', V9 uses 'linear'",
           "periodicity": ("exp(-i phi Z_a Z_b) changes only by a global phase when phi -> phi + pi, and exp(-i u Z) when u -> u + pi; "
                           "so the constant pi^2 acts as a fixed ZZ rotation with phi = pi^2 mod pi = %.4f rad" % (np.pi ** 2 % np.pi)),
           "note_on_linear_terms": ("all three parts (constant, linear, bilinear) multiply Z_a Z_b, so all three are two-qubit interactions; "
                                    "the linear part is an entangling ZZ rotation whose angle depends additively on the data, only the "
                                    "bilinear part couples the two features multiplicatively")}
    for n, bw in ((4, 0.05), (8, 1.0)):
        X = np.random.default_rng(n).uniform(0, np.pi, (5, n))
        a = V9Map(n, 2).states(bw * X)
        b = ZFeatureMap(n, 2, "v9", entanglement="linear", bandwidth=bw).states(X)
        out[f"v10_reproduces_v9_states_n{n}_bw{bw}"] = bool(np.allclose(np.abs(np.sum(np.conj(a) * b, 1)), 1))
    res = {}
    for name, (k, c) in POINT_SETS.items():
        ps = point_set(root, k, c)
        A, n = ps["A"], ps["qubits"]
        pairs = [(q, q + 1) for q in range(n - 1)]
        per = {}
        for bw in (0.01, ps["selected_bandwidth"], 0.25, 1.0):
            parts = v9_phase_parts(A, bw, pairs)
            span = {key: float(np.ptp(v)) for key, v in parts.items()}
            sd = {key: float(v.std()) for key, v in parts.items()}
            Ufull = kernel_states(A, n, bw)
            Kf = KK.fidelity_kernel(Ufull, Ufull)
            comp = {}
            for variant in ("v9_without_bilinear", "v9_without_constant", "v9_bilinear", "none"):
                Kv = KK.fidelity_kernel(*(2 * [kernel_states(A, n, bw, variant)]))
                m = ~np.eye(len(Kf), dtype=bool)
                comp[variant] = {"max_abs_diff": float(np.abs(Kf - Kv)[m].max()), "mean_abs_diff": float(np.abs(Kf - Kv)[m].mean()),
                                 "relative_frobenius_centred": float(np.linalg.norm(KK.centre(Kf) - KK.centre(Kv)) / max(np.linalg.norm(KK.centre(Kf)), 1e-300)),
                                 "centred_alignment_between_kernels": KK.kernel_agreement(Kf, Kv)["centred_alignment"]}
            fs = fubini_study_diag(lambda Z: kernel_states(Z, n, bw), A[:200])
            fs_none = fubini_study_diag(lambda Z: kernel_states(Z, n, bw, "none"), A[:200])
            per[f"bw_{bw:g}"] = {"pair_phase_span_rad (phi, i.e. exp(-i phi ZZ))": span, "pair_phase_sd_rad": sd,
                                 "single_qubit_phase_span_rad": float(np.ptp(bw * A)),
                                 "kernel_change_when_removing_part": comp,
                                 "fubini_study_mean_per_feature_v9": fs.mean(0).tolist(),
                                 "fubini_study_mean_per_feature_no_pair_terms": fs_none.mean(0).tolist(),
                                 "meyer_wallach_mean": float(meyer_wallach(Ufull, n).mean())}
        res[name] = {"n_points": ps["n_unique"], "qubits": n, "per_bandwidth": per}
    out["on_real_points"] = res
    return out


# ------------------------------------------------------------------------------------------------ finding D
def finding_d(root, quick=False) -> dict:
    from src.quantum.models import VQC
    from src.quantum.v9 import VQCv9
    out = {"claim": "final RZ rotations before the CNOT ring and Z measurement have exactly zero gradient"}
    struct = {}
    for n, L, tm, readout in ((8, 2, True, "allz"), (8, 3, False, "z0"), (4, 3, True, "z0"), (8, 2, False, "allz")):
        c = build_v9_vqc(n, L, trainable_map=tm)
        masks = [(q,) for q in range(n)] if readout == "allz" else [(0,)]
        tab = parameter_table(c, masks, n_draws=6 if quick else 15, batch=6, seed=1)
        jr = jacobian_rank(c, masks, n_inputs=24 if quick else 40, n_points=2, seed=2)
        struct[f"n{n}_L{L}_{'map' if tm else 'fixed'}_{readout}"] = {
            "declared_circuit_and_map_weights": c.n_weights, "proved_dead": tab["dead_proved"], "zero_in_all_draws": tab["zero_all_draws"],
            "proved_equals_numerical": tab["dead_proved"] == tab["zero_all_draws"], "jacobian_rank": jr["rank_max"],
            "redundant_directions": c.n_weights - len(tab["dead_proved"]) - jr["rank_max"],
            "effective_readouts": [list(m) for m in effective_readouts(c, masks)], "rows": tab["rows"], "gate_counts": gate_counts(c)}
    out["structural_analysis_random_draws"] = struct
    # trained models: dead weights never moved from their initial values
    trained = {}
    for fam, mid in (("quantum", "vqc_v9_cnn"), ("quantum", "vqc_v9_eff"), ("quantum", "vqc_8q"), ("quantum", "vqc_8q_ft")):
        a = load_npz(root, fam, mid)
        if "a" in a:                                    # V9 model
            n, L, tm, rz = (int(v) for v in a["shape"])
            init = VQCv9(n, L, bool(tm), "allz" if rz else "z0", seed=42)
            w, w0 = a["w"], init.w
            readout = "allz" if rz else "z0"
        else:                                           # V8 model
            n, L = (int(v) for v in a["shape"])
            init = VQC(n, L, seed=42)
            w, w0 = a["w"], init.w
            tm, readout = False, "z0"
        c = build_v9_vqc(n, L)
        masks = [(q,) for q in range(n)] if readout == "allz" else [(0,)]
        dead = dead_by_commutation(c, masks)["dead_weights"]
        live = [k for k in range(len(w)) if k not in dead]
        rec = {"qubits": n, "layers": L, "trainable_map": bool(tm), "readout": readout, "circuit_weights": int(len(w)),
               "proved_dead": dead, "dead_weights_unchanged_within_1e-8": bool(np.abs(w[dead] - w0[dead]).max() < 1e-8),
               "dead_weights_bitwise_equal_initial_values": bool(np.array_equal(w[dead], w0[dead])),
               "why_not_bitwise": ("adjoint gradients of dead weights are ~1e-17 rounding noise, and Adam divides the moment by "
                                   "sqrt(v) + 1e-8, so each step moves them by about lr * 1e-9"),
               "max_abs_change_dead": float(np.abs(w[dead] - w0[dead]).max()),
               "median_abs_change_live": float(np.median(np.abs(w[live] - w0[live]))),
               "init_rule": "seed 42, uniform(-0.1 pi, 0.1 pi) (V9 code defaults used by the training scripts)"}
        if "a" in a:
            rec["map_offsets_trained"] = a["o"].tolist()
            rec["map_scales_trained"] = a["a"].tolist()
            rec["readout_weights"] = a["v"].tolist()
        trained[mid] = rec
    out["trained_models"] = trained
    # gradients at the trained V8 weights on real training inputs (V8 kernel support vectors share the VQC encoder)
    real = {}
    for mid, (k, c2) in (("vqc_8q", POINT_SETS["v8_imagenet_8q"]), ("vqc_8q_ft", POINT_SETS["v8_finetuned_8q"])):
        a = load_npz(root, "quantum", mid)
        enc_ok = all(np.array_equal(a[key], load_npz(root, "quantum", k)[key]) for key in a if key.startswith("enc_"))
        ps = point_set(root, k, c2)
        c = build_v9_vqc(8, 3)
        tab = parameter_table(c, [(0,)], n_draws=10, batch=min(32, ps["n_unique"]), seed=3, fixed_weights=a["w"], inputs=ps["A"])
        g = np.array([r["max_abs_grad"] for r in tab["rows"]])
        live = [r["index"] for r in tab["rows"] if r["status"] == "active"]
        z = c.forward(ps["A"], a["w"], [(0,)])[:, 0]
        real[mid] = {"encoder_shared_with_kernel": bool(enc_ok), "real_inputs": ps["n_unique"], "zero_in_all_draws": tab["zero_all_draws"],
                     "max_abs_grad_live_min_median_max": [float(g[live].min()), float(np.median(g[live])), float(g[live].max())],
                     "z0_on_real_inputs_abs_median": float(np.median(np.abs(z))), "z0_range": [float(z.min()), float(z.max())],
                     "readout_scale_bias": a["readout"].tolist()}
    out["trained_v8_on_real_inputs"] = real
    return out


# ------------------------------------------------------------------------------------------------ finding E
def compare_pair(root, a_mid, a_fam, b_mid, b_fam) -> dict:
    out = {}
    for split in ("test", "val"):
        A, B = preds(root, a_mid, split), preds(root, b_mid, split)
        assert (A.row.to_numpy() == B.row.to_numpy()).all()
        y = A.y.to_numpy()
        ta, tb = meta(root, a_fam, a_mid)["threshold"], meta(root, b_fam, b_mid)["threshold"]
        pa, pb = A.prob.to_numpy(), B.prob.to_numpy()
        sa, sb = A.score.to_numpy(), B.score.to_numpy()
        ca, cb = M.auc_pair_counts(y, sa), M.auc_pair_counts(y, sb)
        pos, neg = y == 1, y == 0
        oa = sa[pos][:, None] > sa[neg][None, :]
        ob = sb[pos][:, None] > sb[neg][None, :]
        from scipy.stats import kendalltau
        da, db = pa >= ta, pb >= tb
        r = {"n": int(len(y)), "auc_a": M.roc_auc(y, sa), "auc_b": M.roc_auc(y, sb), "auc_pair_counts_a": ca, "auc_pair_counts_b": cb,
             "auc_exactly_equal": ca == cb, "pearson_scores": float(np.corrcoef(sa, sb)[0, 1]), "spearman_scores": spearman(sa, sb),
             "kendall_tau_scores": float(kendalltau(sa, sb).statistic),
             "pairs_a_right_b_wrong": int((oa & ~ob).sum()), "pairs_b_right_a_wrong": int((ob & ~oa).sum()),
             "prob_abs_diff_mean": float(np.abs(pa - pb).mean()), "prob_abs_diff_max": float(np.abs(pa - pb).max()),
             "prob_abs_diff_p95": float(np.percentile(np.abs(pa - pb), 95)),
             "decision_disagreements": int((da != db).sum()), "threshold_a": ta, "threshold_b": tb,
             "metrics_a": M.full_metrics(y, sa, pa, ta), "metrics_b": M.full_metrics(y, sb, pb, tb),
             "per_group": {}}
        for g in sorted(A.scan_id.unique()):
            m = A.scan_id.to_numpy() == g
            r["per_group"][g] = {"n": int(m.sum()), "auc_a": M.roc_auc(y[m], sa[m]), "auc_b": M.roc_auc(y[m], sb[m]),
                                 "log_loss_a": M.log_loss(y[m], pa[m]), "log_loss_b": M.log_loss(y[m], pb[m]),
                                 "decision_disagreements": int((da[m] != db[m]).sum())}
        out[split] = r
    return out


def finding_e(root) -> dict:
    out = {"claim": "fusion hybrid and classical twin have identical test AUC (~10 decimals) and rank correlation ~0.98; no-fusion hybrid below twin"}
    out["v9_cnn"] = compare_pair(root, "hybrid_v9_cnn", "hybrid", "twin_v9_cnn", "classical")
    out["v9_eff"] = compare_pair(root, "hybrid_v9_eff", "hybrid", "twin_v9_eff", "classical")
    out["v8_finetuned"] = compare_pair(root, "hybrid_cnn_vqc_8q_ft", "hybrid", "hybrid_classical_twin_ft", "classical")
    out["v8_imagenet"] = compare_pair(root, "hybrid_cnn_vqc_8q", "hybrid", "hybrid_classical_twin", "classical")
    rec = {}
    for tag in ("cnn", "eff"):
        Q = load_json(root, f"results/v9/quantum/quantum_experiments_{tag}.json")
        hy = {(r["qubits"], r["fusion"]): r for r in Q["configurations"] if r["kind"] == "hybrid"}
        tw = {(r["qubits"], r["fusion"]): r for r in Q["configurations"] if r["kind"] == "twin"}
        rec[tag] = [{"qubits": k[0], "fusion": k[1], "hybrid_val": hy[k]["val_auc"], "hybrid_test": hy[k]["test_auc"],
                     "twin_val": tw[k]["val_auc"], "twin_test": tw[k]["test_auc"], "hybrid_minus_twin_test": hy[k]["test_auc"] - tw[k]["test_auc"],
                     "hybrid_params": hy[k]["parameters"], "twin_params": tw[k]["parameters"]} for k in sorted(hy)]
    out["recorded_sweep"] = rec
    w = {}
    for mid in ("hybrid_v9_cnn", "hybrid_v9_eff"):
        a = load_npz(root, "hybrid", mid)
        n, L, fu = (int(v) for v in a["shape"])
        v = a["v"]
        w[mid] = {"qubits": n, "layers": L, "fusion": bool(fu), "readout_abs_quantum_part": np.abs(v[:n]).tolist(),
                  "readout_abs_classical_part": np.abs(v[n:]).tolist() if fu else [],
                  "note": "readout weights alone do not measure contribution: the inputs to each part have different scales and correlations"}
    out["stored_readout_weights"] = w
    out["not_executable"] = {"ablations (classical-only, quantum-only, zeroed branches, random quantum features, frozen vs trainable fusion)": NOT_EXECUTABLE}
    return out


# ------------------------------------------------------------------------------------------------ finding F
def finding_f(root) -> dict:
    out = {"claim": "VQC validation AUC does not predict test AUC (examples 0.9967/0.9965, 0.9994/0.9731, 0.9966/0.9168, 0.7164/0.3642)"}
    for tag in ("cnn", "eff"):
        Q = load_json(root, f"results/v9/quantum/quantum_experiments_{tag}.json")
        res = {}
        for kind in ("vqc", "qksvm", "hybrid", "lr_control", "rbf_control", "twin"):
            rows = [r for r in Q["configurations"] if r["kind"] == kind]
            if len(rows) < 2:
                continue
            v = np.array([r["val_auc"] for r in rows])
            t = np.array([r["test_auc"] for r in rows])
            sel = int(np.argmax(v))
            res[kind] = {"configurations": len(rows), "spearman_val_vs_test": spearman(v, t) if len(rows) > 2 else None,
                         "selected_test_auc": float(t[sel]), "best_test_auc": float(t.max()), "median_test_auc": float(np.median(t)),
                         "val_minus_test_mean": float((v - t).mean()), "val_minus_test_max": float((v - t).max()),
                         "rows": [{k: r[k] for k in r if k in ("qubits", "layers", "trainable_map", "readout", "bandwidth", "C", "fusion", "val_auc", "test_auc")} for r in rows]}
        out[tag] = res
    quoted = [("cnn", 8, 2, False, "z0", 0.9967, 0.9965), ("cnn", 8, 2, True, "allz", 0.9994, 0.9731),
              ("cnn", 6, 3, False, "z0", 0.9966, 0.9168), ("cnn", 8, 1, False, "z0", 0.7164, 0.3642)]
    Q = load_json(root, "results/v9/quantum/quantum_experiments_cnn.json")
    chk = []
    for tag, n, L, tm, ro, v, t in quoted:
        r = next(r for r in Q["configurations"] if r["kind"] == "vqc" and r["qubits"] == n and r["layers"] == L
                 and r["trainable_map"] == tm and r["readout"] == ro)
        chk.append({"config": [n, L, tm, ro], "quoted": [v, t], "recorded": [r["val_auc"], r["test_auc"]],
                    "match": abs(r["val_auc"] - v) < 5e-5 and abs(r["test_auc"] - t) < 5e-5})
    out["quoted_pairs_check"] = chk
    per = {}
    for mid in ("vqc_v9_cnn", "lr_control_v9_cnn", "vqc_v9_eff", "lr_control_v9_eff"):
        g = {}
        for split in ("val", "test"):
            P = preds(root, mid, split)
            for s in sorted(P.scan_id.unique()):
                m = P.scan_id == s
                g[f"{split}:{s}"] = M.roc_auc(P.y[m], P.score[m])
        per[mid] = g
    out["per_group_auc_selected_models"] = per
    enc = {}
    for mid in ("vqc_v9_cnn", "vqc_v9_eff"):
        a = load_npz(root, "quantum", mid)
        enc[mid] = {"encoder": "standardise -> PCA -> min-max to [0, pi] on training rows, clipped",
                    "pca_component_range_training": (a["enc_hi"] - a["enc_lo"]).tolist()}
    out["encoder"] = enc
    occ = {}
    for name, (k, c) in POINT_SETS.items():
        ps = point_set(root, k, c)
        A = ps["A"]
        lo, hi = np.percentile(A, 5, axis=0), np.percentile(A, 95, axis=0)
        occ[name] = {"points": ps["n_unique"], "central_90pct_interval_per_component": np.c_[lo, hi].tolist(),
                     "central_90pct_width_as_share_of_pi": ((hi - lo) / np.pi).tolist(),
                     "note": "support vectors only (margin-biased); min-max scaling maps the training extremes to 0 and pi"}
    out["encoded_range_occupancy"] = occ
    out["one_layer_explanation"] = ("for one layer with <Z_0> readout the output equals prod_{q>=1} cos(x_q + w_q) exactly and does not depend "
                                    "on x_0 (tests/test_v10_circuits.py::test_v9_one_layer_z0_closed_form); a product of seven cosines is small and "
                                    "sign-unstable, consistent with the collapse (val 0.716, test 0.364)")
    out["not_executable"] = {"seed_sensitivity": NOT_EXECUTABLE, "scaling_variants_and_test_clipping_fraction": NOT_EXECUTABLE,
                             "per_group_behaviour_of_non_selected_configurations": "predictions were saved only for the selected configuration of each model type"}
    return out


# ------------------------------------------------------------------------------------------------ finding G
def finding_g(root, quick=False) -> dict:
    from src.quantum.v9 import VQCv9, noisy_expect_z, noisy_kernel
    out = {"claim": "one depolarizing rate for all gates; no readout/T1/T2; 16 trajectories; one random stream; non-monotone results; affine kernel noise; noisy test kernel only"}
    src = (root / "src" / "quantum" / "v9.py").read_text().splitlines()
    scr = (root / "scripts" / "v9_quantum.py").read_text().splitlines()

    def find(lines, needle):
        return [i + 1 for i, l in enumerate(lines) if needle in l]
    out["code_evidence"] = {
        "single_rate_all_gates": {"file": "src/quantum/v9.py", "lines": find(src, "hit = rng.random(len(Xr)) < p"),
                                  "detail": "the same p after every gate on each touched qubit; a CNOT gets two independent one-qubit errors"},
        "no_readout_t1_t2": {"file": "src/quantum/v9.py", "lines": find(src, "def noisy_expect_z") + find(src, "def noisy_kernel"),
                             "detail": "only Pauli X/Y/Z errors and binomial shots; no assignment error, amplitude or phase damping"},
        "trajectories_16": {"file": "scripts/v9_quantum.py", "lines": find(scr, "trajectories=16")},
        "single_random_stream": {"file": "scripts/v9_quantum.py", "lines": find(scr, "rng = np.random.default_rng(7)"),
                                 "detail": "one generator shared by every noise setting in sequence"},
        "affine_kernel_model": {"file": "src/quantum/v9.py", "lines": find(src, "Kn = lam * K + (1 - lam) / 2 ** n_qubits")},
        "noise_on_test_kernel_only": {"file": "scripts/v9_quantum.py", "lines": find(scr, "Kte = fidelity_kernel(m.states(A[te]), m.sv_states)")}}
    rec = {}
    for tag in ("cnn", "eff"):
        Q = load_json(root, f"results/v9/quantum/quantum_experiments_{tag}.json")
        rec[tag] = Q["noise_study_test_auc"]
        nm = []
        for model, rows in Q["noise_study_test_auc"].items():
            for shots in sorted({r["shots"] for r in rows}, key=lambda s: -1 if s is None else s):
                seq = [r["test_auc"] for r in rows if r["shots"] == shots]
                ps = [r["p_depolarizing"] for r in rows if r["shots"] == shots]
                inc = [(ps[i], ps[i + 1]) for i in range(len(seq) - 1) if seq[i + 1] > seq[i] + 1e-12]
                if inc:
                    nm.append({"model": model, "shots": shots, "auc_increases_between_p": inc})
            for pv in sorted({r["p_depolarizing"] for r in rows}):
                base = next(r["test_auc"] for r in rows if r["p_depolarizing"] == pv and r["shots"] is None)
                for r in rows:
                    if r["p_depolarizing"] == pv and r["shots"] is not None and r["test_auc"] > base + 1e-12:
                        nm.append({"model": model, "p": pv, "shots": r["shots"], "auc_without_shots": base,
                                   "auc_with_shots": r["test_auc"], "kind": "adding shot noise raised AUC"})
            const = [r for r in rows if r["shots"] is None]
            if len(const) > 1 and len({round(r["test_auc"], 12) for r in const}) == 1:
                nm.append({"model": model, "kind": "AUC identical for every p without shots (affine kernel map on the test rows only)"})
        rec[tag + "_non_monotone"] = nm
    out["recorded"] = rec
    # Monte-Carlo variability of V9's own noise routine at the trained VQC weights (synthetic inputs in the encoded range)
    a = load_npz(root, "quantum", "vqc_v9_cnn")
    m = VQCv9.from_arrays(a)
    X = np.random.default_rng(5).uniform(0, np.pi, (100, 8))
    ang = m._angles(X)
    clean = m.logits_from_expect(m.circ.forward(ang, m.w))
    mc = {}
    for traj, n_seeds in ((16, 20 if not quick else 5), (256, 6 if not quick else 2)):
        L = np.array([m.logits_from_expect(noisy_expect_z(m.circ, ang, m.w, p=0.005, trajectories=traj,
                                                          rng=np.random.default_rng(1000 + s))) for s in range(n_seeds)])
        sp = [spearman(L[i], L[j]) for i in range(n_seeds) for j in range(i + 1, n_seeds)]
        mc[f"trajectories_{traj}"] = {"seeds": n_seeds, "mean_sd_across_seeds": float(L.std(0).mean()),
                                      "sd_across_inputs_clean": float(clean.std()),
                                      "spearman_between_seeds_min_median": [float(min(sp)), float(np.median(sp))],
                                      "mean_shift_from_clean": float((L.mean(0) - clean).mean())}
    out["v9_noise_routine_monte_carlo_variability_p0.005"] = mc
    # kernel noise algebra with V9's own noisy_kernel on real stored points
    from sklearn.svm import SVC
    ps = point_set(root, *POINT_SETS["v9_eff_4q"])
    A, y = ps["A"], ps["y"]
    rng = np.random.default_rng(6)
    perm = rng.permutation(len(A))
    tr, te = perm[: len(A) * 2 // 3], perm[len(A) * 2 // 3:]
    S = kernel_states(A, 4, 0.05)
    K = KK.fidelity_kernel(S, S)
    from src.quantum.v9 import zz_depth
    gates = zz_depth(4, 2)["one_qubit_gates"] + zz_depth(4, 2)["two_qubit_gates"]
    model = SVC(kernel="precomputed", C=10.0, class_weight="balanced", tol=1e-10).fit(K[np.ix_(tr, tr)], y[tr])
    base = model.decision_function(K[np.ix_(te, tr)])
    alg = []
    for p in (0.001, 0.005, 0.01):
        Kn_te = noisy_kernel(K[np.ix_(te, tr)], 4, gates, p, None)
        Kn_tr = noisy_kernel(K[np.ix_(tr, tr)], 4, gates, p, None)
        s_clean_trained = model.decision_function(Kn_te)
        m2 = SVC(kernel="precomputed", C=10.0, class_weight="balanced", tol=1e-10).fit(Kn_tr, y[tr])
        s_noisy_trained = m2.decision_function(Kn_te)
        lam = (1 - p) ** (2 * gates)
        m3 = SVC(kernel="precomputed", C=10.0 * lam, class_weight="balanced", tol=1e-10).fit(K[np.ix_(tr, tr)], y[tr])
        alg.append({"p": p, "lambda": lam,
                    "clean_trained_noisy_test_auc_equals_clean": bool(np.isclose(M.roc_auc(y[te], s_clean_trained), M.roc_auc(y[te], base))),
                    "clean_trained_scores_are_affine_in_clean_scores": bool(np.allclose(s_clean_trained, lam * (base - model.intercept_[0]) + model.intercept_[0], atol=1e-6)),
                    "noisy_trained_equals_clean_trained_with_C_times_lambda": bool(np.allclose(s_noisy_trained, m3.decision_function(K[np.ix_(te, tr)]), atol=1e-4)),
                    "noisy_trained_auc": M.roc_auc(y[te], s_noisy_trained), "clean_auc": M.roc_auc(y[te], base)})
    out["affine_kernel_algebra_on_real_points"] = {"points": "v9_eff_4q support vectors, random 2/3 train split (illustration only)",
                                                   "results": alg}
    out["shot_noise_standard_error"] = {f"p={p}": {f"shots={N}": float(np.sqrt(p * (1 - p) / N)) for N in (1024, 8192)}
                                        for p in (0.5, 0.9, 0.99, 0.993)}
    return out


# ------------------------------------------------------------------------------------------------ timing
def timing(root) -> dict:
    from src.quantum.v9 import VQCv9
    out = {"recorded": load_json(root, "results/v9/quantum/inference_timing.json")}
    a = load_npz(root, "quantum", "vqc_v9_cnn")
    lr = load_npz(root, "classical", "lr_control_v9_cnn")
    m = VQCv9.from_arrays(a)
    from src.quantum.models import PCAEncoder
    enc = PCAEncoder.from_arrays(a)
    F = np.random.default_rng(0).normal(size=(622, a["enc_mu"].shape[0]))

    def best(fn, reps=7):
        fn()
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return min(ts)
    t_enc = best(lambda: enc.transform(F))
    A = enc.transform(F)
    t_state = best(lambda: m.circ.state(m._angles(A), m.w))
    psi = m.circ.state(m._angles(A), m.w)
    t_read = best(lambda: m.logits_from_expect(m.circ.expect_z(psi)))
    t_vqc = best(lambda: m.logits(enc.transform(F)))
    t_lr = best(lambda: enc.transform(F) @ lr["coef"].ravel() + float(np.ravel(lr["intercept"])[0]))
    t_lr_head = best(lambda: A @ lr["coef"].ravel())
    from src.v10.quantum.circuits import build_v9_vqc as b9
    c10 = b9(8, 2, trainable_map=True)
    wfull = np.concatenate([a["w"], a["a"], a["o"]])
    t_v10 = best(lambda: c10.forward(A, wfull))
    tracemalloc.start()
    m.logits(enc.transform(F))
    _, peak_vqc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    enc.transform(F) @ lr["coef"].ravel()
    _, peak_lr = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # training time per epoch on synthetic data of the V9 training size
    Xtr = np.random.default_rng(1).uniform(0, np.pi, (1543, 8))
    ytr = (np.random.default_rng(2).random(1543) < 0.54).astype(float)
    Xva = np.random.default_rng(3).uniform(0, np.pi, (284, 8))
    yva = (np.random.default_rng(4).random(284) < 0.6).astype(float)
    mm = VQCv9(8, 2, trainable_map=True, readout="allz")
    t0 = time.perf_counter()
    mm.fit(Xtr, ytr, Xva, yva, epochs=1, patience=99, log=lambda s: None)
    epoch = time.perf_counter() - t0
    n = 622
    out["remeasured_this_environment"] = {
        "patches": n, "encoding_ms_per_patch": 1000 * t_enc / n, "state_evolution_ms_per_patch": 1000 * t_state / n,
        "readout_ms_per_patch": 1000 * t_read / n, "vqc_head_total_ms_per_patch": 1000 * t_vqc / n,
        "lr_head_total_ms_per_patch": 1000 * t_lr / n, "lr_dot_product_only_ms_per_patch": 1000 * t_lr_head / n,
        "ratio_vqc_to_lr_including_encoding": t_vqc / t_lr, "v10_simulator_forward_ms_per_patch": 1000 * t_v10 / n,
        "peak_memory_bytes_vqc_head": int(peak_vqc), "peak_memory_bytes_lr_head": int(peak_lr),
        "vqc_one_training_epoch_seconds_synthetic_1543": epoch,
        "method": "best of 7 after one warm-up call; synthetic inputs of the V9 shapes; trained V9 weights",
        "recorded_vqc_train_seconds": 53.4,
        "implied_epochs_if_recorded_time_is_training": 53.4 / epoch}
    out["interpretation"] = ("the ratio compares two NumPy implementations on one CPU; it is not a quantum-versus-classical speed comparison "
                             "and says nothing about hardware execution time")
    return out


# ------------------------------------------------------------------------------------------------ statistics
def statistics_check(root) -> dict:
    C = load_json(root, "results/v9/comparison_v9.json")
    pt = C["paired_tests"]
    q = [p for p in pt if any(s in (p["model_a"] + p["model_b"]) for s in ("vqc", "qksvm", "hybrid", "qkernel"))]
    out = {"paired_tests_reported": len(pt), "involving_quantum_or_hybrid": len(q),
           "recorded_planned_comparisons_per_feature_set": [load_json(root, f"results/v9/quantum/quantum_experiments_{t}.json")["bonferroni_planned_comparisons"] for t in ("cnn", "eff")],
           "multiplicity_correction_applied_in_v9": False}
    re = []
    for a_mid, b_mid in (("vqc_v9_cnn", "lr_control_v9_cnn"), ("qksvm_v9_cnn", "rbf_control_v9_cnn"), ("hybrid_v9_cnn", "twin_v9_cnn"),
                         ("vqc_v9_eff", "lr_control_v9_eff"), ("qksvm_v9_eff", "rbf_control_v9_eff"), ("hybrid_v9_eff", "twin_v9_eff")):
        A, B = preds(root, a_mid), preds(root, b_mid)
        y, g = A.y.to_numpy(), A.scan_id.to_numpy()
        sf = ST.group_sign_flip_test(y, A.score.to_numpy(), B.score.to_numpy(), g)
        bs = ST.group_bootstrap_delta(y, A.score.to_numpy(), B.score.to_numpy(), g, n_boot=2000, seed=0)
        dl = ST.delong(y, A.score.to_numpy(), B.score.to_numpy())
        v9 = next((p for p in pt if p["model_a"] == a_mid and p["model_b"] == b_mid), None)
        re.append({"a": a_mid, "b": b_mid, "delta_auc": bs["delta_auc"],
                   "group_sign_flip": {k: sf[k] for k in ("p_two_sided", "groups_used", "groups_with_nonzero_difference", "smallest_attainable_p")},
                   "group_bootstrap_ci95": bs["ci"], "group_bootstrap_distinct_resamples_max": bs["distinct_group_resamples_max"],
                   "delong_patch_level_reference": {"p": dl["p_two_sided"], "assumption": dl.get("assumption")},
                   "v9_reported": {"ci95": v9["ci95"], "p": v9["p_two_sided"]} if v9 else None,
                   "per_group_delta": [{"group": r["group"], "delta": r["delta"]} for r in sf["groups"]]})
    out["reanalysis_quantum_vs_control_test_set"] = re
    out["conclusion"] = ("with three test groups an exact group-level test cannot reach p < 0.25, and when only G01 differs (G07 and G08 at "
                         "AUC 1.000 for both models) it cannot reach p < 1; patch-level tests treat 622 dependent patches as independent")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out_dir = root / "results" / "v10" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    steps = {"A_feature_saturation": lambda: finding_a(root), "B_kernel_geometry": lambda: finding_b(root, a.quick),
             "C_zz_phase": lambda: finding_c(root), "D_parameters_and_readout": lambda: finding_d(root, a.quick),
             "E_hybrid_vs_twin": lambda: finding_e(root), "F_vqc_selection": lambda: finding_f(root),
             "G_noise": lambda: finding_g(root, a.quick), "timing": lambda: timing(root), "statistics": lambda: statistics_check(root)}
    status = {}
    for name, fn in steps.items():
        t0 = time.time()
        try:
            res = fn()
            write_json(out_dir / f"{name}.json", res)
            status[name] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # recorded, never hidden
            import traceback
            status[name] = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()}
        print(name, status[name]["status"], status[name].get("seconds"), flush=True)
    inputs = ["results/v9/quantum/quantum_experiments_cnn.json", "results/v9/quantum/quantum_experiments_eff.json",
              "results/v9/quantum/inference_timing.json", "results/v9/comparison_v9.json"]
    inputs += [f"results/predictions/{m}.csv" for m in ("cnn25d_v9_a4_res_se_focal", "lr_control_v9_cnn", "vqc_v9_cnn", "qksvm_v9_cnn",
                                                        "rbf_control_v9_cnn", "hybrid_v9_cnn", "twin_v9_cnn", "vqc_v9_eff", "lr_control_v9_eff",
                                                        "qksvm_v9_eff", "rbf_control_v9_eff", "hybrid_v9_eff", "twin_v9_eff")]
    inputs += [f"artifacts/{f}/{m}/model.npz" for f, m in (("quantum", "vqc_v9_cnn"), ("quantum", "vqc_v9_eff"), ("quantum", "vqc_8q"),
                                                          ("quantum", "vqc_8q_ft"), ("quantum", "qksvm_v9_cnn"), ("quantum", "qksvm_v9_eff"),
                                                          ("quantum", "qkernel_svm_8q"), ("quantum", "qkernel_svm_8q_ft"),
                                                          ("classical", "rbf_control_v9_cnn"), ("classical", "rbf_control_v9_eff"),
                                                          ("classical", "pca8_svm_rbf"), ("classical", "pca8_svm_rbf_ft"),
                                                          ("hybrid", "hybrid_v9_cnn"), ("hybrid", "hybrid_v9_eff"))]
    write_json(out_dir / "run_manifest.json", run_manifest(root, inputs, {"global": "fixed per analysis (see code); no training on real data"},
                                                           {"steps": status, "total_seconds": round(time.time() - t_all, 1),
                                                            "command": "python scripts/v10_verify_v9_findings.py" + (" --quick" if a.quick else "")}))
    if any(s["status"] != "ok" for s in status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
