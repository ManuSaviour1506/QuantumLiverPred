"""Trainability, expressibility and parameter-accounting diagnostics for variational circuits (classical simulation).

parameter_table      per trainable weight: max / mean |gradient| and the fraction of exactly-zero gradients over many
                     random (input, weight) draws, for every observable set. A weight with zero gradient in every draw is
                     reported as structurally dead only if dead_by_commutation also proves it; otherwise as
                     "zero in all draws" (possible coincidence).
jacobian_rank        numerical rank of d(observables)/d(weights) stacked over inputs, at random weights: the number of
                     independent parameter directions. declared - rank = dead + redundant directions.
gradient_variance    Var over random initialisations of d<O>/d theta_k, averaged over k, for given qubit counts and depths.
                     Small values at small n are not evidence of a barren plateau; only a consistent decay with n is.
expressibility_kl    KL divergence between the histogram of fidelities |<psi(theta)|psi(theta')>|^2 of random parameter
                     pairs and the Haar distribution P(F) = (N - 1)(1 - F)^(N - 2), N = 2**n (lower = closer to Haar).
meyer_wallach        Q = 2 (1 - mean_q tr(rho_q^2)), averaged over random parameters (0 = product states).
"""
from __future__ import annotations

import numpy as np

from src.v10.quantum.circuits import dead_by_commutation
from src.v10.quantum.kernels import bloch_vectors
from src.v10.quantum.simulator import Circuit


def _random_inputs(rng, B, n_features, low=0.0, high=np.pi):
    return rng.uniform(low, high, (B, n_features))


def parameter_table(circ: Circuit, masks, n_draws: int = 20, batch: int = 8, seed: int = 0,
                    weight_low: float = -np.pi, weight_high: float = np.pi, fixed_weights=None, inputs=None) -> dict:
    rng = np.random.default_rng(seed)
    P = circ.n_weights
    G = np.zeros((n_draws, len(masks), P))
    for d in range(n_draws):
        w = rng.uniform(weight_low, weight_high, P) if fixed_weights is None else np.asarray(fixed_weights, float)
        X = _random_inputs(rng, batch, circ.n_features) if inputs is None else inputs[rng.choice(len(inputs), batch, replace=False)]
        for m in range(len(masks)):
            dL = np.zeros((len(X), len(masks)))
            dL[:, m] = 1.0
            G[d, m] = circ.adjoint_grad(X, w, dL, masks)[0]
    A = np.abs(G)
    zero = A < 1e-12
    proof = dead_by_commutation(circ, masks)
    roles = circ.meta.get("roles", [f"w{k}" for k in range(P)])
    rows = []
    for k in range(P):
        all_zero = bool(zero[:, :, k].all())
        status = ("structurally dead (proved)" if k in proof["dead_weights"] else
                  "zero in all draws (not proved)" if all_zero else "active")
        rows.append({"index": k, "role": roles[k] if k < len(roles) else f"w{k}", "max_abs_grad": float(A[:, :, k].max()),
                     "mean_abs_grad": float(A[:, :, k].mean()), "zero_fraction": float(zero[:, :, k].mean()),
                     "status": status})
    return {"rows": rows, "proof": proof, "n_draws": n_draws, "batch": batch, "masks": [list(m) for m in masks],
            "dead_proved": proof["dead_weights"], "zero_all_draws": [r["index"] for r in rows if r["zero_fraction"] == 1.0]}


def jacobian_rank(circ: Circuit, masks, n_inputs: int = 64, n_points: int = 3, seed: int = 0, extra_weight_fn=None,
                  fixed_weights=None, inputs=None, rel_tol: float = 1e-8) -> dict:
    """Rank of the Jacobian of all observables over n_inputs rows with respect to the weights (max over n_points)."""
    rng = np.random.default_rng(seed)
    P = circ.n_weights
    ranks, svs = [], []
    for _ in range(n_points):
        w = rng.uniform(-np.pi, np.pi, P) if fixed_weights is None else np.asarray(fixed_weights, float)
        X = _random_inputs(rng, n_inputs, circ.n_features) if inputs is None else inputs[rng.choice(len(inputs), min(n_inputs, len(inputs)), replace=False)]
        J = []
        for b in range(len(X)):
            for m in range(len(masks)):
                dL = np.zeros((1, len(masks)))
                dL[0, m] = 1.0
                J.append(circ.adjoint_grad(X[b:b + 1], w, dL, masks)[0])
        J = np.array(J)
        s = np.linalg.svd(J, compute_uv=False)
        r = int((s > rel_tol * s[0]).sum()) if s[0] > 0 else 0
        ranks.append(r)
        svs.append(s.tolist())
    return {"declared": P, "rank_max": max(ranks), "ranks": ranks, "singular_values_first_point": svs[0]}


def gradient_variance(builder, n_list, depth_list, masks_fn, n_init: int = 200, seed: int = 0, n_inputs: int = 1,
                      param_index: str = "all") -> list[dict]:
    """For each (n, depth): Var over n_init random weight draws of d<O>/d w_k, averaged over k (and the first weight)."""
    out = []
    for n in n_list:
        for L in depth_list:
            circ = builder(n, L)
            masks = masks_fn(n)
            rng = np.random.default_rng([seed, n, L])
            G = np.zeros((n_init, circ.n_weights))
            for i in range(n_init):
                w = rng.uniform(-np.pi, np.pi, circ.n_weights)
                X = rng.uniform(0, np.pi, (n_inputs, circ.n_features))
                dL = np.ones((n_inputs, len(masks))) / (n_inputs * len(masks))
                G[i] = circ.adjoint_grad(X, w, dL, masks)[0]
            var = G.var(0)
            live = var > 1e-24
            out.append({"qubits": n, "layers": L, "weights": circ.n_weights, "mean_var_all": float(var.mean()),
                        "mean_var_live": float(var[live].mean()) if live.any() else 0.0,
                        "var_first_weight": float(var[0]), "zero_variance_weights": int((~live).sum()),
                        "mean_grad_norm": float(np.linalg.norm(G, axis=1).mean()), "n_init": n_init})
    return out


def haar_fidelity_pdf(F, N):
    return (N - 1) * (1 - F) ** (N - 2)


def expressibility_kl(state_fn, n_qubits: int, n_weights: int, n_pairs: int = 2000, bins: int = 75, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-np.pi, np.pi, (n_pairs, n_weights))
    W2 = rng.uniform(-np.pi, np.pi, (n_pairs, n_weights))
    F = np.array([np.abs(np.vdot(state_fn(a), state_fn(b))) ** 2 for a, b in zip(W1, W2)])
    N = 2 ** n_qubits
    edges = np.linspace(0, 1, bins + 1)
    p, _ = np.histogram(F, edges)
    p = p / p.sum()
    q = np.array([(1 - a) ** (N - 1) - (1 - b) ** (N - 1) for a, b in zip(edges[:-1], edges[1:])])  # Haar bin masses
    q = q / q.sum()
    m = p > 0
    kl = float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))))
    return {"kl_to_haar": kl, "mean_fidelity": float(F.mean()), "haar_mean_fidelity": 1.0 / N, "n_pairs": n_pairs, "bins": bins}


def meyer_wallach(psi: np.ndarray, n: int) -> np.ndarray:
    r = bloch_vectors(psi, n)
    purity = 0.5 * (1 + (r ** 2).sum(2))  # tr(rho_q^2) = (1 + |r|^2) / 2
    return 2 * (1 - purity.mean(1))


def fubini_study_diag(state_fn, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Feature-map sensitivity: g_qq(x) ~ (1 - |<psi(x)|psi(x + eps e_q)>|^2) / eps^2 for every row and feature."""
    S0 = state_fn(X)
    out = np.zeros(X.shape)
    for q in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, q] += eps
        F = np.abs(np.sum(np.conj(S0) * state_fn(Xp), 1)) ** 2
        out[:, q] = (1 - F) / eps ** 2
    return out
