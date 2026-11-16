"""Quantum and classical kernels, and the kernel diagnostics used in V10.

Kernels
  fidelity_kernel     K_ij = |<psi(x_i)|psi(x_j)>|^2 from precomputed states
  projected_kernel    K_ij = exp(-gamma * sum_q ||rho_q(x_i) - rho_q(x_j)||_F^2), rho_q = one-qubit reduced state
                      (||rho - rho'||_F^2 = |r - r'|^2 / 2 for Bloch vectors r, r')
  rbf / polynomial / linear classical kernels on the same encoded inputs

Diagnostics
  centre(K) = H K H with H = I - 11^T / N (removes the kernel's mean in feature space; rank drops by at most one)
  alignment(K, y, centred) = <K', yy'^T>_F / (||K'||_F ||yy'^T||_F) with K' = HKH and yy' = H y y^T H when centred
      (centred alignment; y in {-1, +1}). The uncentred version (V9's formula) is kept for comparison: for a kernel whose
      entries are all close to 1 it is dominated by the constant part and by class imbalance, not by class structure.
  spectrum(K): eigenvalues, numerical rank, entropy effective rank exp(-sum p_i log p_i) with p = lambda / trace,
      participation ratio (sum lambda)^2 / sum lambda^2, largest-eigenvalue share (1.0 means rank one),
      condition number of K, and of the centred kernel on its non-null space.
  class_similarity: mean off-diagonal kernel value within each class and between classes.
  kernel_agreement(K1, K2): Pearson and Spearman correlation of off-diagonal entries and the centred alignment between
      two kernels. A high correlation does not make two kernels equivalent; the SVM depends on the full matrix.
"""
from __future__ import annotations

import numpy as np


def fidelity_kernel(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    return np.abs(np.conj(SA) @ SB.T) ** 2


def bloch_vectors(psi: np.ndarray, n: int) -> np.ndarray:
    """(B, n, 3) Bloch vectors (<X_q>, <Y_q>, <Z_q>) of the one-qubit reduced states."""
    B = psi.shape[0]
    out = np.zeros((B, n, 3))
    for q in range(n):
        v = psi.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        z = np.sum(np.conj(a0) * a1, axis=(1, 2))
        out[:, q, 0] = 2 * z.real
        out[:, q, 1] = 2 * z.imag
        out[:, q, 2] = np.sum(np.abs(a0) ** 2 - np.abs(a1) ** 2, axis=(1, 2))
    return out


def projected_kernel(RA: np.ndarray, RB: np.ndarray, gamma: float) -> np.ndarray:
    d2 = ((RA[:, None, :, :] - RB[None, :, :, :]) ** 2).sum((2, 3)) / 2.0
    return np.exp(-gamma * d2)


def rbf_kernel(XA, XB, gamma: float) -> np.ndarray:
    XA, XB = np.asarray(XA, float), np.asarray(XB, float)
    d2 = (XA ** 2).sum(1)[:, None] + (XB ** 2).sum(1)[None] - 2 * XA @ XB.T
    return np.exp(-gamma * np.maximum(d2, 0))


def polynomial_kernel(XA, XB, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> np.ndarray:
    return (gamma * np.asarray(XA, float) @ np.asarray(XB, float).T + coef0) ** degree


def linear_kernel(XA, XB) -> np.ndarray:
    return np.asarray(XA, float) @ np.asarray(XB, float).T


def centre(K: np.ndarray) -> np.ndarray:
    N = len(K)
    H = np.eye(N) - np.ones((N, N)) / N
    return H @ K @ H


def alignment(K: np.ndarray, y, centred: bool = True) -> float:
    t = np.where(np.asarray(y) == 1, 1.0, -1.0)
    Y = np.outer(t, t)
    if centred:
        K, Y = centre(K), centre(Y)
    den = np.linalg.norm(K) * np.linalg.norm(Y)
    return float((K * Y).sum() / den) if den > 0 else float("nan")


def kernel_agreement(K1: np.ndarray, K2: np.ndarray) -> dict:
    m = ~np.eye(len(K1), dtype=bool)
    a, b = K1[m], K2[m]

    def rank(v):
        r = np.empty(len(v))
        r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
        return r
    pear = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    spear = float(np.corrcoef(rank(a), rank(b))[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    C1, C2 = centre(K1), centre(K2)
    den = np.linalg.norm(C1) * np.linalg.norm(C2)
    return {"pearson_offdiag": pear, "spearman_offdiag": spear,
            "centred_alignment": float((C1 * C2).sum() / den) if den > 0 else float("nan")}


def spectrum(K: np.ndarray) -> dict:
    K = 0.5 * (K + K.T)
    ev = np.sort(np.linalg.eigvalsh(K))[::-1]
    tr = float(ev.sum())
    tol = max(K.shape) * np.finfo(float).eps * max(abs(ev[0]), 1e-300)
    pos = np.clip(ev, 0, None)
    p = pos / pos.sum() if pos.sum() > 0 else pos
    nz = p[p > 0]
    evc = np.sort(np.linalg.eigvalsh(0.5 * (centre(K) + centre(K).T)))[::-1]
    tolc = max(K.shape) * np.finfo(float).eps * max(abs(evc[0]), 1e-300)
    evc_pos = evc[evc > tolc]
    return {"n": int(len(K)), "eigenvalues_top10": ev[:10].tolist(), "min_eigenvalue": float(ev[-1]),
            "trace": tr, "numerical_rank": int((ev > tol).sum()),
            "effective_rank_entropy": float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else 0.0,
            "participation_ratio": float(pos.sum() ** 2 / (pos ** 2).sum()) if (pos ** 2).sum() > 0 else 0.0,
            "top_eigenvalue_share": float(ev[0] / tr) if tr > 0 else float("nan"),
            "condition_number": float(ev[0] / ev[-1]) if ev[-1] > tol else float("inf"),
            "centred_numerical_rank": int(len(evc_pos)),
            "centred_condition_number_nonnull": float(evc_pos[0] / evc_pos[-1]) if len(evc_pos) else float("nan")}


def offdiag_stats(K: np.ndarray) -> dict:
    m = ~np.eye(len(K), dtype=bool)
    v = K[m]
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v))}


def class_similarity(K: np.ndarray, y) -> dict:
    y = np.asarray(y).astype(int)
    m = ~np.eye(len(K), dtype=bool)
    same = (y[:, None] == y[None, :]) & m
    w1 = same & (y[:, None] == 1)
    w0 = same & (y[:, None] == 0)
    btw = (y[:, None] != y[None, :])
    out = {"within_class1_mean": float(K[w1].mean()) if w1.any() else float("nan"),
           "within_class0_mean": float(K[w0].mean()) if w0.any() else float("nan"),
           "between_mean": float(K[btw].mean()) if btw.any() else float("nan")}
    out["within_minus_between"] = float(K[same].mean() - K[btw].mean()) if same.any() and btw.any() else float("nan")
    return out


def full_report(K: np.ndarray, y=None) -> dict:
    rep = {"offdiag": offdiag_stats(K), "spectrum": spectrum(K),
           "symmetric": bool(np.allclose(K, K.T, atol=1e-10)),
           "diagonal_min": float(np.diag(K).min()), "diagonal_max": float(np.diag(K).max())}
    if y is not None:
        rep["alignment_centred"] = alignment(K, y, True)
        rep["alignment_uncentred"] = alignment(K, y, False)
        rep["class_similarity"] = class_similarity(K, y)
    return rep


def sample_kernel(K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool = True,
                  sample_diagonal: bool = False) -> np.ndarray:
    """Finite-shot estimate of a kernel whose entries are probabilities (compute-uncompute: P(0...0)).
    Each estimated entry is Binomial(shots, K_ij) / shots with standard error sqrt(K_ij (1 - K_ij) / shots).
    symmetric=True estimates i < j once and mirrors it (a square training kernel); the diagonal is kept exact unless
    sample_diagonal=True (with hardware noise the diagonal is below 1 and must be measured)."""
    K = np.clip(np.asarray(K, float), 0, 1)
    est = rng.binomial(shots, K) / shots
    if symmetric and K.shape[0] == K.shape[1]:
        iu = np.triu_indices(len(K), 1)
        out = np.zeros_like(K)
        out[iu] = est[iu]
        out = out + out.T
        np.fill_diagonal(out, np.diag(est) if sample_diagonal else np.diag(K))
        return out
    return est


def bernoulli_se(p, shots: int):
    p = np.asarray(p, float)
    return np.sqrt(p * (1 - p) / shots)


def clip_to_psd(K: np.ndarray) -> tuple[np.ndarray, dict]:
    """Nearest PSD matrix in Frobenius norm (negative eigenvalues set to zero) with a record of what was changed."""
    Ks = 0.5 * (K + K.T)
    ev, V = np.linalg.eigh(Ks)
    neg = ev < 0
    out = (V * np.clip(ev, 0, None)) @ V.T
    return out, {"negative_eigenvalues": int(neg.sum()), "most_negative": float(ev.min()),
                 "frobenius_change": float(np.linalg.norm(out - K))}
