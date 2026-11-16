"""Grouped cross-validation with leakage guards.

Units: groups that may be related (for example two reconstructed scan groups that could belong to one patient) are merged
into one unit with merge_related(); a unit is never split between training and test.

outer_folds       leave-one-unit-out (or k-fold over units)
inner_folds       grouped folds over the outer-training units only
FoldPreprocessor  scaling and PCA fitted on training rows only; transform() never refits. Scalers: standard, robust
                  (median / IQR), minmax (V9 style, with clipping), tanh / arctan of the standardised values (bounded,
                  smooth); output optionally mapped to an angle range [0, angle_max]
"""
from __future__ import annotations

import numpy as np


def merge_related(groups, relations) -> dict:
    """Union-find over group labels. relations: iterable of (g1, g2). Returns {group: unit label}."""
    parent = {g: g for g in groups}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in relations:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    units = {}
    for g in groups:
        r = find(g)
        units.setdefault(r, []).append(g)
    return {g: "+".join(sorted(units[find(g)])) for g in groups}


def outer_folds(units: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    units = np.asarray(units)
    out = []
    for u in np.unique(units):
        te = np.where(units == u)[0]
        tr = np.where(units != u)[0]
        out.append((str(u), tr, te))
    return out


def inner_folds(units: np.ndarray, train_idx: np.ndarray, k: int | None = None, seed: int = 0):
    """Grouped folds within train_idx. k=None -> leave-one-unit-out."""
    units = np.asarray(units)
    tr_units = np.unique(units[train_idx])
    if k is None or k >= len(tr_units):
        blocks = [[u] for u in tr_units]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(tr_units)
        blocks = [list(perm[i::k]) for i in range(k)]
    out = []
    for b in blocks:
        va = train_idx[np.isin(units[train_idx], b)]
        fit = train_idx[~np.isin(units[train_idx], b)]
        out.append((b, fit, va))
    return out


def assert_disjoint(units: np.ndarray, *index_sets) -> None:
    units = np.asarray(units)
    seen = []
    for ix in index_sets:
        s = set(units[ix].tolist())
        for t in seen:
            if s & t:
                raise AssertionError(f"unit leakage between partitions: {sorted(s & t)}")
        seen.append(s)


class FoldPreprocessor:
    def __init__(self, scaler: str = "standard", pca: int | None = None, angle_max: float | None = None, clip: bool = True):
        self.scaler, self.pca, self.angle_max, self.clip = scaler, pca, angle_max, clip
        self.fitted_rows = None

    def fit(self, X, rows=None):
        X = np.asarray(X, float)
        Xt = X if rows is None else X[rows]
        self.fitted_rows = None if rows is None else np.asarray(rows).copy()
        self.mu, self.sd = Xt.mean(0), Xt.std(0) + 1e-12
        Z = (Xt - self.mu) / self.sd
        if self.pca:
            self.pmean = Z.mean(0)
            _, S, Vt = np.linalg.svd(Z - self.pmean, full_matrices=False)
            self.comp = Vt[: self.pca]
            self.explained = float((S[: self.pca] ** 2).sum() / (S ** 2).sum())
            Z = (Z - self.pmean) @ self.comp.T
        self._fit_scaler(Z)
        return self

    def _fit_scaler(self, Z):
        if self.scaler == "standard":
            self.a, self.b = Z.mean(0), Z.std(0) + 1e-12
        elif self.scaler == "robust":
            q1, med, q3 = np.percentile(Z, [25, 50, 75], axis=0)
            self.a, self.b = med, (q3 - q1) + 1e-12
        elif self.scaler == "minmax":
            self.a, self.b = Z.min(0), Z.max(0) - Z.min(0) + 1e-12
        elif self.scaler in ("tanh", "arctan"):
            self.a, self.b = Z.mean(0), Z.std(0) + 1e-12
        else:
            raise ValueError(self.scaler)

    def transform(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        if self.pca:
            Z = (Z - self.pmean) @ self.comp.T
        U = (Z - self.a) / self.b
        if self.scaler == "minmax":
            U = np.clip(U, 0, 1) if self.clip else U
            unit = U
        elif self.scaler == "tanh":
            unit = 0.5 * (1 + np.tanh(U))
        elif self.scaler == "arctan":
            unit = 0.5 + np.arctan(U) / np.pi
        else:
            unit = None
        if self.angle_max is not None:
            if unit is None:
                unit = 0.5 * (1 + np.tanh(U / 2))  # standard / robust -> bounded map before angles
            return unit * self.angle_max
        return unit if unit is not None else U

    def fraction_clipped(self, X) -> float:
        """Share of values outside the training range (only meaningful for minmax)."""
        Z = (np.asarray(X, float) - self.mu) / self.sd
        if self.pca:
            Z = (Z - self.pmean) @ self.comp.T
        U = (Z - self.a) / self.b
        return float(((U < 0) | (U > 1)).mean()) if self.scaler == "minmax" else 0.0
