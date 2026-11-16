"""Out-of-domain detector on raw DenseNet-121 INTERMEDIATE-stage pooled features (V6 audit Problem D).

Design choice (V7_EXPERIMENT_LOG.md): final-GAP features missed most CT slices and CT faked as three phases; the
intermediate stage (more low-level image statistics) caught all of them at the same false-alarm rate. The design was
chosen with 30 CT slices from the excluded scans.zip as negative controls; tests/test_v7.py checks OTHER CT slices.

Fitted on the 767 training slices: standardise -> PCA(k, whitened). Two scores per slice:
  d2       squared Mahalanobis distance inside the PCA subspace,
  residual reconstruction error outside it (catches inputs unlike anything in training).
Thresholds come from leave-one-patient-out: the detector is refitted without each patient and that patient's slices
are scored, so the thresholds reflect how far UNSEEN in-domain patients fall (99th percentile; ~2% of unseen
real slices are flagged and withheld).
A slice is out-of-domain if either score exceeds its threshold. No label is used.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class OODDetector:
    def __init__(self, k: int = 32, q: float = 99.0, margin: float = 1.0, seed: int = 42):
        self.k, self.q, self.margin, self.seed = k, q, margin, seed

    def _fit_core(self, X):
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-8] = 1.0
        Z = (X - mu) / sd
        p = PCA(self.k, random_state=self.seed).fit(Z)
        return mu, sd, p.mean_, p.components_, np.sqrt(p.explained_variance_)

    @staticmethod
    def _scores(X, mu, sd, pm, comp, ev):
        Z = (np.asarray(X, float) - mu) / sd
        P = (Z - pm) @ comp.T
        d2 = ((P / ev) ** 2).sum(1)
        res = (((Z - pm) - P @ comp) ** 2).sum(1) / Z.shape[1]
        return d2, res

    def fit(self, X, groups):
        X, groups = np.asarray(X, float), np.asarray(groups)
        held_d2, held_res = [], []
        for g in np.unique(groups):
            core = self._fit_core(X[groups != g])
            d2, res = self._scores(X[groups == g], *core)
            held_d2.append(d2)
            held_res.append(res)
        self.thr_d2 = float(np.percentile(np.concatenate(held_d2), self.q) * self.margin)
        self.thr_res = float(np.percentile(np.concatenate(held_res), self.q) * self.margin)
        self.mu, self.sd, self.pm, self.comp, self.ev = self._fit_core(X)
        return self

    def score(self, X) -> dict:
        d2, res = self._scores(X, self.mu, self.sd, self.pm, self.comp, self.ev)
        return {"d2": d2, "residual": res, "d2_ratio": d2 / self.thr_d2, "residual_ratio": res / self.thr_res,
                "out_of_domain": (d2 > self.thr_d2) | (res > self.thr_res)}

    def to_arrays(self) -> dict:
        return {"ood_mu": self.mu, "ood_sd": self.sd, "ood_pm": self.pm, "ood_comp": self.comp, "ood_ev": self.ev,
                "ood_thr": np.array([self.thr_d2, self.thr_res]), "ood_params": np.array([self.k, self.q, self.margin, self.seed], float)}

    @classmethod
    def from_arrays(cls, d) -> "OODDetector":
        k, q, m, s = d["ood_params"]
        o = cls(int(k), float(q), float(m), int(s))
        o.mu, o.sd, o.pm, o.comp, o.ev = (np.asarray(d[x]) for x in ("ood_mu", "ood_sd", "ood_pm", "ood_comp", "ood_ev"))
        o.thr_d2, o.thr_res = (float(v) for v in d["ood_thr"])
        return o
