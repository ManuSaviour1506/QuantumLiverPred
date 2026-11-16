"""V7 guards for the V6 audit's threshold and calibration failures (Problem C).

* SafeCalibrator: Platt scaling that RAISES CalibrationCollapsed when the training scores are constant or rank the
  labels inversely (slope <= 0). V6's Calibrator silently replaced such cases with a constant probability; the
  deployed ResNet-50 then output p = 0.39374 for every input. That replacement no longer exists in V7 code.
* scan_standardise: label-free z-scoring of scores within each scan (removes per-patient score shifts and the scale
  difference between inner models and a refit model).
* check_predictions: detects constant scores and all-positive / all-negative predictions, overall and per scan.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibrationCollapsed(RuntimeError):
    """Raised instead of returning a constant probability."""


class ConstantPredictions(RuntimeError):
    """Raised when a model's scores carry no ranking information."""


class SafeCalibrator:
    """Platt scaling on standardised scores. fit() raises CalibrationCollapsed when it cannot calibrate honestly."""

    def __init__(self, min_slope: float = 1e-3):
        self.min_slope = min_slope

    def fit(self, s, y):
        s, y = np.asarray(s, float), np.asarray(y, int)
        if not np.all(np.isfinite(s)):
            raise CalibrationCollapsed("non-finite training scores")
        if len(np.unique(y)) < 2:
            raise CalibrationCollapsed("training labels contain one class only")
        self.mu, self.sd = float(s.mean()), float(s.std())
        if self.sd < 1e-9:
            raise CalibrationCollapsed("training scores are constant")
        z = ((s - self.mu) / self.sd)[:, None]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if self.a < self.min_slope:
            raise CalibrationCollapsed(f"calibration slope {self.a:.4g} <= {self.min_slope}: training scores rank the "
                                       "labels inversely or not at all")
        return self

    def __call__(self, s):
        z = (np.asarray(s, float) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sd": self.sd, "a": self.a, "b": self.b, "min_slope": self.min_slope}

    @classmethod
    def from_dict(cls, d: dict) -> "SafeCalibrator":
        c = cls(d.get("min_slope", 1e-3))
        c.mu, c.sd, c.a, c.b = float(d["mu"]), float(d["sd"]), float(d["a"]), float(d["b"])
        if c.sd < 1e-9 or c.a < c.min_slope:
            raise CalibrationCollapsed(f"stored calibrator is collapsed (sd={c.sd}, slope={c.a})")
        return c


def scan_standardise(scores, scans, min_slices: int = 5) -> np.ndarray:
    """z-score scores within each scan using only that scan's scores (no labels). Scans with fewer than `min_slices`
    slices or constant scores raise ValueError: a per-scan scale cannot be estimated from them."""
    scores, scans = np.asarray(scores, float), np.asarray(scans)
    out = np.empty_like(scores)
    for u in np.unique(scans):
        m = scans == u
        if m.sum() < min_slices:
            raise ValueError(f"scan {u!r} has {int(m.sum())} slices; scan standardisation needs >= {min_slices}")
        sd = scores[m].std()
        if sd < 1e-9:
            raise ConstantPredictions(f"scan {u!r}: constant scores")
        out[m] = (scores[m] - scores[m].mean()) / sd
    return out


def check_predictions(scores, pred=None, scans=None, one_class_frac: float = 0.95) -> dict:
    """Flags constant scores and single-class predictions, overall and per scan."""
    scores = np.asarray(scores, float)
    out = {"n": int(len(scores)), "constant_scores": bool(np.nanstd(scores) < 1e-9),
           "non_finite": int((~np.isfinite(scores)).sum())}
    if pred is not None:
        pred = np.asarray(pred)
        out["all_positive"] = bool((pred == 1).all())
        out["all_negative"] = bool((pred == 0).all())
        if scans is not None:
            scans = np.asarray(scans)
            per = {}
            for u in np.unique(scans):
                f = float((pred[scans == u] == 1).mean())
                per[str(u)] = f
            out["scans_one_class"] = [u for u, f in per.items() if f >= one_class_frac or f <= 1 - one_class_frac]
            out["positive_fraction_per_scan"] = per
    out["ok"] = not (out["constant_scores"] or out["non_finite"] or out.get("all_positive") or out.get("all_negative"))
    return out
