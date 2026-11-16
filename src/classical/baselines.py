"""Classical machine-learning baselines on handcrafted (radiomics-style) patch features.

Features (44 per 96x96 patch, computed identically at training and inference time):
  first-order  mean, SD, skewness, kurtosis, 10/25/50/75/90th percentiles, histogram entropy, fraction of pixels in
               5 grey-level bands (fat/air, low, mid, high, very bright)
  texture      GLCM contrast / dissimilarity / homogeneity / energy / correlation / ASM at distances 1 and 3
               (averaged over 0, 45, 90, 135 degrees), uniform LBP (P=8, R=1) histogram (10 bins)
  edges        Sobel gradient magnitude mean, SD and 90th percentile
Models: RBF-SVM, Random Forest, L2 logistic regression (scikit-learn for training). Hyper-parameters are chosen on
the validation split only. Every fitted model is exported to plain arrays and evaluated by a NumPy predictor, so no
pickle is ever loaded and inference does not depend on the scikit-learn version.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

FEATURE_NAMES = (["mean", "sd", "skew", "kurtosis", "p10", "p25", "p50", "p75", "p90", "entropy"] +
                 [f"band_{i}" for i in range(5)] +
                 [f"glcm_{p}_d{d}" for d in (1, 3) for p in ("contrast", "dissimilarity", "homogeneity", "energy",
                                                            "correlation", "ASM")] +
                 [f"lbp_{i}" for i in range(10)] + ["grad_mean", "grad_sd", "grad_p90"])
BANDS = (0, 20, 80, 150, 220, 256)


def patch_features(p: np.ndarray) -> np.ndarray:
    g = np.asarray(p, np.uint8)
    x = g.astype(np.float64)
    hist = np.bincount(g.ravel(), minlength=256) / g.size
    nz = hist[hist > 0]
    f = [x.mean(), x.std(), float(stats.skew(x.ravel())) if x.std() > 0 else 0.0,
         float(stats.kurtosis(x.ravel())) if x.std() > 0 else 0.0, *np.percentile(x, [10, 25, 50, 75, 90]),
         float(-(nz * np.log2(nz)).sum())]
    f += [float(((g >= a) & (g < b)).mean()) for a, b in zip(BANDS[:-1], BANDS[1:])]
    q = (g // 8).astype(np.uint8)  # 32 grey levels for the GLCM
    glcm = graycomatrix(q, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    for di in range(2):
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
            v = graycoprops(glcm, prop)[di]
            f.append(float(np.nan_to_num(v.mean())))
    lbp = local_binary_pattern(g, 8, 1, method="uniform")
    f += list(np.bincount(lbp.astype(int).ravel(), minlength=10)[:10] / lbp.size)
    gm = np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1))
    f += [gm.mean(), gm.std(), float(np.percentile(gm, 90))]
    return np.asarray(f, np.float64)


def features(patches: np.ndarray) -> np.ndarray:
    return np.stack([patch_features(p) for p in patches])


# --------------------------------------------------------------------------------------------- export / predict
def export_lr(lr, mu, sd) -> dict:
    return {"kind": np.array("logreg"), "mu": mu, "sd": sd, "coef": lr.coef_.ravel(), "intercept": lr.intercept_}


def export_svm(svm, mu, sd) -> dict:
    gamma = svm._gamma if hasattr(svm, "_gamma") else svm.gamma
    return {"kind": np.array("svm_rbf"), "mu": mu, "sd": sd, "sv": svm.support_vectors_, "dual": svm.dual_coef_.ravel(),
            "intercept": svm.intercept_, "gamma": np.array(float(gamma))}


def export_rf(rf, mu, sd) -> dict:
    trees = [e.tree_ for e in rf.estimators_]
    n = max(t.node_count for t in trees)
    L = np.full((len(trees), n), -1, np.int32)
    R, F = L.copy(), L.copy()
    T = np.zeros((len(trees), n), np.float64)
    V = np.zeros((len(trees), n), np.float64)
    for i, t in enumerate(trees):
        k = t.node_count
        L[i, :k], R[i, :k], F[i, :k], T[i, :k] = t.children_left, t.children_right, t.feature, t.threshold
        val = t.value[:, 0, :]
        V[i, :k] = val[:, 1] / np.maximum(val.sum(1), 1e-12)
    return {"kind": np.array("random_forest"), "mu": mu, "sd": sd, "left": L, "right": R, "feature": F, "threshold": T,
            "value": V}


def predict_arrays(a: dict, X: np.ndarray) -> np.ndarray:
    """Decision score (higher = tumour) from exported arrays."""
    kind = str(a["kind"])
    Z = (np.asarray(X, float) - a["mu"]) / a["sd"]
    if kind == "logreg":
        return Z @ a["coef"] + float(a["intercept"][0])
    if kind == "svm_rbf":
        d2 = (Z ** 2).sum(1)[:, None] + (a["sv"] ** 2).sum(1)[None] - 2 * Z @ a["sv"].T
        return np.exp(-float(a["gamma"]) * np.maximum(d2, 0)) @ a["dual"] + float(a["intercept"][0])
    if kind == "random_forest":
        # sklearn trees compare float32 features against float64 thresholds
        Zf = Z.astype(np.float32).astype(np.float64)
        out = np.zeros(len(Z))
        for t in range(a["left"].shape[0]):
            node = np.zeros(len(Z), np.int64)
            for _ in range(64):
                leaf = a["left"][t, node] == -1
                if leaf.all():
                    break
                f = a["feature"][t, node]
                go_left = Zf[np.arange(len(Z)), np.maximum(f, 0)] <= a["threshold"][t, node]
                nxt = np.where(go_left, a["left"][t, node], a["right"][t, node])
                node = np.where(leaf, node, nxt)
            out += a["value"][t, node]
        return out / a["left"].shape[0]
    raise ValueError(f"unknown baseline kind {kind}")


# --------------------------------------------------------------------------------------------- V9 normalisation
def normalise_patches(patches: np.ndarray, norm: str = "none", smooth: float = 0.0) -> np.ndarray:
    """V9 radiomics input normalisation (uses only the patch itself, so it applies identically at inference).

    norm 'patch_z': z-score the patch and map +/-3 SD to 0..255 (removes scan-level brightness/contrast shifts such as
    the darker, more variable liver of test scan G01); smooth > 0: Gaussian blur with that sigma before the features
    (reduces sensitivity to image noise, the largest quality difference found for G01)."""
    import cv2
    out = []
    for p in patches:
        a = p.astype(np.float32)
        if smooth > 0:
            a = cv2.GaussianBlur(a, (0, 0), smooth)
        if norm == "patch_z":
            a = np.clip((a - a.mean()) / (a.std() + 1e-6) / 6.0 + 0.5, 0, 1) * 255.0
        out.append(np.clip(a, 0, 255).round().astype(np.uint8))
    return np.stack(out)
