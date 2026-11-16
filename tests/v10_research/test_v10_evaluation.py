"""V10 evaluation: metrics against scikit-learn, group statistics, grouped CV isolation, fold-only preprocessing."""
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True

from src.v10.evaluation import metrics as M  # noqa: E402
from src.v10.evaluation import statistics as S  # noqa: E402
from src.v10.evaluation.grouped_cv import FoldPreprocessor, assert_disjoint, inner_folds, merge_related, outer_folds  # noqa: E402


def test_metrics_match_scikit_learn():
    from sklearn import metrics as sk
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)          # ties on purpose
    p = 1 / (1 + np.exp(-s))
    assert np.isclose(M.roc_auc(y, s), sk.roc_auc_score(y, s))
    assert np.isclose(M.average_precision(y, s), sk.average_precision_score(y, s))
    assert np.isclose(M.log_loss(y, p), sk.log_loss(y, p))
    t = M.threshold_metrics(y, p, 0.6)
    pred = (p >= 0.6).astype(int)
    assert np.isclose(t["mcc"], sk.matthews_corrcoef(y, pred))
    assert np.isclose(t["f1"], sk.f1_score(y, pred)) and np.isclose(t["balanced_accuracy"], sk.balanced_accuracy_score(y, pred))
    tn, fp, fn, tp = sk.confusion_matrix(y, pred).ravel()
    assert (t["tp"], t["fp"], t["tn"], t["fn"]) == (tp, fp, tn, fn)
    c = M.auc_pair_counts(y, s)
    assert np.isclose((c["correct"] + 0.5 * c["tied"]) / c["pairs"], M.roc_auc(y, s))


def test_sign_flip_test_with_three_groups():
    rng = np.random.default_rng(1)
    groups = np.repeat(["G1", "G2", "G3"], 50)
    y = rng.integers(0, 2, 150)
    s1 = y + rng.normal(0, 0.5, 150)
    s2 = y + rng.normal(0, 0.8, 150)
    r = S.group_sign_flip_test(y, s1, s2, groups)
    assert r["smallest_attainable_p"] == 0.25 and r["p_two_sided"] >= 0.25
    same = S.group_sign_flip_test(y, s1, s1, groups)
    assert same["statistic"] == 0 and same["p_two_sided"] == 1.0


def test_bootstrap_and_tost_and_holm():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(8), 40)
    y = rng.integers(0, 2, 320)
    s = y + rng.normal(0, 1, 320)
    b = S.group_bootstrap_delta(y, s, s, groups, n_boot=200, seed=0)
    assert b["delta_auc"] == 0 and b["ci"] == [0.0, 0.0]
    assert S.tost_from_ci([-0.01, 0.015], 0.02)["equivalent"] and not S.tost_from_ci([-0.03, 0.01], 0.02)["equivalent"]
    h = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(h["a"], 0.03) and np.isclose(h["c"], 0.06) and np.isclose(h["b"], 0.06)


def test_delong_basic():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    s1 = y + rng.normal(0, 0.7, 200)
    s2 = y + rng.normal(0, 0.7, 200)
    d = S.delong(y, s1, s2)
    assert np.isclose(d["delta_auc"], M.roc_auc(y, s1) - M.roc_auc(y, s2)) and 0 <= d["p_two_sided"] <= 1


def test_grouped_cv_isolation():
    groups = np.repeat(["G01", "G02", "G03", "G04", "G05"], 10)
    units_map = merge_related(sorted(set(groups)), [("G02", "G03")])
    assert units_map["G02"] == units_map["G03"] == "G02+G03"
    units = np.array([units_map[g] for g in groups])
    folds = outer_folds(units)
    assert len(folds) == 4
    for name, tr, te in folds:
        assert_disjoint(units, tr, te)
        for b, fit, va in inner_folds(units, tr):
            assert_disjoint(units, fit, va, te)
            assert not set(units[va]) & set(units[te])
    with pytest.raises(AssertionError):
        assert_disjoint(units, np.arange(0, 15), np.arange(12, 20))


def test_fold_preprocessor_uses_training_rows_only():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 6))
    tr = np.arange(60)
    for scaler in ("standard", "robust", "minmax", "tanh", "arctan"):
        a = FoldPreprocessor(scaler, pca=3, angle_max=np.pi).fit(X, tr)
        X2 = X.copy()
        X2[60:] += 100.0                      # change only the held-out rows
        b = FoldPreprocessor(scaler, pca=3, angle_max=np.pi).fit(X2, tr)
        assert np.allclose(a.transform(X[:60]), b.transform(X2[:60]))
        out = a.transform(X)
        assert out.min() >= 0 and out.max() <= np.pi
    mm = FoldPreprocessor("minmax", pca=3).fit(X, tr)
    assert mm.fraction_clipped(X[:60]) == 0.0 and mm.fraction_clipped(X2 := X + 5.0) > 0
