"""Classical model zoo with equal-size, log-spaced search grids (every model gets a comparable tuning budget).

Every estimator is a scikit-learn Pipeline (StandardScaler -> classifier) so that the fitted preprocessing is saved
together with the model. XGBoost is used only if it is installed (it is not in the build environment).
"""
from __future__ import annotations

import importlib.util
import itertools

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


GRIDS = {
    "logreg": _grid(C=[0.01, 0.1, 1.0, 10.0]),
    "decision_tree": _grid(max_depth=[3, 6], min_samples_leaf=[5, 20]),
    "random_forest": _grid(max_depth=[None, 8], min_samples_leaf=[1, 5]),
    "extra_trees": _grid(max_depth=[None, 8], min_samples_leaf=[1, 5]),
    "svm_rbf": _grid(C=[0.3, 3.0], gamma=["scale", 0.01]),
    "knn": _grid(n_neighbors=[7, 21], weights=["uniform", "distance"]),
    "hist_gb": _grid(learning_rate=[0.05, 0.15], max_depth=[3, None]),
    "mlp_small": _grid(alpha=[1e-3, 1e-1, 1.0], hidden=[8]),
    "xgboost": _grid(max_depth=[3, 6], learning_rate=[0.05, 0.15]),
}
STOCHASTIC = {"random_forest", "extra_trees", "mlp_small", "xgboost"}
NAMES = {"logreg": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
         "extra_trees": "Extra Trees", "svm_rbf": "SVM (RBF kernel)", "knn": "K-Nearest Neighbours",
         "hist_gb": "Histogram Gradient Boosting", "mlp_small": "MLP (capacity-matched)", "xgboost": "XGBoost"}


def available() -> list[str]:
    out = [k for k in GRIDS if k != "xgboost"]
    if importlib.util.find_spec("xgboost") is not None:
        out.append("xgboost")
    return out


def make(kind: str, params: dict, seed: int = 0) -> Pipeline:
    p = dict(params)
    if kind == "logreg":
        clf = LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000)
    elif kind == "decision_tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **p)
    elif kind == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=2, random_state=seed, **p)
    elif kind == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", n_jobs=2, random_state=seed, **p)
    elif kind == "svm_rbf":
        clf = SVC(kernel="rbf", class_weight="balanced", **p)
    elif kind == "knn":
        clf = KNeighborsClassifier(**p)
    elif kind == "hist_gb":
        clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed, **p)
    elif kind == "mlp_small":
        clf = MLPClassifier(hidden_layer_sizes=(p["hidden"],), alpha=p["alpha"], max_iter=2000, early_stopping=True,
                            random_state=seed)
    elif kind == "xgboost":
        from xgboost import XGBClassifier  # type: ignore
        clf = XGBClassifier(n_estimators=300, random_state=seed, eval_metric="logloss", **p)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def decision(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Continuous score (higher = tumour): decision_function when available, else the logit of predict_proba."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "decision_function") and not isinstance(clf, (MLPClassifier, HistGradientBoostingClassifier)):
        return np.asarray(model.decision_function(X), float)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def n_parameters(model: Pipeline) -> int | None:
    clf = model.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return int(clf.coef_.size + clf.intercept_.size)
    if isinstance(clf, MLPClassifier):
        return int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    if isinstance(clf, SVC):
        return int(clf.support_vectors_.size + clf.dual_coef_.size + 1)
    return None


def feature_importance(model: Pipeline, names: list[str] | None = None, top: int = 15) -> list[dict]:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        v = np.asarray(clf.feature_importances_, float)
        kind = "impurity importance"
    elif isinstance(clf, LogisticRegression):
        v = np.abs(clf.coef_.ravel())
        kind = "|standardised coefficient|"
    else:
        return []
    names = names or [f"f{i}" for i in range(len(v))]
    order = np.argsort(-v)[:top]
    return [{"feature": names[i], "importance": float(v[i]), "kind": kind} for i in order]
