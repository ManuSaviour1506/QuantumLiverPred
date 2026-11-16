#!/usr/bin/env python3
"""Classical models (scikit-learn pipelines) on patch features — train deployable V10 models on all units and save artifacts/v10/<model_id>/.

    python scripts/train_classical.py --dataset /path/to/liver_ct_lits_verified_v8 [--features radiomics|imagenet_gap]
                            [--models logreg,decision_tree,random_forest,extra_trees,svm_rbf,knn,hist_gb,mlp_small,logreg_pca4,svm_rbf_pca4] [--seed 0]

Configuration = the one most often selected by the nested evaluation's inner loops (run run_nested_evaluation.py
first); calibration and threshold from leave-one-unit-out predictions; each run writes results/v10/runs/<run>/
(config, split, environment, log, metrics, out-of-fold predictions).
"""
import argparse

from _common import dataset_arg, require_dataset, train_specs

DEFAULT = "logreg,decision_tree,random_forest,extra_trees,svm_rbf,knn,hist_gb,mlp_small,logreg_pca4,svm_rbf_pca4"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    dataset_arg(ap)
    ap.add_argument("--features", default="radiomics", choices=["radiomics", "imagenet_gap"])
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train_specs(a.models.split(","), a.features, require_dataset(a.dataset), a.seed)
