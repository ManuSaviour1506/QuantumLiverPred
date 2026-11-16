# Classical machine learning

17 of 29 registered models are servable on the installation that generated this file (`python scripts/v11_build_model_index.py`).

Model code lives in `src/`, trained artifacts in `artifacts/`; this folder is the index. Full machine-readable details: [`../model_index.json`](../model_index.json).

| model | architecture | version | availability | threshold | artifact | implementation |
|---|---|---|---|---|---|---|
| `v10_decision_tree_radiomics` | Decision Tree | v10.1 | available | 0.6790 | `artifacts/v10/v10_decision_tree_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_extra_trees_radiomics` | Extra Trees | v10.0 | available | 0.4699 | `artifacts/v10/v10_extra_trees_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_hist_gb_radiomics` | Histogram Gradient Boosting | v10.0 | available | 0.5870 | `artifacts/v10/v10_hist_gb_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_hybrid_twin_radiomics` | Classical twin | v10.0 | available | 0.7112 | `artifacts/v10/v10_hybrid_twin_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_knn_radiomics` | K-Nearest Neighbours | v10.0 | available | 0.5147 | `artifacts/v10/v10_knn_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_logreg_pca4_radiomics` | Logistic Regression on PCA(4) | v10.0 | available | 0.4301 | `artifacts/v10/v10_logreg_pca4_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_logreg_radiomics` | Logistic Regression | v10.0 | available | 0.5396 | `artifacts/v10/v10_logreg_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_mlp_small_radiomics` | MLP | v10.0 | available | 0.5067 | `artifacts/v10/v10_mlp_small_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_random_forest_radiomics` | Random Forest | v10.0 | available | 0.4741 | `artifacts/v10/v10_random_forest_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_svm_rbf_pca4_radiomics` | RBF-SVM on PCA(4) | v10.0 | available | 0.4205 | `artifacts/v10/v10_svm_rbf_pca4_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `v10_svm_rbf_radiomics` | SVM | v10.0 | available | 0.5880 | `artifacts/v10/v10_svm_rbf_radiomics` | `src/v10/classical/models.py`, `src/v10/features/extract.py` |
| `logreg_radiomics_v9` | Logistic Regression | v9 | available | 0.6340 | `artifacts/classical/logreg_radiomics_v9` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `lr_control_v9_cnn` | Logistic Regression | v9 | requires_context | 0.5759 | `artifacts/classical/lr_control_v9_cnn` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `random_forest_radiomics_v9` | Random Forest | v9 | available | 0.4936 | `artifacts/classical/random_forest_radiomics_v9` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `rbf_control_v9_cnn` | SVM (RBF) | v9 | requires_context | 0.7375 | `artifacts/classical/rbf_control_v9_cnn` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `svm_rbf_radiomics_v9` | SVM (RBF) | v9 | available | 0.7497 | `artifacts/classical/svm_rbf_radiomics_v9` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `twin_v9_cnn` | Classical twin (MLP) | v9 | requires_context | 0.8982 | `artifacts/classical/twin_v9_cnn` | `src/quantum/v9.py`, `src/inference/registry.py` |
| `hybrid_classical_twin_ft` |  | v8 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.9084 | `artifacts/classical/hybrid_classical_twin_ft` | `src/hybrid/hybrid_model.py`, `src/inference/registry.py` |
| `lr_control_v9_eff` |  | v9 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.7217 | `artifacts/classical/lr_control_v9_eff` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `pca8_logreg_ft` |  | v8 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.8232 | `artifacts/classical/pca8_logreg_ft` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `pca8_svm_rbf_ft` |  | v8 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.7872 | `artifacts/classical/pca8_svm_rbf_ft` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `rbf_control_v9_eff` |  | v9 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.7483 | `artifacts/classical/rbf_control_v9_eff` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `twin_v9_eff` |  | v9 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.8766 | `artifacts/classical/twin_v9_eff` | `src/quantum/v9.py`, `src/inference/registry.py` |
| `hybrid_classical_twin` | Classical twin (MLP) | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.4624 | `artifacts/classical/hybrid_classical_twin` | `src/hybrid/hybrid_model.py`, `src/inference/registry.py` |
| `logreg_radiomics` | Logistic Regression | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.8201 | `artifacts/classical/logreg_radiomics` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `pca8_logreg` | Logistic Regression | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.5253 | `artifacts/classical/pca8_logreg` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `pca8_svm_rbf` | SVM (RBF) | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.5383 | `artifacts/classical/pca8_svm_rbf` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `random_forest_radiomics` | Random Forest | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.7441 | `artifacts/classical/random_forest_radiomics` | `src/classical/baselines.py`, `src/inference/registry.py` |
| `svm_rbf_radiomics` | SVM (RBF) | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.7737 | `artifacts/classical/svm_rbf_radiomics` | `src/classical/baselines.py`, `src/inference/registry.py` |
