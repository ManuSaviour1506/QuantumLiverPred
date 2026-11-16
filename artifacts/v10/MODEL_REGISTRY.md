# Model registry

33 servable of 63 registered.

| id | family | architecture | version | status | training | execution | primary metric |
|---|---|---|---|---|---|---|---|
| v10_decision_tree_radiomics | classical | Decision Tree | v10.1 | available | TRAINED | classical_cpu_numpy | 0.7948 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_extra_trees_radiomics | classical | Extra Trees | v10.0 | available | TRAINED | classical_cpu_numpy | 0.8918 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_hist_gb_radiomics | classical | Histogram Gradient Boosting | v10.0 | available | TRAINED | classical_cpu_numpy | 0.9242 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_hybrid_twin_radiomics | classical | Classical twin | v10.0 | available | TRAINED | classical_cpu_numpy | 0.9357 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_knn_radiomics | classical | K-Nearest Neighbours | v10.0 | available | TRAINED | classical_cpu_numpy | 0.8106 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_logreg_pca4_radiomics | classical | Logistic Regression on PCA(4) | v10.0 | available | TRAINED | classical_cpu_numpy | 0.8935 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_logreg_radiomics | classical | Logistic Regression | v10.0 | available | TRAINED | classical_cpu_numpy | 0.9657 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_mlp_small_radiomics | classical | MLP | v10.0 | available | TRAINED | classical_cpu_numpy | 0.9379 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_random_forest_radiomics | classical | Random Forest | v10.0 | available | TRAINED | classical_cpu_numpy | 0.9159 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_svm_rbf_pca4_radiomics | classical | RBF-SVM on PCA(4) | v10.0 | available | TRAINED | classical_cpu_numpy | 0.8518 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_svm_rbf_radiomics | classical | SVM | v10.0 | available | TRAINED | classical_cpu_numpy | 0.9547 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_hybrid_feature_fusion_radiomics | hybrid | Hybrid feature fusion | v10.0 | available | TRAINED | custom_numpy_simulator | 0.9093 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_hybrid_no_fusion_radiomics | hybrid | Hybrid | v10.0 | available | TRAINED | custom_numpy_simulator | 0.9344 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_late_fusion_radiomics | hybrid | Late fusion | v10.0 | available | TRAINED | custom_numpy_simulator | 0.9221 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_qksvm_radiomics | quantum | Quantum kernel SVM | v10.0 | available | TRAINED | custom_numpy_simulator | 0.8928 (Nested leave-one-unit-out evaluation (radiomics)) |
| v10_vqc_radiomics | quantum | VQC | v10.0 | available | TRAINED | custom_numpy_simulator | 0.878 (Nested leave-one-unit-out evaluation (radiomics)) |
| logreg_radiomics_v9 | classical | Logistic Regression | v9 | available | TRAINED | classical_cpu_numpy | 0.9431 (V9 held-out test (recorded)) |
| lr_control_v9_cnn | classical | Logistic Regression | v9 | available | TRAINED | classical_cpu_numpy | 0.9983 (V9 held-out test (recorded)) |
| random_forest_radiomics_v9 | classical | Random Forest | v9 | available | TRAINED | classical_cpu_numpy | 0.943 (V9 held-out test (recorded)) |
| rbf_control_v9_cnn | classical | SVM (RBF) | v9 | available | TRAINED | classical_cpu_numpy | 0.982 (V9 held-out test (recorded)) |
| svm_rbf_radiomics_v9 | classical | SVM (RBF) | v9 | available | TRAINED | classical_cpu_numpy | 0.8903 (V9 held-out test (recorded)) |
| twin_v9_cnn | classical | Classical twin (MLP) | v9 | available | TRAINED | classical_cpu_numpy | 0.9986 (V9 held-out test (recorded)) |
| cnn25d_v9_a1_aug | deep_learning | CNN (NumPy) | v9 | available | TRAINED | classical_cpu_numpy | 0.9952 (V9 held-out test (recorded)) |
| cnn25d_v9_a2_aug_focal | deep_learning | CNN (NumPy) | v9 | available | TRAINED | classical_cpu_numpy | 0.9969 (V9 held-out test (recorded)) |
| cnn25d_v9_a3_res_se | deep_learning | CNN (NumPy) | v9 | available | TRAINED | classical_cpu_numpy | 0.9981 (V9 held-out test (recorded)) |
| cnn25d_v9_a4_res_se_focal | deep_learning | CNN (NumPy) | v9 | available | TRAINED | classical_cpu_numpy | 0.9987 (V9 held-out test (recorded)) |
| cnn2d_v9 | deep_learning | CNN (NumPy) | v9 | available | TRAINED | classical_cpu_numpy | 0.9981 (V9 held-out test (recorded)) |
| cnn3d_v9 | deep_learning | CNN (NumPy) | v9 | available | TRAINED | classical_cpu_numpy | 0.9965 (V9 held-out test (recorded)) |
| cnn5s_v9 | deep_learning | CNN (NumPy) | v9 | available | TRAINED | classical_cpu_numpy | 0.9986 (V9 held-out test (recorded)) |
| densenet121_ft_v9 | deep_learning | DenseNet-121 | v9 | available | PARTIALLY TRAINED | classical_cpu_numpy | 0.8643 (V9 held-out test (recorded)) |
| hybrid_v9_cnn | hybrid | Hybrid quantum-classical | v9 | available | TRAINED | custom_numpy_simulator | 0.9986 (V9 held-out test (recorded)) |
| qksvm_v9_cnn | quantum | Quantum kernel SVM | v9 | available | TRAINED | custom_numpy_simulator | 0.9977 (V9 held-out test (recorded)) |
| vqc_v9_cnn | quantum | VQC | v9 | available | TRAINED | custom_numpy_simulator | 0.9731 (V9 held-out test (recorded)) |
| convnext_tiny_ft_v9 | classical |  | v9 | not_trained | NOT TRAINED | None |  (—) |
| efficientnet_b2_ft_v9 | classical |  | v9 | not_trained | NOT TRAINED | None |  (—) |
| efficientnet_b0_ft | classical | EFFICIENTNET-B0 | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| efficientnet_b0_ft_v9 | classical | EFFICIENTNET-B0 | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| hybrid_classical_twin_ft | classical |  | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| lr_control_v9_eff | classical |  | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| pca8_logreg_ft | classical |  | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| pca8_svm_rbf_ft | classical |  | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| rbf_control_v9_eff | classical |  | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| resnet50_ft | classical | RESNET50 | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| resnet50_ft_v9 | classical | RESNET50 | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| twin_v9_eff | classical |  | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| hybrid_cnn_vqc_8q_ft | hybrid |  | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| hybrid_v9_eff | hybrid |  | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| qkernel_svm_8q_ft | quantum |  | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| qksvm_v9_eff | quantum |  | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| vqc_8q_ft | quantum |  | v8 | failed | TRAINED (weights not in this installation) | None |  (—) |
| vqc_v9_eff | quantum |  | v9 | failed | TRAINED (weights not in this installation) | None |  (—) |
| hybrid_classical_twin | classical | Classical twin (MLP) | v9 | legacy | PARTIALLY TRAINED | classical_cpu_numpy | 0.7335 (V9 held-out test (recorded)) |
| logreg_radiomics | classical | Logistic Regression | v9 | legacy | TRAINED | classical_cpu_numpy | 0.8202 (V9 held-out test (recorded)) |
| pca8_logreg | classical | Logistic Regression | v9 | legacy | PARTIALLY TRAINED | classical_cpu_numpy | 0.6341 (V9 held-out test (recorded)) |
| pca8_svm_rbf | classical | SVM (RBF) | v9 | legacy | PARTIALLY TRAINED | classical_cpu_numpy | 0.5836 (V9 held-out test (recorded)) |
| random_forest_radiomics | classical | Random Forest | v9 | legacy | TRAINED | classical_cpu_numpy | 0.7512 (V9 held-out test (recorded)) |
| svm_rbf_radiomics | classical | SVM (RBF) | v9 | legacy | TRAINED | classical_cpu_numpy | 0.4674 (V9 held-out test (recorded)) |
| cnn25d_scratch | deep_learning | CNN (NumPy) | v9 | legacy | TRAINED | classical_cpu_numpy | 0.9977 (V9 held-out test (recorded)) |
| cnn2d_scratch | deep_learning | CNN (NumPy) | v9 | legacy | TRAINED | classical_cpu_numpy | 0.9986 (V9 held-out test (recorded)) |
| densenet121_ft | deep_learning | DenseNet-121 | v9 | legacy | PARTIALLY TRAINED | classical_cpu_numpy | 0.8882 (V9 held-out test (recorded)) |
| hybrid_cnn_vqc_8q | hybrid | Hybrid quantum-classical | v9 | legacy | PARTIALLY TRAINED | custom_numpy_simulator | 0.6693 (V9 held-out test (recorded)) |
| qkernel_svm_8q | quantum | Quantum kernel SVM | v9 | legacy | PARTIALLY TRAINED | custom_numpy_simulator | 0.61 (V9 held-out test (recorded)) |
| vqc_8q | quantum | VQC | v9 | legacy | PARTIALLY TRAINED | custom_numpy_simulator | 0.6246 (V9 held-out test (recorded)) |
