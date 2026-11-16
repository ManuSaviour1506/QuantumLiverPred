# V9 results (generated)

Task A: tumour vs tumour-free liver tissue patches (96 x 96 px), V8 patient-independent split. Two-stage cluster bootstrap (scan -> slice), 2,000 resamples; only 3 test scans, so between-patient uncertainty is under-represented.

Selected V9 2.5-D CNN: **2.5-D residual-SE CNN + augmentation + focal loss (ablation a4)** (highest validation ROC-AUC; candidates within 0.001 of the best are separated by lower validation ECE).

| Model | Val AUC | Test AUC (95% CI) | Sens. | Spec. | Missed | False alarms | ECE | AUC G01 |
|---|---|---|---|---|---|---|---|---|
| V8 · 2.5-D CNN (from scratch) | 0.992 | 0.998 (0.991–1.000) | 0.995 | 0.972 | 2 | 7 | 0.021 | 1.000 |
| V9 · 2.5-D residual-SE CNN + augmentation + focal loss (ablation a4) | 1.000 | 0.999 (0.997–1.000) | 0.995 | 0.919 | 2 | 20 | 0.048 | 0.999 |
| V8 · 2-D CNN (from scratch, ablation) | 0.999 | 0.999 (0.989–1.000) | 0.997 | 0.850 | 1 | 37 | 0.054 | 1.000 |
| V9 · 2-D CNN, V9 recipe | 0.998 | 0.998 (0.996–1.000) | 0.995 | 0.846 | 2 | 38 | 0.050 | 0.999 |
| V9 · 5-slice 2.5-D CNN, V9 recipe | 1.000 | 0.999 (0.995–1.000) | 1.000 | 0.801 | 0 | 49 | 0.084 | 0.999 |
| V9 · 3-D CNN (3x3x3 stem over 5 slices), V9 recipe | 0.998 | 0.997 (0.991–1.000) | 0.968 | 0.988 | 12 | 3 | 0.013 | 0.996 |
| V8 · EfficientNet-B0 (fine-tuned) | 0.975 | 0.936 (0.919–0.989) | 0.654 | 0.963 | 130 | 9 | 0.128 | 0.930 |
| V9 · EfficientNet-B0 (fine-tuned, V9 CT recipe) | 0.983 | 0.933 (0.915–0.988) | 0.657 | 0.951 | 129 | 12 | 0.123 | 0.926 |
| V8 · DenseNet-121 (fine-tuned) | 0.946 | 0.888 (0.858–1.000) | 0.551 | 0.963 | 169 | 9 | 0.135 | 0.866 |
| V9 · DenseNet-121 (fine-tuned, V9 CT recipe) | 0.935 | 0.864 (0.823–1.000) | 0.628 | 0.890 | 140 | 27 | 0.094 | 0.829 |
| V8 · ResNet-50 (fine-tuned) | 0.818 | 0.691 (0.634–0.988) | 0.324 | 0.915 | 254 | 21 | 0.191 | 0.645 |
| V9 · ResNet-50 (fine-tuned, V9 CT recipe) | 0.849 | 0.683 (0.624–0.985) | 0.221 | 0.967 | 293 | 8 | 0.231 | 0.633 |
| V8 · Logistic Regression (radiomics) | 0.912 | 0.820 (0.767–1.000) | 0.941 | 0.423 | 22 | 142 | 0.263 | 0.779 |
| V9 · Logistic Regression (radiomics, V9) | 0.950 | 0.943 (0.923–1.000) | 0.920 | 0.687 | 30 | 77 | 0.127 | 0.929 |
| V8 · Random Forest (radiomics) | 0.826 | 0.751 (0.665–1.000) | 0.835 | 0.581 | 62 | 103 | 0.114 | 0.666 |
| V9 · Random Forest (radiomics, V9) | 0.905 | 0.943 (0.924–1.000) | 0.997 | 0.589 | 1 | 101 | 0.127 | 0.934 |
| V8 · RBF-SVM (radiomics) | 0.904 | 0.467 (0.291–0.991) | 0.489 | 0.419 | 192 | 143 | 0.223 | 0.282 |
| V9 · RBF-SVM (radiomics, V9) | 0.937 | 0.890 (0.857–1.000) | 0.612 | 0.882 | 146 | 29 | 0.213 | 0.861 |
| V8 · Variational Quantum Classifier (8 qubits, fine-tuned features) | 0.834 | 0.847 (0.813–0.957) | 0.638 | 0.870 | 136 | 32 | 0.060 | 0.843 |
| V9 · Variational Quantum Classifier (8 qubits, V9, CNN features) | 0.999 | 0.973 (0.960–1.000) | 0.867 | 0.915 | 50 | 21 | 0.061 | 0.966 |
| V9 · Logistic regression (control for the VQC) (8 qubits, CNN features) | 0.999 | 0.998 (0.997–1.000) | 0.995 | 0.919 | 2 | 20 | 0.045 | 0.998 |
| V8 · Quantum Kernel SVM (8 qubits, fine-tuned features) | 0.936 | 0.871 (0.843–0.968) | 0.535 | 0.931 | 175 | 17 | 0.130 | 0.857 |
| V9 · Quantum Kernel SVM (4 qubits, V9, CNN features) | 1.000 | 0.998 (0.996–1.000) | 0.992 | 0.943 | 3 | 14 | 0.042 | 0.998 |
| V9 · RBF-SVM (control for the quantum kernel) (4 qubits, CNN features) | 0.999 | 0.982 (0.971–1.000) | 0.878 | 0.943 | 46 | 14 | 0.047 | 0.977 |
| V8 · Hybrid CNN + Quantum Layer (8 qubits, fine-tuned features) | 0.947 | 0.897 (0.864–0.983) | 0.612 | 0.967 | 146 | 8 | 0.121 | 0.889 |
| V9 · Hybrid CNN + quantum layer (8 qubits, V9, CNN features) | 0.999 | 0.999 (0.997–1.000) | 0.992 | 0.947 | 3 | 13 | 0.045 | 0.999 |
| V9 · Classical twin (control for the hybrid) (8 qubits, CNN features) | 0.999 | 0.999 (0.997–1.000) | 0.992 | 0.947 | 3 | 13 | 0.048 | 0.999 |
| V9 · Variational Quantum Classifier (4 qubits, V9, EFF features) | 0.957 | 0.907 (0.881–0.996) | 0.535 | 0.967 | 175 | 8 | 0.138 | 0.902 |
| V9 · Quantum Kernel SVM (4 qubits, V9, EFF features) | 0.949 | 0.900 (0.877–0.985) | 0.439 | 0.992 | 211 | 2 | 0.140 | 0.891 |
| V9 · Hybrid CNN + quantum layer (8 qubits, V9, EFF features) | 0.965 | 0.914 (0.893–1.000) | 0.620 | 0.959 | 143 | 10 | 0.125 | 0.904 |

Paired tests (test set, model A − model B):

| Comparison | ΔAUC (95% CI) | p | ΔSens. | p | ΔSpec. | p |
|---|---|---|---|---|---|---|
| V9 selected 2.5-D CNN vs V8 2.5-D CNN | +0.001 (−0.001 to 0.009) | 0.798 | 0.000 | 1.000 | −0.053 | 0.569 |
| V9 2-D CNN vs V8 2-D CNN | 0.000 (−0.002 to 0.011) | 0.749 | −0.003 | 0.998 | −0.004 | 1.000 |
| EfficientNet-B0: V9 vs V8 | −0.003 (−0.011 to 0.004) | 0.439 | +0.003 | 1.000 | −0.012 | 0.565 |
| DenseNet-121: V9 vs V8 | −0.024 (−0.038 to 0.005) | 0.389 | +0.077 | 0.003 | −0.073 | 0.569 |
| ResNet-50: V9 vs V8 | −0.008 (−0.019 to 0.018) | 0.594 | −0.104 | 0.000 | +0.053 | 0.167 |
| Logistic regression radiomics: V9 vs V8 | +0.123 (−0.002 to 0.163) | 0.089 | −0.021 | 0.594 | +0.264 | 0.569 |
| Random forest radiomics: V9 vs V8 | +0.192 (−0.027 to 0.268) | 0.289 | +0.162 | 0.000 | +0.008 | 1.000 |
| RBF-SVM radiomics: V9 vs V8 | +0.423 (−0.011 to 0.571) | 0.170 | +0.122 | 0.526 | +0.463 | 0.569 |
| Ablation a1 (CT augmentation, AUC stopping) vs V8 recipe | −0.003 (−0.009 to 0.005) | 0.467 | +0.003 | 0.945 | −0.179 | 0.569 |
| Ablation: + focal loss (a2 vs a1) | +0.002 (−0.002 to 0.004) | 0.572 | 0.000 | 1.000 | −0.053 | 0.569 |
| Ablation: + residual SE blocks (a3 vs a1) | +0.003 (0.000 to 0.012) | 0.184 | −0.021 | 0.569 | +0.203 | 0.569 |
| Ablation: + focal loss on residual SE (a4 vs a3) | +0.001 (0.000 to 0.003) | 0.315 | +0.019 | 0.570 | −0.077 | 0.569 |
| 5-slice 2.5-D CNN vs selected 2.5-D CNN | 0.000 (−0.004 to 0.001) | 1.000 | +0.005 | 0.345 | −0.118 | 0.156 |
| 3-D CNN vs selected 2.5-D CNN | −0.002 (−0.007 to 0.001) | 0.510 | −0.027 | 0.077 | +0.069 | 0.569 |
| V9: 2.5-D vs 2-D (adjacent-slice context) | +0.001 (0.000 to 0.002) | 0.861 | 0.000 | 1.000 | +0.073 | 0.572 |
| VQC on the same EfficientNet features: V9 design vs V8 | +0.060 (0.023 to 0.076) | <0.001 | −0.104 | 0.158 | +0.098 | 0.304 |
| Quantum kernel on the same EfficientNet features: V9 vs V8 | +0.029 (0.004 to 0.038) | 0.002 | −0.096 | 0.009 | +0.061 | 0.090 |
| Hybrid on the same EfficientNet features: V9 vs V8 | +0.017 (−0.005 to 0.059) | 0.088 | +0.008 | 0.601 | −0.008 | 0.868 |
| VQC vs logistic-regression control (CNN features) | −0.025 (−0.038 to 0.000) | 0.529 | −0.128 | 0.003 | −0.004 | 1.000 |
| Quantum kernel vs RBF-SVM control (CNN features) | +0.016 (0.000 to 0.026) | 0.511 | +0.114 | 0.071 | 0.000 | 1.000 |
| Hybrid vs classical twin (CNN features) | 0.000 (0.000 to 0.000) | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| VQC (CNN features) vs selected 2.5-D CNN | −0.026 (−0.039 to 0.000) | 0.531 | −0.128 | 0.003 | −0.004 | 1.000 |
| QKSVM (CNN features) vs selected 2.5-D CNN | −0.001 (−0.003 to 0.000) | 0.571 | −0.003 | 1.000 | +0.024 | 0.573 |
| HYBRID (CNN features) vs selected 2.5-D CNN | 0.000 (−0.001 to 0.000) | 0.805 | −0.003 | 1.000 | +0.028 | 0.570 |
| VQC vs logistic-regression control (EFF features) | −0.003 (−0.006 to 0.005) | 0.419 | −0.045 | 0.569 | 0.000 | 1.000 |
| Quantum kernel vs RBF-SVM control (EFF features) | −0.015 (−0.022 to 0.017) | 0.407 | −0.122 | 0.006 | +0.024 | 0.099 |
| Hybrid vs classical twin (EFF features) | +0.004 (−0.003 to 0.027) | 0.238 | −0.005 | 0.894 | +0.012 | 0.448 |
| VQC (EFF features) vs selected 2.5-D CNN | −0.091 (−0.118 to −0.004) | 0.008 | −0.460 | 0.000 | +0.049 | 0.598 |
| QKSVM (EFF features) vs selected 2.5-D CNN | −0.099 (−0.122 to −0.015) | <0.001 | −0.556 | 0.000 | +0.073 | 0.569 |
| HYBRID (EFF features) vs selected 2.5-D CNN | −0.085 (−0.106 to 0.000) | 0.069 | −0.375 | 0.007 | +0.041 | 0.610 |

Segmentation (test, 64 × 64 cells): liver Dice 0.799, tumour Dice 0.734 (IoU 0.580), lesion sensitivity 0.600 (6/10), 4.0 false-positive lesions per scan.

Volume pipeline (test groups): segmentation alone 6/10 lesions, 4.0 FP/volume; with CNN confirmation 6/10, 3.0 FP/volume.

All circuits were run on a NumPy exact state-vector simulator (noise study uses simulated depolarizing noise and shots). No quantum hardware was used.

Not trained / not done: convnext_tiny_ft_v9 (ImageNet weights not available offline (no network access)); efficientnet_b2_ft_v9 (ImageNet weights not available offline (no network access)); unet3d_v9 (no contiguous labelled volumes with spacing; 2 CPU cores; the 2.5-D U-Net is the trained segmentation model); nnunet_v9 (requires PyTorch, which cannot be installed offline); pennylane_qiskit (not installable offline; all circuits use the project's exact NumPy state-vector simulator); external_validation (no external CT dataset reachable (no network access)); genuine_volume_ids (original LiTS volumes not available; scan groups remain reconstructed (scripts/v9_recover_volume_ids.py ready)).
