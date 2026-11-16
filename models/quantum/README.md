# Quantum machine learning (CPU state-vector simulation — never quantum hardware)

4 of 10 registered models are servable on the installation that generated this file (`python scripts/v11_build_model_index.py`).

Model code lives in `src/`, trained artifacts in `artifacts/`; this folder is the index. Full machine-readable details: [`../model_index.json`](../model_index.json).

| model | architecture | version | availability | threshold | artifact | implementation |
|---|---|---|---|---|---|---|
| `v10_qksvm_radiomics` | Quantum kernel SVM | v10.0 | available | 0.4067 | `artifacts/v10/v10_qksvm_radiomics` | `src/v10/quantum/qksvm.py`, `src/v10/quantum/simulator.py` |
| `v10_vqc_radiomics` | VQC | v10.0 | available | 0.3865 | `artifacts/v10/v10_vqc_radiomics` | `src/v10/quantum/vqc.py`, `src/v10/quantum/simulator.py` |
| `qksvm_v9_cnn` | Quantum kernel SVM | v9 | requires_context | 0.8369 | `artifacts/quantum/qksvm_v9_cnn` | `src/quantum/v9.py`, `src/quantum/simulator.py` |
| `vqc_v9_cnn` | VQC | v9 | requires_context | 0.6027 | `artifacts/quantum/vqc_v9_cnn` | `src/quantum/v9.py`, `src/quantum/simulator.py` |
| `qkernel_svm_8q_ft` |  | v8 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.7303 | `artifacts/quantum/qkernel_svm_8q_ft` | `src/quantum/models.py`, `src/quantum/simulator.py` |
| `qksvm_v9_eff` |  | v9 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.8470 | `artifacts/quantum/qksvm_v9_eff` | `src/quantum/v9.py`, `src/quantum/simulator.py` |
| `vqc_8q_ft` |  | v8 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.6169 | `artifacts/quantum/vqc_8q_ft` | `src/quantum/models.py`, `src/quantum/simulator.py` |
| `vqc_v9_eff` |  | v9 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.7811 | `artifacts/quantum/vqc_v9_eff` | `src/quantum/v9.py`, `src/quantum/simulator.py` |
| `qkernel_svm_8q` | Quantum kernel SVM | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.5680 | `artifacts/quantum/qkernel_svm_8q` | `src/quantum/models.py`, `src/quantum/simulator.py` |
| `vqc_8q` | VQC | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.7463 | `artifacts/quantum/vqc_8q` | `src/quantum/models.py`, `src/quantum/simulator.py` |

Every quantum circuit is executed by an exact CPU state-vector simulator (`custom_numpy_simulator`). No model in this platform runs on quantum hardware.
