# Hybrid quantum-classical (simulated quantum layer)

4 of 7 registered models are servable on the installation that generated this file (`python scripts/v11_build_model_index.py`).

Model code lives in `src/`, trained artifacts in `artifacts/`; this folder is the index. Full machine-readable details: [`../model_index.json`](../model_index.json).

| model | architecture | version | availability | threshold | artifact | implementation |
|---|---|---|---|---|---|---|
| `v10_hybrid_feature_fusion_radiomics` | Hybrid feature fusion | v10.0 | available | 0.6587 | `artifacts/v10/v10_hybrid_feature_fusion_radiomics` | `src/v10/hybrid/models.py`, `src/v10/quantum/simulator.py` |
| `v10_hybrid_no_fusion_radiomics` | Hybrid | v10.0 | available | 0.5399 | `artifacts/v10/v10_hybrid_no_fusion_radiomics` | `src/v10/hybrid/models.py`, `src/v10/quantum/simulator.py` |
| `v10_late_fusion_radiomics` | Late fusion | v10.0 | available | 0.5695 | `artifacts/v10/v10_late_fusion_radiomics` | `src/v10/hybrid/models.py`, `src/v10/quantum/simulator.py` |
| `hybrid_v9_cnn` | Hybrid quantum-classical | v9 | requires_context | 0.9060 | `artifacts/hybrid/hybrid_v9_cnn` | `src/quantum/v9.py`, `src/inference/registry.py` |
| `hybrid_cnn_vqc_8q_ft` |  | v8 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.7219 | `artifacts/hybrid/hybrid_cnn_vqc_8q_ft` | `src/hybrid/hybrid_model.py`, `src/inference/registry.py` |
| `hybrid_v9_eff` |  | v9 | missing_dependency — ModelIntegrityError: needs the fine-tuned EfficientNet-B0 (efficientnet_b0_ft), which is not loaded | 0.7814 | `artifacts/hybrid/hybrid_v9_eff` | `src/quantum/v9.py`, `src/inference/registry.py` |
| `hybrid_cnn_vqc_8q` | Hybrid quantum-classical | v9 | legacy — V8-era model kept for reproducibility; not offered | 0.5657 | `artifacts/hybrid/hybrid_cnn_vqc_8q` | `src/hybrid/hybrid_model.py`, `src/inference/registry.py` |

Every quantum circuit is executed by an exact CPU state-vector simulator (`custom_numpy_simulator`). No model in this platform runs on quantum hardware.
