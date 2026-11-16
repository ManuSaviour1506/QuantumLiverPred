#!/usr/bin/env python3
"""Export the model registry (every model the engine knows, with status, version, artifact hash and metrics) to
artifacts/v10/MODEL_REGISTRY.json and MODEL_REGISTRY.md.

    python scripts/export_model_registry.py
"""
import json

from _common import ROOT

if __name__ == "__main__":
    from src.v10.engine.registry import ModelEngine
    from src.v10.utils import clean
    eng = ModelEngine(ROOT).load()
    models = eng.list(include_legacy=True)
    (ROOT / "artifacts" / "v10").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "v10" / "MODEL_REGISTRY.json").write_text(json.dumps(clean(
        {"summary": eng.summary(), "models": models}), indent=1, default=str))
    lines = ["# Model registry", "", f"{eng.summary()['models_servable']} servable of {len(models)} registered.", "",
             "| id | family | architecture | version | status | training | execution | primary metric |",
             "|---|---|---|---|---|---|---|---|"]
    for m in models:
        ev = ((m.get("metrics") or {}).get("evaluations") or [{}])[0]
        met = ev.get("metrics") or {}
        val = met.get("primary_mean_within_unit_auc", met.get("auc"))
        lines.append(f"| {m['id']} | {m.get('family')} | {m.get('architecture')} | {m.get('version')} | "
                     f"{m.get('status')} | {m.get('training_status')} | {m.get('execution_mode')} | "
                     f"{'' if val is None else round(val, 4)} ({ev.get('name', '—')}) |")
    (ROOT / "artifacts" / "v10" / "MODEL_REGISTRY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
