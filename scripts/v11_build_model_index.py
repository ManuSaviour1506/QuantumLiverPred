#!/usr/bin/env python3
"""Build models/model_index.json and models/<family>/README.md from the live model registry.

The V11 folder layout asks for models/{classical,deep_learning,quantum,hybrid}. The model code already lives in src/
and the trained artifacts in artifacts/ (moving them would break every import path and artifact hash), so models/ is
an index: for each family it lists every registered model with its implementation files, artifact folder, SHA-256,
version, availability (as the engine determines it on THIS installation), dependencies and measured load time.

    python scripts/v11_build_model_index.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from src.v10.engine.registry import ModelEngine  # noqa: E402

FAMILIES = ("classical", "deep_learning", "quantum", "hybrid")
TITLES = {"classical": "Classical machine learning", "deep_learning": "Deep learning",
          "quantum": "Quantum machine learning (CPU state-vector simulation — never quantum hardware)",
          "hybrid": "Hybrid quantum-classical (simulated quantum layer)"}
AVAIL_TEXT = {
    "available": "runs on a single uploaded slice",
    "requires_context": "trained and loaded; needs neighbouring slices (2.5-D / 3-D / 5-slice input) — upload them too",
    "missing_dependency": "trained, but a required upstream artifact is not in this installation",
    "missing_weights": "trained, but its weight file is not in this installation",
    "legacy": "superseded earlier version, kept for reproducibility (not run in comparisons)",
    "not_trained": "registered architecture with no trained weights (never trained)",
    "failed_integrity": "artifact present but failed its integrity / self test",
    "non_deployable": "loaded but cannot produce a confident call (see reason)",
}


def implementation(d: dict) -> list[str]:
    mid, fam, src = d["id"], d["compare_family"], d.get("source")
    if src == "v10":
        files = {"classical": ["src/v10/classical/models.py"],
                 "hybrid": ["src/v10/hybrid/models.py", "src/v10/quantum/simulator.py"],
                 "quantum": [("src/v10/quantum/qksvm.py" if "qksvm" in mid else "src/v10/quantum/vqc.py"),
                             "src/v10/quantum/simulator.py"]}.get(fam, [])
        return files + ["src/v10/features/extract.py", "src/v10/engine/v10_models.py"]
    v9_recipe = "_v9" in mid
    if fam == "deep_learning":
        if any(b in mid for b in ("densenet", "resnet", "efficientnet", "convnext")):
            files = ["src/classical/backbones.py", "src/classical/cnn_finetune.py"]
        else:
            files = ["src/classical/numpy_cnn.py", "src/classical/nn.py"]
    elif fam == "quantum":
        files = ["src/quantum/v9.py", "src/quantum/simulator.py"] if v9_recipe else ["src/quantum/models.py",
                                                                                      "src/quantum/simulator.py"]
    elif fam == "hybrid":
        files = ["src/quantum/v9.py"] if v9_recipe else ["src/hybrid/hybrid_model.py"]
    elif "twin" in mid:
        files = ["src/quantum/v9.py"] if v9_recipe else ["src/hybrid/hybrid_model.py"]
    else:
        files = ["src/classical/baselines.py"]
    return files + ["src/inference/registry.py", "src/v10/engine/v9_adapter.py"]


def main() -> int:
    eng = ModelEngine(ROOT).load()
    summ = eng.summary()
    models = eng.list(include_legacy=True)
    out = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "note": "Availability is what the model engine determined on the installation that generated this file; "
                   "the running platform re-checks it at every start (Model Registry page, GET /api/v1/models).",
           "engine": {k: summ.get(k) for k in ("models_total", "models_servable", "weights", "unet_liver_validation",
                                                "ood_detector", "liver_gate", "engine_load_seconds")},
           "availability_categories": AVAIL_TEXT, "families": {}}
    for fam in FAMILIES:
        rows = []
        for d in models:
            if d["compare_family"] != fam:
                continue
            ic = d.get("input_contract") or {}
            rows.append({
                "id": d["id"], "name": d.get("name"), "architecture": d.get("architecture"),
                "version": d.get("version"), "source": d.get("source"), "role": d.get("role"),
                "training_status": d.get("training_status"), "training_date": d.get("training_date"),
                "dataset_version": d.get("dataset_version"), "execution_mode": d.get("execution_mode"),
                "availability": d.get("availability"), "servable": d.get("servable"), "reason": d.get("reason"),
                "threshold": d.get("threshold"), "calibrated": d.get("calibrated"),
                "inputs": {"context_slices_each_side": ic.get("context_slices_each_side", 0),
                           "features": ic.get("features"), "patch_px": ic.get("patch_px"),
                           "formats": ic.get("formats")},
                "artifact_path": d.get("artifact_path"), "artifact_sha256": d.get("artifact_sha256"),
                "size_bytes": d.get("size_bytes"), "parameters": d.get("parameters"),
                "dependencies": d.get("dependencies"), "load_ms": d.get("load_ms"),
                "implementation": implementation(d)})
        out["families"][fam] = rows
        fdir = ROOT / "models" / fam
        fdir.mkdir(parents=True, exist_ok=True)
        lines = [f"# {TITLES[fam]}", "",
                 f"{sum(1 for r in rows if r['servable'])} of {len(rows)} registered models are servable on the "
                 "installation that generated this file (`python scripts/v11_build_model_index.py`).", "",
                 "Model code lives in `src/`, trained artifacts in `artifacts/`; this folder is the index. "
                 "Full machine-readable details: [`../model_index.json`](../model_index.json).", "",
                 "| model | architecture | version | availability | threshold | artifact | implementation |",
                 "|---|---|---|---|---|---|---|"]
        for r in rows:
            thr = "—" if r["threshold"] is None else f"{r['threshold']:.4f}"
            av = r["availability"] + (f" — {r['reason']}" if r["reason"] and r["availability"] != "available" else "")
            lines.append(f"| `{r['id']}` | {r['architecture']} | {r['version']} | {av} | {thr} | "
                         f"`{r['artifact_path'] or '—'}` | {', '.join('`%s`' % f for f in r['implementation'][:2])} |")
        if fam in ("quantum", "hybrid"):
            lines += ["", "Every quantum circuit is executed by an exact CPU state-vector simulator "
                          "(`custom_numpy_simulator`). No model in this platform runs on quantum hardware."]
        (fdir / "README.md").write_text("\n".join(lines) + "\n")
    (ROOT / "models" / "model_index.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    counts = {f: {} for f in FAMILIES}
    for f in FAMILIES:
        for r in out["families"][f]:
            counts[f][r["availability"]] = counts[f].get(r["availability"], 0) + 1
    readme = ["# Model index (V11)", "",
              "The requested `models/{classical,deep_learning,quantum,hybrid}` layout, mapped onto the existing "
              "project without moving code or artifacts:", "",
              "| family | index | implementation | trained artifacts |", "|---|---|---|---|",
              "| classical | [classical/](classical/README.md) | `src/classical/`, `src/v10/classical/` | "
              "`artifacts/classical/*` (radiomics, CNN-embedding heads), `artifacts/v10/v10_*` |",
              "| deep learning | [deep_learning/](deep_learning/README.md) | `src/classical/numpy_cnn.py`, "
              "`src/classical/backbones.py`, `src/v10/deep_learning/` | `artifacts/classical/cnn*_v9`, "
              "`artifacts/classical/*_ft*`, `artifacts/weights/` (ImageNet backbones) |",
              "| quantum | [quantum/](quantum/README.md) | `src/quantum/`, `src/v10/quantum/` | `artifacts/quantum/*`, "
              "`artifacts/v10/v10_qksvm_radiomics`, `artifacts/v10/v10_vqc_radiomics` |",
              "| hybrid | [hybrid/](hybrid/README.md) | `src/hybrid/`, `src/v10/hybrid/` | `artifacts/hybrid/*`, "
              "`artifacts/v10/v10_*fusion*` |", "",
              "Shared front end (every family): OOD detector `artifacts/ood/`, liver gate `artifacts/classical/"
              "liver_gate`, U-Net liver validation `artifacts/segmentation/unet25d_v9`.", "",
              f"Engine on the generating installation: {summ['models_servable']} of {summ['models_total']} models "
              "servable. Counts by availability:", "",
              "| family | " + " | ".join(AVAIL_TEXT) + " |", "|---|" + "---|" * len(AVAIL_TEXT)]
    for f in FAMILIES:
        readme.append(f"| {f} | " + " | ".join(str(counts[f].get(a, 0)) for a in AVAIL_TEXT) + " |")
    readme += ["", "Models reported as `missing_weights` / `missing_dependency` need V9 files that are not in this "
                   "package — see [`../MISSING_ARTIFACTS.md`](../MISSING_ARTIFACTS.md)."]
    (ROOT / "models" / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
