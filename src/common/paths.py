"""Project paths. Every script resolves files relative to the project root so the folder can be moved freely."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LABELS = DATA / "labels"
MASKS = DATA / "masks"
MANIFESTS = DATA / "manifests"
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ARTIFACTS / "weights"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"


def scans_dir() -> Path:
    """Location of the extracted scans.zip ('scans' folder with normal/steatosis/cirrhosis/focal_nodules/hcc).

    Set V9_SCANS_DIR (or V8_SCANS_DIR) to point at an existing copy instead of copying 400 MB into data/raw/scans.
    """
    env = os.environ.get("V9_SCANS_DIR") or os.environ.get("V8_SCANS_DIR")
    return Path(env).expanduser() if env else RAW / "scans"


def ensure_dirs() -> None:
    for p in (PROCESSED, LABELS, MASKS, MANIFESTS, ARTIFACTS / "classical", ARTIFACTS / "quantum", ARTIFACTS / "hybrid",
              RESULTS, RESULTS / "figures", RESULTS / "training_logs"):
        p.mkdir(parents=True, exist_ok=True)
