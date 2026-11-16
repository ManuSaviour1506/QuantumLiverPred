"""SYNTHETIC fixtures for software and pipeline tests only.

These are abstract geometric phantoms (an ellipse with or without a bright disc, plus noise). They are not CT images,
are never presented as medical data, and every file and manifest row is labelled `synthetic_phantom`. Models trained on
them are tagged `training_data: synthetic` and are refused by the server unless ALLOW_SYNTHETIC_MODELS=true (tests).
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = ("phantom_A", "phantom_B")


def phantom(rng: np.random.Generator, cls: int, size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2 + rng.normal(0, 6), size / 2 + rng.normal(0, 6)
    img = np.zeros((size, size))
    img[((yy - cy) / (size * 0.38)) ** 2 + ((xx - cx) / (size * 0.30)) ** 2 < 1] = 90 + rng.normal(0, 5)
    if cls == 1:
        r = size * rng.uniform(0.06, 0.12)
        oy, ox = cy + rng.normal(0, size * 0.1), cx + rng.normal(0, size * 0.08)
        img[(yy - oy) ** 2 + (xx - ox) ** 2 < r ** 2] = 200
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def png_bytes(img: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="PNG")
    return b.getvalue()


def make_dataset(n_per_class: int = 12, n_groups: int = 4, seed: int = 0) -> list[tuple[str, bytes, dict]]:
    rng = np.random.default_rng(seed)
    out = []
    for cls in (0, 1):
        for i in range(n_per_class):
            g = f"SYN{(i % n_groups) + 1:02d}"
            name = f"images/synthetic_phantom_{LABELS[cls]}_{i:03d}.png"
            out.append((name, png_bytes(phantom(rng, cls)),
                        {"image_path": name, "label": LABELS[cls], "patient_id": "", "scan_id": g,
                         "volume_id": "", "slice_id": str(i), "modality": "synthetic", "mask_path": "",
                         "split": "train" if (i % n_groups) < n_groups - 1 else "test",
                         "data_source": "synthetic_phantom (software test fixture, not medical data)"}))
    return out


def dataset_zip(entries, extra: dict[str, bytes] | None = None, manifest: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data, _ in entries:
            z.writestr(name, data)
        if manifest:
            s = io.StringIO()
            w = csv.DictWriter(s, fieldnames=list(entries[0][2]))
            w.writeheader()
            for _, _, row in entries:
                w.writerow(row)
            z.writestr("manifest.csv", s.getvalue())
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def train_synthetic_model(artifacts_v10: Path, model_id: str = "synthetic_test_logreg", seed: int = 0) -> Path:
    """Train the software-test model (logistic regression on 32x32 down-sampled phantoms) and save it."""
    from sklearn.linear_model import LogisticRegression
    from src.v10.engine import artifacts as A
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(160):
        cls = i % 2
        img = phantom(rng, cls)
        X.append(np.asarray(Image.fromarray(img).resize((32, 32), Image.BILINEAR), float).ravel() / 255.0)
        y.append(cls)
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lr = LogisticRegression(C=0.1, max_iter=2000).fit((X - mu) / sd, y)
    acc = float(lr.score((X - mu) / sd, y))
    card = {"name": "Synthetic software-test model (phantom A vs B)", "kind": "synthetic_image_logreg",
            "family": "classical", "architecture": "Logistic Regression",
            "task": "SOFTWARE TEST ONLY: distinguishes two synthetic phantom classes; not a medical model",
            "version": "v10.0-test", "training_data": "synthetic", "dataset_version": "synthetic_phantoms_seed0",
            "threshold": 0.5, "seeds": [seed],
            "evaluations": [{"name": "training accuracy on synthetic phantoms (software check only)",
                             "status": "software check", "metrics": {"accuracy": acc},
                             "evaluation_unit": "synthetic images"}]}
    return A.save(artifacts_v10, model_id, card, {"mu": mu, "sd": sd, "coef": lr.coef_.ravel(),
                                                  "b": lr.intercept_})
