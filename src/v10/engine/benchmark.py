"""New-evaluation benchmarking engine (V11): runs registered models on ground-truth-labelled CT slices through the same
front end and concurrent family pipelines as the live comparison, and computes slice-level metrics.

Kept strictly separate from the historical (recorded) evaluations in the model cards:
  * results are stored as benchmark runs with their own id, dataset, split and date;
  * every model x slice result is labelled with the ROLE the slice's scan group played for that model:
        test        group held out from the model's training, threshold and calibration      -> valid estimate
        validation  group used to choose the model's threshold / calibration (V9 split)      -> optimistic
        train       group used to fit the model (V10 final models were fit on ALL units)     -> in-sample, not an estimate
    and metrics are computed per role, never pooled across roles;
  * a slice's ground truth is the expert-mask label (`label_tumour`); slices withheld by the front end (no liver,
    out-of-distribution) are reported as coverage, not counted as negatives;
  * ROC-AUC / PR-AUC / ECE use the model's slice score (largest liver-patch probability) and need both classes;
    threshold metrics use only decisive calls (Tumour / No tumour) and report Uncertain separately.
Nothing is tuned on these results.
"""
from __future__ import annotations

import csv
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from src.v10.engine.compare import ComparisonRunner, FAMILIES

V9_SPLIT = {"train": {"G03", "G04", "G05", "G06"}, "validation": {"G02", "G09", "G10"}, "test": {"G01", "G07", "G08"}}
ROLES = ("test", "validation", "train", "unknown")


def labelled_datasets(root: Path) -> dict[str, dict]:
    """Labelled slice sets available in this installation."""
    out = {}
    base = root / "dataset" / "sample_data" / "lits_sample"
    if (base / "labels" / "slices.csv").is_file():
        rows = list(csv.DictReader(open(base / "labels" / "slices.csv")))
        out["lits_sample"] = {"label": "Bundled LiTS sample (8 real CT slices)", "root": base, "path_col": "image_path",
                              "rows": rows}
    env = os.environ.get("LTI_DATASET_DIR")         # e.g. the Docker mount; falls back to the in-project copy
    full = Path(env) if env and Path(env).is_dir() else root / "dataset" / "liver_ct_lits_verified_v8"
    lab = root / "data" / "labels" / "verified_slice_labels.csv"
    if full.is_dir() and lab.is_file():
        rows = [r for r in csv.DictReader(open(lab)) if (full / r["path"]).is_file()]
        if rows:
            out["liver_ct_lits_verified_v8"] = {"label": "Verified LiTS-derived dataset (full)", "root": full,
                                                "path_col": "path", "rows": rows}
    return out


def describe_datasets(root: Path) -> list[dict]:
    res = []
    for key, ds in labelled_datasets(root).items():
        rows = ds["rows"]
        splits = sorted({r.get("split") or "" for r in rows})
        res.append({"id": key, "label": ds["label"], "n_slices": len(rows),
                    "n_tumour": sum(int(r["label_tumour"]) for r in rows),
                    "n_liver_free": sum(1 for r in rows if int(r.get("liver_px_512") or 0) == 0),
                    "splits": {s: sum(1 for r in rows if (r.get("split") or "") == s) for s in splits},
                    "groups": sorted({r["group_id"] for r in rows})})
    return res


def model_role(desc: dict, group: str) -> str:
    if desc.get("source") == "v9" or desc.get("version") in ("v9", "v8"):
        for role, groups in V9_SPLIT.items():
            if group in groups:
                return role
        return "unknown"
    if desc.get("source") == "v10":
        return "train"                       # final V10 models: fit on all units (model_card data_split.final_fit)
    return "unknown"


def _auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def _pr_auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())          # average precision


def _ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return None
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == k).mean() * abs(y[i == k].mean() - p[i == k].mean()) for k in range(bins) if (i == k).any()))


def metrics_for(rows: list[dict]) -> dict:
    """rows: per-slice results of one model and one role."""
    ok = [r for r in rows if r["status"] == "ok"]
    y = [r["y"] for r in ok]
    s = [r["p"] for r in ok]
    dec = [r for r in ok if r["label"] in ("Tumour detected", "No tumour detected")]
    tp = sum(1 for r in dec if r["y"] == 1 and r["label"] == "Tumour detected")
    fn = sum(1 for r in dec if r["y"] == 1 and r["label"] == "No tumour detected")
    fp = sum(1 for r in dec if r["y"] == 0 and r["label"] == "Tumour detected")
    tn = sum(1 for r in dec if r["y"] == 0 and r["label"] == "No tumour detected")
    div = lambda a, b: round(a / b, 4) if b else None
    inf = [r["inference_ms"] for r in ok if r["inference_ms"] is not None]
    cpu = [r["cpu_ms"] for r in ok if r.get("cpu_ms") is not None]
    return {"n_slices": len(rows), "n_scored": len(ok), "n_tumour": sum(y), "n_no_tumour": len(y) - sum(y),
            "n_uncertain": sum(1 for r in ok if r["label"] == "Uncertain / unable to classify"),
            "n_not_scored": len(rows) - len(ok),
            "not_scored_reasons": sorted({r["status"] for r in rows if r["status"] != "ok"}),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "accuracy": div(tp + tn, len(dec)), "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "f1": div(2 * tp, 2 * tp + fp + fn),
            "roc_auc": _auc(y, s), "pr_auc": _pr_auc(y, s), "ece": _ece(y, s) if ok else None,
            "inference_ms_median": round(statistics.median(inf), 3) if inf else None,
            "inference_ms_p95": round(sorted(inf)[min(len(inf) - 1, int(round(0.95 * (len(inf) - 1))))], 3) if inf else None,
            "thread_cpu_ms_median": round(statistics.median(cpu), 3) if cpu else None}


def run_benchmark(engine, root: Path, dataset: str, split: str | None = None, max_slices: int = 0,
                  model_ids: list[str] | None = None, progress=None, cancel: threading.Event | None = None) -> dict:
    ds = labelled_datasets(root).get(dataset)
    if ds is None:
        raise ValueError(f"labelled dataset {dataset!r} is not available in this installation")
    rows = [r for r in ds["rows"] if not split or (r.get("split") or "") == split]
    rows.sort(key=lambda r: (r["group_id"], int(r["upstream_index"])))
    if max_slices:
        rows = rows[:max_slices]
    by = {(r.get("scan_id") or r["group_id"], int(r["upstream_index"])): r for r in ds["rows"]}
    runner = ComparisonRunner(engine)
    cancel = cancel or threading.Event()
    descs = {d["id"]: d for d in engine.list(include_legacy=False)}
    per_model: dict[str, list] = {}
    resources = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        if cancel.is_set():
            break
        scan, idx = r.get("scan_id") or r["group_id"], int(r["upstream_index"])

        def nb(off):
            n = by.get((scan, idx + off))
            return (ds["root"] / n[ds["path_col"]]).read_bytes() if n else None
        prev, nxt, prev2, nxt2 = nb(-1), nb(1), nb(-2), nb(2)
        if prev is None or nxt is None:
            prev = nxt = prev2 = nxt2 = None
        elif prev2 is None or nxt2 is None:
            prev2 = nxt2 = None
        data = (ds["root"] / r[ds["path_col"]]).read_bytes()
        prep = runner.prepare(data, Path(r[ds["path_col"]]).name, prev, nxt, prev2, nxt2)
        res = runner.run(prep, runner.plan(prep, model_ids))
        resources.append(res["resources"]["process_cpu_percent_mean"])
        for m in res["models"]:
            if m["status"] == "unavailable" and model_ids is None:
                continue
            d = descs.get(m["model_id"], {})
            per_model.setdefault(m["model_id"], []).append({
                "slice": Path(r[ds["path_col"]]).name, "group": r["group_id"], "role": model_role(d, r["group_id"]),
                "y": int(r["label_tumour"]), "status": m["status"], "label": m.get("label"), "p": m.get("probability"),
                "inference_ms": (m.get("timings_ms") or {}).get("inference"),
                "cpu_ms": (m.get("resources") or {}).get("thread_cpu_ms"), "family": m["family"]})
        if progress:
            progress(i + 1, len(rows))
    models = []
    for mid, rs in per_model.items():
        d = descs.get(mid, {})
        roles = {role: metrics_for([x for x in rs if x["role"] == role]) for role in ROLES
                 if any(x["role"] == role for x in rs)}
        models.append({"model_id": mid, "family": rs[0]["family"], "source": d.get("source"), "version": d.get("version"),
                       "threshold": d.get("threshold"), "by_role": roles, "slices": rs})
    models.sort(key=lambda m: (FAMILIES.index(m["family"]) if m["family"] in FAMILIES else 9, m["model_id"]))
    return {"dataset": dataset, "dataset_label": ds["label"], "split": split or "all", "n_slices": len(rows),
            "n_processed": len(next(iter(per_model.values()), [])), "cancelled": cancel.is_set(),
            "seconds": round(time.perf_counter() - t0, 2), "models": models,
            "mean_process_cpu_percent": round(float(np.mean(resources)), 1) if resources else None,
            "roles": {"test": "slice's scan group held out from the model's training, threshold and calibration",
                      "validation": "group used to choose the model's threshold / calibration (optimistic)",
                      "train": "group used to fit the model (V10 final models were fit on all units): in-sample, "
                               "not an accuracy estimate", "unknown": "training groups of the model not recorded"},
            "notes": ["Slice-level evaluation (expert-mask label per slice); the historical recorded metrics are "
                      "patch-level and are not comparable with these numbers.",
                      "Threshold metrics use decisive calls only; Uncertain and not-scored slices are counted separately.",
                      "Scores for ROC-AUC / PR-AUC / ECE are each model's largest liver-patch probability."]}
