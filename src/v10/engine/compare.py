"""Four-paradigm comparison engine: one uploaded slice, every eligible model, four family pipelines run concurrently.

Workflow
  1. prepare()  — ONE shared front end for every model: decode -> image QC -> OOD check -> 96-px patch grid -> liver
                  gate -> U-Net liver validation -> liver patches and adjacent-slice stacks (3 and 5 slices). Its time is
                  reported separately as common preprocessing.
  2. plan()     — decides for every registered model whether it runs, and if not, why (unavailable in the registry,
                  needs neighbouring slices that were not supplied, input withheld by QC / OOD / liver checks).
  3. run()      — starts the four family pipelines (classical, deep learning, quantum, hybrid) at the same moment on a
                  thread pool of four workers. Inside a family, models run one after another by default
                  (COMPARE_MODEL_WORKERS_PER_FAMILY=1): the models are CPU-bound NumPy / scikit-learn code, so running
                  all of them at once would oversubscribe the CPU and distort the timings. This scheduling strategy is
                  reported with every run ("family_parallel_model_sequential"); it is never described as every model
                  running simultaneously.

Timing (time.perf_counter, monotonic, per model)
  queue_ms      run start (after common preprocessing) -> model start; waiting for its family worker / locks
  features_ms   model-specific feature extraction (radiomics, frozen backbone features, CNN embeddings); features
                shared between models are computed once per run, so the first model that needs them pays the cost
  inference_ms  the model's own computation on prepared inputs (classifier / CNN forward / quantum-circuit simulation)
  total_ms      queue + features + inference + slice-level decision
  cold_start    True for the first inference of that model in this process (lazy caches, memory allocation)
GPU: none of the models uses a GPU (NumPy / scikit-learn on CPU), so no device synchronisation is needed.

Quantum models are simulated exactly on a classical CPU (custom NumPy state-vector simulator); no quantum hardware is
used, and every result says so.
"""
from __future__ import annotations

import os
import platform
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.v10.engine.interface import LABELS, ModelUnavailable, slice_call

FAMILIES = ("classical", "deep_learning", "quantum", "hybrid")
FAMILY_LABEL = {"classical": "Classical ML", "deep_learning": "Deep Learning", "quantum": "Quantum ML",
                "hybrid": "Hybrid ML"}
BAND = 0.10
SCHEDULING = "family_parallel_model_sequential"

# Named locks shared by every run in the process: the same model object (or backbone) is never used by two threads at
# once, so concurrent comparison runs cannot corrupt model state (some NumPy layers cache activations on the object).
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_WARM: set[str] = set()


def named_lock(name: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if name not in _LOCKS:
            _LOCKS[name] = threading.Lock()
        return _LOCKS[name]


def confidence_margin(p: float, thr: float) -> float:
    """Decision margin in [0, 1]: 0 at the threshold, 1 at probability 0 or 1. Not a probability of being correct."""
    if p >= thr:
        return round((p - thr) / max(1 - thr, 1e-9), 4)
    return round((thr - p) / max(thr, 1e-9), 4)


def rss_mb() -> float | None:
    """Resident memory of this process in MB (psutil when installed, /proc on Linux, else None)."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1048576, 1)
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def peak_rss_mb() -> float | None:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / (1048576 if platform.system() == "Darwin" else 1024), 1)
    except Exception:
        return None


def gpu_info() -> dict:
    smi = shutil.which("nvidia-smi")
    return {"detected": bool(smi), "used": False,
            "note": ("an NVIDIA driver is present, but every model runs on the CPU (NumPy / scikit-learn)" if smi else
                     "no GPU detected; every model runs on the CPU (NumPy / scikit-learn)")}


class ResourceSampler:
    """Samples process CPU utilisation (process CPU time / wall time / logical CPUs) and resident memory every
    `every` seconds while a run executes. CPU time includes every thread of the process (BLAS threads too)."""

    def __init__(self, every: float = 0.1):
        self.every, self.samples, self._stop = every, [], threading.Event()
        self.ncpu = os.cpu_count() or 1
        self._t = threading.Thread(target=self._loop, name="cmp-resources", daemon=True)

    def _loop(self):
        t0 = w_prev = time.perf_counter()
        c_prev = time.process_time()
        while not self._stop.wait(self.every):
            w, c = time.perf_counter(), time.process_time()
            pct = 100.0 * (c - c_prev) / max(w - w_prev, 1e-9) / self.ncpu
            self.samples.append((round((w - t0) * 1000, 1), round(pct, 1), rss_mb()))
            w_prev, c_prev = w, c

    def __enter__(self):
        self.rss_start, self.cpu0, self.w0 = rss_mb(), time.process_time(), time.perf_counter()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(1)
        self.rss_end, self.cpu1, self.w1 = rss_mb(), time.process_time(), time.perf_counter()

    def summary(self) -> dict:
        cpu = [s[1] for s in self.samples]
        rss = [s[2] for s in self.samples if s[2] is not None]
        step = max(1, len(self.samples) // 150)
        wall = max(self.w1 - self.w0, 1e-9)
        return {"cpu_count": self.ncpu, "process_cpu_seconds": round(self.cpu1 - self.cpu0, 3),
                "process_cpu_percent_mean": round(100 * (self.cpu1 - self.cpu0) / wall / self.ncpu, 1),
                "process_cpu_percent_peak": max(cpu) if cpu else None,
                "rss_mb_start": self.rss_start, "rss_mb_peak": max(rss) if rss else peak_rss_mb(),
                "rss_mb_end": self.rss_end, "samples": self.samples[::step],
                "sample_fields": ["ms_since_run_start", "process_cpu_percent_of_all_cpus", "rss_mb"],
                "gpu": gpu_info(),
                "method": "process CPU time / wall time / logical CPUs, sampled every 100 ms; memory = resident set "
                          "size of the API process (shared by concurrent runs)"}


class FeatureCache:
    """Per-run cache of features shared between models; one lock per key, so two families never compute the same
    features twice and never wait for unrelated features."""

    def __init__(self):
        self._vals: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.compute_ms: dict[str, float] = {}

    def get(self, key: str, fn: Callable[[], object]):
        with self._guard:
            lk = self._locks.setdefault(key, threading.Lock())
        with lk:
            if key not in self._vals:
                t0 = time.perf_counter()
                with named_lock("feature:" + key.split(":")[0] + ":" + key.split(":")[-1]):
                    self._vals[key] = fn()
                self.compute_ms[key] = round((time.perf_counter() - t0) * 1000, 2)
            return self._vals[key]


@dataclass
class PreparedInput:
    run_id: str
    fe: object                         # patch_pipeline.FrontEnd
    ctx: object | None                 # src.inference.registry.PatchContext shared by the V9 models
    cache: FeatureCache
    has_context3: bool
    has_context5: bool
    timings_ms: dict = field(default_factory=dict)
    stop_status: str | None = None     # withheld | rejected (front end stopped)
    stop_reason: str | None = None

    def summary(self) -> dict:
        fe = self.fe
        return {"image": fe.meta, "quality": fe.quality, "ood": fe.ood, "liver_gate": fe.gate,
                "liver_check": fe.liver_check, "context": {"adjacent_slices": self.has_context3,
                                                          "five_slice_context": self.has_context5,
                                                          "note": fe.context_note},
                "n_liver_patches": int(len(fe.liver_idx)) if fe.liver_idx is not None and not fe.stop_reason else 0,
                "stopped": self.stop_reason, "stop_status": self.stop_status, "timings_ms": self.timings_ms}


class ComparisonRunner:
    def __init__(self, engine, model_workers_per_family: int | None = None):
        self.engine = engine
        self.model_workers = max(1, int(model_workers_per_family or os.environ.get("COMPARE_MODEL_WORKERS_PER_FAMILY", 1)))

    # ------------------------------------------------------------------ 1. shared front end
    def prepare(self, data: bytes, filename: str | None = None, prev: bytes | None = None, nxt: bytes | None = None,
                prev2: bytes | None = None, nxt2: bytes | None = None, run_id: str | None = None) -> PreparedInput:
        from src.v10.engine import patch_pipeline as PP
        from src.inference.registry import PatchContext
        if self.engine.v9 is None:
            raise ModelUnavailable("the V9 front end (OOD detector, liver gate) is not loaded")
        eff = (self.engine.weights or {}).get("efficientnet_b0", {})
        if not eff.get("ok"):
            raise ModelUnavailable("EfficientNet-B0 ImageNet weights are missing or invalid: " + str(eff.get("reason")))
        t0 = time.perf_counter()
        fe = PP.run(self.engine.v9, data, filename, prev, nxt, prev2, nxt2)
        timings = dict(fe.timings_ms)
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        stop_status = None
        if fe.stop_reason:
            stop_status = "rejected" if fe.quality.get("status") != "ok" else (
                "out_of_distribution" if fe.ood.get("status") == "out_of_distribution" else "withheld")
        ctx = None
        if not fe.stop_reason:
            reg = self.engine.v9
            emb = reg.models[reg.selected_cnn].m if reg.selected_cnn in reg.models else None
            stacks = fe.stacks if fe.stacks is not None else np.repeat(fe.patches[:, None], 3, 1)
            ctx = PatchContext(fe.patches, stacks, reg.ft_model(), fe.stacks5, emb)
        return PreparedInput(run_id or uuid.uuid4().hex, fe, ctx, FeatureCache(), fe.stacks is not None,
                             fe.stacks5 is not None, timings, stop_status, fe.stop_reason)

    # ------------------------------------------------------------------ 2. eligibility
    def model_info(self, d: dict) -> dict:
        circ = d.get("circuit") if isinstance(d.get("circuit"), dict) else {}
        conf = d.get("configuration") if isinstance(d.get("configuration"), dict) else {}
        qubits = circ.get("qubits") or conf.get("qubits") or (circ.get("vqc") or {}).get("qubits")
        fam = d.get("compare_family") or self.engine.compare_family(d)
        quantum = fam in ("quantum", "hybrid") and d.get("execution_mode") == "custom_numpy_simulator"
        return {"model_id": d["id"], "name": d.get("name") or d["id"], "family": fam,
                "architecture": d.get("architecture"), "version": d.get("version"), "source": d.get("source"),
                "role": d.get("role"), "threshold": d.get("threshold"), "calibrated": d.get("calibrated"),
                "context_slices_each_side": int(d.get("context_slices_each_side") or 0),
                "execution_mode": d.get("execution_mode"), "load_ms": d.get("load_ms"),
                "quantum": ({"qubits": qubits, "backend": "exact state-vector simulation on CPU (custom NumPy)",
                             "hardware": False, "circuit_depth": (circ.get("gate_counts") or {}).get("depth") or
                             circ.get("depth")} if quantum else None),
                "features": (d.get("input_contract") or {}).get("features")}

    def plan(self, prepared: PreparedInput | None, model_ids: list[str] | None = None,
             include_unavailable: bool = True) -> list[dict]:
        out = []
        descs = self.engine.list(include_legacy=False)
        wanted = set(model_ids) if model_ids else None
        for d in descs:
            if d.get("family") == "segmentation" or d.get("role") == "software_test":
                continue
            if wanted is not None and d["id"] not in wanted:
                continue
            item = self.model_info(d)
            item.update(eligible=False, skip_status=None, skip_reason=None)
            if not d.get("servable"):
                if not include_unavailable and wanted is None:
                    continue
                item.update(skip_status="unavailable", skip_reason=f"{d.get('availability') or d.get('status')}: "
                                                                   f"{d.get('reason') or 'not servable'}")
            elif prepared is not None and prepared.stop_reason:
                item.update(skip_status=prepared.stop_status, skip_reason=prepared.stop_reason)
            elif item["context_slices_each_side"] >= 2 and (prepared is None or not prepared.has_context5):
                item.update(skip_status="needs_context",
                            skip_reason="reads 5 slices: supply two slices above and two below the target slice")
            elif item["context_slices_each_side"] == 1 and (prepared is None or not prepared.has_context3):
                item.update(skip_status="needs_context",
                            skip_reason="reads adjacent slices: supply the previous and next slices")
            else:
                item["eligible"] = True
            out.append(item)
        if wanted:
            unknown = wanted - {o["model_id"] for o in out}
            for mid in sorted(unknown):
                out.append({"model_id": mid, "name": mid, "family": "classical", "eligible": False,
                            "skip_status": "unavailable", "skip_reason": "unknown model id"})
        return out

    # ------------------------------------------------------------------ 3. per-model execution
    def _features_v9(self, lm, prepared: PreparedInput) -> None:
        """Pre-compute (once per run) the features this V9 model reads, outside its timed inference."""
        ctx, cache = prepared.ctx, prepared.cache
        k, meta = lm.kind, lm.meta
        if k == "cnn_finetune":
            arch = meta["arch"]
            ctx._frozen[arch] = cache.get(f"frozen:{arch}", lambda: ctx.frozen(arch))
        elif k == "radiomics_baseline":
            ctx._rad = cache.get("radiomics:v8", ctx.radiomics)
        elif str(k).startswith("v9_"):
            if meta.get("features") == "cnn25d_v9":
                ctx._emb = cache.get("embedding:cnn25d_v9", ctx.cnn_embedding)
            else:
                ctx._ftgap = cache.get("gap:finetuned", ctx.finetuned_gap)
        elif k in ("pca8_control", "vqc", "quantum_kernel", "hybrid", "classical_twin"):
            if meta.get("feature_source") == "finetuned":
                ctx._ftgap = cache.get("gap:finetuned", ctx.finetuned_gap)
            else:
                ctx._frozen["efficientnet_b0"] = cache.get("frozen:efficientnet_b0", lambda: ctx.frozen("efficientnet_b0"))
                ctx._gap = cache.get("gap:imagenet", ctx.imagenet_gap)
        # numpy_cnn: down-sampling to 64 px happens inside the model; radiomics_v9: normalisation-specific features
        # are computed inside the model (both are part of inference_ms for those models).

    def _execute(self, item: dict, prepared: PreparedInput, t_run0: float) -> dict:
        from src.v10.engine.v9_adapter import V9Model
        from src.v10.engine.v10_models import V10PatchModel
        mid = item["model_id"]
        res = dict(item)
        t_start = time.perf_counter()
        res["timings_ms"] = {"queue": round((t_start - t_run0) * 1000, 2)}
        model = self.engine.get(mid)
        fe = prepared.fe
        c0 = time.thread_time()
        positions = [fe.positions[i] for i in fe.liver_idx]
        t_f0 = time.perf_counter()
        if isinstance(model, V9Model):
            lm = self.engine.v9.models[mid]
            self._features_v9(lm, prepared)
            t_f1 = time.perf_counter()
            with named_lock("model:" + mid):
                cold = mid not in _WARM
                t_i0 = time.perf_counter()
                _, p = lm.probabilities(prepared.ctx)
                t_i1 = time.perf_counter()
                _WARM.add(mid)
            thr = float(lm.threshold)
            note = ("calibrated (Platt scaling on V9 validation scans)" if lm.calibrator is not None else
                    "uncalibrated logistic transform of the model score")
        elif isinstance(model, V10PatchModel):
            model.load_model()
            if model.input_kind == "stacks":
                from src.inference.registry import _down64
                F = prepared.cache.get("stacks64", lambda: _down64(fe.stacks).astype(np.float32) / 255.0)
            elif model.input_kind == "patches":
                F = fe.patches
            else:
                fs = model.card["feature_set"]
                F = prepared.cache.get(f"v10features:{fs}", lambda: model.features(fe.patches))
            t_f1 = time.perf_counter()
            with named_lock("model:" + mid):
                cold = mid not in _WARM
                t_i0 = time.perf_counter()
                p = model.calibrate(model._scores(np.asarray(F, float) if model.input_kind == "features" else F))
                t_i1 = time.perf_counter()
                _WARM.add(mid)
            thr = float(model.card["threshold"])
            note = model.calibration_note
        else:
            raise ModelUnavailable(f"{type(model).__name__} is not a patch model")
        p = np.asarray(p, float)
        k = int(np.argmax(p))
        pmax = float(p[k])
        label = slice_call(pmax, thr, BAND)
        t_end = time.perf_counter()
        res.update(status="ok", label=label, probability=round(pmax, 4), threshold=round(thr, 4), band=BAND,
                   confidence=confidence_margin(pmax, thr), n_liver_patches=int(len(p)),
                   saturated=bool(pmax >= 0.999), probability_note=note, cold_start=cold,
                   top_patch={"y": int(positions[k][0]), "x": int(positions[k][1])},
                   patch_probabilities=[{"y": int(y), "x": int(x), "p": round(float(v), 4)}
                                        for (y, x), v in zip(positions, p)])
        res["timings_ms"].update(features=round((t_f1 - t_f0) * 1000, 2), inference=round((t_i1 - t_i0) * 1000, 2),
                                 lock_wait=round((t_i0 - t_f1) * 1000, 2), total=round((t_end - t_start) * 1000, 2))
        cpu_ms = (time.thread_time() - c0) * 1000
        busy = max((t_f1 - t_f0) + (t_i1 - t_i0), 1e-9) * 1000
        res["resources"] = {"thread_cpu_ms": round(cpu_ms, 2), "thread_cpu_utilisation": round(min(cpu_ms / busy, 9.99), 3),
                            "note": "CPU time of the executing thread (features + inference); BLAS worker threads "
                                    "are counted only in the run-level process CPU"}
        return res

    # ------------------------------------------------------------------ 4. concurrent run
    def run(self, prepared: PreparedInput, plan: list[dict], emit: Callable[[str, dict], None] | None = None,
            cancel: threading.Event | None = None, deadline_s: float | None = None) -> dict:
        emit = emit or (lambda ev, payload: None)
        cancel = cancel or threading.Event()
        t_run0 = time.perf_counter()
        t_deadline = t_run0 + deadline_s if deadline_s else None
        results: dict[str, dict] = {}
        lock = threading.Lock()
        fam_meta: dict[str, dict] = {}

        def skipped(item: dict, status: str, reason: str) -> dict:
            r = dict(item, status=status, label=None if status != "withheld" else LABELS[2], probability=None,
                     reason=reason, timings_ms={})
            return r

        for item in plan:                                  # report non-runnable models up front
            if not item["eligible"]:
                r = skipped(item, item["skip_status"], item["skip_reason"])
                results[item["model_id"]] = r
                emit("model_skipped", r)

        def family_task(fam: str):
            items = [i for i in plan if i["family"] == fam and i["eligible"]]
            t_f0 = time.perf_counter()
            fam_meta[fam] = {"started_ms": round((t_f0 - t_run0) * 1000, 2), "thread": threading.current_thread().name}
            emit("family_started", {"family": fam, "label": FAMILY_LABEL[fam], "n_models": len(items),
                                    "started_ms": fam_meta[fam]["started_ms"]})

            def one(item):
                mid = item["model_id"]
                if cancel.is_set():
                    r = skipped(item, "cancelled", "run cancelled before this model started")
                elif t_deadline and time.perf_counter() > t_deadline:
                    r = skipped(item, "timeout", f"run time limit ({deadline_s:.0f} s) reached before this model started")
                else:
                    emit("model_started", {"model_id": mid, "family": fam,
                                           "started_ms": round((time.perf_counter() - t_run0) * 1000, 2)})
                    try:
                        r = self._execute(item, prepared, t_run0)
                        r["started_ms"] = round(r["timings_ms"]["queue"], 2)
                        r["finished_ms"] = round((time.perf_counter() - t_run0) * 1000, 2)
                    except Exception as e:                  # one model failing never stops the others
                        r = skipped(item, "error", f"{type(e).__name__}: {e}")
                        r["trace"] = traceback.format_exc(limit=3)
                with lock:
                    results[mid] = r
                emit("model_completed" if r["status"] == "ok" else "model_failed", r)
                return r

            if self.model_workers == 1:
                done = [one(i) for i in items]
            else:
                with ThreadPoolExecutor(max_workers=self.model_workers, thread_name_prefix=f"cmp-{fam}") as ex:
                    done = list(ex.map(one, items))
            t_f1 = time.perf_counter()
            fs = family_summary(fam, [results[i["model_id"]] for i in plan if i["family"] == fam])
            fs.update(started_ms=fam_meta[fam]["started_ms"], finished_ms=round((t_f1 - t_run0) * 1000, 2),
                      wall_ms=round((t_f1 - t_f0) * 1000, 2), thread=fam_meta[fam]["thread"])
            fam_meta[fam]["summary"] = fs
            emit("family_completed", fs)
            return done

        with ResourceSampler() as sampler:
            with ThreadPoolExecutor(max_workers=len(FAMILIES), thread_name_prefix="cmp-family") as ex:
                futures = [ex.submit(family_task, fam) for fam in FAMILIES]
                for f in futures:
                    f.result()
        t_run1 = time.perf_counter()
        ordered = [results[i["model_id"]] for i in plan]
        fams = {f: fam_meta[f]["summary"] for f in FAMILIES}
        out = {"run_id": prepared.run_id, "scheduling": {"strategy": SCHEDULING, "family_workers": len(FAMILIES),
                                                         "model_workers_per_family": self.model_workers,
                                                         "description": scheduling_text(self.model_workers)},
               "preprocessing": prepared.summary(), "feature_cache_ms": dict(prepared.cache.compute_ms),
               "models": ordered, "families": fams, "agreement": agreement(ordered),
               "ranking": ranking(ordered, fams), "wall_ms": round((t_run1 - t_run0) * 1000, 2),
               "cancelled": cancel.is_set(),
               "quantum_execution": "simulated: exact state-vector simulation on a classical CPU (no quantum hardware)",
               "family_overlap": family_overlap(fams), "resources": sampler.summary()}
        return out


# ---------------------------------------------------------------------- aggregation helpers
def scheduling_text(model_workers: int) -> str:
    inner = "one model at a time" if model_workers == 1 else f"up to {model_workers} models at a time"
    return (f"The four family pipelines start together on four worker threads and run concurrently; inside each family "
            f"models run {inner} to avoid CPU oversubscription. Shared features are computed once per run.")


def family_summary(fam: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    attempted = [r for r in rows if r["status"] in ("ok", "error", "timeout", "cancelled")]
    failed = [r for r in rows if r["status"] == "error"]
    inf = [(r["timings_ms"]["inference"], r["model_id"]) for r in ok]
    counts = {lab: sum(1 for r in ok if r["label"] == lab) for lab in LABELS}
    combined = None
    if ok:
        top = max(counts.values())
        winners = [lab for lab, c in counts.items() if c == top]
        combined = winners[0] if len(winners) == 1 else LABELS[2]
    return {"family": fam, "label": FAMILY_LABEL[fam], "n_registered": len(rows), "n_attempted": len(attempted),
            "n_completed": len(ok), "n_failed": len(failed),
            "n_skipped": sum(1 for r in rows if r["status"] in ("unavailable", "needs_context", "withheld",
                                                                  "rejected", "out_of_distribution")),
            "success_rate": round(len(ok) / len(attempted), 4) if attempted else None,
            "inference_ms_sum": round(sum(v for v, _ in inf), 2) if inf else None,
            "inference_ms_mean": round(float(np.mean([v for v, _ in inf])), 2) if inf else None,
            "features_ms_sum": round(sum(r["timings_ms"].get("features", 0) for r in ok), 2) if ok else None,
            "thread_cpu_ms_sum": round(sum((r.get("resources") or {}).get("thread_cpu_ms", 0) for r in ok), 2) if ok else None,
            "queue_ms_mean": round(float(np.mean([r["timings_ms"]["queue"] for r in ok])), 2) if ok else None,
            "fastest_model": min(inf)[1] if inf else None, "fastest_ms": min(inf)[0] if inf else None,
            "slowest_model": max(inf)[1] if inf else None, "slowest_ms": max(inf)[0] if inf else None,
            "label_counts": counts, "combined_label": combined,
            "combined_rule": "majority vote of completed models in this family (ties -> Uncertain); not a validated "
                             "ensemble", "mean_probability": round(float(np.mean([r["probability"] for r in ok])), 4)
            if ok else None}


def agreement(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    counts = {lab: sum(1 for r in ok if r["label"] == lab) for lab in LABELS}
    top = max(counts.values()) if ok else 0
    return {"tumour": counts[LABELS[0]], "no_tumour": counts[LABELS[1]], "uncertain": counts[LABELS[2]],
            "failed_or_skipped": len(rows) - len(ok), "completed": len(ok),
            "agreement_pct": round(100 * top / len(ok), 1) if ok else None,
            "majority_label": (max(counts, key=counts.get) if ok and list(counts.values()).count(top) == 1 else None),
            "note": "Agreement between models is not evidence that the prediction is correct."}


def ranking(rows: list[dict], fams: dict) -> dict:
    ok = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: r["timings_ms"]["inference"])
    fam_rank = sorted([f for f in fams.values() if f["inference_ms_mean"] is not None],
                      key=lambda f: f["inference_ms_mean"])
    return {"models_by_inference_ms": [{"model_id": r["model_id"], "family": r["family"],
                                        "inference_ms": r["timings_ms"]["inference"],
                                        "features_ms": r["timings_ms"].get("features"),
                                        "total_ms": r["timings_ms"]["total"]} for r in ok],
            "families_by_mean_inference_ms": [{"family": f["family"], "inference_ms_mean": f["inference_ms_mean"],
                                               "wall_ms": f.get("wall_ms")} for f in fam_rank],
            "fastest_model": ok[0]["model_id"] if ok else None, "slowest_model": ok[-1]["model_id"] if ok else None,
            "fastest_family": fam_rank[0]["family"] if fam_rank else None,
            "slowest_family": fam_rank[-1]["family"] if fam_rank else None,
            "rule": "families ranked by mean per-model inference time of completed models (raw model execution, "
                    "excluding shared preprocessing and feature extraction)"}


def family_overlap(fams: dict) -> dict:
    """Evidence of concurrency: intervals [started, finished] of the family pipelines and how much they overlapped."""
    iv = {f: (s.get("started_ms"), s.get("finished_ms")) for f, s in fams.items() if s.get("started_ms") is not None}
    if not iv:
        return {}
    start_spread = max(a for a, _ in iv.values()) - min(a for a, _ in iv.values())
    lo, hi = max(a for a, _ in iv.values()), min(b for _, b in iv.values())
    return {"intervals_ms": iv, "start_spread_ms": round(start_spread, 2),
            "all_four_overlap_ms": round(max(0.0, hi - lo), 2)}
