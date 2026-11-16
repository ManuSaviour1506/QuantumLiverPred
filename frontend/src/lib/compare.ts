// Four-paradigm comparison: types, API calls and the live event stream (Server-Sent Events read with fetch so the
// bearer token can be sent; reconnects with last_event_id and falls back to polling the run).
import { useEffect, useReducer, useRef } from "react";
import { API_BASE, api, refreshSession, tokens, upload } from "./api";

export type Family = "classical" | "deep_learning" | "quantum" | "hybrid";
export const FAMILIES: Family[] = ["classical", "deep_learning", "quantum", "hybrid"];
export const FAMILY_LABEL: Record<Family, string> = { classical: "Classical ML", deep_learning: "Deep Learning",
  quantum: "Quantum ML", hybrid: "Hybrid ML" };
export const FAMILY_COLOR: Record<Family, string> = { classical: "#60a5fa", deep_learning: "#a78bfa",
  quantum: "#22d3ee", hybrid: "#f472b6" };
export const FAMILY_BLURB: Record<Family, string> = {
  classical: "Radiomics features with logistic regression, SVM, forests, boosting, k-NN, MLP",
  deep_learning: "CNNs trained from scratch (2-D, 2.5-D, 3-D, 5-slice) and fine-tuned ImageNet backbones",
  quantum: "Variational quantum classifiers and quantum-kernel SVMs (exact CPU simulation)",
  hybrid: "Classical + quantum feature fusion, CNN + quantum heads and late fusion",
};
export const LABEL_SHORT: Record<string, string> = { "Tumour detected": "Tumour", "No tumour detected": "No tumour",
  "Uncertain / unable to classify": "Uncertain" };

export interface Timings { queue?: number; features?: number; inference?: number; lock_wait?: number; total?: number }
export interface ModelResult {
  model_id: string; name: string; family: Family; architecture?: string; version?: string; source?: string;
  status: string; label?: string | null; probability?: number | null; threshold?: number | null; band?: number;
  confidence?: number | null; saturated?: boolean; cold_start?: boolean; reason?: string | null;
  probability_note?: string; timings_ms?: Timings; started_ms?: number; finished_ms?: number;
  n_liver_patches?: number; execution_mode?: string | null; context_slices_each_side?: number;
  quantum?: { qubits?: number | null; backend: string; hardware: boolean; circuit_depth?: number | null } | null;
  load_ms?: number | null; resources?: { thread_cpu_ms: number; thread_cpu_utilisation: number; note?: string };
  top_patch?: { y: number; x: number }; patch_probabilities?: { y: number; x: number; p: number }[];
  eligible?: boolean; skip_status?: string | null; skip_reason?: string | null; features?: string | null;
}
export interface FamilySummary {
  family: Family; label: string; n_registered: number; n_attempted: number; n_completed: number; n_failed: number;
  n_skipped: number; success_rate: number | null; inference_ms_sum: number | null; inference_ms_mean: number | null;
  features_ms_sum: number | null; queue_ms_mean: number | null; fastest_model: string | null; fastest_ms: number | null;
  slowest_model: string | null; slowest_ms: number | null; label_counts: Record<string, number>;
  combined_label: string | null; combined_rule: string; mean_probability: number | null;
  started_ms?: number; finished_ms?: number; wall_ms?: number;
}
export interface Agreement { tumour: number; no_tumour: number; uncertain: number; failed_or_skipped: number;
  completed: number; agreement_pct: number | null; majority_label: string | null; note: string }
export interface Preprocessing {
  image?: Record<string, any>; quality?: { status: string; reasons: string[]; warnings: string[]; stats?: Record<string, number> };
  ood?: { status: string; d2_ratio?: number; residual_ratio?: number; rule?: string };
  liver_gate?: { status: string; n_candidates?: number; n_liver_patches?: number; threshold?: number };
  liver_check?: { status: string; unet_liver_px?: number; min_px?: number; method?: string; context?: string;
    gate_patch_liver_coverage?: number[] };
  context?: { adjacent_slices: boolean; five_slice_context: boolean; note?: string | null };
  n_liver_patches?: number; stopped?: string | null; stop_status?: string | null; timings_ms?: Record<string, number>;
}
export interface RunResult {
  run_id: string; scheduling: { strategy: string; family_workers: number; model_workers_per_family: number; description: string };
  preprocessing: Preprocessing; feature_cache_ms: Record<string, number>; models: ModelResult[];
  families: Record<Family, FamilySummary>; agreement: Agreement;
  ranking: { models_by_inference_ms: { model_id: string; family: Family; inference_ms: number; features_ms?: number; total_ms: number }[];
    families_by_mean_inference_ms: { family: Family; inference_ms_mean: number; wall_ms?: number }[];
    fastest_family: Family | null; slowest_family: Family | null; rule: string;
    fastest_model?: string | null; slowest_model?: string | null };
  wall_ms: number; end_to_end_ms?: number; cancelled: boolean; quantum_execution: string;
  family_overlap: { intervals_ms?: Record<string, [number, number]>; start_spread_ms?: number; all_four_overlap_ms?: number };
  hardware?: Record<string, any>; upload?: Record<string, any>; engine?: Record<string, any>;
  resources?: { cpu_count: number; process_cpu_seconds: number; process_cpu_percent_mean: number;
    process_cpu_percent_peak: number | null; rss_mb_start: number | null; rss_mb_peak: number | null; rss_mb_end: number | null;
    samples: [number, number, number | null][]; gpu: { detected: boolean; used: boolean; note: string }; method: string };
}
export interface Run {
  id: string; upload_id: string; mode: string; status: string; n_models?: number | null; n_completed?: number | null;
  n_failed?: number | null; wall_ms?: number | null; fastest_family?: Family | null; slowest_family?: Family | null;
  majority_label?: string | null; agreement_pct?: number | null; error?: string | null; created_at: string;
  started_at?: string | null; finished_at?: string | null; result?: RunResult | null; hardware?: Record<string, any>;
  requested_models?: string[];
}
export interface Compatibility { model_id: string; name: string; family: Family; eligible: boolean;
  skip_status?: string | null; skip_reason?: string | null; context_slices_each_side?: number }
export interface UploadOut {
  id: string; input_name?: string | null; input_format: string; input_bytes: number; input_sha256: string;
  modality: string; source: string; sample_id?: string | null; context_slices: string[]; created_at: string;
  analysis: Preprocessing & { compatibility: Compatibility[]; eligible_by_family: Record<Family, number>;
    preview_png_b64?: string | null; modality?: string; modality_note?: string };
}
export interface Sample { id: string; source: string; category: "real_ct" | "synthetic_phantom"; label_tumour: number | null;
  liver_px_512?: number; tumour_px_512?: number; scan_id: string; split?: string | null; note?: string | null;
  context: { adjacent: boolean; five_slice: boolean; ids: Record<string, string> } }

export const compareApi = {
  samples: () => api.get<Sample[]>("/api/v1/compare/samples"),
  uploadFile: (form: FormData, onProgress?: (f: number) => void) => upload<UploadOut>("/api/v1/upload", form, onProgress),
  uploadSample: (sample_id: string, modality = "CT") => api.post<UploadOut>("/api/v1/upload/sample", { sample_id, modality }),
  getUpload: (id: string) => api.get<UploadOut>(`/api/v1/uploads/${id}`),
  runAll: (upload_id: string) => api.post<{ run_id: string }>("/api/v1/compare/run-all", { upload_id }),
  runSelected: (upload_id: string, model_ids: string[]) =>
    api.post<{ run_id: string }>("/api/v1/compare/run-selected", { upload_id, model_ids }),
  run: (id: string) => api.get<Run>(`/api/v1/compare/${id}`),
  cancel: (id: string) => api.post(`/api/v1/compare/${id}/cancel`),
  history: (limit = 50, offset = 0) => api.get<{ items: any[]; total: number }>("/api/v1/history", { limit, offset }),
  compareRuns: (a: string, b: string) => api.get<any>("/api/v1/history/compare", { a, b }),
  benchmarks: () => api.get<any>("/api/v1/benchmarks"),
  availability: () => api.get<any>("/api/v1/models/availability"),
  status: () => api.get<any>("/api/v1/system/status"),
  metrics: () => api.get<any>("/api/v1/system/metrics"),
  benchDatasets: () => api.get<any[]>("/api/v1/benchmarks/datasets"),
  benchRuns: () => api.get<any[]>("/api/v1/benchmarks/runs"),
  benchRun: (id: string) => api.get<any>(`/api/v1/benchmarks/runs/${id}`),
  startBench: (body: { dataset: string; split?: string | null; max_slices?: number; model_ids?: string[] | null }) =>
    api.post<{ id: string }>("/api/v1/benchmarks/runs", body),
  cancelBench: (id: string) => api.post(`/api/v1/benchmarks/runs/${id}/cancel`),
};

// ------------------------------------------------------------------------------------------- live state
export type Phase = "connecting" | "queued" | "preprocessing" | "running" | "completed" | "cancelled" | "failed";
export interface FamilyLive { status: "waiting" | "running" | "completed"; started_ms?: number; summary?: FamilySummary;
  n_models?: number; startedAt?: number }
export interface LiveState {
  phase: Phase; t0?: number; preprocessing?: Preprocessing; plan: ModelResult[]; models: Record<string, ModelResult>;
  running: Record<string, number>; families: Record<Family, FamilyLive>; result?: RunResult; error?: string;
  events: { t: number; event: string; label: string }[]; cancelRequested?: boolean; transport: "sse" | "polling";
  runT0Perf?: number;          // performance.now() when the family pipelines started (client clock)
}

export const initialLive = (): LiveState => ({
  phase: "connecting", plan: [], models: {}, running: {}, events: [], transport: "sse",
  families: { classical: { status: "waiting" }, deep_learning: { status: "waiting" }, quantum: { status: "waiting" },
    hybrid: { status: "waiting" } },
});

type Action = { type: "event"; event: string; data: any; t: number } | { type: "snapshot"; run: Run } |
  { type: "error"; message: string } | { type: "transport"; value: "sse" | "polling" } | { type: "reset" };

function describe(event: string, d: any): string {
  switch (event) {
    case "family_started": return `${FAMILY_LABEL[d.family as Family]} pipeline started (${d.n_models} models)`;
    case "family_completed": return `${FAMILY_LABEL[d.family as Family]} finished in ${Math.round(d.wall_ms)} ms`;
    case "model_started": return `${d.model_id} started`;
    case "model_completed": return `${d.model_id}: ${LABEL_SHORT[d.label] || d.label} (p=${d.probability?.toFixed(3)}, ${d.timings_ms?.inference?.toFixed(1)} ms)`;
    case "model_failed": return `${d.model_id}: ${d.status} — ${d.reason || ""}`;
    case "preprocessing_completed": return d.stopped ? `Input withheld: ${d.stopped}` : `Shared preprocessing done (${Math.round(d.timings_ms?.total || 0)} ms)`;
    case "run_completed": return "Comparison completed";
    case "run_cancelled": return "Run cancelled";
    case "run_failed": return `Run failed: ${d.error}`;
    case "cancel_requested": return "Cancel requested";
    case "run_started": return "Run started — four family pipelines launching";
    default: return event.replace(/_/g, " ");
  }
}

export type LiveAction = Action;

/** Pure reducer of the live run state (exported for tests). */
export function liveReducer(s: LiveState, a: Action): LiveState {
  if (a.type === "reset") return initialLive();
  if (a.type === "error") return { ...s, error: a.message };
  if (a.type === "transport") return { ...s, transport: a.value };
  if (a.type === "snapshot") {
    const r = a.run;
    if (r.result && ["completed", "cancelled"].includes(r.status)) return fromResult(s, r.result, r.status as Phase);
    if (r.status === "failed") return { ...s, phase: "failed", error: r.error || "failed" };
    return { ...s, phase: r.status === "queued" ? "queued" : s.phase === "connecting" ? "running" : s.phase };
  }
  const { event, data: d, t } = a;
  const ev = event === "model_skipped" ? s.events : [...s.events, { t, event, label: describe(event, d) }].slice(-400);
  const n: LiveState = { ...s, events: ev, t0: s.t0 ?? t };
  switch (event) {
    case "run_queued": return { ...n, phase: "queued" };
    case "run_started": return { ...n, phase: "preprocessing" };
    case "preprocessing_completed": return { ...n, preprocessing: d, phase: "running" };
    case "plan": {
      const models = { ...n.models };
      for (const m of d.models as ModelResult[]) models[m.model_id] = { ...m, ...(models[m.model_id] || {}),
        status: models[m.model_id]?.status || (m.eligible ? "pending" : m.skip_status || "skipped") };
      return { ...n, plan: d.models, models };
    }
    case "family_started": return { ...n, runT0Perf: n.runT0Perf ?? t - (d.started_ms || 0), families: { ...n.families, [d.family]: { status: "running", started_ms: d.started_ms,
      n_models: d.n_models, startedAt: t } } };
    case "model_started": return { ...n, running: { ...n.running, [d.model_id]: t },
      models: { ...n.models, [d.model_id]: { ...(n.models[d.model_id] || { model_id: d.model_id, family: d.family }), status: "running",
        started_ms: d.started_ms } as ModelResult } };
    case "model_completed": case "model_failed": case "model_skipped": {
      const running = { ...n.running };
      delete running[d.model_id];
      return { ...n, running, models: { ...n.models, [d.model_id]: { ...(n.models[d.model_id] || {}), ...d } } };
    }
    case "family_completed": return { ...n, families: { ...n.families, [d.family]: { ...n.families[d.family as Family],
      status: "completed", summary: d } } };
    case "cancel_requested": return { ...n, cancelRequested: true };
    case "run_completed": case "run_cancelled":
      return d.result ? fromResult(n, d.result, event === "run_completed" ? "completed" : "cancelled") : { ...n, phase: "completed" };
    case "run_failed": return { ...n, phase: "failed", error: d.error };
    default: return n;
  }
}

function fromResult(s: LiveState, r: RunResult, phase: Phase): LiveState {
  const models: Record<string, ModelResult> = {};
  for (const m of r.models) models[m.model_id] = m;
  const families = { ...s.families };
  for (const f of FAMILIES) families[f] = { status: "completed", summary: r.families[f], started_ms: r.families[f]?.started_ms };
  return { ...s, phase, result: r, models, running: {}, families, preprocessing: r.preprocessing,
    plan: s.plan.length ? s.plan : r.models };
}

async function* sseLines(res: Response) {
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      yield block;
    }
  }
}

/** Live state of a comparison run: SSE stream with automatic reconnect, then a final snapshot. */
export function useComparisonRun(runId: string | null): [LiveState, () => Promise<void>] {
  const [state, dispatch] = useReducer(liveReducer, undefined, initialLive);
  const stop = useRef(false);
  useEffect(() => {
    if (!runId) return;
    stop.current = false;
    dispatch({ type: "reset" });
    let lastId = -1;
    const ctl = new AbortController();
    let failures = 0;
    (async () => {
      while (!stop.current) {
        try {
          let res = await fetch(`${API_BASE}/api/v1/compare/${runId}/stream?last_event_id=${lastId}`, {
            headers: tokens.access ? { Authorization: `Bearer ${tokens.access}` } : {}, signal: ctl.signal });
          if (res.status === 401 && (await refreshSession())) continue;
          if (!res.ok || !res.body) throw new Error(`stream HTTP ${res.status}`);
          dispatch({ type: "transport", value: "sse" });
          let terminal = false;
          for await (const block of sseLines(res)) {
            let event = "message", data = "", id: number | null = null;
            for (const line of block.split("\n")) {
              if (line.startsWith("event: ")) event = line.slice(7);
              else if (line.startsWith("data: ")) data += line.slice(6);
              else if (line.startsWith("id: ")) id = Number(line.slice(4));
            }
            if (!data) continue;
            if (id !== null) lastId = id;
            dispatch({ type: "event", event, data: JSON.parse(data), t: performance.now() });
            if (["run_completed", "run_cancelled", "run_failed"].includes(event)) terminal = true;
          }
          if (terminal) break;
        } catch (e: any) {
          if (stop.current || e?.name === "AbortError") break;
          failures += 1;
          dispatch({ type: "transport", value: "polling" });
          try {                                     // fallback: poll the run while the stream is unavailable
            const run = await compareApi.run(runId);
            dispatch({ type: "snapshot", run });
            if (["completed", "cancelled", "failed"].includes(run.status)) break;
          } catch (pe: any) {
            if (failures > 5) { dispatch({ type: "error", message: pe?.message || "connection lost" }); break; }
          }
          await new Promise((r) => setTimeout(r, Math.min(4000, 500 * failures)));
        }
      }
    })();
    return () => { stop.current = true; ctl.abort(); };
  }, [runId]);
  const cancel = async () => { if (runId) await compareApi.cancel(runId); };
  return [state, cancel];
}

export function fmtMsShort(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (v >= 100) return `${v.toFixed(0)} ms`;
  return `${v.toFixed(1)} ms`;
}

export function statusColor(status: string): string {
  if (status === "ok" || status === "completed") return "text-emerald-400";
  if (status === "running" || status === "pending" || status === "queued") return "text-amber-300";
  if (status === "error" || status === "failed" || status === "timeout") return "text-red-400";
  return "text-muted";
}

export const STATUS_TEXT: Record<string, string> = {
  ok: "Completed", running: "Running", pending: "Queued", unavailable: "Not available", needs_context: "Needs neighbouring slices",
  withheld: "Input withheld", out_of_distribution: "Out-of-distribution", rejected: "Input rejected", error: "Error",
  cancelled: "Cancelled", timeout: "Timed out", skipped: "Not evaluated",
};
