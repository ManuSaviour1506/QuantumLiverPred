import "./setup";
import assert from "node:assert/strict";
import { test } from "node:test";
import { fmtMsShort, initialLive, liveReducer, type LiveState } from "../src/lib/compare";

const ev = (s: LiveState, event: string, data: any, t = 0) => liveReducer(s, { type: "event", event, data, t });

test("live reducer follows the four family pipelines independently", () => {
  let s = initialLive();
  s = ev(s, "run_started", {}, 1);
  assert.equal(s.phase, "preprocessing");
  s = ev(s, "preprocessing_completed", { timings_ms: { total: 400 } }, 2);
  assert.equal(s.phase, "running");
  s = ev(s, "plan", { models: [
    { model_id: "a", family: "classical", eligible: true }, { model_id: "q", family: "quantum", eligible: true },
    { model_id: "d3", family: "deep_learning", eligible: false, skip_status: "needs_context", skip_reason: "5 slices" }] }, 3);
  assert.equal(s.models.a.status, "pending");
  assert.equal(s.models.d3.status, "needs_context");
  for (const f of ["classical", "deep_learning", "quantum", "hybrid"]) s = ev(s, "family_started", { family: f, n_models: 1, started_ms: 1 }, 4);
  assert.ok(Object.values(s.families).every((f) => f.status === "running"));
  s = ev(s, "model_started", { model_id: "q", family: "quantum", started_ms: 2 }, 5);
  assert.equal(s.models.q.status, "running");
  s = ev(s, "model_completed", { model_id: "q", family: "quantum", status: "ok", label: "Tumour detected", probability: 0.9,
    timings_ms: { inference: 3 } }, 6);
  assert.equal(s.models.q.status, "ok");
  assert.equal(s.running.q, undefined);
  s = ev(s, "family_completed", { family: "quantum", wall_ms: 10, n_completed: 1 }, 7);
  assert.equal(s.families.quantum.status, "completed");
  assert.equal(s.families.classical.status, "running");      // other families unaffected
  s = ev(s, "model_failed", { model_id: "a", family: "classical", status: "error", reason: "boom" }, 8);
  assert.equal(s.models.a.status, "error");
  assert.equal(s.models.q.status, "ok");                     // a failure does not touch other models
});

test("terminal events and formatting", () => {
  let s = ev(initialLive(), "run_failed", { error: "engine down" });
  assert.equal(s.phase, "failed");
  assert.equal(s.error, "engine down");
  assert.equal(fmtMsShort(1500), "1.50 s");
  assert.equal(fmtMsShort(12.34), "12.3 ms");
  assert.equal(fmtMsShort(null), "—");
});

test("V11 benchmark and system-monitor calls hit the real backend endpoints", async () => {
  const { compareApi } = await import("../src/lib/compare");
  const calls: { url: string; method: string; body?: string }[] = [];
  (globalThis as any).fetch = async (url: string, init: any = {}) => {
    calls.push({ url, method: init.method || "GET", body: init.body });
    return new Response(JSON.stringify(url.includes("/runs") && (init.method || "GET") === "POST" ? { id: "b1" } : []),
      { status: 200, headers: { "content-type": "application/json" } });
  };
  await compareApi.metrics();
  await compareApi.benchDatasets();
  const started = await compareApi.startBench({ dataset: "lits_sample", split: "test", max_slices: 8 });
  await compareApi.benchRun("b1");
  await compareApi.cancelBench("b1");
  assert.deepEqual(calls.map((c) => `${c.method} ${c.url}`), [
    "GET /api/v1/system/metrics", "GET /api/v1/benchmarks/datasets", "POST /api/v1/benchmarks/runs",
    "GET /api/v1/benchmarks/runs/b1", "POST /api/v1/benchmarks/runs/b1/cancel"]);
  assert.deepEqual(JSON.parse(calls[2].body!), { dataset: "lits_sample", split: "test", max_slices: 8 });
  assert.equal(started.id, "b1");
});
