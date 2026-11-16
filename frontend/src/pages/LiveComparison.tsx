import { useEffect, useMemo, useState } from "react";
import { LuActivity, LuCircleStop, LuDownload, LuFileText, LuListTree, LuPlus, LuTrophy, LuTurtle, LuZap,
  LuAtom, LuTriangleAlert, LuGitCompareArrows, LuTimer, LuChevronDown, LuChevronUp } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Alert, EmptyState, Skeleton } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { ExecutionTimeline, FamilyTimeBars, OutcomeDonut, ProbabilityBar, type TimelineRow } from "../components/charts/live";
import { LineChart } from "../components/charts/charts";
import { FamilyChip, FAMILY_ICON, PredictionPill, SectionTitle, StatusPill } from "../components/compare/ui";
import { download } from "../lib/api";
import { compareApi, FAMILIES, FAMILY_BLURB, FAMILY_COLOR, FAMILY_LABEL, fmtMsShort, LABEL_SHORT, useComparisonRun,
  type Family, type LiveState, type ModelResult } from "../lib/compare";
import { fmtNum } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link, navigate, useParams } from "../lib/router";
import { cn } from "../lib/utils";

function useNow(active: boolean) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

const ACTIVE = ["connecting", "queued", "preprocessing", "running"];

function FamilyPanel({ fam, s, now }: { fam: Family; s: LiveState; now: number }) {
  const live = s.families[fam];
  const models = Object.values(s.models).filter((m) => m.family === fam);
  const executed = models.filter((m) => ["ok", "error", "running", "pending", "cancelled", "timeout"].includes(m.status));
  const skipped = models.filter((m) => !["ok", "error", "running", "pending", "cancelled", "timeout"].includes(m.status));
  const done = models.filter((m) => m.status === "ok").length;
  const errors = models.filter((m) => m.status === "error");
  const sum = live.summary;
  const elapsed = live.status === "running" && live.startedAt ? now - live.startedAt : sum?.wall_ms ?? null;
  const Icon = FAMILY_ICON[fam];
  const total = live.n_models ?? executed.length;
  const progress = total ? (done + errors.length) / total : live.status === "completed" ? 1 : 0;
  const order = (m: ModelResult) => ({ running: 0, ok: 1, error: 2, pending: 3 } as Record<string, number>)[m.status] ?? 4;
  return (
    <section aria-label={FAMILY_LABEL[fam]} className="card panel-glow flex min-h-[420px] flex-col overflow-hidden"
      style={{ "--glow": `${FAMILY_COLOR[fam]}aa` } as any}>
      <header className="border-b border-line px-4 py-3" style={{ background: `linear-gradient(90deg, ${FAMILY_COLOR[fam]}14, transparent 70%)` }}>
        <div className="flex items-center gap-3">
          <div className="rounded-lg p-2" style={{ background: `${FAMILY_COLOR[fam]}22`, color: FAMILY_COLOR[fam] }}><Icon className="h-5 w-5" /></div>
          <div className="min-w-0 flex-1">
            <h3 className="font-semibold text-ink">{FAMILY_LABEL[fam]}</h3>
            <p className="truncate text-[11px] text-muted">{FAMILY_BLURB[fam]}</p>
          </div>
          <div className="text-right">
            <p className={cn("font-mono text-lg font-semibold tabular-nums", live.status === "running" ? "text-amber-300" : "text-ink")}>
              {elapsed === null ? "—" : fmtMsShort(elapsed)}</p>
            <StatusPill status={live.status === "completed" ? "completed" : live.status === "running" ? "running" : "pending"}>
              {live.status === "completed" ? "Completed" : live.status === "running" ? "Executing" : "Waiting"}</StatusPill>
          </div>
        </div>
        <div className="mt-2.5 flex items-center gap-3">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
            <div className={cn("h-full rounded-full transition-all duration-300", live.status === "running" && "bar-running")}
              style={{ width: `${Math.max(live.status === "running" ? 6 : 0, progress * 100)}%`, background: live.status === "running" ? undefined : FAMILY_COLOR[fam] }} />
          </div>
          <span className="text-[11px] tabular-nums text-muted">{done}/{total} done{errors.length ? ` · ${errors.length} failed` : ""}{skipped.length ? ` · ${skipped.length} not run` : ""}</span>
        </div>
      </header>
      {sum && sum.n_completed > 0 && (
        <div className="grid grid-cols-3 divide-x divide-line border-b border-line text-center">
          <div className="px-2 py-2"><p className="text-[10px] uppercase tracking-wide text-muted">{fam === "hybrid" ? "Combined" : "Majority"}</p>
            <div className="mt-0.5"><PredictionPill label={sum.combined_label} status="ok" /></div></div>
          <div className="px-2 py-2"><p className="text-[10px] uppercase tracking-wide text-muted">Mean inference</p>
            <p className="font-mono text-sm text-ink">{fmtMsShort(sum.inference_ms_mean)}</p></div>
          <div className="px-2 py-2"><p className="text-[10px] uppercase tracking-wide text-muted">Fastest</p>
            <p className="truncate font-mono text-[11px] text-ink" title={sum.fastest_model || ""}>{sum.fastest_model ?? "—"}</p></div>
        </div>
      )}
      <ul className="flex-1 divide-y divide-line/60 overflow-auto">
        {executed.sort((a, b) => order(a) - order(b) || a.model_id.localeCompare(b.model_id)).map((m) => (
          <li key={m.model_id} className={cn("px-4 py-2 animate-fade-in", m.status === "running" && "bg-amber-400/[0.04]")}>
            <div className="flex items-center gap-2">
              <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink" title={m.name || m.model_id}>{m.model_id}</span>
              {m.quantum && <span className="inline-flex items-center gap-0.5 rounded bg-cyan-500/10 px-1.5 text-[10px] text-cyan-300" title={m.quantum.backend}>
                <LuAtom className="h-3 w-3" />{m.quantum.qubits ?? "?"}q · sim</span>}
              {m.saturated && <span className="rounded bg-amber-500/10 px-1.5 text-[10px] text-amber-300" title="p ≥ 0.999: saturated output, see calibration report">sat.</span>}
              {m.cold_start && <span className="rounded bg-white/5 px-1.5 text-[10px] text-muted" title="first inference of this model in the server process">cold</span>}
              <PredictionPill label={m.label} status={m.status} />
            </div>
            {m.status === "ok" && m.probability != null && (
              <div className="mt-1.5 grid grid-cols-[1fr_auto] items-center gap-3">
                <ProbabilityBar p={m.probability} thr={m.threshold} band={m.band} />
                <span className="whitespace-nowrap font-mono text-[11px] tabular-nums text-muted">
                  p {m.probability.toFixed(3)} · thr {m.threshold?.toFixed(2)} · margin {fmtNum(m.confidence, 2)} · <span className="text-ink">{fmtMsShort(m.timings_ms?.inference)}</span>
                </span>
              </div>
            )}
            {m.status === "error" && <p className="mt-1 text-[11px] text-red-400">{m.reason}</p>}
          </li>
        ))}
        {executed.length === 0 && live.status !== "waiting" && <li className="px-4 py-6 text-center text-xs text-muted">No model of this family could run on this input.</li>}
        {executed.length === 0 && live.status === "waiting" && <li className="space-y-2 px-4 py-4"><Skeleton className="h-4" /><Skeleton className="h-4 w-2/3" /></li>}
      </ul>
      {skipped.length > 0 && (
        <details className="border-t border-line px-4 py-2 text-[11px] text-muted">
          <summary className="cursor-pointer select-none">{skipped.length} model(s) not executed — why?</summary>
          <ul className="mt-1.5 space-y-1">
            {skipped.map((m) => <li key={m.model_id}><span className="font-mono text-ink">{m.model_id}</span>: <StatusPill status={m.status} /> {m.reason || m.skip_reason}</li>)}
          </ul>
        </details>
      )}
    </section>
  );
}

function Highlight({ icon, title, value, sub, color }: { icon: React.ReactNode; title: string; value: React.ReactNode; sub?: React.ReactNode; color: string }) {
  return (
    <div className="card relative overflow-hidden p-4">
      <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full blur-2xl" style={{ background: `${color}33` }} />
      <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.12em]" style={{ color }}>{icon}{title}</p>
      <p className="mt-2 text-xl font-semibold text-ink">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </div>
  );
}

export function RunView({ runId }: { runId: string }) {
  const toast = useToast();
  const [s, cancel] = useComparisonRun(runId);
  const active = ACTIVE.includes(s.phase);
  const now = useNow(active);
  const bench = useQuery(["benchmarks"], compareApi.benchmarks);
  const runMeta = useQuery(["run-meta", runId], () => compareApi.run(runId));
  const upload = useQuery(["upload", runMeta.data?.upload_id], () => runMeta.data ? compareApi.getUpload(runMeta.data.upload_id) : Promise.resolve(null));
  const [showLog, setShowLog] = useState(false);
  const r = s.result;
  const models = Object.values(s.models);
  const pre = s.preprocessing;
  const serverNow = s.runT0Perf !== undefined ? now - s.runT0Perf : undefined;

  const famBars = FAMILIES.map((f) => {
    const sum = s.families[f].summary;
    const running = s.families[f].status === "running";
    const doneInf = models.filter((m) => m.family === f && m.status === "ok").map((m) => m.timings_ms?.inference || 0);
    const mean = sum?.inference_ms_mean ?? (doneInf.length ? doneInf.reduce((a, b) => a + b, 0) / doneInf.length : null);
    return { family: f, value: mean, running: running && !doneInf.length,
      highlight: r?.ranking.fastest_family === f ? "fastest" as const : r?.ranking.slowest_family === f ? "slowest" as const : undefined };
  });
  const wallBars = FAMILIES.map((f) => {
    const fl = s.families[f];
    return { family: f, value: fl.status === "running" && fl.startedAt ? now - fl.startedAt : fl.summary?.wall_ms ?? null, running: fl.status === "running" };
  });
  const tl: TimelineRow[] = models.filter((m) => m.started_ms !== undefined || m.status === "ok")
    .map((m) => ({ id: m.model_id, family: m.family, start: m.started_ms ?? m.timings_ms?.queue ?? null,
      end: m.finished_ms ?? (m.status === "running" ? null : (m.started_ms ?? 0) + (m.timings_ms?.total || 0)), status: m.status,
      label: m.label ? LABEL_SHORT[m.label] : undefined }))
    .sort((a, b) => FAMILIES.indexOf(a.family) - FAMILIES.indexOf(b.family) || (a.start ?? 0) - (b.start ?? 0));
  const cnt = (st: string[]) => models.filter((m) => st.includes(m.status)).length;
  const ok = models.filter((m) => m.status === "ok");
  const counts = { tumour: ok.filter((m) => m.label === "Tumour detected").length, none: ok.filter((m) => m.label === "No tumour detected").length,
    unc: ok.filter((m) => m.label === "Uncertain / unable to classify").length };
  const evalBy = useMemo(() => Object.fromEntries(((bench.data?.evaluations || []) as any[]).map((e) => [e.model_id, e])), [bench.data]);
  const best = useMemo(() => {
    const cands = ok.map((m) => ({ m, e: evalBy[m.model_id] })).filter((x) => x.e && x.e.protocol === "v10_nested" && x.e.metrics.primary_mean_within_unit_auc != null);
    if (!cands.length) return null;
    return cands.reduce((a, b) => (b.e.metrics.primary_mean_within_unit_auc > a.e.metrics.primary_mean_within_unit_auc ? b : a));
  }, [ok.length, evalBy]);

  const phaseText: Record<string, string> = { connecting: "Connecting…", queued: "Queued", preprocessing: "Shared preprocessing",
    running: "Executing four family pipelines", completed: "Completed", cancelled: "Cancelled", failed: "Failed" };

  return (
    <div className="space-y-6">
      <PageHeader eyebrow={<span className="inline-flex items-center gap-1.5"><LuActivity className="h-3.5 w-3.5" /> Live four-model comparison</span>}
        title="Classical · Deep Learning · Quantum · Hybrid"
        description={<span>Run <span className="font-mono">{runId}</span> · <span className={active ? "text-amber-300" : s.phase === "completed" ? "text-emerald-400" : "text-red-400"}>{phaseText[s.phase]}</span>
          {s.transport === "polling" && active && <span className="ml-2 text-amber-300">(live stream unavailable — polling)</span>}</span>}
        actions={<>
          {active && <Button variant="danger" onClick={() => cancel().then(() => toast("info", "Cancel requested")).catch((e) => toast("error", e.message))} disabled={s.cancelRequested}><LuCircleStop /> Cancel</Button>}
          {r && <Button variant="outline" onClick={() => download(`/api/v1/compare/${runId}/export.csv`, `comparison_${runId}.csv`).catch((e) => toast("error", e.message))}><LuDownload /> CSV</Button>}
          {r && <Button variant="outline" onClick={() => download(`/api/v1/compare/${runId}/report`, `comparison_${runId}.pdf`).catch((e) => toast("error", e.message))}><LuFileText /> PDF report</Button>}
          {r && <Button variant="outline" onClick={() => navigate(`/app/runs/${runId}`)}><LuListTree /> Details</Button>}
          <Button onClick={() => navigate("/app/upload")}><LuPlus /> New run</Button>
        </>} />

      {s.error && <Alert tone="danger" title="Run error">{s.error}</Alert>}

      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        <div className="card overflow-hidden">
          {upload.data?.analysis.preview_png_b64
            ? <img src={`data:image/png;base64,${upload.data.analysis.preview_png_b64}`} alt="Input slice" className="aspect-square w-full object-cover" />
            : <div className="grid-bg aspect-square w-full" />}
          <div className="p-3 text-xs">
            <p className="truncate font-medium text-ink">{upload.data?.input_name || upload.data?.id || "input"}</p>
            <p className="text-muted">{upload.data?.modality} · context: {upload.data?.context_slices.length ? upload.data.context_slices.join(", ") : "none"}</p>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {[
            ["Quality control", pre?.quality?.status, pre?.quality?.status === "ok"],
            ["Out-of-distribution", pre?.ood?.status?.replace(/_/g, " "), pre?.ood?.status === "in_distribution"],
            ["Liver gate + U-Net", pre ? (pre.liver_check?.status || (pre.liver_gate?.n_liver_patches ? "gate ok" : "no liver")) : undefined, pre?.liver_check?.status === "confirmed"],
            ["Shared preprocessing", pre?.timings_ms?.total !== undefined ? fmtMsShort(pre.timings_ms.total) : undefined, !!pre && !pre.stopped],
          ].map(([k, v, good]) => (
            <div key={k as string} className="card p-3">
              <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">{k as string}</p>
              <p className={cn("mt-1 font-medium", v === undefined ? "text-muted" : good ? "text-emerald-400" : "text-amber-300")}>{(v as string) ?? "pending…"}</p>
            </div>
          ))}
          <div className="card p-3 sm:col-span-2 xl:col-span-4">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">Scheduling</p>
            <p className="mt-1 text-sm text-ink">{r?.scheduling.description || "Four family pipelines start together on four worker threads; inside a family models run one at a time to avoid CPU oversubscription."}</p>
            {r?.family_overlap?.start_spread_ms !== undefined && (
              <p className="mt-1 text-xs text-muted">Measured: all four families started within <b className="text-ink">{fmtMsShort(r.family_overlap.start_spread_ms)}</b> of each other and ran concurrently for <b className="text-ink">{fmtMsShort(r.family_overlap.all_four_overlap_ms)}</b>. {r.quantum_execution}.</p>
            )}
            {pre?.stopped && <p className="mt-1 flex items-center gap-1.5 text-sm text-amber-300"><LuTriangleAlert className="h-4 w-4" /> {pre.stopped}</p>}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {FAMILIES.map((f) => <FamilyPanel key={f} fam={f} s={s} now={now} />)}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Highlight icon={<LuZap className="h-3.5 w-3.5" />} title="Fastest model family" color="#34d399"
          value={r?.ranking.fastest_family ? FAMILY_LABEL[r.ranking.fastest_family] : active ? "measuring…" : "—"}
          sub={r?.ranking.fastest_family ? `${fmtMsShort(r.families[r.ranking.fastest_family].inference_ms_mean)} mean per-model inference` : "lowest mean per-model inference time"} />
        <Highlight icon={<LuTurtle className="h-3.5 w-3.5" />} title="Slowest model family" color="#f87171"
          value={r?.ranking.slowest_family ? FAMILY_LABEL[r.ranking.slowest_family] : active ? "measuring…" : "—"}
          sub={r?.ranking.slowest_family ? `${fmtMsShort(r.families[r.ranking.slowest_family].inference_ms_mean)} mean per-model inference` : "highest mean per-model inference time"} />
        <Highlight icon={<LuTrophy className="h-3.5 w-3.5" />} title="Best performing model" color="#fbbf24"
          value={best ? <span className="font-mono text-base">{best.m.model_id}</span> : "—"}
          sub={best ? `highest recorded mean within-unit AUC (${best.e.metrics.primary_mean_within_unit_auc.toFixed(3)}, V10 nested evaluation) among models that ran — not derived from this image` : "from recorded evaluation on labelled data, never from one image"} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_1fr]">
        <Card>
          <CardHeader title="Execution time by model family" description="Measured on this server · raw model inference excludes shared preprocessing and feature extraction" />
          <CardBody className="space-y-6">
            <FamilyTimeBars title="Mean per-model inference time" metric="Mean per-model inference time of completed models (ms)." rows={famBars} />
            <FamilyTimeBars title="Family pipeline wall time" metric="Family wall time: start of the family pipeline → its last model finished (includes feature extraction and queueing)." rows={wallBars} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Outcomes & agreement" description="Agreement between models is not evidence that a prediction is correct" />
          <CardBody className="space-y-4">
            <OutcomeDonut title="Model outcomes" parts={[
              { label: "Completed", value: cnt(["ok"]), color: "#34d399" },
              { label: "Running / queued", value: cnt(["running", "pending"]), color: "#fbbf24" },
              { label: "Not evaluated (unavailable / context / withheld)", value: cnt(["unavailable", "needs_context", "withheld", "out_of_distribution", "rejected", "skipped"]), color: "#64748b" },
              { label: "Failed / cancelled / timeout", value: cnt(["error", "cancelled", "timeout"]), color: "#f87171" },
            ]} />
            <div className="grid grid-cols-4 gap-2 text-center">
              {[["Tumour", counts.tumour, "text-rose-300"], ["No tumour", counts.none, "text-teal-300"], ["Uncertain", counts.unc, "text-amber-300"],
                ["Agreement", r?.agreement.agreement_pct != null ? `${r.agreement.agreement_pct}%` : ok.length ? `${Math.round(100 * Math.max(counts.tumour, counts.none, counts.unc) / ok.length)}%` : "—", "text-ink"]].map(([k, v, c]) => (
                <div key={k as string} className="rounded-lg border border-line bg-panel2 py-2">
                  <p className={cn("text-xl font-semibold tabular-nums", c as string)}>{v as any}</p>
                  <p className="text-[10px] uppercase tracking-wide text-muted">{k as string}</p>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-muted">Agreement = share of completed models giving the most common call. {r?.agreement.failed_or_skipped ?? cnt(["unavailable", "needs_context", "withheld", "out_of_distribution", "error"])} models failed or were not evaluated and are excluded.</p>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Prediction comparison by family" description="Each family's majority call of its completed models; individual model calls use each model's own threshold and ±0.10 uncertainty band. p values come from different calibration methods (V9: Platt on V9 validation scans; V10: Platt on inner out-of-fold predictions)." />
        <CardBody className="overflow-x-auto p-0">
          <table className="w-full min-w-[720px] text-sm">
            <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
              {["Model family", "Prediction", "Mean probability", "Mean margin", "Mean inference", "Models (done / run / registered)", "Status"].map((h) => <th key={h} className="px-4 py-2.5 font-semibold">{h}</th>)}
            </tr></thead>
            <tbody>
              {FAMILIES.map((f) => {
                const fm = models.filter((m) => m.family === f);
                const fok = fm.filter((m) => m.status === "ok");
                const sum = s.families[f].summary;
                const meanConf = fok.length ? fok.reduce((a, m) => a + (m.confidence || 0), 0) / fok.length : null;
                const stopped = !fok.length && fm.some((m) => ["withheld", "out_of_distribution", "rejected"].includes(m.status)) ? fm.find((m) => ["withheld", "out_of_distribution", "rejected"].includes(m.status))!.status : null;
                return (
                  <tr key={f} className="border-b border-line/60">
                    <td className="px-4 py-2.5"><FamilyChip family={f} /></td>
                    <td className="px-4 py-2.5">{sum?.combined_label ? <PredictionPill label={sum.combined_label} status="ok" /> : <StatusPill status={stopped || (s.families[f].status === "running" ? "running" : sum ? "skipped" : "pending")} />}</td>
                    <td className="px-4 py-2.5 font-mono tabular-nums">{sum?.mean_probability != null ? sum.mean_probability.toFixed(3) : "—"}</td>
                    <td className="px-4 py-2.5 font-mono tabular-nums">{meanConf != null ? meanConf.toFixed(2) : "—"}</td>
                    <td className="px-4 py-2.5 font-mono tabular-nums">{fmtMsShort(sum?.inference_ms_mean)}</td>
                    <td className="px-4 py-2.5 tabular-nums">{fok.length} / {sum?.n_attempted ?? fm.filter((m) => !["unavailable", "needs_context", "withheld", "out_of_distribution", "rejected"].includes(m.status)).length} / {fm.length}</td>
                    <td className="px-4 py-2.5"><StatusPill status={s.families[f].status === "completed" ? (sum?.n_failed ? "error" : "completed") : s.families[f].status === "running" ? "running" : "pending"}>
                      {s.families[f].status === "completed" ? (sum?.n_failed ? `${sum.n_failed} failed` : "Completed") : undefined}</StatusPill></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardBody>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card>
          <CardHeader title="Execution timeline" description="Per-model start → finish (ms since the family pipelines started); colour = family" />
          <CardBody>{tl.length ? <ExecutionTimeline rows={tl} now={active ? serverNow : undefined} title="Execution timeline" /> : <p className="text-sm text-muted">No model has started yet.</p>}
            <div className="mt-2 flex flex-wrap gap-3">{FAMILIES.map((f) => <span key={f} className="inline-flex items-center gap-1.5 text-xs text-muted"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: FAMILY_COLOR[f] }} />{FAMILY_LABEL[f]}</span>)}</div>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Ranking: fastest → slowest model" description={r?.ranking.fastest_model
            ? <>Fastest <span className="font-mono text-emerald-400">{r.ranking.fastest_model}</span> · slowest <span className="font-mono text-red-400">{r.ranking.slowest_model}</span> · raw inference time (features excluded)</>
            : "Raw inference time (features excluded)"} />
          <CardBody className="max-h-[440px] overflow-auto p-0">
            <table className="w-full text-sm">
              <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted"><th className="px-4 py-2">#</th><th className="px-2 py-2">Model</th><th className="px-2 py-2 text-right">Inference</th><th className="px-4 py-2 text-right">Features</th></tr></thead>
              <tbody>
                {(r?.ranking.models_by_inference_ms || [...ok].sort((a, b) => (a.timings_ms?.inference || 0) - (b.timings_ms?.inference || 0))
                  .map((m) => ({ model_id: m.model_id, family: m.family, inference_ms: m.timings_ms?.inference || 0, features_ms: m.timings_ms?.features }))).map((x, i) => (
                  <tr key={x.model_id} className="border-b border-line/50">
                    <td className="px-4 py-1.5 text-muted tabular-nums">{i + 1}</td>
                    <td className="px-2 py-1.5"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: FAMILY_COLOR[x.family] }} /><span className="font-mono text-xs text-ink">{x.model_id}</span></td>
                    <td className="px-2 py-1.5 text-right font-mono text-xs tabular-nums text-ink">{fmtMsShort(x.inference_ms)}</td>
                    <td className="px-4 py-1.5 text-right font-mono text-xs tabular-nums text-muted">{fmtMsShort(x.features_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardBody>
        </Card>
      </div>

      {r?.resources && (
        <Card>
          <CardHeader title="Resource utilisation during the run" description={r.resources.method} />
          <CardBody className="grid gap-4 lg:grid-cols-[300px_1fr]">
            <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-sm">
              <dt className="text-muted">Process CPU (mean)</dt><dd className="font-mono text-ink">{r.resources.process_cpu_percent_mean}% of {r.resources.cpu_count} CPU</dd>
              <dt className="text-muted">Process CPU (peak)</dt><dd className="font-mono text-ink">{r.resources.process_cpu_percent_peak ?? "—"}%</dd>
              <dt className="text-muted">CPU time</dt><dd className="font-mono text-ink">{r.resources.process_cpu_seconds.toFixed(2)} s</dd>
              <dt className="text-muted">Memory (RSS)</dt><dd className="font-mono text-ink">{r.resources.rss_mb_start ?? "—"} → peak {r.resources.rss_mb_peak ?? "—"} MB</dd>
              <dt className="text-muted">GPU</dt><dd className="text-xs text-ink">{r.resources.gpu.note}</dd>
              <dt className="text-muted">Model loading</dt><dd className="text-xs text-ink">once at server start ({r.engine?.load_seconds ?? "—"} s); per-model times in Details</dd>
            </dl>
            {r.resources.samples.length > 1 ? <LineChart height={200} title="Process CPU % and memory during the run" xLabel="ms since run start"
              series={[{ name: "CPU % (all cores)", color: "#fbbf24", points: r.resources.samples.map((x) => ({ x: x[0], y: x[1] })) },
                { name: "RSS MB ÷ 10", color: "#22d3ee", points: r.resources.samples.map((x) => ({ x: x[0], y: x[2] == null ? null : x[2] / 10 })) }]} />
              : <p className="text-sm text-muted">The run was too short for more than one 100 ms sample.</p>}
          </CardBody>
        </Card>
      )}

      <Card>
        <button onClick={() => setShowLog((v) => !v)} className="flex w-full items-center justify-between px-5 py-3 text-left">
          <span className="flex items-center gap-2 text-sm font-semibold text-ink"><LuTimer className="h-4 w-4 text-muted" /> Live event log ({s.events.length})</span>
          {showLog ? <LuChevronUp /> : <LuChevronDown />}
        </button>
        {showLog && (
          <ol className="max-h-72 overflow-auto border-t border-line px-5 py-3 font-mono text-[11px] leading-relaxed text-muted">
            {s.events.map((e, i) => <li key={i}><span className="text-slate-500">{s.t0 !== undefined ? `+${((e.t - s.t0) / 1000).toFixed(2)}s` : ""}</span> <span className="text-ink">{e.label}</span></li>)}
          </ol>
        )}
      </Card>
    </div>
  );
}

export default function LiveComparison() {
  useDocumentTitle("Live comparison");
  const { id } = useParams<{ id: string }>();
  const hist = useQuery(["history-latest"], () => compareApi.history(1), { enabled: !id } as any);
  if (id) return <RunView key={id} runId={id} />;
  const latest = hist.data?.items?.[0];
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Live four-model comparison" title="Classical · Deep Learning · Quantum · Hybrid"
        description="Upload an image and click Run all models to watch the four family pipelines execute concurrently." />
      <Card><CardBody>
        <EmptyState icon={<LuGitCompareArrows className="h-6 w-6" />} title={latest ? "Open the latest run or start a new one" : "No comparison run yet"}
          action={<div className="flex gap-2">
            {latest && <Button variant="outline" onClick={() => navigate(`/app/live/${latest.id}`)}>Latest run</Button>}
            <Button onClick={() => navigate("/app/upload")}><LuPlus /> Upload an image</Button></div>}>
          Each run sends the same validated input to every eligible classical, deep-learning, quantum and hybrid model and measures their predictions and execution time.
        </EmptyState>
      </CardBody></Card>
      {hist.data?.items?.length ? <p className="text-xs text-muted">Older runs: <Link to="/app/history" className="text-brand-400 hover:underline">execution history</Link></p> : null}
    </div>
  );
}
