import { useMemo, useState } from "react";
import { LuDownload, LuInfo } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Alert, Skeleton, Tabs } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { BarChart } from "../components/charts/charts";
import { FamilyChip } from "../components/compare/ui";
import { NewEvaluations } from "../components/compare/NewEvaluations";
import { download } from "../lib/api";
import { compareApi, FAMILIES, FAMILY_COLOR, FAMILY_LABEL, fmtMsShort, type Family } from "../lib/compare";
import { fmtNum } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { cn } from "../lib/utils";

const METRICS: [string, string][] = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["specificity", "Specificity"],
  ["f1", "F1"], ["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["ece", "ECE"]];

export default function Benchmarks() {
  useDocumentTitle("Benchmarking");
  const toast = useToast();
  const b = useQuery(["benchmarks"], compareApi.benchmarks, { staleMs: 5000 });
  const [proto, setProto] = useState<"v10_nested" | "v9_fixed_split">("v10_nested");
  const [famSel, setFamSel] = useState<string>("all");
  const rows = useMemo(() => ((b.data?.evaluations || []) as any[]).filter((e) => e.protocol === proto && (famSel === "all" || e.family === famSel)), [b.data, proto, famSel]);
  const key = proto === "v10_nested" ? "primary_mean_within_unit_auc" : "roc_auc";
  const chart = rows.filter((e) => e.metrics[key] != null).sort((x, y) => y.metrics[key] - x.metrics[key])
    .map((e) => ({ label: e.model_id, value: e.metrics[key], color: FAMILY_COLOR[e.family as Family] }));
  const lat = (b.data?.latency?.models || []) as any[];
  const latChart = lat.filter((m) => m.warm_median_ms != null).map((m) => ({ label: m.model_id, value: m.warm_median_ms, color: FAMILY_COLOR[m.family as Family] }));
  const famLat = FAMILIES.map((f) => ({ label: FAMILY_LABEL[f], value: b.data?.latency?.families?.[f]?.median_of_model_warm_medians_ms ?? null, color: FAMILY_COLOR[f] }));
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Performance benchmarking" title="Accuracy & latency by model family"
        description="Accuracy comes only from recorded evaluations on ground-truth-labelled, scan-group-held-out data. Latency comes from comparison runs measured on this server. A single uploaded image never establishes accuracy."
        actions={<Button variant="outline" onClick={() => download("/api/v1/benchmarks/summary.csv", "benchmark_summary.csv").catch((e) => toast("error", e.message))}><LuDownload /> Benchmark CSV</Button>} />
      {b.isLoading && <Skeleton className="h-64" />}
      {b.data && <>
        <Alert tone="info" title="How to read this page">
          <ul className="list-disc space-y-0.5 pl-4">{(b.data.notes as string[]).map((n) => <li key={n}>{n}</li>)}</ul>
        </Alert>
        <NewEvaluations />
        <h2 className="pt-2 text-sm font-semibold uppercase tracking-[0.08em] text-muted">Historical (recorded) evaluations</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {FAMILIES.map((f) => { const fam = b.data.families[f]; const best = fam.best_by_protocol[proto]; return (
            <div key={f} className="card p-4">
              <div className="flex items-center justify-between"><FamilyChip family={f} /><span className="text-xs text-muted">{fam.n_servable}/{fam.n_models} servable</span></div>
              <p className="mt-3 text-[11px] uppercase tracking-wide text-muted">Best ({proto === "v10_nested" ? "within-unit AUC" : "test ROC-AUC"})</p>
              <p className="truncate font-mono text-sm text-ink" title={best?.model_id}>{best?.model_id ?? "— no model under this protocol"}</p>
              <p className="text-2xl font-semibold tabular-nums text-ink">{best ? best.value.toFixed(3) : "—"}</p>
              <p className="mt-2 text-xs text-muted">Median warm inference: <span className="font-mono text-ink">{fmtMsShort(fam.latency?.median_of_model_warm_medians_ms)}</span></p>
            </div>); })}
        </div>
        <Card>
          <CardHeader title="Recorded evaluation metrics" description={b.data.protocols[proto]}
            action={<div className="flex flex-wrap gap-2">
              <Tabs tabs={[{ id: "v10_nested", label: "V10 nested (LOUO)" }, { id: "v9_fixed_split", label: "V9 fixed split" }]} value={proto} onChange={setProto} />
              <select value={famSel} onChange={(e) => setFamSel(e.target.value)} className="h-9 rounded-lg border border-line bg-panel px-2 text-sm" aria-label="Family">
                <option value="all">All families</option>{FAMILIES.map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}</select>
            </div>} />
          <CardBody className="space-y-5">
            {chart.length > 0 && <BarChart horizontal data={chart} domain={[0.5, 1]} format={(v) => v.toFixed(3)}
              title={proto === "v10_nested" ? "Mean within-unit ROC AUC (primary metric)" : "Test ROC AUC"} />}
            <div className="overflow-x-auto">
              <table className="w-full min-w-[980px] text-sm">
                <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
                  <th className="px-3 py-2">Model</th><th className="px-3 py-2">Family</th>
                  {proto === "v10_nested" && <th className="px-3 py-2">Within-unit AUC</th>}
                  {METRICS.map(([, l]) => <th key={l} className="px-3 py-2 text-right">{l}</th>)}<th className="px-3 py-2">Confusion (TP/FN/FP/TN)</th>
                </tr></thead>
                <tbody>
                  {rows.map((e) => (
                    <tr key={e.model_id} className="border-b border-line/50">
                      <td className="px-3 py-2"><p className="font-mono text-xs text-ink">{e.model_id}</p><p className={cn("text-[10px]", e.servable ? "text-emerald-400" : "text-muted")}>{e.availability}</p></td>
                      <td className="px-3 py-2"><FamilyChip family={e.family} /></td>
                      {proto === "v10_nested" && <td className="px-3 py-2 font-mono text-xs font-semibold tabular-nums text-ink">{fmtNum(e.metrics.primary_mean_within_unit_auc, 3)}</td>}
                      {METRICS.map(([k]) => <td key={k} className="px-3 py-2 text-right font-mono text-xs tabular-nums">{fmtNum(e.metrics[k], 3)}</td>)}
                      <td className="px-3 py-2 font-mono text-[11px] text-muted">{e.confusion ? `${e.confusion.tp}/${e.confusion.fn}/${e.confusion.fp}/${e.confusion.tn}` : "—"}</td>
                    </tr>
                  ))}
                  {!rows.length && <tr><td colSpan={12} className="px-3 py-6 text-center text-sm text-muted">No model has a recorded evaluation under this protocol and filter.</td></tr>}
                </tbody>
              </table>
            </div>
            <p className="flex items-start gap-1.5 text-xs text-muted"><LuInfo className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              Metrics from the two protocols are not comparable with each other. A higher probability on one image does not make a model better. Thresholds were chosen on validation / inner folds only — never on test data.</p>
          </CardBody>
        </Card>
        <div className="grid gap-4 xl:grid-cols-[1fr_1.4fr]">
          <Card>
            <CardHeader title="Measured latency by family" description={`${b.data.latency.n_measurements} model executions recorded on this server`} />
            <CardBody>{famLat.some((x) => x.value != null) ? <BarChart horizontal data={famLat} format={(v) => fmtMsShort(v)} title="Median of per-model warm inference medians (ms)" />
              : <p className="text-sm text-muted">{b.data.latency.n_measurements > 0
                ? "Only cold-start executions so far (each model has run once in this server process). Warm latencies appear from the second run on; cold times are in the per-model table."
                : "No comparison run yet — latencies appear after the first run."}</p>}</CardBody>
          </Card>
          <Card>
            <CardHeader title="Per-model latency (warm vs cold)" description="Warm = the model had already run in this server process. Inference only; features and preprocessing excluded." />
            <CardBody className="space-y-4">
              {latChart.length > 0 && <BarChart horizontal data={latChart} format={(v) => fmtMsShort(v)} title="Median warm inference time (ms)" />}
              <div className="max-h-80 overflow-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
                    {["Model", "Family", "Warm n", "Warm median", "Warm p95", "Cold n", "Cold median", "Features median"].map((h) => <th key={h} className="px-3 py-2">{h}</th>)}</tr></thead>
                  <tbody>{lat.map((m) => (
                    <tr key={m.model_id} className="border-b border-line/50">
                      <td className="px-3 py-1.5 font-mono text-xs text-ink">{m.model_id}</td><td className="px-3 py-1.5 text-xs">{FAMILY_LABEL[m.family as Family]}</td>
                      <td className="px-3 py-1.5 tabular-nums">{m.n_warm}</td><td className="px-3 py-1.5 font-mono text-xs">{fmtMsShort(m.warm_median_ms)}</td>
                      <td className="px-3 py-1.5 font-mono text-xs">{fmtMsShort(m.warm_p95_ms)}</td><td className="px-3 py-1.5 tabular-nums">{m.n_cold}</td>
                      <td className="px-3 py-1.5 font-mono text-xs">{fmtMsShort(m.cold_median_ms)}</td><td className="px-3 py-1.5 font-mono text-xs text-muted">{fmtMsShort(m.features_median_ms)}</td>
                    </tr>))}</tbody>
                </table>
              </div>
            </CardBody>
          </Card>
        </div>
      </>}
    </div>
  );
}
