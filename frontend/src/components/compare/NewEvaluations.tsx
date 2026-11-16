// V11 new-evaluation benchmark runs: start a run on a labelled dataset, follow its progress, read metrics per
// evaluation role (held-out test / validation / in-sample). Kept visually separate from the recorded (historical) metrics.
import { useEffect, useMemo, useState } from "react";
import { LuCircleStop, LuDownload, LuFlaskConical, LuPlay } from "react-icons/lu";
import { Button } from "../ui/button";
import { Card, CardBody, CardHeader } from "../ui/card";
import { Alert, Progress } from "../ui/misc";
import { useToast } from "../ui/toast";
import { FamilyChip, StatusPill } from "./ui";
import { download } from "../../lib/api";
import { compareApi, fmtMsShort, type Family } from "../../lib/compare";
import { fmtDate, fmtNum } from "../../lib/format";
import { useQuery } from "../../lib/query";
import { cn } from "../../lib/utils";

const ROLE_STYLE: Record<string, string> = {
  test: "bg-emerald-500/15 text-emerald-300 ring-emerald-400/30",
  validation: "bg-amber-500/15 text-amber-300 ring-amber-400/30",
  train: "bg-red-500/15 text-red-300 ring-red-400/30",
  unknown: "bg-slate-500/15 text-slate-300 ring-slate-400/30",
};
const ROLE_LABEL: Record<string, string> = { test: "held-out test", validation: "validation (optimistic)",
  train: "in-sample (training data)", unknown: "unknown" };

export function NewEvaluations() {
  const toast = useToast();
  const ds = useQuery(["bench-datasets"], compareApi.benchDatasets);
  const runs = useQuery(["bench-runs"], compareApi.benchRuns, { refetchInterval: 4000 });
  const [dataset, setDataset] = useState("lits_sample");
  const [split, setSplit] = useState("");
  const [maxSlices, setMaxSlices] = useState(0);
  const [sel, setSel] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const current = sel || runs.data?.[0]?.id || null;
  const run = useQuery(["bench-run", current], () => current ? compareApi.benchRun(current) : Promise.resolve(null),
    { enabled: !!current, refetchInterval: 2000 });
  const dsInfo = (ds.data || []).find((d: any) => d.id === dataset);
  useEffect(() => { if (ds.data?.length && !ds.data.some((d: any) => d.id === dataset)) setDataset(ds.data[0].id); }, [ds.data]);
  const start = async () => {
    setBusy(true);
    try { const r = await compareApi.startBench({ dataset, split: split || null, max_slices: maxSlices }); setSel(r.id); runs.refetch(); toast("success", "Evaluation started"); }
    catch (e: any) { toast("error", e.message); }
    finally { setBusy(false); }
  };
  const res = run.data?.result;
  const rows = useMemo(() => (res?.models || []).flatMap((m: any) => Object.entries(m.by_role).map(([role, mt]: any) => ({ m, role, mt }))), [res]);
  return (
    <Card>
      <CardHeader title={<span className="flex items-center gap-2"><LuFlaskConical className="h-4 w-4 text-brand-400" /> New evaluation runs</span>}
        description="Runs the models on ground-truth-labelled slices through the same front end as the live comparison. Results are kept separate from the recorded metrics and split by each slice's role for each model." />
      <CardBody className="space-y-4">
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <label className="flex flex-col gap-1"><span className="text-xs text-muted">Labelled dataset</span>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)} className="h-9 rounded-lg border border-line bg-panel px-2">
              {(ds.data || []).map((d: any) => <option key={d.id} value={d.id}>{d.label} ({d.n_slices})</option>)}</select></label>
          <label className="flex flex-col gap-1"><span className="text-xs text-muted">Split</span>
            <select value={split} onChange={(e) => setSplit(e.target.value)} className="h-9 rounded-lg border border-line bg-panel px-2">
              <option value="">all</option>{Object.keys(dsInfo?.splits || {}).map((s) => <option key={s} value={s}>{s} ({dsInfo.splits[s]})</option>)}</select></label>
          <label className="flex flex-col gap-1"><span className="text-xs text-muted">Max slices (0 = all)</span>
            <input type="number" min={0} value={maxSlices} onChange={(e) => setMaxSlices(Number(e.target.value))} className="h-9 w-28 rounded-lg border border-line bg-panel px-2" /></label>
          <Button onClick={start} loading={busy} disabled={!dsInfo}><LuPlay /> Start evaluation</Button>
          {dsInfo && <span className="text-xs text-muted">{dsInfo.n_tumour} tumour · {dsInfo.n_liver_free} liver-free · groups {dsInfo.groups.join(", ")}</span>}
        </div>
        {dataset === "lits_sample" && <Alert tone="warning">The bundled sample has 8 slices from 2 scan groups and no tumour-free liver slice: it exercises the engine but cannot estimate accuracy. Add dataset/liver_ct_lits_verified_v8 for meaningful numbers.</Alert>}
        {!!runs.data?.length && (
          <div className="flex flex-wrap gap-2">
            {runs.data.slice(0, 8).map((r: any) => (
              <button key={r.id} onClick={() => setSel(r.id)} className={cn("rounded-lg border px-2.5 py-1 text-xs", current === r.id ? "border-brand-400 bg-brand-500/10 text-ink" : "border-line text-muted")}>
                {r.dataset}{r.split ? `/${r.split}` : ""} · {fmtDate(r.created_at)} · <StatusPill status={r.status} /></button>))}
          </div>
        )}
        {run.data && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <StatusPill status={run.data.status} />
              <span className="text-muted">{run.data.processed ?? Math.round((run.data.progress || 0) * (run.data.n_slices || 0))} / {run.data.n_slices ?? "?"} slices</span>
              {run.data.status === "running" && <Button size="sm" variant="ghost" onClick={() => compareApi.cancelBench(run.data.id).catch((e) => toast("error", e.message))}><LuCircleStop /> Stop</Button>}
              {res && <Button size="sm" variant="outline" onClick={() => download(`/api/v1/benchmarks/runs/${run.data.id}/export.csv`, `benchmark_${run.data.id}.csv`).catch((e) => toast("error", e.message))}><LuDownload /> CSV</Button>}
              {res && <span className="text-xs text-muted">{res.seconds} s · mean process CPU {res.mean_process_cpu_percent}%</span>}
            </div>
            {run.data.status === "running" && <Progress value={run.data.progress || 0} label="evaluation progress" />}
            {run.data.error && <Alert tone="danger">{run.data.error}</Alert>}
            {res && (
              <div className="max-h-[520px] overflow-auto">
                <table className="w-full min-w-[1100px] text-sm">
                  <thead className="sticky top-0 bg-panel"><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
                    {["Model", "Family", "Role", "Slices (scored)", "Tumour / no", "Uncertain", "Acc.", "Prec.", "Recall", "Spec.", "F1", "ROC-AUC", "PR-AUC", "ECE", "TP/FN/FP/TN", "Inference (med)"].map((h) => <th key={h} className="px-2 py-2">{h}</th>)}</tr></thead>
                  <tbody>{rows.map(({ m, role, mt }: any) => (
                    <tr key={m.model_id + role} className="border-b border-line/50">
                      <td className="px-2 py-1.5 font-mono text-xs text-ink">{m.model_id}</td>
                      <td className="px-2 py-1.5"><FamilyChip family={m.family as Family} /></td>
                      <td className="px-2 py-1.5"><span className={cn("rounded px-1.5 py-0.5 text-[11px] ring-1", ROLE_STYLE[role])}>{ROLE_LABEL[role]}</span></td>
                      <td className="px-2 py-1.5 tabular-nums">{mt.n_slices} ({mt.n_scored})</td>
                      <td className="px-2 py-1.5 tabular-nums">{mt.n_tumour} / {mt.n_no_tumour}</td>
                      <td className="px-2 py-1.5 tabular-nums">{mt.n_uncertain}</td>
                      {["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "ece"].map((k) => <td key={k} className="px-2 py-1.5 font-mono text-xs tabular-nums">{fmtNum(mt[k], 3)}</td>)}
                      <td className="px-2 py-1.5 font-mono text-[11px] text-muted">{mt.confusion.tp}/{mt.confusion.fn}/{mt.confusion.fp}/{mt.confusion.tn}</td>
                      <td className="px-2 py-1.5 font-mono text-xs">{fmtMsShort(mt.inference_ms_median)}</td>
                    </tr>))}</tbody>
                </table>
              </div>
            )}
            {res && <ul className="list-disc space-y-0.5 pl-5 text-xs text-muted">{(res.notes as string[]).map((n) => <li key={n}>{n}</li>)}
              {Object.entries(res.roles as Record<string, string>).map(([k, v]) => <li key={k}><b className="text-ink">{ROLE_LABEL[k]}</b>: {v}</li>)}</ul>}
          </div>
        )}
      </CardBody>
    </Card>
  );
}
