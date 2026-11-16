import { useState } from "react";
import { LuGitCompareArrows, LuDownload, LuFileText, LuHistory } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Alert, EmptyState, Pagination, Skeleton } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { FamilyChip, PredictionPill, StatusPill } from "../components/compare/ui";
import { download } from "../lib/api";
import { compareApi, FAMILIES, FAMILY_LABEL, fmtMsShort, type Family } from "../lib/compare";
import { fmtDate } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link, navigate } from "../lib/router";
import { cn } from "../lib/utils";

function Diff({ a, b }: { a: string; b: string }) {
  const d = useQuery(["cmp2", a, b], () => compareApi.compareRuns(a, b));
  if (d.isLoading) return <Skeleton className="h-40" />;
  if (!d.data) return <Alert tone="danger">Could not compare these runs.</Alert>;
  const x = d.data;
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">{x.same_input ? "Same input image: differences reflect timing noise / model changes." : "Different inputs: prediction differences are expected."}</p>
      <div className="grid gap-3 md:grid-cols-4">
        {FAMILIES.map((f) => { const fa = x.families[f]?.a, fb = x.families[f]?.b; return (
          <div key={f} className="rounded-xl border border-line bg-panel2 p-3 text-xs">
            <FamilyChip family={f} />
            <p className="mt-2 text-muted">Mean inference: <span className="font-mono text-ink">{fmtMsShort(fa?.inference_ms_mean)}</span> → <span className="font-mono text-ink">{fmtMsShort(fb?.inference_ms_mean)}</span></p>
            <p className="text-muted">Completed: {fa?.n_completed ?? "—"} → {fb?.n_completed ?? "—"}</p>
          </div>); })}
      </div>
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full min-w-[820px] text-sm">
          <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
            {["Model", "Family", "Run A", "Run B", "Δ probability", "Inference A → B", "Δ ms"].map((h) => <th key={h} className="px-3 py-2">{h}</th>)}</tr></thead>
          <tbody>{x.models.map((m: any) => (
            <tr key={m.model_id} className={cn("border-b border-line/50", m.label_changed && "bg-amber-400/[0.05]")}>
              <td className="px-3 py-1.5 font-mono text-xs text-ink">{m.model_id}</td>
              <td className="px-3 py-1.5 text-xs">{m.family ? FAMILY_LABEL[m.family as Family] : "—"}</td>
              <td className="px-3 py-1.5"><PredictionPill label={m.a.label} status={m.a.status || "unavailable"} /></td>
              <td className="px-3 py-1.5"><PredictionPill label={m.b.label} status={m.b.status || "unavailable"} /></td>
              <td className="px-3 py-1.5 font-mono text-xs tabular-nums">{m.probability_delta == null ? "—" : (m.probability_delta > 0 ? "+" : "") + m.probability_delta.toFixed(4)}</td>
              <td className="px-3 py-1.5 font-mono text-xs tabular-nums">{fmtMsShort(m.a.inference_ms)} → {fmtMsShort(m.b.inference_ms)}</td>
              <td className="px-3 py-1.5 font-mono text-xs tabular-nums">{m.inference_ms_delta == null ? "—" : (m.inference_ms_delta > 0 ? "+" : "") + m.inference_ms_delta.toFixed(1)}</td>
            </tr>))}</tbody>
        </table>
      </div>
    </div>
  );
}

export default function RunHistory() {
  useDocumentTitle("Execution history");
  const toast = useToast();
  const [page, setPage] = useState(1);
  const size = 20;
  const h = useQuery(["history", page], () => compareApi.history(size, (page - 1) * size));
  const [sel, setSel] = useState<string[]>([]);
  const [cmp, setCmp] = useState<[string, string] | null>(null);
  const toggle = (id: string) => setSel((s) => s.includes(id) ? s.filter((x) => x !== id) : [...s.slice(-1), id]);
  const pages = Math.max(1, Math.ceil((h.data?.total || 0) / size));
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Execution history" title="Saved comparison runs"
        description="Every run is stored with its input, model versions, predictions, probabilities, timings, failures and hardware information. Select two runs to compare them."
        actions={<>
          <Button variant="outline" onClick={() => navigate("/app/predictions")}><LuHistory /> Single-model predictions</Button>
          <Button disabled={sel.length !== 2} onClick={() => setCmp([sel[0], sel[1]])}><LuGitCompareArrows /> Compare selected ({sel.length}/2)</Button>
        </>} />
      {cmp && <Card><CardHeader title="Run comparison" description={<span className="font-mono text-xs">A {cmp[0]} · B {cmp[1]}</span>}
        action={<Button size="sm" variant="ghost" onClick={() => setCmp(null)}>Close</Button>} /><CardBody><Diff a={cmp[0]} b={cmp[1]} /></CardBody></Card>}
      <Card>
        <CardBody className="overflow-x-auto p-0">
          {h.isLoading && <div className="p-5"><Skeleton className="h-32" /></div>}
          {h.data && !h.data.items.length && <EmptyState title="No runs saved yet" action={<Button onClick={() => navigate("/app/upload")}>Run a comparison</Button>} />}
          {!!h.data?.items.length && (
            <table className="w-full min-w-[1100px] text-sm">
              <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
                {["", "Run", "Input", "Modality", "Mode", "Status", "Models", "Majority", "Agreement", ...FAMILIES.map((f) => FAMILY_LABEL[f]), "Wall", "When", ""].map((x, i) => <th key={i} className="px-3 py-2.5 font-semibold">{x}</th>)}
              </tr></thead>
              <tbody>{h.data.items.map((r: any) => (
                <tr key={r.id} className="border-b border-line/50 hover:bg-white/[0.02]">
                  <td className="px-3 py-2"><input type="checkbox" checked={sel.includes(r.id)} onChange={() => toggle(r.id)} aria-label={`select ${r.id}`} className="accent-indigo-500" /></td>
                  <td className="px-3 py-2"><Link to={`/app/live/${r.id}`} className="font-mono text-xs text-brand-400 hover:underline">{r.id.slice(0, 14)}…</Link></td>
                  <td className="max-w-[160px] truncate px-3 py-2 text-xs text-muted" title={r.input_name || r.sample_id}>{r.input_name || r.sample_id}</td>
                  <td className="px-3 py-2 text-xs">{r.modality}</td>
                  <td className="px-3 py-2 text-xs">{r.mode.replace("_", " ")}</td>
                  <td className="px-3 py-2"><StatusPill status={r.status} /></td>
                  <td className="px-3 py-2 text-xs tabular-nums">{r.n_completed ?? 0}/{r.n_models ?? 0}{r.n_failed ? <span className="text-red-400"> · {r.n_failed} err</span> : null}</td>
                  <td className="px-3 py-2">{r.majority_label ? <PredictionPill label={r.majority_label} status="ok" /> : <span className="text-xs text-muted">{r.stopped ? "withheld" : "—"}</span>}</td>
                  <td className="px-3 py-2 text-xs tabular-nums">{r.agreement_pct != null ? `${r.agreement_pct}%` : "—"}</td>
                  {FAMILIES.map((f) => <td key={f} className={cn("px-3 py-2 font-mono text-[11px] tabular-nums", r.fastest_family === f && "text-emerald-400", r.slowest_family === f && "text-red-400")}>{fmtMsShort(r.families?.[f]?.inference_ms_mean)}</td>)}
                  <td className="px-3 py-2 font-mono text-[11px]">{fmtMsShort(r.wall_ms)}</td>
                  <td className="px-3 py-2 text-[11px] text-muted">{fmtDate(r.created_at)}</td>
                  <td className="whitespace-nowrap px-3 py-2">
                    <button className="mr-2 text-muted hover:text-ink" title="CSV" onClick={() => download(`/api/v1/compare/${r.id}/export.csv`, `comparison_${r.id}.csv`).catch((e) => toast("error", e.message))}><LuDownload className="h-4 w-4" /></button>
                    {r.status === "completed" && <button className="text-muted hover:text-ink" title="PDF report" onClick={() => download(`/api/v1/compare/${r.id}/report`, `comparison_${r.id}.pdf`).catch((e) => toast("error", e.message))}><LuFileText className="h-4 w-4" /></button>}
                  </td>
                </tr>))}</tbody>
            </table>
          )}
        </CardBody>
        {pages > 1 && <div className="border-t border-line px-4 py-3"><Pagination page={page} pages={pages} total={h.data?.total || 0} onPage={setPage} /></div>}
      </Card>
      <p className="text-xs text-muted">Family columns show mean per-model inference time; green = fastest family of that run, red = slowest.</p>
    </div>
  );
}
