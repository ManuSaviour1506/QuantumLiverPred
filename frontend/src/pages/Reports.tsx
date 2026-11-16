import { LuAtom, LuDownload, LuFileText, LuFlaskConical, LuGauge, LuShieldAlert } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { EmptyState, Skeleton } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { PredictionPill, StatusPill } from "../components/compare/ui";
import { download } from "../lib/api";
import { compareApi, fmtMsShort } from "../lib/compare";
import { fmtDate } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { navigate } from "../lib/router";

export default function Reports() {
  useDocumentTitle("Research reports");
  const toast = useToast();
  const h = useQuery(["history", "reports"], () => compareApi.history(30));
  const dl = (path: string, name: string) => download(path, name).catch((e) => toast("error", e.message));
  const done = (h.data?.items || []).filter((r: any) => r.status === "completed");
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Research reports" title="Reports & exports"
        description="PDF research reports and CSV exports of comparison runs, the benchmark summary, and the experiment reports of the research module. None of these is a clinical report." />
      <div className="grid gap-4 md:grid-cols-3">
        <Card><CardBody className="space-y-2">
          <LuGauge className="h-5 w-5 text-brand-400" /><p className="font-semibold text-ink">Benchmark summary</p>
          <p className="text-sm text-muted">Recorded metrics (both protocols) and measured latencies of every model.</p>
          <Button variant="outline" size="sm" onClick={() => dl("/api/v1/benchmarks/summary.csv", "benchmark_summary.csv")}><LuDownload /> CSV</Button>
        </CardBody></Card>
        <Card><CardBody className="space-y-2">
          <LuFlaskConical className="h-5 w-5 text-brand-400" /><p className="font-semibold text-ink">Experiments & statistics</p>
          <p className="text-sm text-muted">Nested evaluation, quantum-vs-classical tests, noise study and experiment reports.</p>
          <Button variant="outline" size="sm" onClick={() => navigate("/app/research")}><LuAtom /> Open research module</Button>
        </CardBody></Card>
        <Card><CardBody className="space-y-2">
          <LuShieldAlert className="h-5 w-5 text-brand-400" /><p className="font-semibold text-ink">Audit & calibration</p>
          <p className="text-sm text-muted">In the repository: reports/v10/CALIBRATION_AND_SATURATION.md, LIVER_GATE_VALIDATION.md and V10_UPGRADE_AUDIT.md.</p>
          <Button variant="outline" size="sm" onClick={() => navigate("/app/compare")}>Model metric comparison</Button>
        </CardBody></Card>
      </div>
      <Card>
        <CardHeader title="Comparison run reports" description="Each report covers all four model families with their measured execution metrics." />
        <CardBody className="p-0">
          {h.isLoading && <div className="p-5"><Skeleton className="h-24" /></div>}
          {h.data && !done.length && <EmptyState title="No completed run yet" />}
          <ul className="divide-y divide-line">
            {done.map((r: any) => (
              <li key={r.id} className="flex flex-wrap items-center gap-3 px-5 py-3">
                <LuFileText className="h-4 w-4 text-muted" />
                <div className="min-w-0">
                  <p className="font-mono text-xs text-ink">{r.id}</p>
                  <p className="text-[11px] text-muted">{r.input_name || r.sample_id} · {fmtDate(r.created_at)} · {r.n_completed}/{r.n_models} models · wall {fmtMsShort(r.wall_ms)}</p>
                </div>
                <div className="ml-auto flex items-center gap-2">
                  <StatusPill status={r.status} />{r.majority_label && <PredictionPill label={r.majority_label} status="ok" />}
                  <Button size="sm" variant="outline" onClick={() => dl(`/api/v1/compare/${r.id}/report`, `comparison_${r.id}.pdf`)}><LuFileText /> PDF</Button>
                  <Button size="sm" variant="ghost" onClick={() => dl(`/api/v1/compare/${r.id}/export.csv`, `comparison_${r.id}.csv`)}><LuDownload /> CSV</Button>
                </div>
              </li>
            ))}
          </ul>
        </CardBody>
      </Card>
    </div>
  );
}
