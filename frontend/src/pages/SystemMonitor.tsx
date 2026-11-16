import { LuActivity, LuCpu, LuMemoryStick, LuTimer, LuServer } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Skeleton } from "../components/ui/misc";
import { LineChart } from "../components/charts/charts";
import { Kpi } from "../components/compare/ui";
import { compareApi, fmtMsShort } from "../lib/compare";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";

export default function SystemMonitor() {
  useDocumentTitle("System monitor");
  const m = useQuery(["metrics"], compareApi.metrics, { refetchInterval: 2000, staleMs: 1000 });
  const st = useQuery(["status"], compareApi.status, { refetchInterval: 10000 });
  const d = m.data;
  const tNow = d?.samples?.at(-1)?.t ?? 0;          // x axis: seconds relative to the latest sample (0 = now)
  const up = d ? `${Math.floor(d.uptime_s / 3600)}h ${Math.floor((d.uptime_s % 3600) / 60)}m` : "—";
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="System monitoring" title="API server & model engine"
        description="Live measurements of the API process (refreshed every 2 s). All models run on the CPU; quantum models are simulated on the CPU." />
      {!d && <Skeleton className="h-40" />}
      {d && <>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <Kpi label="Process CPU" value={`${d.samples?.at(-1)?.cpu_percent ?? 0}%`} hint={`of ${d.cpu_count} logical CPUs`} icon={<LuCpu className="h-5 w-5" />} accent="#fbbf24" />
          <Kpi label="Memory (RSS)" value={d.rss_mb != null ? `${d.rss_mb} MB` : "n/a"} hint={d.peak_rss_mb ? `peak ${d.peak_rss_mb} MB` : undefined} icon={<LuMemoryStick className="h-5 w-5" />} accent="#22d3ee" />
          <Kpi label="Active comparison runs" value={d.active_runs} icon={<LuActivity className="h-5 w-5" />} accent="#a78bfa" />
          <Kpi label="Threads" value={d.threads} icon={<LuServer className="h-5 w-5" />} />
          <Kpi label="Uptime" value={up} icon={<LuTimer className="h-5 w-5" />} />
        </div>
        <Card>
          <CardHeader title="CPU and memory (last 5 minutes)" description="CPU = process CPU time / wall time / logical CPUs; memory = resident set size of the API process" />
          <CardBody>
            {d.samples?.length > 1 ? <LineChart height={240} title="Process CPU % and memory" xLabel="seconds (0 = latest sample)"
              xFormat={(v) => String(Math.round(v))}
              series={[{ name: "CPU %", color: "#fbbf24", points: d.samples.map((s: any) => ({ x: s.t - tNow, y: s.cpu_percent })) },
                { name: "RSS MB ÷ 10", color: "#22d3ee", points: d.samples.map((s: any) => ({ x: s.t - tNow, y: s.rss_mb == null ? null : s.rss_mb / 10 })) }]} />
              : <p className="text-sm text-muted">Collecting samples…</p>}
          </CardBody>
        </Card>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader title="Model engine" description="Models are loaded once at start-up and reused by every run" />
            <CardBody className="space-y-2 text-sm">
              <p>Engine load: <b className="font-mono text-ink">{d.engine?.load_seconds ?? "—"} s</b> · {d.engine?.models_loaded ?? 0} models timed · sum of per-model loads {fmtMsShort(d.engine?.total_model_load_ms)}</p>
              <table className="w-full text-xs"><tbody>
                {(d.engine?.slowest_loads_ms || []).map(([k, v]: [string, number]) => (
                  <tr key={k} className="border-b border-line/50"><td className="py-1 font-mono text-ink">{k}</td><td className="py-1 text-right font-mono">{fmtMsShort(v)}</td></tr>))}
              </tbody></table>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Hardware & execution" />
            <CardBody className="space-y-1.5 text-sm">
              <p><span className="text-muted">Platform:</span> <span className="text-ink">{d.hardware?.platform}</span></p>
              <p><span className="text-muted">Python:</span> <span className="text-ink">{d.hardware?.python}</span> · <span className="text-muted">BLAS:</span> <span className="text-ink">{d.hardware?.blas || "—"}</span></p>
              <p><span className="text-muted">GPU:</span> <span className="text-ink">{d.gpu?.note}</span></p>
              <p><span className="text-muted">Quantum:</span> <span className="text-ink">{d.hardware?.quantum}</span></p>
              <p><span className="text-muted">Comparison scheduling:</span> <span className="text-ink">{st.data?.comparison?.scheduling?.replace(/_/g, " ")} · max {st.data?.comparison?.max_concurrent_runs} runs · timeout {st.data?.comparison?.run_timeout_s} s</span></p>
              <p><span className="text-muted">Weights:</span> <span className="text-ink">{Object.entries(st.data?.engine?.weights || {}).map(([k, w]: any) => `${k} ${w.ok ? "✓" : "✗"}`).join(" · ")}</span></p>
            </CardBody>
          </Card>
        </div>
      </>}
    </div>
  );
}
