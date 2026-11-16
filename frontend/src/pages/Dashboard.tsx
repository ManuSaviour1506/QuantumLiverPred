import { LuActivity, LuBoxes, LuDatabase, LuFlaskConical, LuScanLine, LuUsers, LuArrowRight } from "react-icons/lu";
import { BarChart, PALETTE } from "../components/charts/charts";
import { PageHeader } from "../components/layout/AppShell";
import { Disclaimer } from "../components/layout/Disclaimer";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Alert, Badge, EmptyState, Skeleton, Stat } from "../components/ui/misc";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtDate, fmtInt, fmtNum, humanize, statusTone } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link } from "../lib/router";

export default function Dashboard() {
  useDocumentTitle("Dashboard");
  const { user } = useAuth();
  const ov = useQuery(["overview"], () => api.get("/api/v1/analytics/overview"), { refetchInterval: 15000 });
  const dist = useQuery(["dist"], () => api.get("/api/v1/analytics/prediction-distribution"));
  const cmp = useQuery(["comparison"], () => api.get("/api/v1/analytics/model-comparison"));
  const o = ov.data;
  const engineState = o?.engine?.state;
  const top = (cmp.data?.rows || []).filter((r: any) => r.roc_auc !== null).slice(0, 10);
  return (
    <>
      <PageHeader title={`Welcome${user ? `, ${user.full_name.split(" ")[0]}` : ""}`}
        description="Live figures from the database, the model registry and saved evaluations."
        actions={<Link to="/app/workspace"><Button><LuScanLine /> New prediction</Button></Link>} />
      {ov.isError && <Alert tone="danger" className="mb-4">Could not load the overview. Is the API running?</Alert>}
      {engineState && engineState !== "ready" && (
        <Alert tone={engineState === "failed" ? "danger" : "info"} className="mb-4" title={`Model engine: ${engineState}`}>
          {engineState === "failed" ? o?.engine?.error : "Models are loading; prediction will be available shortly."}
        </Alert>
      )}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {o ? (
          <>
            <Stat label={user?.role === "admin" ? "Prediction requests (all users)" : "My prediction requests"}
              value={fmtInt(o.predictions_total)} icon={<LuActivity className="h-5 w-5" />}
              hint={Object.entries(o.predictions_by_status || {}).map(([k, v]) => `${v} ${k}`).join(" · ") || "none yet"} />
            <Stat label="Servable models" value={`${o.models_servable} / ${o.models_registered}`} icon={<LuBoxes className="h-5 w-5" />}
              hint={Object.entries(o.models_by_family || {}).map(([k, v]) => `${v} ${k}`).join(" · ")} />
            <Stat label="Datasets" value={fmtInt(o.datasets_total)} icon={<LuDatabase className="h-5 w-5" />}
              hint={`${fmtInt(o.dataset_files_total)} files in latest versions`} />
            {o.users_total !== null && o.users_total !== undefined
              ? <Stat label="Registered users" value={fmtInt(o.users_total)} icon={<LuUsers className="h-5 w-5" />} hint="administrators only" />
              : <Stat label="Experiments" value={fmtInt(o.experiments_total)} icon={<LuFlaskConical className="h-5 w-5" />}
                hint={`${o.jobs_pending} background job(s) pending`} />}
          </>
        ) : [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-24" />)}
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-3">
          <CardHeader title="Model comparison (saved evaluations)" description="Primary metric per model; V9 and V10 evaluations differ — see the comparison page."
            action={<Link to="/app/compare" className="text-sm text-brand-700 hover:underline dark:text-brand-300">Details →</Link>} />
          <CardBody>
            {cmp.isLoading ? <Skeleton className="h-64" /> : top.length ? (
              <BarChart horizontal title="ROC-AUC by model" domain={[0.5, 1]} format={(v) => v.toFixed(2)}
                data={top.map((r: any) => ({ label: r.name.replace(" — V10, radiomics", " (V10)"), value: r.roc_auc,
                  color: r.family === "quantum" ? PALETTE[1] : r.family === "hybrid" ? PALETTE[4] : r.family === "deep_learning" ? PALETTE[2] : PALETTE[0] }))} />
            ) : <EmptyState title="No evaluated models yet">Trained and evaluated models will appear here.</EmptyState>}
          </CardBody>
        </Card>
        <Card className="xl:col-span-2">
          <CardHeader title="My prediction outcomes" description="Slice-level calls of completed predictions" />
          <CardBody>
            {dist.isLoading ? <Skeleton className="h-48" /> : (dist.data?.by_label || []).length ? (
              <BarChart title="Predictions by outcome" format={(v) => String(Math.round(v))}
                data={dist.data.by_label.map((r: any, i: number) => ({ label: r.label, value: r.n, color: r.label.startsWith("Tumour") ? "#c2410c" : r.label.startsWith("No") ? "#14a08e" : r.label.startsWith("Uncertain") ? "#e0a526" : PALETTE[6 - (i % 2)] }))} />
            ) : <EmptyState title="No predictions yet" action={<Link to="/app/workspace"><Button variant="outline" size="sm">Run your first prediction <LuArrowRight /></Button></Link>}>Upload a CT slice in the imaging workspace.</EmptyState>}
          </CardBody>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Recent activity" />
          <CardBody className="p-0">
            {(o?.recent_activity || []).length ? (
              <ul className="divide-y divide-line">
                {o.recent_activity.map((a: any, i: number) => (
                  <li key={i} className="flex items-center justify-between gap-3 px-5 py-2.5 text-sm">
                    <span className="truncate">{humanize(a.action.replace(".", " — "))}{a.details?.model_id ? ` · ${a.details.model_id}` : ""}</span>
                    <span className="shrink-0 text-xs text-muted">{fmtDate(a.created_at)}</span>
                  </li>
                ))}
              </ul>
            ) : <EmptyState title="No activity recorded yet" />}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Processing status" />
          <CardBody className="space-y-3 text-sm">
            <div className="flex items-center justify-between"><span className="text-muted">Model engine</span>
              <Badge tone={statusTone(engineState || "unknown")}>{engineState || "unknown"}</Badge></div>
            <div className="flex items-center justify-between"><span className="text-muted">V9 engine</span><span>{o?.engine?.v9_engine || "—"}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">Engine load time</span><span>{o?.engine?.load_seconds ? `${fmtNum(o.engine.load_seconds, 1)} s` : "—"}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">Background jobs pending</span><span className="tabular-nums">{o ? o.jobs_pending : "—"}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">Experiments (recorded + yours)</span><span className="tabular-nums">{o ? o.experiments_total : "—"}</span></div>
            <p className="pt-2 text-xs text-muted">Updated {fmtDate(o?.generated_at)}</p>
          </CardBody>
        </Card>
      </div>
      <Disclaimer compact className="mt-6" />
    </>
  );
}
