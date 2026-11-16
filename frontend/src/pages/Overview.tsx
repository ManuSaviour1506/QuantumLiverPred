import { LuActivity, LuBoxes, LuDatabase, LuGauge, LuLayers, LuPlay, LuTimer, LuUpload, LuCircleCheck,
  LuCircleX, LuTriangleAlert, LuCpu } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { EmptyState, Skeleton } from "../components/ui/misc";
import { FamilyChip, Kpi, PredictionPill, StatusPill } from "../components/compare/ui";
import { compareApi, FAMILIES, FAMILY_COLOR, FAMILY_LABEL, fmtMsShort, type Family } from "../lib/compare";
import { fmtDate } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link, navigate } from "../lib/router";

export default function Overview() {
  useDocumentTitle("Overview");
  const st = useQuery(["status"], compareApi.status, { refetchInterval: 15000 });
  const hist = useQuery(["history", "recent"], () => compareApi.history(8));
  const bench = useQuery(["benchmarks"], compareApi.benchmarks);
  const s = st.data;
  const eng = s?.engine || {};
  const fams = s?.families || {};
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Research & model benchmarking" title="Liver Tumour Intelligence Platform"
        description="One CT slice, every eligible classical, deep-learning, quantum and hybrid model, executed concurrently — with measured predictions, execution times and the reasons for every model that cannot run."
        actions={<><Button variant="outline" onClick={() => navigate("/app/upload?tab=sample")}><LuDatabase /> Use a dataset sample</Button>
          <Button onClick={() => navigate("/app/upload")}><LuUpload /> Upload image</Button></>} />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <Kpi label="Registered models" value={eng.models_total ?? "—"} icon={<LuBoxes className="h-5 w-5" />} hint="V9 + V10 artifacts, incl. legacy" />
        <Kpi label="Available now" value={eng.models_servable ?? "—"} icon={<LuCircleCheck className="h-5 w-5" />} accent="#34d399"
          hint={s?.availability ? `${s.availability.requires_context ?? 0} need neighbouring slices` : undefined} />
        <Kpi label="Model families" value={4} icon={<LuLayers className="h-5 w-5" />} hint="classical · DL · quantum · hybrid" />
        <Kpi label="Completed runs" value={s?.stats?.completed_runs ?? "—"} icon={<LuActivity className="h-5 w-5" />} accent="#a78bfa" />
        <Kpi label="Avg inference" value={fmtMsShort(s?.stats?.avg_inference_ms)} icon={<LuTimer className="h-5 w-5" />} accent="#22d3ee" hint="per model, measured in runs" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card>
          <CardHeader title="Model families" description="Servable models per family and best recorded evaluation (labelled data, not single images)"
            action={<Link to="/app/models" className="text-xs text-brand-400 hover:underline">Registry →</Link>} />
          <CardBody className="grid gap-3 sm:grid-cols-2">
            {FAMILIES.map((f) => {
              const fam = fams[f] || {};
              const b = bench.data?.families?.[f];
              const best = b?.best_by_protocol?.v10_nested || b?.best_by_protocol?.v9_fixed_split;
              const lat = b?.latency?.median_of_model_warm_medians_ms;
              return (
                <div key={f} className="relative overflow-hidden rounded-xl border border-line bg-panel2 p-4">
                  <div className="absolute inset-x-0 top-0 h-0.5" style={{ background: FAMILY_COLOR[f] }} />
                  <div className="flex items-center justify-between"><FamilyChip family={f as Family} />
                    <span className="text-xs text-muted"><b className="text-ink">{fam.servable ?? "—"}</b> / {fam.registered ?? "—"} servable</span></div>
                  <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
                    <div><dt className="text-muted">Best recorded</dt><dd className="truncate font-mono text-ink" title={best?.model_id}>{best ? best.model_id : "—"}</dd></div>
                    <div><dt className="text-muted">{best?.metric === "primary_mean_within_unit_auc" ? "Within-unit AUC" : "Test AUC"}</dt><dd className="font-mono text-ink">{best ? best.value.toFixed(3) : "—"}</dd></div>
                    <div><dt className="text-muted">Median warm inference</dt><dd className="font-mono text-ink">{fmtMsShort(lat)}</dd></div>
                    <div><dt className="text-muted">Measured models</dt><dd className="font-mono text-ink">{b?.latency?.n_models_measured ?? 0}</dd></div>
                  </dl>
                </div>
              );
            })}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="System status" />
          <CardBody className="space-y-2.5 text-sm">
            {!s && <Skeleton className="h-32" />}
            {s && <>
              {[
                ["Model engine", eng.state === "ready", eng.state],
                ["ImageNet backbone weights", Object.values(eng.weights || {}).every((w: any) => w.ok), Object.entries(eng.weights || {}).map(([k, w]: any) => `${k} ${w.ok ? "ok" : "missing"}`).join(" · ")],
                ["OOD detector", !!eng.ood_detector, eng.ood_detector ? "loaded" : "unavailable"],
                ["Liver gate + U-Net validation", !!eng.liver_gate && !!eng.unet_liver_validation, `${eng.liver_gate ? "gate" : "no gate"} · ${eng.unet_liver_validation ? "U-Net" : "no U-Net"}`],
              ].map(([k, ok, v]) => (
                <div key={k as string} className="flex items-center gap-2.5 rounded-lg border border-line bg-panel2 px-3 py-2">
                  {ok ? <LuCircleCheck className="h-4 w-4 text-emerald-400" /> : <LuCircleX className="h-4 w-4 text-red-400" />}
                  <span className="text-ink">{k as string}</span><span className="ml-auto text-xs text-muted">{v as string}</span>
                </div>
              ))}
              <div className="flex items-center gap-2.5 rounded-lg border border-line bg-panel2 px-3 py-2">
                <LuCpu className="h-4 w-4 text-brand-400" /><span className="text-ink">Execution</span>
                <span className="ml-auto text-right text-xs text-muted">{s.hardware?.cpu_count} CPU · {s.comparison?.scheduling?.replace(/_/g, " ")} · max {s.comparison?.max_concurrent_runs} runs</span>
              </div>
              {(s.availability?.missing_dependency || s.availability?.missing_weights) ? (
                <p className="flex items-start gap-1.5 text-xs text-amber-300"><LuTriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  {(s.availability.missing_dependency || 0) + (s.availability.missing_weights || 0)} models cannot run in this installation (missing weights or dependencies) — see the registry for the reasons.</p>) : null}
            </>}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Recent comparison runs" action={<Link to="/app/history" className="text-xs text-brand-400 hover:underline">All history →</Link>} />
        <CardBody className="overflow-x-auto p-0">
          {hist.data && !hist.data.items.length && <EmptyState title="No comparison run yet" action={<Button onClick={() => navigate("/app/upload")}><LuPlay /> Start the first run</Button>}>
            Upload a CT slice or pick a dataset sample, then click Run all models.</EmptyState>}
          {!!hist.data?.items.length && (
            <table className="w-full min-w-[860px] text-sm">
              <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
                {["Run", "Input", "Status", "Completed", "Majority call", "Agreement", "Fastest family", "Wall time", "When"].map((h) => <th key={h} className="px-4 py-2.5 font-semibold">{h}</th>)}
              </tr></thead>
              <tbody>
                {hist.data.items.map((r: any) => (
                  <tr key={r.id} className="cursor-pointer border-b border-line/50 hover:bg-white/[0.02]" onClick={() => navigate(`/app/live/${r.id}`)}>
                    <td className="px-4 py-2.5 font-mono text-xs text-ink">{r.id.slice(0, 12)}…</td>
                    <td className="px-4 py-2.5 text-xs text-muted">{r.input_name || r.sample_id}</td>
                    <td className="px-4 py-2.5"><StatusPill status={r.status} /></td>
                    <td className="px-4 py-2.5 tabular-nums">{r.n_completed ?? 0} / {r.n_models ?? 0}</td>
                    <td className="px-4 py-2.5">{r.majority_label ? <PredictionPill label={r.majority_label} status="ok" /> : <span className="text-xs text-muted">{r.stopped ? "input withheld" : "—"}</span>}</td>
                    <td className="px-4 py-2.5 tabular-nums">{r.agreement_pct != null ? `${r.agreement_pct}%` : "—"}</td>
                    <td className="px-4 py-2.5 text-xs">{r.fastest_family ? FAMILY_LABEL[r.fastest_family as Family] : "—"}</td>
                    <td className="px-4 py-2.5 font-mono text-xs">{fmtMsShort(r.wall_ms)}</td>
                    <td className="px-4 py-2.5 text-xs text-muted">{fmtDate(r.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardBody>
      </Card>
      <div className="grid gap-4 md:grid-cols-3">
        {[[LuUpload, "1 · Upload & validate", "QC, out-of-distribution and two-stage liver checks; see which models are compatible.", "/app/upload"],
          [LuActivity, "2 · Run all models", "Four family pipelines execute concurrently; panels update as each model finishes.", "/app/live"],
          [LuGauge, "3 · Benchmark", "Recorded accuracy on labelled scan-group-held-out data and measured latencies per family.", "/app/benchmarks"]].map(([Icon, t, d, to]: any) => (
          <Link key={t} to={to} className="card group p-4 transition hover:border-brand-500/40">
            <Icon className="h-5 w-5 text-brand-400" /><p className="mt-2 font-semibold text-ink">{t}</p><p className="mt-1 text-sm text-muted">{d}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
