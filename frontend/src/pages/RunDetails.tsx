import { useMemo, useState } from "react";
import { LuActivity, LuDownload, LuFileText, LuListTree, LuSearch } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Input, Select } from "../components/ui/input";
import { Alert, EmptyState, Skeleton } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { ProbabilityBar } from "../components/charts/live";
import { FamilyChip, PredictionPill, StatusPill } from "../components/compare/ui";
import { download } from "../lib/api";
import { compareApi, FAMILIES, FAMILY_COLOR, FAMILY_LABEL, fmtMsShort, type Family, type ModelResult } from "../lib/compare";
import { fmtDate, fmtNum } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link, navigate, useParams } from "../lib/router";
import { cn } from "../lib/utils";

/** Patch probability overlay of one model on the input preview (positions are on the canonical 512 px grid). */
function PatchOverlay({ src, m }: { src: string; m: ModelResult }) {
  const pp = m.patch_probabilities || [];
  return (
    <div className="relative aspect-square w-full overflow-hidden rounded-xl border border-line">
      <img src={src} alt="input slice" className="h-full w-full object-cover" />
      <svg viewBox="0 0 512 512" className="absolute inset-0 h-full w-full" role="img" aria-label={`patch probabilities of ${m.model_id}`}>
        {pp.map((p, i) => {
          const hot = m.threshold != null && p.p >= m.threshold + (m.band ?? 0.1);
          return <rect key={i} x={p.x} y={p.y} width={96} height={96} fill={hot ? "#fb7185" : "#22d3ee"} fillOpacity={0.12 + 0.45 * p.p}
            stroke={m.top_patch && m.top_patch.x === p.x && m.top_patch.y === p.y ? "#fff" : "none"} strokeWidth={3}>
            <title>{`patch (${p.y}, ${p.x}): p = ${p.p.toFixed(3)}`}</title></rect>;
        })}
      </svg>
    </div>
  );
}

function RunList() {
  const hist = useQuery(["history", 0], () => compareApi.history(50));
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Detailed model results" title="Model results" description="Pick a run to inspect every model's prediction, probability, threshold, timings and the reason for any model that did not run." />
      <Card><CardBody className="p-0">
        {hist.isLoading && <div className="p-5"><Skeleton className="h-24" /></div>}
        {hist.data && !hist.data.items.length && <EmptyState title="No runs yet" action={<Button onClick={() => navigate("/app/upload")}>Upload an image</Button>} />}
        <ul className="divide-y divide-line">
          {(hist.data?.items || []).map((r: any) => (
            <li key={r.id}><Link to={`/app/runs/${r.id}`} className="flex flex-wrap items-center gap-3 px-5 py-3 hover:bg-white/[0.02]">
              <LuListTree className="h-4 w-4 text-muted" />
              <span className="font-mono text-xs text-ink">{r.id}</span>
              <span className="text-xs text-muted">{r.input_name || r.sample_id || r.upload_id} · {fmtDate(r.created_at)}</span>
              <span className="ml-auto flex items-center gap-3 text-xs"><StatusPill status={r.status} />
                <span className="text-muted">{r.n_completed ?? 0}/{r.n_models ?? 0} completed</span>
                {r.majority_label && <PredictionPill label={r.majority_label} status="ok" />}</span>
            </Link></li>
          ))}
        </ul>
      </CardBody></Card>
    </div>
  );
}

export default function RunDetails() {
  useDocumentTitle("Model results");
  const { id } = useParams<{ id: string }>();
  const toast = useToast();
  const run = useQuery(["run", id], () => compareApi.run(id!), { enabled: !!id, refetchInterval: 4000 });
  const up = useQuery(["upload", run.data?.upload_id], () => run.data ? compareApi.getUpload(run.data.upload_id) : Promise.resolve(null));
  const [fam, setFam] = useState<string>("all");
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  const res = run.data?.result;
  const rows = useMemo(() => (res?.models || []).filter((m) => (fam === "all" || m.family === fam) &&
    (!q || m.model_id.toLowerCase().includes(q.toLowerCase()))), [res, fam, q]);
  if (!id) return <RunList />;
  const selected = res?.models.find((m) => m.model_id === sel) || res?.models.find((m) => m.status === "ok");
  const preview = up.data?.analysis.preview_png_b64 ? `data:image/png;base64,${up.data.analysis.preview_png_b64}` : null;
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Detailed model results" title={`Run ${id.slice(0, 16)}…`}
        description={run.data ? `${run.data.mode.replace("_", " ")} · ${fmtDate(run.data.created_at)} · ${run.data.status}` : "loading…"}
        actions={<>
          <Button variant="outline" onClick={() => navigate(`/app/live/${id}`)}><LuActivity /> Live view</Button>
          <Button variant="outline" onClick={() => download(`/api/v1/compare/${id}/export.csv`, `comparison_${id}.csv`).catch((e) => toast("error", e.message))}><LuDownload /> CSV</Button>
          <Button variant="outline" onClick={() => download(`/api/v1/compare/${id}/report`, `comparison_${id}.pdf`).catch((e) => toast("error", e.message))}><LuFileText /> PDF</Button>
        </>} />
      {run.data && !res && <Alert tone="info">This run has not finished yet; the table appears when it completes. <Link to={`/app/live/${id}`} className="underline">Watch it live</Link>.</Alert>}
      {res && (
        <>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
            <Card>
              <CardHeader title="Per-model results" description="Probability = largest liver-patch probability; call uses the model's own threshold ± band. Margin is the distance from the threshold, not a probability of being correct."
                action={<div className="flex gap-2">
                  <div className="relative"><LuSearch className="absolute left-2.5 top-2.5 h-4 w-4 text-muted" /><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter models" className="h-9 w-44 pl-8" aria-label="Filter models" /></div>
                  <Select value={fam} onChange={(e) => setFam(e.target.value)} className="h-9 w-40" aria-label="Family"><option value="all">All families</option>{FAMILIES.map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}</Select>
                </div>} />
              <CardBody className="overflow-x-auto p-0">
                <table className="w-full min-w-[1120px] text-sm">
                  <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
                    {["Model", "Family", "Status / call", "Probability", "Thr", "Margin", "Queue", "Features", "Inference", "Total", "CPU", "Load", "Notes"].map((h) => <th key={h} className="px-3 py-2.5 font-semibold">{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {rows.map((m) => (
                      <tr key={m.model_id} onClick={() => setSel(m.model_id)} className={cn("cursor-pointer border-b border-line/50 hover:bg-white/[0.02]", selected?.model_id === m.model_id && "bg-brand-500/[0.06]")}>
                        <td className="px-3 py-2"><p className="font-mono text-xs text-ink">{m.model_id}</p><p className="text-[11px] text-muted">{m.architecture} · {m.version}</p></td>
                        <td className="px-3 py-2"><FamilyChip family={m.family} /></td>
                        <td className="px-3 py-2"><PredictionPill label={m.label} status={m.status} /></td>
                        <td className="w-36 px-3 py-2">{m.probability != null ? <><p className="mb-1 font-mono text-xs tabular-nums">{m.probability.toFixed(4)}</p><ProbabilityBar p={m.probability} thr={m.threshold} band={m.band} /></> : "—"}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums">{fmtNum(m.threshold, 3)}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums">{fmtNum(m.confidence, 2)}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums text-muted">{fmtMsShort(m.timings_ms?.queue)}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums text-muted">{fmtMsShort(m.timings_ms?.features)}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums text-ink">{fmtMsShort(m.timings_ms?.inference)}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums text-muted">{fmtMsShort(m.timings_ms?.total)}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums text-muted" title="CPU time of the executing thread (features + inference)">{fmtMsShort(m.resources?.thread_cpu_ms)}</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums text-muted" title="model load time at server start (artifact read + integrity + self-test)">{fmtMsShort(m.load_ms)}</td>
                        <td className="max-w-xs px-3 py-2 text-[11px] text-muted">
                          {m.status !== "ok" ? m.reason : [m.cold_start && "cold start", m.saturated && "saturated (p ≥ 0.999)", m.quantum && `${m.quantum.qubits} qubits · simulated on CPU`].filter(Boolean).join(" · ") || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardBody>
            </Card>
            <div className="space-y-4">
              <Card>
                <CardHeader title="Patch map" description={selected ? selected.model_id : "select a completed model"} />
                <CardBody className="space-y-3">
                  {preview && selected?.patch_probabilities ? <PatchOverlay src={preview} m={selected} /> : preview ? <img src={preview} alt="input" className="rounded-xl border border-line" /> : <Skeleton className="aspect-square" />}
                  {selected?.status === "ok" && <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                    <dt className="text-muted">Liver patches</dt><dd className="text-ink">{selected.n_liver_patches}</dd>
                    <dt className="text-muted">Calibration</dt><dd className="text-ink">{selected.probability_note}</dd>
                    <dt className="text-muted">Execution</dt><dd className="text-ink">{selected.quantum ? selected.quantum.backend : "CPU (NumPy / scikit-learn)"}</dd>
                  </dl>}
                  <p className="text-[11px] text-muted">Patch colours show where the model's probability is high; this is not a segmentation and not a diagnosis.</p>
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="Families" />
                <CardBody className="space-y-2 text-xs">
                  {FAMILIES.map((f) => { const s = res.families[f as Family]; return (
                    <div key={f} className="flex items-center gap-2 rounded-lg border border-line bg-panel2 px-3 py-2">
                      <span className="h-2 w-2 rounded-full" style={{ background: FAMILY_COLOR[f] }} />
                      <span className="w-24 text-ink">{FAMILY_LABEL[f]}</span>
                      <span className="text-muted">{s.n_completed}/{s.n_registered}</span>
                      <span className="ml-auto font-mono text-ink">{fmtMsShort(s.inference_ms_mean)}</span>
                    </div>); })}
                  <p className="text-muted">Feature extraction shared by several models: {Object.entries(res.feature_cache_ms).map(([k, v]) => `${k} ${fmtMsShort(v)}`).join(" · ") || "none"}</p>
                </CardBody>
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
