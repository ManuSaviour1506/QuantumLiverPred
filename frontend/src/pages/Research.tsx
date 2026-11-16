import { useState } from "react";
import { LuAtom, LuFlaskConical, LuDownload, LuPlay } from "react-icons/lu";
import { BarChart, LineChart, PALETTE } from "../components/charts/charts";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Field, Input, Select } from "../components/ui/input";
import { Alert, Badge, EmptyState, Skeleton, Table, Tabs, td, th } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, download } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useZodForm } from "../lib/form";
import { fmtDate, fmtNum, humanize, statusTone } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { invalidate, useQuery } from "../lib/query";
import { experimentSchema } from "../lib/schemas";
import type { Dataset, Experiment } from "../lib/types";

type Tab = "quantum" | "nested" | "ablation" | "noise" | "validation" | "experiments";

function Json({ value }: { value: unknown }) {
  return <pre className="max-h-96 overflow-auto rounded-lg bg-surface p-3 text-xs leading-relaxed text-muted">{JSON.stringify(value, null, 1)}</pre>;
}

function NestedPanel({ data }: { data: any }) {
  const sets = Object.entries(data?.nested || {});
  if (!sets.length) return <Card><EmptyState title="No nested evaluation saved">Run scripts/run_nested_evaluation.py; results appear here.</EmptyState></Card>;
  return (
    <div className="space-y-6">
      {sets.map(([fs, n]: [string, any]) => (
        <Card key={fs}>
          <CardHeader title={`Nested evaluation — ${fs === "radiomics" ? "radiomics (confirmatory family)" : "frozen ImageNet features (secondary family)"}`}
            description={`${n.n_patches} patches · units ${n.units?.join(", ")} · primary metric over ${n.qualifying_units?.length} qualifying units · seeds ${n.seeds?.join(", ")}`} />
          <CardBody className="space-y-6">
            <BarChart horizontal title={`Mean within-unit AUC (${fs})`} domain={[0.5, 1]} format={(v) => v.toFixed(3)}
              data={[...n.models].sort((a: any, b: any) => (b.mean_within_unit_auc ?? 0) - (a.mean_within_unit_auc ?? 0)).map((m: any) => ({
                label: m.name, value: m.mean_within_unit_auc, note: `units ${fmtNum(m.min_unit_auc, 2)}–${fmtNum(m.max_unit_auc, 2)}`,
                color: m.family === "quantum" ? PALETTE[1] : m.family === "hybrid" ? PALETTE[4] : m.family === "control" ? PALETTE[6] : PALETTE[0] }))} />
            <div>
              <p className="mb-2 text-sm font-medium">Pre-declared comparisons (exact sign-flip test over units, Holm-corrected)</p>
              <Table>
                <thead><tr>{["Comparison", "Mean Δ AUC", "Units favouring first / second", "p (two-sided)", "p (Holm)", "Smallest attainable p", "Interpretation"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
                <tbody>{Object.entries(n.comparisons?.comparisons || {}).map(([k, c]: [string, any]) => (
                  <tr key={k}><td className={td + " font-medium"}>{k}</td><td className={td + " tabular-nums"}>{fmtNum(c.mean_difference, 4)}</td>
                    <td className={td}>{c.units_favouring_first} / {c.units_favouring_second}</td><td className={td + " tabular-nums"}>{fmtNum(c.p_two_sided, 4)}</td>
                    <td className={td + " tabular-nums"}>{fmtNum(c.p_holm, 4)}</td><td className={td + " tabular-nums"}>{fmtNum(c.smallest_attainable_p, 4)}</td>
                    <td className={td + " text-xs"}>{c.interpretation}</td></tr>))}</tbody>
              </Table>
            </div>
            <div>
              <p className="mb-2 text-sm font-medium">Per-unit within-unit AUC</p>
              <Table>
                <thead><tr><th className={th}>Model</th>{n.units.map((u: string) => <th key={u} className={th}>{u}{!n.qualifying_units.includes(u) && "*"}</th>)}<th className={th}>Seed SD</th><th className={th}>Params</th></tr></thead>
                <tbody>{n.models.map((m: any) => (
                  <tr key={m.key}><td className={td}>{m.name}</td>{n.units.map((u: string) => <td key={u} className={td + " tabular-nums"}>{fmtNum(m.per_unit[u], 3)}</td>)}
                    <td className={td + " tabular-nums"}>{fmtNum(m.seed_sd, 4)}</td><td className={td + " tabular-nums"}>{m.n_parameters ?? "—"}</td></tr>))}</tbody>
              </Table>
              <p className="mt-2 text-xs text-muted">* unit with fewer than 10 patches of one class: shown, excluded from the primary mean. Units are reconstructed scan groups, not patients.</p>
            </div>
          </CardBody>
        </Card>
      ))}
    </div>
  );
}

function ExperimentsPanel() {
  const { user } = useAuth();
  const toast = useToast();
  const list = useQuery(["experiments"], () => api.get<Experiment[]>("/api/v1/experiments"), { refetchInterval: 4000 });
  const datasets = useQuery(["datasets"], () => api.get<Dataset[]>("/api/v1/datasets"));
  const [sel, setSel] = useState<string | null>(null);
  const res = useQuery(["experiments", sel, "results"], () => api.get(`/api/v1/experiments/${sel}/results`), { enabled: !!sel });
  const f = useZodForm(experimentSchema, { name: "", kind: "quantum_diagnostics", n_qubits: 4, layers: 2, entangler: "brick", dataset_id: "" });
  return (
    <div className="grid gap-6 xl:grid-cols-[0.8fr_1.2fr]">
      <div className="space-y-6">
        {user?.role !== "viewer" && (
          <Card>
            <CardHeader title="Launch an experiment" description="Runs as a background job; results are stored with the configuration." />
            <CardBody>
              <form noValidate className="space-y-4" onSubmit={f.handleSubmit(async (v) => {
                const config: any = v.kind === "quantum_diagnostics" ? { n_qubits: v.n_qubits, layers: v.layers, entangler: v.entangler }
                  : v.kind === "noise_study" ? { model_id: "v10_vqc_radiomics", p2_levels: [0, 0.002, 0.01, 0.03] }
                    : { models: ["logreg", "logreg_pca4", "vqc", "qksvm"], seeds: [0] };
                const e = await api.post<Experiment>("/api/v1/experiments", { name: v.name, kind: v.kind, dataset_id: v.dataset_id || null, config });
                invalidate(["experiments"]);
                setSel(e.id);
                toast("success", "Experiment queued");
              })}>
                {f.errors._form && <Alert tone="danger">{f.errors._form}</Alert>}
                <Field label="Name" name="name" error={f.errors.name}><Input {...f.register("name")} /></Field>
                <Field label="Kind" name="kind"><Select {...f.register("kind")}>
                  <option value="quantum_diagnostics">Quantum circuit diagnostics</option>
                  <option value="noise_study">Noise study (deployed V10 VQC)</option>
                  <option value="nested_evaluation">Nested evaluation on a dataset</option></Select></Field>
                {f.values.kind === "quantum_diagnostics" && (
                  <div className="grid grid-cols-3 gap-3">
                    <Field label="Qubits" name="n_qubits" error={f.errors.n_qubits}><Input type="number" min={2} max={8} {...f.register("n_qubits")} /></Field>
                    <Field label="Layers" name="layers" error={f.errors.layers}><Input type="number" min={1} max={4} {...f.register("layers")} /></Field>
                    <Field label="Entangler" name="entangler"><Select {...f.register("entangler")}>{["brick", "ring", "linear", "full"].map((e) => <option key={e}>{e}</option>)}</Select></Field>
                  </div>
                )}
                {f.values.kind === "nested_evaluation" && (
                  <Field label="Dataset (validated)" name="dataset_id"><Select {...f.register("dataset_id")}>
                    <option value="">Select…</option>{(datasets.data || []).map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</Select></Field>
                )}
                <Button type="submit" loading={f.submitting}><LuPlay /> Run</Button>
              </form>
            </CardBody>
          </Card>
        )}
        <Card>
          <CardHeader title="Experiments" description="Recorded results from saved files, and your runs" />
          {list.isLoading ? <Skeleton className="m-4 h-32" /> : (list.data || []).length === 0 ? <EmptyState title="No experiments" /> : (
            <ul className="divide-y divide-line">
              {list.data!.map((e) => (
                <li key={e.id}><button onClick={() => setSel(e.id)} className={"flex w-full items-center justify-between gap-3 px-5 py-3 text-left text-sm hover:bg-surface " + (sel === e.id ? "bg-surface" : "")}>
                  <span className="min-w-0"><span className="block truncate font-medium">{e.name}</span><span className="text-xs text-muted">{humanize(e.kind)} · {e.source} · {fmtDate(e.created_at)}</span></span>
                  <Badge tone={statusTone(e.status)}>{e.status}</Badge></button></li>))}
            </ul>
          )}
        </Card>
      </div>
      <Card>
        <CardHeader title="Results" action={sel ? <Button size="sm" variant="outline" onClick={() => download(`/api/v1/experiments/${sel}/report`, `experiment_${sel}.pdf`).catch((e) => toast("error", e.message))}><LuDownload /> PDF</Button> : undefined} />
        <CardBody>
          {!sel ? <EmptyState icon={<LuFlaskConical className="h-6 w-6" />} title="Select an experiment" /> : res.isLoading ? <Skeleton className="h-64" /> : (
            <div className="space-y-4">
              {res.data?.status !== "succeeded" && <Alert tone={res.data?.status === "failed" ? "danger" : "info"}>Status: {res.data?.status}</Alert>}
              {Array.isArray(res.data?.results?.summary) && res.data.results.summary.length > 0 && typeof res.data.results.summary[0] === "object" && (
                <Table>
                  <thead><tr>{Object.keys(res.data.results.summary[0]).slice(0, 8).map((k) => <th key={k} className={th}>{humanize(k)}</th>)}</tr></thead>
                  <tbody>{res.data.results.summary.slice(0, 60).map((r: any, i: number) => <tr key={i}>{Object.keys(res.data.results.summary[0]).slice(0, 8).map((k) => <td key={k} className={td + " tabular-nums"}>{typeof r[k] === "number" ? fmtNum(r[k], 4) : String(r[k] ?? "—")}</td>)}</tr>)}</tbody>
                </Table>
              )}
              <Json value={res.data?.results} />
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

export default function Research() {
  useDocumentTitle("Research");
  const [tab, setTab] = useState<Tab>("quantum");
  const s = useQuery(["research-summary"], () => api.get("/api/v1/research/summary"));
  const d = s.data;
  const noise = d?.noise_study?.exact || d?.noise_study;
  return (
    <>
      <PageHeader title="Research and experiment dashboard" description="Quantum circuits, nested comparisons, hybrid ablations, noise, verification results and experiment logs." />
      <Alert tone="info" className="mb-4" title="Execution mode">{d?.execution_note || "All quantum results are simulations on the custom NumPy simulator; no quantum hardware was used."}</Alert>
      <div className="mb-5"><Tabs value={tab} onChange={setTab} tabs={[{ id: "quantum", label: "Quantum circuits" }, { id: "nested", label: "Classical vs quantum" },
        { id: "ablation", label: "Hybrid ablations" }, { id: "noise", label: "Noise" }, { id: "validation", label: "Verification" }, { id: "experiments", label: "Experiments & logs" }]} /></div>
      {s.isLoading && <Skeleton className="h-96" />}
      {d && tab === "quantum" && (
        <Card>
          <CardHeader title="Quantum and hybrid models" description="Circuit resources as deployed" />
          {(d.quantum_models || []).length === 0 ? <EmptyState icon={<LuAtom className="h-6 w-6" />} title="No quantum models available" /> : (
            <Table>
              <thead><tr>{["Model", "Source", "Qubits", "Layers", "Depth", "2-qubit gates", "Trainable params", "Feature map / readout", "Execution"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
              <tbody>{d.quantum_models.map((m: any) => (
                <tr key={m.model_id}><td className={td + " font-medium"}>{m.name}</td><td className={td}>{m.source}</td><td className={td}>{m.qubits ?? "—"}</td><td className={td}>{m.layers ?? "—"}</td>
                  <td className={td}>{m.depth ?? "—"}</td><td className={td}>{m.two_qubit_gates ?? "—"}</td><td className={td}>{m.trainable_parameters ?? "—"}</td>
                  <td className={td + " text-xs"}>{typeof m.feature_map === "object" && m.feature_map ? `${m.feature_map.pair} (reps ${m.feature_map.reps})` : (m.feature_map || m.readout || "—")}</td>
                  <td className={td}><Badge tone="brand">simulator</Badge></td></tr>))}</tbody>
            </Table>
          )}
        </Card>
      )}
      {d && tab === "nested" && <NestedPanel data={d} />}
      {d && tab === "ablation" && (
        <Card>
          <CardHeader title="Hybrid ablations (radiomics, nested evaluation)" description="Same inputs, seeds and evaluation; differences isolate the contribution of each component" />
          <CardBody>{(d.hybrid_ablation || []).length ? (
            <BarChart horizontal title="Hybrid ablations" domain={[0.5, 1]} format={(v) => v.toFixed(3)}
              data={d.hybrid_ablation.map((m: any) => ({ label: m.name, value: m.mean_within_unit_auc, color: m.key.includes("twin") || m.key === "logreg_pca4" ? PALETTE[6] : PALETTE[4] }))} />
          ) : <EmptyState title="No ablation results saved" />}</CardBody>
        </Card>
      )}
      {d && tab === "noise" && (
        <Card>
          <CardHeader title="Noise study — deployed V10 VQC" description={noise ? `${noise.noise_model}. ${noise.inputs}` : undefined} />
          <CardBody>{noise?.summary?.length ? (
            <div className="grid gap-6 lg:grid-cols-2">
              <LineChart title="AUC versus two-qubit depolarising probability" xLabel="p₂ (two-qubit depolarising)" yLabel="AUC"
                series={[{ name: "noisy", points: noise.summary.map((r: any) => ({ x: r.p2, y: r.auc ?? null })) },
                  { name: "noiseless", points: noise.summary.map((r: any) => ({ x: r.p2, y: r.auc_noiseless ?? null })), color: PALETTE[6] }]}
                xFormat={(v) => v.toFixed(3)} yFormat={(v) => v.toFixed(3)} />
              <LineChart title="Mean |logit change| versus noise" xLabel="p₂" yLabel="mean |Δ logit|"
                series={[{ name: "logit change", points: noise.summary.map((r: any) => ({ x: r.p2, y: r.mean_abs_logit_change })) }]}
                xFormat={(v) => v.toFixed(3)} yFormat={(v) => v.toFixed(2)} />
            </div>
          ) : <EmptyState title="No noise study saved">Run scripts/run_noise_experiments.py or launch a noise study under Experiments.</EmptyState>}</CardBody>
        </Card>
      )}
      {d && tab === "validation" && (
        <div className="grid gap-6 lg:grid-cols-2">
          {Object.entries({ ...d.stage2_verification, ...d.mathematical_validation }).filter(([, v]) => v).map(([k, v]) => (
            <Card key={k}><CardHeader title={humanize(k)} /><CardBody><Json value={v} /></CardBody></Card>))}
        </div>
      )}
      {tab === "experiments" && (
        <div className="space-y-6">
          <ExperimentsPanel />
          {d && (
            <Card><CardHeader title="Training and evaluation run logs" description="results/v10/runs/*" />
              {(d.runs || []).length === 0 ? <EmptyState title="No runs recorded" /> : (
                <Table><thead><tr>{["Run", "Status", "Started", "Seconds", "Artifact"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
                  <tbody>{d.runs.map((r: any) => <tr key={r.run}><td className={td + " font-mono text-xs"}>{r.run}</td><td className={td}><Badge tone={statusTone(r.status)}>{r.status}</Badge></td><td className={td}>{fmtDate(r.started_utc)}</td><td className={td}>{r.seconds}</td><td className={td + " font-mono text-xs"}>{r.artifact || "—"}</td></tr>)}</tbody></Table>
              )}
            </Card>
          )}
        </div>
      )}
    </>
  );
}
