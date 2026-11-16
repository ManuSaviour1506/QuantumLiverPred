import { LuBoxes, LuDownload, LuSearch, LuX, LuAtom, LuPlay } from "react-icons/lu";
import { useMemo, useState } from "react";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Input, Select } from "../components/ui/input";
import { Alert, Badge, EmptyState, Skeleton, Table, td, th } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, download } from "../lib/api";
import { fmtBytes, fmtDate, fmtNum, humanize, statusTone } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link, navigate, useParams } from "../lib/router";
import type { Model } from "../lib/types";

function Metric({ k, v }: { k: string; v: any }) {
  if (v === undefined || v === null || typeof v === "object") return null;
  return <div className="rounded-lg bg-surface px-3 py-2"><p className="text-[11px] uppercase tracking-wide text-muted">{humanize(k)}</p><p className="font-semibold tabular-nums">{typeof v === "number" ? fmtNum(v, 3) : String(v)}</p></div>;
}

function ModelDetail({ m, onClose }: { m: Model; onClose: () => void }) {
  const toast = useToast();
  const evs = m.metrics?.evaluations || [];
  const circuit = m.metadata?.circuit;
  return (
    <Card className="animate-fade-in">
      <CardHeader title={m.name} description={<span className="font-mono text-xs">{m.id}</span>}
        action={<div className="flex gap-2">
          {m.servable && <Button size="sm" onClick={() => navigate(`/app/workspace?model=${m.id}`)}><LuPlay /> Use</Button>}
          <Button size="sm" variant="outline" onClick={() => download(`/api/v1/models/${m.id}/metadata.json`, `${m.id}_metadata.json`).catch((e) => toast("error", e.message))}><LuDownload /> JSON</Button>
          <Button size="sm" variant="ghost" onClick={onClose} aria-label="Close details"><LuX /></Button></div>} />
      <CardBody className="space-y-5 text-sm">
        <div className="flex flex-wrap gap-1.5">
          <Badge tone={statusTone(m.status)}>{m.status}</Badge>
          <Badge tone={m.deployment_status === "deployed" ? "success" : "neutral"}>{m.deployment_status}</Badge>
          <Badge tone="navy">{m.architecture}</Badge><Badge>{humanize(m.family)}</Badge><Badge>{m.version}</Badge>
          {m.training_status && <Badge tone={m.training_status === "TRAINED" ? "success" : "warning"}>{m.training_status}</Badge>}
          {m.execution_mode === "custom_numpy_simulator" && <Badge tone="brand"><LuAtom /> custom NumPy simulator</Badge>}
        </div>
        {m.reason && <Alert tone="warning">{m.reason}</Alert>}
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
          {[["Task", m.task], ["Training date", fmtDate(m.training_date)], ["Training dataset", m.dataset_version],
            ["Execution", humanize(m.execution_mode)], ["Artifact", m.artifact_path], ["Size", fmtBytes(m.size_bytes)],
            ["SHA-256", m.artifact_sha256 ? m.artifact_sha256.slice(0, 16) + "…" : "—"],
            ["Input", m.input_contract.requires_context ? "CT slice + previous and next slice" : "CT slice"],
            ["Threshold", fmtNum(m.metadata?.threshold, 3)], ["Parameters", m.metadata?.parameters ?? "—"]].map(([k, v]) => (
            <div key={k as string} className="flex justify-between gap-3 border-b border-line/60 py-1.5"><dt className="text-muted">{k}</dt><dd className="max-w-[60%] truncate text-right" title={String(v ?? "")}>{v || "—"}</dd></div>
          ))}
        </dl>
        {typeof m.metadata?.architecture_detail === "string" && <p className="rounded-lg bg-surface p-3 text-xs leading-relaxed text-muted">{m.metadata.architecture_detail.slice(0, 600)}</p>}
        {circuit && (
          <div>
            <p className="mb-2 font-medium">Quantum circuit</p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {["qubits", "layers", "entangler", "readout", "trainable_circuit_parameters"].map((k) => <Metric key={k} k={k} v={circuit[k] ?? circuit.configuration?.[k]} />)}
              {circuit.gate_counts && Object.entries(circuit.gate_counts).slice(0, 4).map(([k, v]) => <Metric key={k} k={k} v={v} />)}
            </div>
          </div>
        )}
        <div>
          <p className="mb-2 font-medium">Saved evaluations</p>
          {evs.length === 0 ? <p className="text-muted">No saved evaluation for this model.</p> : evs.map((ev, i) => (
            <div key={i} className="mb-3 rounded-xl border border-line p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="font-medium">{ev.name}</p><Badge tone="info">{ev.status}</Badge>
              </div>
              <p className="mt-1 text-xs text-muted">Evaluation unit: {ev.evaluation_unit || "—"}{typeof ev.split === "object" && ev.split ? ` · split ${Object.entries(ev.split).map(([k, v]) => `${k}: ${(v as string[]).join(", ")}`).join("; ")}` : ""}</p>
              <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                {Object.entries(ev.metrics || {}).filter(([, v]) => typeof v === "number").slice(0, 12).map(([k, v]) => <Metric key={k} k={k} v={v} />)}
              </div>
              {(ev.caveats || []).length > 0 && <ul className="mt-2 list-disc pl-5 text-xs text-muted">{ev.caveats!.map((c) => <li key={c}>{c}</li>)}</ul>}
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

export default function Models() {
  useDocumentTitle("Model registry");
  const { id } = useParams<{ id?: string }>();
  const q = useQuery(["models", "all"], () => api.get<Model[]>("/api/v1/models"));
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("");
  const [status, setStatus] = useState("");
  const rows = useMemo(() => (q.data || []).filter((m) =>
    (!family || m.family === family) && (!status || m.status === status) &&
    (!search || `${m.id} ${m.name} ${m.architecture}`.toLowerCase().includes(search.toLowerCase()))), [q.data, search, family, status]);
  const selected = (q.data || []).find((m) => m.id === id);
  const families = [...new Set((q.data || []).map((m) => m.family))];
  return (
    <>
      <PageHeader title="Model registry" description="Every model known to the engine. A model serves predictions only after its artifact passes the integrity and self-tests." />
      <div className="grid gap-6 xl:grid-cols-[1.25fr_1fr]">
        <Card>
          <div className="grid gap-3 border-b border-line p-4 sm:grid-cols-3">
            <div className="relative"><LuSearch className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-muted" />
              <Input aria-label="Search models" placeholder="Search" className="pl-9" value={search} onChange={(e) => setSearch(e.target.value)} /></div>
            <Select aria-label="Family" value={family} onChange={(e) => setFamily(e.target.value)}><option value="">All families</option>{families.map((f) => <option key={f} value={f}>{humanize(f)}</option>)}</Select>
            <Select aria-label="Status" value={status} onChange={(e) => setStatus(e.target.value)}><option value="">Any status</option>{["available", "not_trained", "failed", "synthetic_only"].map((s) => <option key={s} value={s}>{humanize(s)}</option>)}</Select>
          </div>
          {q.isLoading ? <div className="space-y-2 p-4">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-10" />)}</div> :
            rows.length === 0 ? <EmptyState icon={<LuBoxes className="h-6 w-6" />} title="No models match" /> : (
              <Table>
                <thead><tr>{["Model", "Family", "Version", "Status", "Primary metric"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
                <tbody>{rows.map((m) => {
                  const ev = m.metrics?.evaluations?.[0];
                  const met: any = ev?.metrics || {};
                  const v = met.primary_mean_within_unit_auc ?? met.auc;
                  return (
                    <tr key={m.id} className={id === m.id ? "bg-brand-50/60 dark:bg-brand-500/5" : "hover:bg-surface"}>
                      <td className={td}><Link to={`/app/models/${m.id}`} className="font-medium text-brand-700 hover:underline dark:text-brand-300">{m.name}</Link><p className="font-mono text-[11px] text-muted">{m.id}</p></td>
                      <td className={td}>{humanize(m.family)}</td>
                      <td className={td}>{m.version}</td>
                      <td className={td}><Badge tone={statusTone(m.status)}>{m.status}</Badge></td>
                      <td className={td + " tabular-nums"}>{v !== undefined && v !== null ? <span title={ev?.name}>{fmtNum(v, 3)} <span className="text-xs text-muted">{met.primary_mean_within_unit_auc != null ? "within-unit AUC" : "V9 test AUC"}</span></span> : "—"}</td>
                    </tr>);
                })}</tbody>
              </Table>
            )}
        </Card>
        <div>{selected ? <ModelDetail m={selected} onClose={() => navigate("/app/models")} /> :
          <Card><EmptyState icon={<LuBoxes className="h-6 w-6" />} title="Select a model">Choose a model to see its architecture, artifact, input contract and saved evaluations.</EmptyState></Card>}</div>
      </div>
    </>
  );
}
