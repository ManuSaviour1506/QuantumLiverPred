import { useMemo, useState } from "react";
import { LuCircleCheck, LuCircleX, LuSearch, LuInfo } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Skeleton } from "../components/ui/misc";
import { FamilyChip } from "../components/compare/ui";
import { compareApi, FAMILIES, FAMILY_LABEL, type Family } from "../lib/compare";
import { fmtDate } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link } from "../lib/router";
import { cn } from "../lib/utils";

const CAT_STYLE: Record<string, string> = {
  available: "bg-emerald-500/15 text-emerald-300 ring-emerald-400/30",
  requires_context: "bg-sky-500/15 text-sky-300 ring-sky-400/30",
  missing_dependency: "bg-amber-500/15 text-amber-300 ring-amber-400/30",
  missing_weights: "bg-amber-500/15 text-amber-300 ring-amber-400/30",
  non_deployable: "bg-red-500/15 text-red-300 ring-red-400/30",
  failed_integrity: "bg-red-500/15 text-red-300 ring-red-400/30",
  legacy: "bg-slate-500/15 text-slate-300 ring-slate-400/30",
  not_trained: "bg-slate-500/15 text-slate-300 ring-slate-400/30",
};
const CAT_LABEL: Record<string, string> = { available: "Available", requires_context: "Requires neighbouring slices",
  missing_dependency: "Missing dependency", missing_weights: "Missing weights", non_deployable: "Non-deployable",
  failed_integrity: "Failed integrity", legacy: "Legacy", not_trained: "Not trained" };

export default function Registry() {
  useDocumentTitle("Model registry");
  const a = useQuery(["availability"], compareApi.availability);
  const [cat, setCat] = useState("all");
  const [fam, setFam] = useState("all");
  const [q, setQ] = useState("");
  const models = useMemo(() => ((a.data?.models || []) as any[]).filter((m) => (cat === "all" || m.availability === cat) &&
    (fam === "all" || m.family === fam) && (!q || (m.id + " " + m.name + " " + m.architecture).toLowerCase().includes(q.toLowerCase()))), [a.data, cat, fam, q]);
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Model registry" title="Every trained model and whether it can run"
        description="The registry reflects real runtime availability: weights are verified at start-up, dependencies are checked, and models that cannot produce meaningful predictions are marked non-deployable. Nothing is hidden: every model that cannot run states why." />
      {a.isLoading && <Skeleton className="h-64" />}
      {a.data && <>
        <div className="flex flex-wrap gap-2">
          <button onClick={() => setCat("all")} className={cn("rounded-full px-3 py-1 text-xs ring-1 ring-line", cat === "all" ? "bg-brand-500/20 text-ink" : "text-muted")}>All ({a.data.models.length})</button>
          {Object.entries(a.data.counts as Record<string, number>).map(([k, v]) => (
            <button key={k} onClick={() => setCat(k)} title={a.data.categories[k]}
              className={cn("rounded-full px-3 py-1 text-xs ring-1", CAT_STYLE[k], cat === k && "outline outline-2 outline-offset-1 outline-brand-400")}>{CAT_LABEL[k] || k} ({v})</button>
          ))}
        </div>
        <Card>
          <CardHeader title={`${models.length} models`} description="Click a model id for its full metadata and recorded metrics."
            action={<div className="flex gap-2">
              <div className="relative"><LuSearch className="absolute left-2.5 top-2.5 h-4 w-4 text-muted" /><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search" className="h-9 w-48 pl-8" aria-label="Search models" /></div>
              <select value={fam} onChange={(e) => setFam(e.target.value)} className="h-9 rounded-lg border border-line bg-panel px-2 text-sm" aria-label="Family">
                <option value="all">All families</option>{FAMILIES.map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}</select>
            </div>} />
          <CardBody className="overflow-x-auto p-0">
            <table className="w-full min-w-[1100px] text-sm">
              <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-muted">
                {["Model", "Family", "Architecture", "Version", "Training data", "Weights", "Dependencies", "Input", "Status", "Last success"].map((h) => <th key={h} className="px-3 py-2.5 font-semibold">{h}</th>)}
              </tr></thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id} className="border-b border-line/50 align-top">
                    <td className="px-3 py-2"><Link to={`/app/models/${m.id}`} className="font-mono text-xs text-brand-400 hover:underline">{m.id}</Link>
                      <p className="max-w-[220px] truncate text-[11px] text-muted" title={m.name}>{m.name}</p></td>
                    <td className="px-3 py-2"><FamilyChip family={m.family as Family} /></td>
                    <td className="px-3 py-2 text-xs">{m.architecture}</td>
                    <td className="px-3 py-2 text-xs">{m.version}</td>
                    <td className="max-w-[160px] px-3 py-2 text-[11px] text-muted">{m.training_dataset || "—"}</td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted">{m.weights_location || "—"}</td>
                    <td className="px-3 py-2"><ul className="space-y-0.5">{(m.dependencies || []).map((d: any) => (
                      <li key={d.name} className="flex items-center gap-1 text-[11px]" title={d.detail}>
                        {d.ok ? <LuCircleCheck className="h-3 w-3 text-emerald-400" /> : <LuCircleX className={cn("h-3 w-3", d.optional ? "text-amber-400" : "text-red-400")} />}
                        <span className="text-muted">{d.name.split(" (")[0]}</span></li>))}</ul></td>
                    <td className="px-3 py-2 text-[11px] text-muted">{m.input_requirements.context_slices_each_side ? `${m.input_requirements.context_slices_each_side * 2 + 1}-slice context` : "single slice"}{m.input_requirements.features ? ` · ${m.input_requirements.features}` : ""}</td>
                    <td className="max-w-[300px] px-3 py-2">
                      <span className={cn("inline-flex rounded-md px-2 py-0.5 text-[11px] font-semibold ring-1", CAT_STYLE[m.availability])}>{CAT_LABEL[m.availability] || m.availability}</span>
                      {m.reason && <p className="mt-1 text-[11px] leading-snug text-muted">{m.reason}</p>}
                    </td>
                    <td className="px-3 py-2 text-[11px] text-muted">{m.last_successful_inference ? fmtDate(m.last_successful_inference) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardBody>
        </Card>
        <Card><CardHeader title="Category definitions" /><CardBody className="grid gap-2 text-xs sm:grid-cols-2">
          {Object.entries(a.data.categories as Record<string, string>).map(([k, v]) => (
            <p key={k} className="flex items-start gap-2"><span className={cn("rounded px-1.5 py-0.5 font-semibold ring-1", CAT_STYLE[k])}>{CAT_LABEL[k]}</span><span className="text-muted">{v}</span></p>))}
          <p className="flex items-start gap-1.5 text-muted sm:col-span-2"><LuInfo className="mt-0.5 h-3.5 w-3.5" />Weights: {Object.entries(a.data.weights || {}).map(([k, w]: any) => `${k} ${w.ok ? "verified (SHA-256)" : "MISSING — " + w.reason}`).join(" · ")}</p>
        </CardBody></Card>
      </>}
    </div>
  );
}
