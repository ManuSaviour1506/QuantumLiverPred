import { LuDownload, LuSearch, LuHistory } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Input, Select } from "../components/ui/input";
import { Badge, EmptyState, Pagination, Skeleton, Table, td, th, Alert } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, download } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtDate, fmtMs, fmtNum, labelTone, statusTone } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link, useSearchParams } from "../lib/router";
import type { Model, Page, PredictionSummary } from "../lib/types";

const SORTS = [["-created_at", "Newest first"], ["created_at", "Oldest first"], ["-probability", "Highest probability"],
  ["probability", "Lowest probability"], ["model_id", "Model A–Z"]];

export default function History() {
  useDocumentTitle("Prediction history");
  const { user } = useAuth();
  const toast = useToast();
  const [sp, set] = useSearchParams();
  const query = { q: sp.get("q") || "", model_id: sp.get("model_id") || "", status: sp.get("status") || "",
    label: sp.get("label") || "", date_from: sp.get("from") || "", date_to: sp.get("to") || "",
    sort: sp.get("sort") || "-created_at", page: Number(sp.get("page") || 1), page_size: 15,
    all_users: sp.get("all") === "1" };
  const apiQuery = { ...query, date_from: query.date_from || undefined, date_to: query.date_to || undefined };
  const list = useQuery(["predictions", apiQuery], () => api.get<Page<PredictionSummary>>("/api/v1/predictions", apiQuery as any));
  const models = useQuery(["models", "all"], () => api.get<Model[]>("/api/v1/models"));
  const update = (k: string, v: string) => set({ [k]: v, page: 1 });
  return (
    <>
      <PageHeader title="Prediction history" description="Your predictions (administrators can include all users). Search, filter, sort and export."
        actions={<Button variant="outline" onClick={() => download("/api/v1/predictions/export.csv", "prediction_history.csv", apiQuery as any).catch((e) => toast("error", e.message))}><LuDownload /> Export CSV</Button>} />
      <Card className="mb-4 p-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <div className="relative lg:col-span-2">
            <LuSearch className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-muted" />
            <Input aria-label="Search" placeholder="Search file name, model or label" className="pl-9" defaultValue={query.q}
              onKeyDown={(e) => { if (e.key === "Enter") update("q", (e.target as HTMLInputElement).value); }}
              onBlur={(e) => e.target.value !== query.q && update("q", e.target.value)} />
          </div>
          <Select aria-label="Model" value={query.model_id} onChange={(e) => update("model_id", e.target.value)}>
            <option value="">All models</option>
            {(models.data || []).map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </Select>
          <Select aria-label="Status" value={query.status} onChange={(e) => update("status", e.target.value)}>
            <option value="">Any status</option>{["queued", "running", "succeeded", "failed"].map((s) => <option key={s}>{s}</option>)}
          </Select>
          <Select aria-label="Outcome" value={query.label} onChange={(e) => update("label", e.target.value)}>
            <option value="">Any outcome</option>
            {["Tumour detected", "No tumour detected", "Uncertain / unable to classify"].map((s) => <option key={s}>{s}</option>)}
          </Select>
          <Select aria-label="Sort" value={query.sort} onChange={(e) => update("sort", e.target.value)}>
            {SORTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </Select>
          <Input type="date" aria-label="From date" value={query.date_from} onChange={(e) => update("from", e.target.value)} />
          <Input type="date" aria-label="To date" value={query.date_to} onChange={(e) => update("to", e.target.value)} />
          {user?.role === "admin" && <label className="flex items-center gap-2 text-sm"><input type="checkbox" className="h-4 w-4 accent-brand-600" checked={query.all_users} onChange={(e) => update("all", e.target.checked ? "1" : "")} /> All users</label>}
        </div>
      </Card>
      {list.isError && <Alert tone="danger">Could not load the history.</Alert>}
      <Card>
        {list.isLoading ? <div className="space-y-2 p-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-10" />)}</div> :
          (list.data?.items.length || 0) === 0 ? <EmptyState icon={<LuHistory className="h-6 w-6" />} title="No predictions match" action={<Link to="/app/workspace"><Button variant="outline" size="sm">Open the workspace</Button></Link>}>Try clearing the filters, or run a prediction.</EmptyState> : (
            <>
              <Table>
                <thead><tr>{["Submitted", "File", "Model", "Status", "Outcome", "Probability", "Time"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
                <tbody>
                  {list.data!.items.map((p) => (
                    <tr key={p.id} className="hover:bg-surface">
                      <td className={td}><Link to={`/app/predictions/${p.id}`} className="font-medium text-brand-700 hover:underline dark:text-brand-300">{fmtDate(p.created_at)}</Link></td>
                      <td className={td + " max-w-48 truncate"}>{p.input_name || "—"}</td>
                      <td className={td}>{p.model_name || p.model_id}</td>
                      <td className={td}><Badge tone={statusTone(p.status)}>{p.status}</Badge></td>
                      <td className={td}>{p.label ? <Badge tone={labelTone(p.label)}>{p.label}</Badge> : "—"}</td>
                      <td className={td + " tabular-nums"}>{fmtNum(p.probability, 3)}</td>
                      <td className={td + " tabular-nums"}>{fmtMs(p.processing_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
              <div className="px-4"><Pagination page={list.data!.meta.page} pages={list.data!.meta.pages} total={list.data!.meta.total} onPage={(p) => set({ page: p })} /></div>
            </>
          )}
      </Card>
    </>
  );
}
