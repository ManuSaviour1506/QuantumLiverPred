import { useState } from "react";
import { LuDatabase, LuUpload, LuRefreshCw, LuTriangleAlert, LuArrowLeft } from "react-icons/lu";
import { BarChart } from "../components/charts/charts";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Field, Input, Select, Textarea } from "../components/ui/input";
import { Alert, Badge, Dialog, EmptyState, Pagination, Progress, Skeleton, Table, td, th } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, upload } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useZodForm } from "../lib/form";
import { fmtDate, fmtInt, humanize, statusTone } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { invalidate, useQuery } from "../lib/query";
import { Link, navigate, useParams, useSearchParams } from "../lib/router";
import { datasetSchema } from "../lib/schemas";
import type { Dataset, DatasetFile, Page } from "../lib/types";

function UploadDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const f = useZodForm(datasetSchema, { name: "", description: "", license: "", data_source: "" });
  return (
    <Dialog open={open} onClose={onClose} title="Upload a dataset">
      <form noValidate className="space-y-4" onSubmit={f.handleSubmit(async (v) => {
        if (!file) { f.setErrors({ _form: "Choose a .zip archive" }); return; }
        const fd = new FormData();
        fd.append("file", file);
        Object.entries(v).forEach(([k, val]) => fd.append(k, val));
        setProgress(0);
        try {
          const d = await upload<Dataset>("/api/v1/datasets/upload", fd, setProgress);
          invalidate(["datasets"]);
          toast("success", "Dataset uploaded — validation is running");
          onClose();
          navigate(`/app/datasets/${d.id}`);
        } finally { setProgress(null); }
      })}>
        <Alert tone="info">Upload a <b>.zip</b> with the images and a <code className="kbd">manifest.csv</code> (columns: image_path, label, patient_id, scan_id, volume_id, slice_id, modality, mask_path, split, data_source). Only upload data you are allowed to use; files stay private to you.</Alert>
        {f.errors._form && <Alert tone="danger">{f.errors._form}</Alert>}
        <div>
          <label htmlFor="ds-file" className="mb-1.5 block text-sm font-medium">Archive (.zip)</label>
          <input id="ds-file" type="file" accept=".zip" onChange={(e) => setFile(e.target.files?.[0] || null)} className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-brand-50 file:px-3 file:py-2 file:text-brand-700" />
        </div>
        <Field label="Name" name="name" error={f.errors.name}><Input {...f.register("name")} /></Field>
        <Field label="Description" name="description" error={f.errors.description}><Textarea {...f.register("description")} /></Field>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Licence" name="license" error={f.errors.license}><Input placeholder="e.g. CC BY-NC-SA 4.0" {...f.register("license")} /></Field>
          <Field label="Data source" name="data_source" error={f.errors.data_source}><Input placeholder="where the images come from" {...f.register("data_source")} /></Field>
        </div>
        {progress !== null && <Progress value={progress < 1 ? progress : null} label="Upload progress" />}
        <div className="flex justify-end gap-2"><Button type="button" variant="outline" onClick={onClose}>Cancel</Button><Button type="submit" loading={f.submitting}><LuUpload /> Upload</Button></div>
      </form>
    </Dialog>
  );
}

export default function Datasets() {
  useDocumentTitle("Datasets");
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  const q = useQuery(["datasets"], () => api.get<Dataset[]>("/api/v1/datasets"), { refetchInterval: 10000 });
  return (
    <>
      <PageHeader title="Dataset management" description="Uploaded and registered datasets, their versions and validation results."
        actions={user?.role !== "viewer" ? <Button onClick={() => setOpen(true)}><LuUpload /> Upload dataset</Button> : undefined} />
      {q.isError && <Alert tone="danger">Could not load datasets.</Alert>}
      <Card>
        {q.isLoading ? <div className="space-y-2 p-4">{[0, 1].map((i) => <Skeleton key={i} className="h-12" />)}</div> :
          (q.data || []).length === 0 ? <EmptyState icon={<LuDatabase className="h-6 w-6" />} title="No datasets yet">Upload a zip with a manifest, or ask an administrator to register the verified LiTS dataset folder (<code className="kbd">python -m app.manage register-dataset</code>).</EmptyState> : (
            <Table>
              <thead><tr>{["Name", "Source", "Version", "Files", "Status", "Licence", "Updated"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
              <tbody>{q.data!.map((d) => (
                <tr key={d.id} className="hover:bg-surface">
                  <td className={td}><Link to={`/app/datasets/${d.id}`} className="font-medium text-brand-700 hover:underline dark:text-brand-300">{d.name}</Link>{d.visibility === "shared" && <Badge className="ml-2">shared</Badge>}</td>
                  <td className={td}>{humanize(d.source)}</td>
                  <td className={td}>{d.latest_version ? `v${d.latest_version.version}` : "—"}</td>
                  <td className={td + " tabular-nums"}>{fmtInt(d.latest_version?.n_files)}</td>
                  <td className={td}>{d.latest_version && <Badge tone={statusTone(d.latest_version.status)}>{humanize(d.latest_version.status)}</Badge>}</td>
                  <td className={td}>{d.license || "—"}</td>
                  <td className={td}>{fmtDate(d.updated_at)}</td>
                </tr>))}</tbody>
            </Table>
          )}
      </Card>
      <UploadDialog open={open} onClose={() => setOpen(false)} />
    </>
  );
}

function Dist({ title, data }: { title: string; data?: Record<string, number> }) {
  const entries = Object.entries(data || {});
  if (!entries.length) return null;
  return (
    <Card><CardHeader title={title} /><CardBody>
      <BarChart horizontal title={title} format={(v) => String(Math.round(v))} data={entries.slice(0, 15).map(([k, v]) => ({ label: k, value: v }))} />
    </CardBody></Card>
  );
}

export function DatasetDetail() {
  const { id } = useParams<{ id: string }>();
  useDocumentTitle("Dataset");
  const toast = useToast();
  const { user } = useAuth();
  const [sp, set] = useSearchParams();
  const page = Number(sp.get("page") || 1);
  const status = sp.get("status") || "";
  const d = useQuery(["datasets", id], () => api.get<Dataset>(`/api/v1/datasets/${id}`), { refetchInterval: 4000 });
  const files = useQuery(["datasets", id, "files", page, status], () => api.get<Page<DatasetFile>>(`/api/v1/datasets/${id}/files`, { page, page_size: 20, status }));
  if (d.isError) return <Alert tone="danger" title="Dataset not found">It may not exist or you may not have access.</Alert>;
  if (!d.data) return <Skeleton className="h-96" />;
  const ds = d.data;
  const v = ds.latest_version;
  const val = v?.validation || {};
  const st = v?.statistics || {};
  return (
    <>
      <PageHeader title={ds.name} description={ds.description || humanize(ds.source)}
        actions={<>
          <Link to="/app/datasets"><Button variant="outline"><LuArrowLeft /> Datasets</Button></Link>
          {user?.role !== "viewer" && <Button variant="outline" onClick={async () => { try { await api.post(`/api/v1/datasets/${id}/validate`, {}); toast("info", "Validation queued"); invalidate(["datasets", id]); } catch (e) { toast("error", (e as Error).message); } }}><LuRefreshCw /> Re-validate</Button>}
        </>} />
      <div className="mb-6 grid gap-4 md:grid-cols-4">
        <Card className="p-4"><p className="text-xs uppercase text-muted">Status</p><div className="mt-1">{v ? <Badge tone={statusTone(v.status)}>{humanize(v.status)}</Badge> : "—"}</div></Card>
        <Card className="p-4"><p className="text-xs uppercase text-muted">Version</p><p className="mt-1 font-semibold">{v ? `v${v.version}` : "—"} <span className="font-mono text-xs font-normal text-muted">{v?.content_sha256?.slice(0, 12)}</span></p></Card>
        <Card className="p-4"><p className="text-xs uppercase text-muted">Files</p><p className="mt-1 text-xl font-semibold tabular-nums">{fmtInt(v?.n_files)}</p></Card>
        <Card className="p-4"><p className="text-xs uppercase text-muted">Licence</p><p className="mt-1 font-medium">{ds.license || "—"}</p></Card>
      </div>
      {(val.warnings || []).map((w: string) => <Alert key={w} tone="warning" className="mb-3"><span className="inline-flex gap-1"><LuTriangleAlert className="hidden" />{w}</span></Alert>)}
      {v && ["uploaded", "validating"].includes(v.status) && <Alert tone="info" className="mb-4">Validation is running in the background…</Alert>}
      {val.rows !== undefined && (
        <div className="mb-6 grid gap-6 lg:grid-cols-3">
          <Card className="lg:col-span-1"><CardHeader title="Validation results" /><CardBody className="space-y-2 text-sm">
            {[["Passed", val.passed ? "yes" : "no"], ["Rows", val.rows], ["File status", Object.entries(val.file_status || {}).map(([k, n]) => `${n} ${k}`).join(", ")],
              ["Missing labels", val.label_distribution?.["(missing)"] || 0], ["Exact duplicate groups", val.exact_duplicate_groups],
              ["Near-duplicate groups", val.near_duplicate_groups], ["Unlisted images", val.unlisted_images],
              ["Patient ids present", val.identifier_availability?.patient_id], ["Scan / group ids present", val.identifier_availability?.scan_id],
              ["Scan ids in several splits (leakage)", val.leakage?.scan_id?.ids_in_multiple_splits],
              ["Pixel-hash verification", val.pixel_hash_verification ? `${val.pixel_hash_verification.matched} / ${val.pixel_hash_verification.checked} matched` : "not applicable"],
              ["Validated", fmtDate(v?.validated_at)]].map(([k, x]) => (
              <div key={k as string} className="flex justify-between gap-3 border-b border-line/60 pb-1.5"><span className="text-muted">{k}</span><span className="text-right font-medium">{String(x ?? "—")}</span></div>))}
          </CardBody></Card>
          <div className="space-y-6 lg:col-span-2">
            <Dist title="Label distribution" data={st.label_distribution} />
            <div className="grid gap-6 md:grid-cols-2"><Dist title="Group (scan) distribution" data={st.group_distribution} /><Dist title="Split distribution" data={st.split_distribution} /></div>
          </div>
        </div>
      )}
      <Card>
        <CardHeader title="Manifest viewer" action={<Select aria-label="File status" className="w-44" value={status} onChange={(e) => set({ status: e.target.value, page: 1 })}>
          <option value="">All files</option>{["ok", "warning", "missing", "corrupt", "unchecked"].map((s) => <option key={s} value={s}>{s}</option>)}</Select>} />
        {files.isLoading ? <Skeleton className="m-4 h-40" /> : (files.data?.items.length || 0) === 0 ? <EmptyState title="No files match" /> : (
          <>
            <Table>
              <thead><tr>{["#", "Image path", "Label", "Scan / group", "Patient", "Slice", "Split", "Size", "Status", "Issues"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
              <tbody>{files.data!.items.map((f) => (
                <tr key={f.row_index}>
                  <td className={td + " tabular-nums text-muted"}>{f.row_index + 1}</td>
                  <td className={td + " max-w-64 truncate font-mono text-xs"} title={f.image_path}>{f.image_path}</td>
                  <td className={td}>{f.label || <span className="text-red-600">missing</span>}</td>
                  <td className={td}>{f.scan_id || "—"}</td><td className={td}>{f.patient_id || "—"}</td><td className={td}>{f.slice_id || "—"}</td>
                  <td className={td}>{f.split || "—"}</td><td className={td + " tabular-nums"}>{f.width ? `${f.width}×${f.height}` : "—"}</td>
                  <td className={td}><Badge tone={statusTone(f.file_status)}>{f.file_status}</Badge></td>
                  <td className={td + " max-w-56 text-xs text-muted"}>{f.issues.join("; ") || "—"}</td>
                </tr>))}</tbody>
            </Table>
            <div className="px-4"><Pagination page={files.data!.meta.page} pages={files.data!.meta.pages} total={files.data!.meta.total} onPage={(p) => set({ page: p })} /></div>
          </>
        )}
      </Card>
      {Object.keys(ds.provenance || {}).length > 0 && (
        <Card className="mt-6"><CardHeader title="Provenance" /><CardBody><pre className="overflow-x-auto text-xs text-muted">{JSON.stringify({ ...ds.provenance, root: undefined }, null, 2)}</pre></CardBody></Card>
      )}
    </>
  );
}
