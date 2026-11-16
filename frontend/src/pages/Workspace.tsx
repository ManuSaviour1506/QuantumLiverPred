import { useMemo, useState, type DragEvent } from "react";
import { LuUpload, LuImage, LuFileWarning, LuPlay, LuX, LuInfo, LuAtom } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Disclaimer } from "../components/layout/Disclaimer";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Field, Label, Select } from "../components/ui/input";
import { Alert, Badge, Progress, Skeleton } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, ApiError, upload } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtBytes, humanize } from "../lib/format";
import { useDocumentTitle, useObjectUrl } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { navigate, useSearchParams } from "../lib/router";
import { checkImageFile, IMAGE_ACCEPT } from "../lib/schemas";
import type { Model, SystemInfo } from "../lib/types";
import { cn } from "../lib/utils";

function FileDrop({ label, file, onFile, maxMb, optional }: { label: string; file: File | null;
  onFile: (f: File | null) => void; maxMb: number; optional?: boolean }) {
  const [drag, setDrag] = useState(false);
  const err = file ? checkImageFile(file, maxMb) : null;
  const preview = useObjectUrl(file && !err ? file : null);
  const id = `file-${label.replace(/\W+/g, "-").toLowerCase()}`;
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files?.[0] || null); };
  return (
    <div>
      <Label htmlFor={id} hint={optional ? "optional" : undefined}>{label}</Label>
      <label htmlFor={id} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
        className={cn("flex min-h-28 cursor-pointer items-center gap-4 rounded-xl border-2 border-dashed p-4 transition-colors",
          drag ? "border-brand-500 bg-brand-50/50 dark:bg-brand-500/5" : "border-line hover:border-brand-400", err && "border-red-400")}>
        {preview ? <img src={preview} alt="" className="h-20 w-20 rounded-lg object-cover ring-1 ring-line" />
          : <div className="flex h-20 w-20 items-center justify-center rounded-lg bg-surface text-muted">{file ? <LuImage className="h-7 w-7" /> : <LuUpload className="h-7 w-7" />}</div>}
        <div className="min-w-0 text-sm">
          {file ? (
            <>
              <p className="truncate font-medium">{file.name}</p>
              <p className="text-muted">{fmtBytes(file.size)} · {file.type || "unknown type"}</p>
              {!preview && !err && <p className="text-xs text-muted">Preview is shown for PNG / JPEG; DICOM and NIfTI are decoded on the server.</p>}
            </>
          ) : <p className="text-muted">Drop a file or <span className="font-medium text-brand-700 dark:text-brand-300">browse</span> — PNG, JPEG, DICOM (.dcm), NIfTI (.nii, .nii.gz), up to {maxMb} MB</p>}
          {err && <p className="mt-1 flex items-center gap-1 text-xs text-red-600" role="alert"><LuFileWarning /> {err}</p>}
        </div>
        {file && <button type="button" className="ml-auto rounded-md p-1 text-muted hover:bg-black/5" aria-label={`Remove ${label}`}
          onClick={(e) => { e.preventDefault(); onFile(null); }}><LuX /></button>}
      </label>
      <input id={id} type="file" accept={IMAGE_ACCEPT} className="sr-only" onChange={(e) => onFile(e.target.files?.[0] || null)} />
    </div>
  );
}

export default function Workspace() {
  useDocumentTitle("Imaging workspace");
  const { user } = useAuth();
  const toast = useToast();
  const [params] = useSearchParams();
  const models = useQuery(["models", "servable"], () => api.get<Model[]>("/api/v1/models", { servable_only: true }));
  const info = useQuery(["system-info"], () => api.get<SystemInfo>("/api/v1/system/info"));
  const maxMb = info.data?.limits.max_upload_mb || 20;
  const [modelId, setModelId] = useState(params.get("model") || "");
  const [file, setFile] = useState<File | null>(null);
  const [prev, setPrev] = useState<File | null>(null);
  const [next, setNext] = useState<File | null>(null);
  const [explain, setExplain] = useState(true);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const list = (models.data || []).filter((m) => (m.input_contract.context_slices_each_side || 0) <= 1);
  const groups = useMemo(() => {
    const g: Record<string, Model[]> = {};
    for (const m of list) (g[m.family] = g[m.family] || []).push(m);
    return g;
  }, [list]);
  const selected = list.find((m) => m.id === modelId) || null;
  const needsContext = !!selected?.input_contract.requires_context;
  const fileErr = file ? checkImageFile(file, maxMb) : "Choose an image";
  const ctxErr = needsContext && (!prev || !next) ? "This model reads adjacent slices: add the previous and next slice" : null;
  const disabled = !selected || !!fileErr || !!ctxErr || (prev && !!checkImageFile(prev, maxMb)) || (next && !!checkImageFile(next, maxMb)) || progress !== null;

  async function run() {
    if (!selected || !file) return;
    setError(null);
    setProgress(0);
    const fd = new FormData();
    fd.append("model_id", selected.id);
    fd.append("file", file);
    if (prev) fd.append("prev_file", prev);
    if (next) fd.append("next_file", next);
    fd.append("explain", explain ? "true" : "false");
    try {
      const r = await upload<{ id: string }>("/api/v1/predictions", fd, setProgress);
      toast("success", "Prediction queued — running the model");
      navigate(`/app/predictions/${r.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Upload failed");
      setProgress(null);
    }
  }

  return (
    <>
      <PageHeader title="Medical imaging workspace" description="Upload a CT slice, choose a trained model and run a real prediction. Results are experimental research outputs." />
      {user?.role === "viewer" && <Alert tone="info" className="mb-4">Your role (viewer) can browse results but cannot run predictions.</Alert>}
      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card>
          <CardHeader title="1 · Image" description="Pixels are decoded on the server; DICOM / NIfTI with Hounsfield units are windowed like the training data. PNG / JPEG carry no DICOM metadata." />
          <CardBody className="space-y-4">
            <FileDrop label="CT slice" file={file} onFile={setFile} maxMb={maxMb} />
            <details className="rounded-xl border border-line p-3" open={needsContext}>
              <summary className="cursor-pointer text-sm font-medium">Adjacent slices (2.5-D models) {needsContext && <Badge tone="warning" className="ml-2">required for this model</Badge>}</summary>
              <div className="mt-3 grid gap-4 md:grid-cols-2">
                <FileDrop label="Previous slice" file={prev} onFile={setPrev} maxMb={maxMb} optional={!needsContext} />
                <FileDrop label="Next slice" file={next} onFile={setNext} maxMb={maxMb} optional={!needsContext} />
              </div>
            </details>
          </CardBody>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader title="2 · Model" />
            <CardBody className="space-y-4">
              {models.isLoading ? <Skeleton className="h-10" /> : list.length === 0 ? (
                <Alert tone="warning" title="No servable model">The model registry has no model that can serve predictions. Check the API status page.</Alert>
              ) : (
                <Field label="Model" name="model">
                  <Select id="f-model" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                    <option value="">Select a trained model…</option>
                    {Object.entries(groups as Record<string, Model[]>).map(([fam, ms]) => (
                      <optgroup key={fam} label={humanize(fam)}>
                        {ms.map((m) => <option key={m.id} value={m.id}>{m.name}{m.input_contract.requires_context ? " · needs adjacent slices" : ""}</option>)}
                      </optgroup>
                    ))}
                  </Select>
                </Field>
              )}
              {selected && (
                <div className="space-y-2 rounded-xl bg-surface p-3 text-sm">
                  <div className="flex flex-wrap gap-1.5">
                    <Badge tone="navy">{selected.architecture}</Badge>
                    <Badge>{selected.version}</Badge>
                    <Badge tone={selected.training_status === "TRAINED" ? "success" : "warning"}>{selected.training_status}</Badge>
                    {selected.execution_mode === "custom_numpy_simulator" && <Badge tone="brand"><LuAtom /> NumPy simulator</Badge>}
                  </div>
                  <p className="text-muted">{selected.task}</p>
                  <p className="text-xs text-muted">Training data: {selected.dataset_version || "—"}</p>
                </div>
              )}
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" className="h-4 w-4 accent-brand-600" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
                Compute explanation maps when the model supports them (Grad-CAM for CNNs, sensitivities for feature models)
              </label>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="3 · Run" />
            <CardBody className="space-y-3">
              {ctxErr && <Alert tone="warning">{ctxErr}</Alert>}
              {error && <Alert tone="danger">{error}</Alert>}
              {progress !== null && <div className="space-y-1"><Progress value={progress < 1 ? progress : null} label="Upload progress" />
                <p className="text-xs text-muted">{progress < 1 ? `Uploading… ${Math.round(progress * 100)}%` : "Queuing the prediction…"}</p></div>}
              <Button className="w-full" size="lg" onClick={run} disabled={!!disabled || user?.role === "viewer"} loading={progress !== null}>
                <LuPlay /> Run prediction
              </Button>
              <p className="flex gap-2 text-xs text-muted"><LuInfo className="mt-0.5 shrink-0" />The output is a model score on liver-tissue patches, aggregated to a slice-level call. It is not a diagnosis and not a calibrated clinical confidence.</p>
            </CardBody>
          </Card>
        </div>
      </div>
      <Disclaimer compact className="mt-6" />
    </>
  );
}
