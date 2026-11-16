import { useMemo, useRef, useState, type DragEvent } from "react";
import { LuCircleCheck, LuCircleX, LuDatabase, LuFileImage, LuImageUp, LuPlay, LuListChecks, LuScanLine,
  LuTriangleAlert, LuCircleHelp, LuLayers } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Select } from "../components/ui/input";
import { Alert, Badge, Progress, Tabs } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { FamilyChip, StatusPill } from "../components/compare/ui";
import { compareApi, FAMILIES, FAMILY_LABEL, type Family, type Sample, type UploadOut } from "../lib/compare";
import { fmtBytes } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { navigate, useSearchParams } from "../lib/router";
import { cn } from "../lib/utils";

const CTX = [["prev2_file", "2 slices above"], ["prev_file", "Previous slice"], ["next_file", "Next slice"],
  ["next2_file", "2 slices below"]] as const;

function Check({ ok, warn, label, detail }: { ok: boolean | null; warn?: boolean; label: string; detail?: string }) {
  const Icon = ok === null ? LuCircleHelp : ok ? (warn ? LuTriangleAlert : LuCircleCheck) : LuCircleX;
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-line bg-panel2 px-3 py-2.5">
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", ok === null ? "text-muted" : ok ? (warn ? "text-amber-400" : "text-emerald-400") : "text-red-400")} />
      <div className="min-w-0 text-sm">
        <p className="font-medium text-ink">{label}</p>
        {detail && <p className="text-xs text-muted">{detail}</p>}
      </div>
    </div>
  );
}

export function AnalysisPanel({ up }: { up: UploadOut }) {
  const a = up.analysis;
  const lc = a.liver_check;
  return (
    <div className="grid gap-2.5 sm:grid-cols-2">
      <Check ok={a.quality?.status === "ok"} label="Image quality control"
        detail={a.quality?.status === "ok" ? `passed${a.quality.warnings?.length ? ` · warnings: ${a.quality.warnings.join(", ")}` : ""}` : a.quality?.reasons?.join(", ")} />
      <Check ok={a.ood?.status === "in_distribution" ? true : a.ood?.status ? false : null} label="Out-of-distribution check"
        detail={a.ood?.status ? `${a.ood.status.replace(/_/g, " ")}${a.ood.d2_ratio !== undefined ? ` · d² ratio ${a.ood.d2_ratio} (>1 = beyond 99th pct)` : ""}` : "not evaluated"} />
      <Check ok={a.liver_gate?.n_liver_patches ? true : a.liver_gate?.status ? false : null} label="Liver gate (patch classifier)"
        detail={a.liver_gate?.status ? `${a.liver_gate.n_liver_patches ?? 0} of ${a.liver_gate.n_candidates ?? 0} body patches look like liver` : "not evaluated"} />
      <Check ok={lc?.status === "confirmed" ? true : lc?.status === "rejected" ? false : null} label="Liver region validation (U-Net)"
        detail={lc?.status ? `${lc.status}: U-Net liver area ${lc.unet_liver_px ?? "—"} px (minimum ${lc.min_px} px)` : "skipped — earlier check stopped"} />
      <Check ok={a.context?.adjacent_slices ?? false} warn={!a.context?.five_slice_context} label="Neighbouring-slice context"
        detail={a.context?.five_slice_context ? "two slices on each side: 2.5-D, 3-D and 5-slice models can run"
          : a.context?.adjacent_slices ? "previous + next: 2.5-D models run; 3-D / 5-slice models need two each side"
          : "single slice: models that read neighbours will be reported as 'needs context'"} />
      <Check ok={!a.stopped} label="Ready for model execution" detail={a.stopped || `${a.n_liver_patches} liver patches will be analysed`} />
    </div>
  );
}

function CompatibilityTable({ up }: { up: UploadOut }) {
  const by = useMemo(() => {
    const m: Record<string, typeof up.analysis.compatibility> = {};
    for (const c of up.analysis.compatibility) (m[c.family] ||= []).push(c);
    return m;
  }, [up]);
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {FAMILIES.map((f) => {
        const rows = by[f] || [];
        const ok = rows.filter((r) => r.eligible).length;
        return (
          <div key={f} className="rounded-xl border border-line bg-panel2 p-3">
            <div className="mb-2 flex items-center justify-between"><FamilyChip family={f} />
              <span className="text-xs text-muted"><b className="text-ink">{ok}</b> / {rows.length} will run</span></div>
            <ul className="max-h-48 space-y-1 overflow-auto pr-1 text-xs">
              {rows.map((r) => (
                <li key={r.model_id} className="flex items-start justify-between gap-2">
                  <span className="truncate font-mono text-[11px] text-ink" title={r.name}>{r.model_id}</span>
                  {r.eligible ? <StatusPill status="ok">eligible</StatusPill>
                    : <span title={r.skip_reason || ""}><StatusPill status={r.skip_status || "unavailable"} /></span>}
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

function SelectModels({ up, onRun, busy }: { up: UploadOut; onRun: (ids: string[]) => void; busy: boolean }) {
  const eligible = up.analysis.compatibility.filter((c) => c.eligible);
  const [sel, setSel] = useState<Set<string>>(() => new Set(eligible.map((e) => e.model_id)));
  const toggle = (id: string) => setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2">
        {FAMILIES.map((f) => (
          <fieldset key={f} className="rounded-xl border border-line bg-panel2 p-3">
            <legend className="px-1"><FamilyChip family={f} /></legend>
            {eligible.filter((e) => e.family === f).length === 0 && <p className="text-xs text-muted">No eligible model for this input.</p>}
            {eligible.filter((e) => e.family === f).map((e) => (
              <label key={e.model_id} className="flex cursor-pointer items-center gap-2 py-0.5 text-xs">
                <input type="checkbox" checked={sel.has(e.model_id)} onChange={() => toggle(e.model_id)} className="accent-indigo-500" />
                <span className="font-mono text-[11px] text-ink">{e.model_id}</span>
              </label>
            ))}
          </fieldset>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <Button onClick={() => onRun([...sel])} disabled={!sel.size} loading={busy}><LuPlay /> Run {sel.size} selected models</Button>
        <Button variant="ghost" size="sm" onClick={() => setSel(new Set())}>Clear</Button>
        <Button variant="ghost" size="sm" onClick={() => setSel(new Set(eligible.map((e) => e.model_id)))}>All eligible</Button>
      </div>
    </div>
  );
}

export default function Upload() {
  useDocumentTitle("Image upload");
  const toast = useToast();
  const [params] = useSearchParams();
  const [tab, setTab] = useState<"file" | "sample">(params.get("tab") === "sample" ? "sample" : "file");
  const [file, setFile] = useState<File | null>(null);
  const [ctx, setCtx] = useState<Record<string, File | null>>({});
  const [modality, setModality] = useState("CT");
  const [progress, setProgress] = useState<number | null>(null);
  const [up, setUp] = useState<UploadOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [showSelect, setShowSelect] = useState(false);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const samples = useQuery(["samples"], compareApi.samples);
  const existing = params.get("upload");
  useQuery(["upload", existing], () => existing ? compareApi.getUpload(existing).then((u) => { setUp(u); return u; }) : Promise.resolve(null));

  const doUpload = async () => {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    for (const [k] of CTX) if (ctx[k]) fd.append(k, ctx[k] as File);
    fd.append("modality", modality);
    setBusy(true); setProgress(0); setUp(null);
    try { setUp(await compareApi.uploadFile(fd, setProgress)); }
    catch (e: any) { toast("error", e.message); }
    finally { setBusy(false); setProgress(null); }
  };
  const useSample = async (s: Sample) => {
    setBusy(true); setUp(null);
    try { setUp(await compareApi.uploadSample(s.id, modality)); }
    catch (e: any) { toast("error", e.message); }
    finally { setBusy(false); }
  };
  const runAll = async () => {
    if (!up) return;
    setBusy(true);
    try { const r = await compareApi.runAll(up.id); navigate(`/app/live/${r.run_id}`); }
    catch (e: any) { toast("error", e.message); setBusy(false); }
  };
  const runSelected = async (ids: string[]) => {
    if (!up) return;
    setBusy(true);
    try { const r = await compareApi.runSelected(up.id, ids); navigate(`/app/live/${r.run_id}`); }
    catch (e: any) { toast("error", e.message); setBusy(false); }
  };
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) setFile(f); };
  const eligible = up?.analysis.eligible_by_family;
  const nEligible = eligible ? Object.values(eligible).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Step 1" title="Image upload & analysis"
        description="Upload a CT slice (or pick a real slice from the dataset). The platform validates it — quality, out-of-distribution, liver region — and shows which models can run before anything executes." />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        <Card>
          <CardHeader title="Input" action={<Tabs tabs={[{ id: "file", label: "Upload file" }, { id: "sample", label: "Dataset sample" }]} value={tab} onChange={setTab} />} />
          <CardBody className="space-y-4">
            <div className="flex items-center gap-3 text-sm">
              <label htmlFor="modality" className="text-muted">Modality</label>
              <Select id="modality" value={modality} onChange={(e) => setModality(e.target.value)} className="w-32"><option>CT</option><option>MRI</option></Select>
              {modality === "MRI" && <span className="text-xs text-amber-400">All models were trained on CT; MRI is expected to be rejected as out-of-distribution.</span>}
            </div>
            {tab === "file" ? (
              <>
                <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
                  onClick={() => input.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
                  className={cn("grid-bg flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition",
                    drag ? "border-brand-400 bg-brand-500/10" : "border-line hover:border-brand-400/60")}>
                  <LuImageUp className="mb-2 h-9 w-9 text-brand-400" />
                  <p className="font-medium text-ink">{file ? file.name : "Drop a CT slice here or click to browse"}</p>
                  <p className="mt-1 text-xs text-muted">{file ? fmtBytes(file.size) : "PNG, JPG/JPEG, uncompressed DICOM (.dcm) or NIfTI (.nii / .nii.gz)"}</p>
                  <input ref={input} type="file" className="hidden" accept=".png,.jpg,.jpeg,.dcm,.nii,.gz,image/*"
                    onChange={(e) => setFile(e.target.files?.[0] || null)} />
                </div>
                <div>
                  <p className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted"><LuLayers className="h-3.5 w-3.5" /> Neighbouring slices (optional)</p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {CTX.map(([k, lab]) => (
                      <label key={k} className="flex items-center gap-2 rounded-lg border border-line bg-panel2 px-3 py-2 text-xs">
                        <LuFileImage className="h-4 w-4 text-muted" />
                        <span className="w-28 shrink-0 text-muted">{lab}</span>
                        <input type="file" accept=".png,.jpg,.jpeg,.dcm,.nii,.gz,image/*" className="min-w-0 flex-1 text-[11px] file:mr-2 file:rounded file:border-0 file:bg-brand-500/15 file:px-2 file:py-1 file:text-brand-300"
                          onChange={(e) => setCtx((c) => ({ ...c, [k]: e.target.files?.[0] || null }))} aria-label={lab} />
                      </label>
                    ))}
                  </div>
                  <p className="mt-1.5 text-[11px] text-muted">2.5-D models need the previous and next slices; 3-D and 5-slice models need two on each side.</p>
                </div>
                {progress !== null && <Progress value={progress} label="upload progress" />}
                <Button onClick={doUpload} disabled={!file} loading={busy} className="w-full"><LuScanLine /> Upload &amp; validate</Button>
              </>
            ) : (
              <div className="space-y-2">
                <p className="text-xs text-muted">Real LiTS-derived CT slices with expert-mask ground truth; neighbouring slices from the same scan are attached automatically. Synthetic phantoms are software test images (not medical data) used to check out-of-distribution rejection.</p>
                <div className="max-h-[420px] space-y-1.5 overflow-auto pr-1">
                  {(samples.data || []).map((s) => (
                    <button key={s.id} onClick={() => useSample(s)} disabled={busy}
                      className="flex w-full items-center gap-3 rounded-lg border border-line bg-panel2 px-3 py-2 text-left text-sm hover:border-brand-400/60 disabled:opacity-50">
                      <LuDatabase className={cn("h-4 w-4 shrink-0", s.category === "real_ct" ? "text-brand-400" : "text-slate-500")} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-mono text-xs text-ink">{s.id}</span>
                        <span className="text-[11px] text-muted">{s.category === "real_ct" ? `${s.scan_id} · ${s.split} split` : "synthetic phantom · not medical data"}</span>
                      </span>
                      {s.category === "real_ct" && <Badge tone={s.label_tumour ? "danger" : "neutral"}>{s.label_tumour ? "tumour (mask)" : (s.liver_px_512 ? "no tumour" : "no liver")}</Badge>}
                      <Badge tone={s.context.five_slice ? "success" : s.context.adjacent ? "info" : "neutral"}>
                        {s.context.five_slice ? "5-slice ctx" : s.context.adjacent ? "3-slice ctx" : "single"}</Badge>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Validation & model compatibility" description={up ? `${up.input_name || up.id} · ${up.input_format.toUpperCase()} · ${fmtBytes(up.input_bytes)} · ${up.modality}` : "Upload an image or pick a sample to see the checks"} />
          <CardBody className="space-y-5">
            {!up && !busy && <div className="grid-bg flex h-64 items-center justify-center rounded-xl border border-dashed border-line text-sm text-muted">No image yet</div>}
            {busy && !up && <div className="flex h-64 items-center justify-center text-sm text-muted"><span className="animate-pulse">Validating image: QC → OOD → liver gate → U-Net…</span></div>}
            {up && (
              <>
                <div className="grid gap-4 md:grid-cols-[220px_1fr]">
                  <div>
                    {up.analysis.preview_png_b64 ? <img src={`data:image/png;base64,${up.analysis.preview_png_b64}`} alt="Uploaded CT slice (canonical 512×512, shown at 384 px)"
                      className="aspect-square w-full rounded-xl border border-line object-cover" /> : <div className="aspect-square rounded-xl bg-panel2" />}
                    <dl className="mt-2 grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px] text-muted">
                      <dt>Format</dt><dd className="text-ink">{up.analysis.image?.format}</dd>
                      <dt>Original</dt><dd className="text-ink">{(up.analysis.image?.original_shape || []).join(" × ")}</dd>
                      <dt>HU</dt><dd className="text-ink">{up.analysis.image?.hu_available ? "yes" : "no (display PNG)"}</dd>
                      <dt>Context</dt><dd className="text-ink">{up.context_slices.length ? up.context_slices.join(", ") : "none"}</dd>
                      <dt>SHA-256</dt><dd className="truncate font-mono text-ink">{up.input_sha256.slice(0, 12)}</dd>
                    </dl>
                  </div>
                  <AnalysisPanel up={up} />
                </div>
                {up.analysis.modality_note && <Alert tone="warning">{up.analysis.modality_note}</Alert>}
                <div className="rounded-xl border border-brand-500/30 bg-gradient-to-r from-brand-500/10 via-transparent to-cyan-500/5 p-4">
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="mr-auto">
                      <p className="font-semibold text-ink">{nEligible} models eligible for this input</p>
                      <p className="text-xs text-muted">{FAMILIES.map((f) => `${FAMILY_LABEL[f as Family]} ${eligible?.[f as Family] ?? 0}`).join(" · ")}. Non-eligible models are still listed in the results with the reason.</p>
                    </div>
                    <Button size="lg" onClick={runAll} loading={busy}><LuPlay /> Run all models</Button>
                    <Button size="lg" variant="outline" onClick={() => setShowSelect((v) => !v)}><LuListChecks /> Run selected</Button>
                  </div>
                </div>
                {showSelect && <SelectModels up={up} onRun={runSelected} busy={busy} />}
                <div>
                  <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">Model compatibility</p>
                  <CompatibilityTable up={up} />
                </div>
              </>
            )}
          </CardBody>
        </Card>
      </div>
      <p className="text-xs text-muted">Single-model analysis with explanation maps is still available in the <a className="text-brand-400 hover:underline" href="/app/workspace" onClick={(e) => { e.preventDefault(); navigate("/app/workspace"); }}>imaging workspace</a>.</p>
    </div>
  );
}
