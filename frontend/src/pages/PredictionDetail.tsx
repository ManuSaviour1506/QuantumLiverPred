import { LuDownload, LuTrash2, LuArrowLeft, LuClock, LuAtom, LuCpu, LuCircleAlert } from "react-icons/lu";
import { BarChart } from "../components/charts/charts";
import { PageHeader } from "../components/layout/AppShell";
import { Disclaimer } from "../components/layout/Disclaimer";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Alert, Badge, Dialog, Progress, Skeleton } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, ApiError, download } from "../lib/api";
import { fmtDate, fmtMs, fmtNum, humanize, labelTone, statusTone } from "../lib/format";
import { useAuthImage, useDocumentTitle } from "../lib/hooks";
import { invalidate, useQuery } from "../lib/query";
import { Link, navigate, useParams } from "../lib/router";
import type { Prediction } from "../lib/types";
import { useEffect, useState, type ReactNode } from "react";

function ArtifactImage({ url, alt, caption }: { url: string; alt: string; caption: string }) {
  const { src, error } = useAuthImage(url);
  return (
    <figure className="space-y-2">
      {src ? <img src={src} alt={alt} className="aspect-square w-full rounded-xl bg-black object-contain ring-1 ring-line" />
        : error ? <div className="flex aspect-square items-center justify-center rounded-xl bg-surface text-sm text-muted">Image unavailable</div>
          : <Skeleton className="aspect-square w-full rounded-xl" />}
      <figcaption className="text-xs text-muted">{caption}</figcaption>
    </figure>
  );
}

function Row({ k, v }: { k: string; v: ReactNode }) {
  return <div className="flex justify-between gap-4 border-b border-line/60 py-2 text-sm last:border-0"><dt className="text-muted">{k}</dt><dd className="text-right font-medium">{v}</dd></div>;
}

export default function PredictionDetail() {
  const { id } = useParams<{ id: string }>();
  useDocumentTitle("Prediction");
  const toast = useToast();
  const [confirm, setConfirm] = useState(false);
  const [poll, setPoll] = useState<number | false>(1500);
  const q = useQuery(["prediction", id], () => api.get<Prediction>(`/api/v1/predictions/${id}`), {
    refetchInterval: poll, staleMs: 1000 });
  const p = q.data;
  const final = !!p && (p.status === "succeeded" || p.status === "failed");
  useEffect(() => { if (final) { setPoll(false); invalidate(["predictions"]); invalidate(["overview"]); } }, [final]);
  if (q.isError) {
    const e = q.error as ApiError;
    return <Alert tone="danger" title={e?.status === 404 ? "Prediction not found" : "Could not load the prediction"}>{e?.message}</Alert>;
  }
  if (!p) return <Skeleton className="h-96" />;
  const r = p.result || {};
  const d = r.details || {};
  const img = (k: string) => p.artifacts.find((a) => a.kind === k);
  const patches: { y: number; x: number; p: number }[] = d.patch_probabilities || [];
  const expl = r.explanation || {};
  const items = expl.feature_contributions?.items || expl.input_sensitivity?.items || expl.feature_importance?.items || [];
  return (
    <>
      <PageHeader title="Prediction result" description={<span className="font-mono text-xs">{p.id}</span>}
        actions={<>
          <Link to="/app/predictions"><Button variant="outline"><LuArrowLeft /> History</Button></Link>
          {p.status === "succeeded" && <Button variant="outline" onClick={() => download(`/api/v1/predictions/${p.id}/report`, `prediction_${p.id}.pdf`).catch((e) => toast("error", e.message))}><LuDownload /> PDF report</Button>}
          <Button variant="outline" onClick={() => setConfirm(true)}><LuTrash2 /> Delete</Button>
        </>} />
      {!final && (
        <Card className="mb-6"><CardBody className="space-y-3">
          <div className="flex items-center gap-2 text-sm"><LuClock className="text-brand-600" /> Status: <Badge tone="info">{p.status}</Badge> — the model is running in a background worker.</div>
          <Progress value={null} label="Inference progress" />
        </CardBody></Card>
      )}
      {p.status === "failed" && <Alert tone="danger" className="mb-6" title="The prediction failed">{p.error}</Alert>}
      {p.status === "succeeded" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
          <Card>
            <CardHeader title="Model output" description="Slice-level call from the largest liver-patch probability versus the model's operating threshold ± 0.10" />
            <CardBody>
              <div className="flex flex-wrap items-center gap-3">
                <Badge tone={labelTone(p.label)} className="px-3 py-1 text-sm">{p.label || "No output"}</Badge>
                {p.execution_mode === "custom_numpy_simulator" ? <Badge tone="brand"><LuAtom /> custom NumPy simulator (not hardware)</Badge> : <Badge><LuCpu /> {humanize(p.execution_mode)}</Badge>}
              </div>
              {r.reason && <Alert tone="warning" className="mt-4" title="Output withheld">{r.reason}</Alert>}
              {p.probability !== null && p.probability !== undefined && (
                <div className="mt-5">
                  <div className="mb-1 flex justify-between text-sm"><span className="text-muted">Model probability (top liver patch)</span><span className="font-semibold tabular-nums">{fmtNum(p.probability, 3)}</span></div>
                  <div className="relative h-3 rounded-full bg-gradient-to-r from-emerald-200 via-amber-200 to-red-300 dark:from-emerald-900 dark:via-amber-900 dark:to-red-900">
                    <div className="absolute -top-1 h-5 w-1 rounded bg-navy-900 dark:bg-white" style={{ left: `calc(${p.probability * 100}% - 2px)` }} title="probability" />
                    {p.threshold !== null && p.threshold !== undefined && <div className="absolute -top-2 h-7 border-l-2 border-dashed border-navy-500" style={{ left: `${p.threshold * 100}%` }} title="threshold" />}
                  </div>
                  <p className="mt-2 text-xs text-muted">Threshold {fmtNum(p.threshold, 3)} (dashed). {r.probability_note}</p>
                </div>
              )}
              <dl className="mt-5">
                <Row k="Model" v={<Link to={`/app/models/${p.model_id}`} className="text-brand-700 hover:underline dark:text-brand-300">{p.model_name || p.model_id}</Link>} />
                <Row k="Model version" v={p.model_version || "—"} />
                <Row k="Training dataset" v={p.dataset_version || "—"} />
                <Row k="Liver patches analysed" v={d.n_liver_patches ?? "—"} />
                <Row k="Processing time" v={fmtMs(p.processing_ms)} />
                <Row k="Submitted" v={fmtDate(p.created_at)} />
                <Row k="Completed" v={fmtDate(p.finished_at)} />
                <Row k="Input" v={`${(p.input_format || "").toUpperCase()} · ${p.input_name || "file"} · SHA-256 ${p.input_sha256_prefix}…`} />
              </dl>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Images" description="Explanation visualisations, not segmentations or validated tumour localisation" />
            <CardBody className="grid gap-4 sm:grid-cols-2">
              {img("preview") && <ArtifactImage url={img("preview")!.url} alt="Pre-processed input slice" caption="Uploaded slice after decoding and windowing" />}
              {img("patch_map") && <ArtifactImage url={img("patch_map")!.url} alt="Patch probability map" caption="This model's patch probabilities (red = higher); outlined patches exceed the threshold" />}
              {img("gradcam") && <ArtifactImage url={img("gradcam")!.url} alt="Grad-CAM heat map" caption="Grad-CAM: pixels that influenced the CNN score (not proof of correctness)" />}
              {!img("patch_map") && <p className="text-sm text-muted">No patch map: the model produced no patch scores for this input.</p>}
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Input checks" />
            <CardBody>
              <dl>
                <Row k="Image quality" v={<Badge tone={statusTone(d.quality?.status || "")}>{d.quality?.status || "—"}</Badge>} />
                {(d.quality?.warnings || []).length > 0 && <Row k="Quality warnings" v={d.quality.warnings.join(", ")} />}
                <Row k="Out-of-distribution check" v={<Badge tone={d.ood?.status === "in_distribution" ? "success" : d.ood?.status === "out_of_distribution" ? "warning" : "neutral"}>{humanize(d.ood?.status)}</Badge>} />
                {d.ood?.d2_ratio !== undefined && <Row k="Mahalanobis ratio (≤ 1 typical)" v={fmtNum(d.ood.d2_ratio, 3)} />}
                <Row k="Liver-tissue gate" v={d.liver_gate?.n_liver_patches !== undefined ? `${d.liver_gate.n_liver_patches} of ${d.liver_gate.n_candidates} patches` : humanize(d.liver_gate?.status)} />
                <Row k="Adjacent slices" v={(r.context || []).length ? r.context.join(" + ") : "not supplied"} />
                <Row k="HU available" v={d.image?.hu_available ? "yes (DICOM / NIfTI)" : "no (display image)"} />
              </dl>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Patch scores and explanation" />
            <CardBody className="space-y-4">
              {patches.length ? (
                <BarChart horizontal title="Liver-patch probabilities" domain={[0, 1]} format={(v) => v.toFixed(2)}
                  threshold={p.threshold != null ? { value: p.threshold, label: "operating threshold" } : undefined}
                  data={[...patches].sort((a, b) => b.p - a.p).slice(0, 12).map((x) => ({ label: `patch y${x.y} x${x.x}`, value: x.p, color: x.p >= (p.threshold ?? 0.5) ? "#c2410c" : "#14a08e" }))} />
              ) : <p className="text-sm text-muted">No patch-level scores for this prediction.</p>}
              {items.length > 0 && (
                <div>
                  <p className="mb-2 text-sm font-medium">{expl.feature_contributions?.method || expl.input_sensitivity?.method || expl.feature_importance?.method}</p>
                  <BarChart horizontal title="Feature explanation" format={(v) => v.toFixed(3)}
                    data={items.slice(0, 8).map((it: any) => ({ label: it.feature, value: it.contribution ?? it.sensitivity ?? it.importance, color: (it.contribution ?? it.sensitivity ?? 1) >= 0 ? "#305c89" : "#c2410c" }))} />
                </div>
              )}
              {expl.gradcam?.status === "unavailable" && <p className="flex gap-2 text-xs text-muted"><LuCircleAlert className="mt-0.5" /> Grad-CAM unavailable: {expl.gradcam.reason}</p>}
            </CardBody>
          </Card>
        </div>
      )}
      <Disclaimer className="mt-6" />
      <Dialog open={confirm} onClose={() => setConfirm(false)} title="Delete this prediction?"
        footer={<><Button variant="outline" onClick={() => setConfirm(false)}>Cancel</Button>
          <Button variant="danger" onClick={async () => {
            try { await api.del(`/api/v1/predictions/${p.id}`); invalidate(["predictions"]); toast("success", "Prediction deleted"); navigate("/app/predictions"); }
            catch (e) { toast("error", (e as Error).message); }
          }}>Delete</Button></>}>
        <p className="text-sm text-muted">The record, the uploaded image and the generated images are permanently removed.</p>
      </Dialog>
    </>
  );
}
