import { useMemo, useState } from "react";
import { LuDownload, LuInfo } from "react-icons/lu";
import { BarChart, PALETTE } from "../components/charts/charts";
import { PageHeader } from "../components/layout/AppShell";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Select } from "../components/ui/input";
import { Alert, Badge, EmptyState, Skeleton, Table, Tabs, td, th } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, download } from "../lib/api";
import { fmtBytes, fmtMs, fmtNum, humanize } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { useQuery } from "../lib/query";
import { Link } from "../lib/router";

const COLS: [string, string][] = [["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["accuracy", "Accuracy"],
  ["balanced_accuracy", "Bal. acc."], ["sensitivity", "Sensitivity"], ["specificity", "Specificity"], ["f1", "F1"],
  ["mcc", "MCC"], ["log_loss", "Log loss"]];
const FAMILY_COLOR: Record<string, string> = { classical: PALETTE[0], quantum: PALETTE[1], deep_learning: PALETTE[2], hybrid: PALETTE[4] };

export default function Compare() {
  useDocumentTitle("Model comparison");
  const toast = useToast();
  const q = useQuery(["comparison"], () => api.get("/api/v1/analytics/model-comparison"));
  const [source, setSource] = useState<"v10" | "v9">("v10");
  const [family, setFamily] = useState("");
  const rows: any[] = useMemo(() => (q.data?.rows || []).filter((r: any) => r.source === source && (!family || r.family === family)), [q.data, source, family]);
  const v10count = (q.data?.rows || []).filter((r: any) => r.source === "v10").length;
  return (
    <>
      <PageHeader title="Model comparison" description="Only trained and evaluated models, each shown with the evaluation that produced its numbers."
        actions={<>
          <Button variant="outline" onClick={() => download("/api/v1/analytics/model-comparison.csv", "model_comparison.csv").catch((e) => toast("error", e.message))}><LuDownload /> CSV</Button>
          <Button variant="outline" onClick={() => download("/api/v1/analytics/model-comparison.pdf", "model_comparison.pdf").catch((e) => toast("error", e.message))}><LuDownload /> PDF report</Button>
        </>} />
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Tabs value={source} onChange={setSource} tabs={[{ id: "v10", label: `V10 nested evaluation (${v10count})` }, { id: "v9", label: "V9 fixed split (recorded)" }]} />
        <Select aria-label="Family" className="w-48" value={family} onChange={(e) => setFamily(e.target.value)}>
          <option value="">All families</option>{["classical", "deep_learning", "quantum", "hybrid"].map((f) => <option key={f} value={f}>{humanize(f)}</option>)}
        </Select>
      </div>
      <Alert tone="info" className="mb-4" title={source === "v10" ? "V10: nested leave-one-unit-out evaluation" : "V9: recorded results on V9's fixed split"}>
        {source === "v10"
          ? "Primary metric: mean within-unit ROC-AUC over the reconstructed scan-group units with ≥ 10 patches of each class. Other metrics are pooled out-of-fold at each fold's inner-chosen threshold. Patch level; not patient level."
          : "Test groups G01, G07 and G08 (622 patches, 531 from G01), patch level. The V9 validation set is saturated and models on V9 CNN embeddings are exploratory (Stage 2). Do not compare these numbers directly with V10."}
      </Alert>
      {q.isLoading ? <Skeleton className="h-80" /> : rows.length === 0 ? (
        <Card><EmptyState title={source === "v10" ? "No V10 evaluations saved yet" : "No V9 results"}>{source === "v10" ? "Run scripts/run_nested_evaluation.py and the training scripts; results appear here automatically." : ""}</EmptyState></Card>
      ) : (
        <>
          <Card className="mb-6">
            <CardHeader title={source === "v10" ? "Mean within-unit ROC-AUC" : "Test ROC-AUC"} description="Colour = family (teal classical, navy quantum, amber deep learning, violet hybrid)" />
            <CardBody>
              <BarChart horizontal title="ROC-AUC by model" domain={[0.5, 1]} format={(v) => v.toFixed(3)}
                data={rows.map((r) => ({ label: r.name.replace(/ — V10, \w+/, ""), value: r.roc_auc, color: FAMILY_COLOR[r.family] || PALETTE[6] }))} />
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="All metrics" description={<span className="flex items-center gap-1"><LuInfo /> Hover a model name for its evaluation; sizes and times as recorded.</span>} />
            <Table>
              <thead><tr><th className={th}>Model</th><th className={th}>Family</th>{COLS.map(([, l]) => <th key={l} className={th}>{l}</th>)}<th className={th}>Train</th><th className={th}>Inference / patch</th><th className={th}>Size</th></tr></thead>
              <tbody>{rows.map((r) => (
                <tr key={r.model_id} className="hover:bg-surface">
                  <td className={td}><Link to={`/app/models/${r.model_id}`} title={`${r.evaluation} — ${r.evaluation_unit}`} className="font-medium text-brand-700 hover:underline dark:text-brand-300">{r.name}</Link>
                    {r.role === "control" && <Badge className="ml-1.5">control</Badge>}</td>
                  <td className={td}>{humanize(r.family)}</td>
                  {COLS.map(([k]) => <td key={k} className={td + " tabular-nums"}>{fmtNum(k === "roc_auc" ? r.roc_auc : r.metrics[k], 3)}</td>)}
                  <td className={td + " tabular-nums"}>{r.train_seconds ? `${fmtNum(r.train_seconds, 1)} s` : "—"}</td>
                  <td className={td + " tabular-nums"}>{fmtMs(r.inference_ms_per_patch)}</td>
                  <td className={td + " tabular-nums"}>{fmtBytes(r.size_bytes)}</td>
                </tr>))}</tbody>
            </Table>
            <CardBody><ul className="list-disc space-y-1 pl-5 text-xs text-muted">{(q.data?.notes || []).map((n: string) => <li key={n}>{n}</li>)}</ul></CardBody>
          </Card>
        </>
      )}
    </>
  );
}
