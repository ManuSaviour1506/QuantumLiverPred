export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return new Intl.NumberFormat("en-GB").format(v);
}

export function fmtBytes(v: number | null | undefined): string {
  if (!v && v !== 0) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = v;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`;
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(v < 10 ? 1 : 0)} ms`;
}

export function fmtDate(iso: string | null | undefined, withTime = true): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-GB", withTime ? { dateStyle: "medium", timeStyle: "short" } : { dateStyle: "medium" });
}

export function labelTone(label: string | null | undefined): "danger" | "success" | "warning" | "neutral" {
  if (!label) return "neutral";
  if (label.startsWith("Tumour detected")) return "danger";
  if (label.startsWith("No tumour")) return "success";
  if (label.startsWith("Uncertain")) return "warning";
  return "neutral";
}

export function statusTone(status: string): "success" | "danger" | "warning" | "info" | "neutral" {
  if (["succeeded", "available", "validated", "ok", "ready", "deployed"].includes(status)) return "success";
  if (["failed", "corrupt", "missing", "validation_failed"].includes(status)) return "danger";
  if (["queued", "running", "validating", "loading", "uploaded"].includes(status)) return "info";
  if (["missing_files", "warning", "synthetic_only", "not_trained", "legacy", "withheld"].includes(status)) return "warning";
  return "neutral";
}

export function humanize(s: string | null | undefined): string {
  if (!s) return "—";
  return s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}
