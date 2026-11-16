// Shared visual vocabulary of the comparison pages: family chips, prediction pills, status pills.
import type { ReactNode } from "react";
import { LuAtom, LuBrain, LuCircleCheck, LuCircleSlash, LuCircleX, LuClock, LuCombine, LuLayers, LuLoaderCircle,
  LuScanLine, LuTriangleAlert } from "react-icons/lu";
import { FAMILY_COLOR, FAMILY_LABEL, LABEL_SHORT, STATUS_TEXT, type Family } from "../../lib/compare";
import { cn } from "../../lib/utils";

export const FAMILY_ICON: Record<Family, typeof LuBrain> = { classical: LuLayers, deep_learning: LuBrain,
  quantum: LuAtom, hybrid: LuCombine };

export function FamilyChip({ family, className }: { family: Family; className?: string }) {
  const Icon = FAMILY_ICON[family];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium", className)}
      style={{ color: FAMILY_COLOR[family], background: `${FAMILY_COLOR[family]}1a`, boxShadow: `inset 0 0 0 1px ${FAMILY_COLOR[family]}33` }}>
      <Icon className="h-3.5 w-3.5" aria-hidden />{FAMILY_LABEL[family]}
    </span>
  );
}

const LABEL_STYLE: Record<string, string> = {
  "Tumour detected": "bg-rose-50 text-rose-700 ring-rose-600/20 dark:bg-rose-500/15 dark:text-rose-300 dark:ring-rose-400/30",
  "No tumour detected": "bg-teal-50 text-teal-700 ring-teal-600/20 dark:bg-teal-500/15 dark:text-teal-300 dark:ring-teal-400/30",
  "Uncertain / unable to classify": "bg-amber-50 text-amber-800 ring-amber-600/20 dark:bg-amber-500/15 dark:text-amber-300 dark:ring-amber-400/30",
};

export function PredictionPill({ label, status }: { label?: string | null; status: string }) {
  if (status !== "ok" || !label) return <StatusPill status={status} />;
  return (
    <span className={cn("inline-flex items-center rounded-md px-2 py-0.5 text-xs font-semibold ring-1", LABEL_STYLE[label])}>
      {LABEL_SHORT[label] || label}
    </span>
  );
}

export function StatusPill({ status, children }: { status: string; children?: ReactNode }) {
  const map: Record<string, [string, typeof LuClock]> = {
    ok: ["text-emerald-500 dark:text-emerald-400", LuCircleCheck], completed: ["text-emerald-500 dark:text-emerald-400", LuCircleCheck],
    running: ["text-amber-500 dark:text-amber-300", LuLoaderCircle], pending: ["text-muted", LuClock],
    queued: ["text-muted", LuClock], error: ["text-red-500 dark:text-red-400", LuCircleX], failed: ["text-red-500 dark:text-red-400", LuCircleX],
    timeout: ["text-red-500 dark:text-red-400", LuClock], cancelled: ["text-muted", LuCircleSlash],
    needs_context: ["text-sky-600 dark:text-sky-300", LuScanLine], withheld: ["text-amber-600 dark:text-amber-300", LuTriangleAlert],
    out_of_distribution: ["text-amber-600 dark:text-amber-300", LuTriangleAlert], rejected: ["text-red-500 dark:text-red-400", LuTriangleAlert],
    unavailable: ["text-muted", LuCircleSlash],
  };
  const [cls, Icon] = map[status] || ["text-muted", LuCircleSlash];
  return (
    <span className={cn("inline-flex items-center gap-1 text-xs font-medium", cls)}>
      <Icon className={cn("h-3.5 w-3.5", status === "running" && "animate-spin")} aria-hidden />
      {children ?? (STATUS_TEXT[status] || status)}
    </span>
  );
}

export function Kpi({ label, value, hint, icon, accent }: { label: string; value: ReactNode; hint?: ReactNode;
  icon?: ReactNode; accent?: string }) {
  return (
    <div className="card panel-glow relative overflow-hidden p-4" style={accent ? ({ "--glow": `${accent}88` } as any) : undefined}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted">{label}</p>
          <p className="mt-1.5 text-2xl font-semibold tabular-nums tracking-tight text-ink">{value}</p>
          {hint && <p className="mt-0.5 text-xs text-muted">{hint}</p>}
        </div>
        {icon && <div className="rounded-lg bg-brand-500/10 p-2 text-brand-500 dark:text-brand-300">{icon}</div>}
      </div>
    </div>
  );
}

export function SectionTitle({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3">
      <h2 className="text-sm font-semibold uppercase tracking-[0.08em] text-muted">{children}</h2>{action}
    </div>
  );
}
