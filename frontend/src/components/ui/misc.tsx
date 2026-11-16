import { cva, type VariantProps } from "class-variance-authority";
import { useEffect, useRef, type ReactNode } from "react";
import { LuCircleAlert, LuInfo, LuTriangleAlert, LuX, LuCircleCheck, LuInbox } from "react-icons/lu";
import { cn } from "../../lib/utils";

export const badgeVariants = cva("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium whitespace-nowrap", {
  variants: {
    tone: {
      neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
      success: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-600/15 dark:bg-emerald-500/10 dark:text-emerald-300",
      danger: "bg-red-50 text-red-700 ring-1 ring-red-600/15 dark:bg-red-500/10 dark:text-red-300",
      warning: "bg-amber-50 text-amber-800 ring-1 ring-amber-600/20 dark:bg-amber-500/10 dark:text-amber-300",
      info: "bg-sky-50 text-sky-700 ring-1 ring-sky-600/15 dark:bg-sky-500/10 dark:text-sky-300",
      brand: "bg-brand-50 text-brand-700 ring-1 ring-brand-600/20 dark:bg-brand-500/10 dark:text-brand-300",
      navy: "bg-navy-50 text-navy-700 ring-1 ring-navy-600/15 dark:bg-navy-400/10 dark:text-navy-200",
    },
  },
  defaultVariants: { tone: "neutral" },
});

export function Badge({ tone, className, children }: VariantProps<typeof badgeVariants> & { className?: string;
  children: ReactNode }) {
  return <span className={cn(badgeVariants({ tone }), className)}>{children}</span>;
}

const alertTone = {
  info: ["border-sky-200 bg-sky-50 text-sky-900 dark:border-sky-900 dark:bg-sky-950/40 dark:text-sky-100", LuInfo],
  warning: ["border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-100", LuTriangleAlert],
  danger: ["border-red-200 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950/40 dark:text-red-100", LuCircleAlert],
  success: ["border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-100", LuCircleCheck],
} as const;

export function Alert({ tone = "info", title, children, className }: { tone?: keyof typeof alertTone;
  title?: ReactNode; children?: ReactNode; className?: string }) {
  const [cls, Icon] = alertTone[tone];
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={cn("flex gap-3 rounded-xl border px-4 py-3 text-sm", cls, className)}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="min-w-0">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={cn(title && "mt-0.5", "leading-relaxed")}>{children}</div>}
      </div>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-slate-200/70 dark:bg-slate-700/40", className)} aria-hidden />;
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string;
  children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      <div className="mb-3 rounded-full bg-surface p-3 text-muted">{icon || <LuInbox className="h-6 w-6" />}</div>
      <p className="font-medium text-ink">{title}</p>
      {children && <div className="mt-1 max-w-md text-sm text-muted">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Progress({ value, label }: { value: number | null; label?: string }) {
  return (
    <div className="w-full" role="progressbar" aria-label={label || "progress"} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={value === null ? undefined : Math.round(value * 100)}>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700/60">
        <div className={cn("h-full rounded-full bg-brand-500 transition-all", value === null && "w-1/3 animate-pulse")}
          style={value === null ? undefined : { width: `${Math.max(2, value * 100)}%` }} />
      </div>
    </div>
  );
}

export function Stat({ label, value, hint, icon }: { label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="card flex items-start gap-3 p-4">
      {icon && <div className="rounded-lg bg-brand-50 p-2 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300">{icon}</div>}
      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">{label}</p>
        <p className="mt-1 text-2xl font-semibold tabular-nums text-ink">{value}</p>
        {hint && <p className="mt-0.5 text-xs text-muted">{hint}</p>}
      </div>
    </div>
  );
}

export function Dialog({ open, onClose, title, children, footer }: { open: boolean; onClose: () => void;
  title: string; children: ReactNode; footer?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("keydown", onKey); prev?.focus(); };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-navy-950/40 p-4 backdrop-blur-[2px] sm:items-center"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={title}
        className="card animate-fade-in w-full max-w-lg outline-none">
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <h2 className="font-semibold">{title}</h2>
          <button onClick={onClose} className="rounded-md p-1 text-muted hover:bg-black/5" aria-label="Close"><LuX /></button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto px-5 py-4">{children}</div>
        {footer && <div className="flex justify-end gap-2 border-t border-line px-5 py-3">{footer}</div>}
      </div>
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string }[]; value: T;
  onChange: (t: T) => void }) {
  return (
    <div role="tablist" className="flex gap-1 overflow-x-auto rounded-lg bg-surface p-1">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          className={cn("rounded-md px-3 py-1.5 text-sm font-medium whitespace-nowrap transition-colors",
            value === t.id ? "bg-panel text-ink shadow-sm" : "text-muted hover:text-ink")}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Table({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full min-w-max border-collapse text-sm">{children}</table>
    </div>
  );
}

export const th = "border-b border-line px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted";
export const td = "border-b border-line/70 px-3 py-2.5 align-top";

export function Pagination({ page, pages, total, onPage }: { page: number; pages: number; total: number;
  onPage: (p: number) => void }) {
  return (
    <div className="flex items-center justify-between gap-3 px-1 py-3 text-sm text-muted">
      <span>{total} result{total === 1 ? "" : "s"}</span>
      <div className="flex items-center gap-2">
        <button className="rounded-md border border-line px-2.5 py-1 disabled:opacity-40" disabled={page <= 1}
          onClick={() => onPage(page - 1)}>Previous</button>
        <span className="tabular-nums">Page {page} of {pages}</span>
        <button className="rounded-md border border-line px-2.5 py-1 disabled:opacity-40" disabled={page >= pages}
          onClick={() => onPage(page + 1)}>Next</button>
      </div>
    </div>
  );
}
