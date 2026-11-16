import { useEffect, useState, type ReactNode } from "react";
import { LuLayoutDashboard, LuUpload, LuActivity, LuListTree, LuGauge, LuBoxes, LuDatabase, LuHistory, LuFileText,
  LuSettings, LuLogOut, LuMenu, LuX, LuMoon, LuSun, LuMonitor, LuCpu, LuWifi, LuWifiOff, LuServerCog } from "react-icons/lu";
import { API_BASE } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import { compareApi } from "../../lib/compare";
import { Link, navigate, useLocation } from "../../lib/router";
import { useTheme } from "../../lib/theme";
import { cn } from "../../lib/utils";
import { Logo } from "./Logo";

const NAV = [
  { group: "Workflow", items: [
    { to: "/app", label: "Overview", icon: LuLayoutDashboard, exact: true },
    { to: "/app/upload", label: "Image Upload", icon: LuUpload },
    { to: "/app/live", label: "Live Comparison", icon: LuActivity, highlight: true },
    { to: "/app/runs", label: "Model Results", icon: LuListTree },
  ] },
  { group: "Research", items: [
    { to: "/app/benchmarks", label: "Benchmarking", icon: LuGauge },
    { to: "/app/models", label: "Model Registry", icon: LuBoxes },
    { to: "/app/datasets", label: "Datasets", icon: LuDatabase },
    { to: "/app/history", label: "Execution History", icon: LuHistory },
    { to: "/app/reports", label: "Research Reports", icon: LuFileText },
  ] },
  { group: "System", items: [{ to: "/app/system", label: "System Monitor", icon: LuServerCog },
    { to: "/app/settings", label: "System Settings", icon: LuSettings }] },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const next = theme === "light" ? "dark" : theme === "dark" ? "system" : "light";
  const Icon = theme === "light" ? LuSun : theme === "dark" ? LuMoon : LuMonitor;
  return (
    <button onClick={() => setTheme(next)} className="rounded-lg p-2 text-muted hover:bg-black/5 hover:text-ink dark:hover:bg-white/5"
      aria-label={`Theme: ${theme}. Switch to ${next}`} title={`Theme: ${theme}`}>
      <Icon className="h-[18px] w-[18px]" />
    </button>
  );
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  const loc = useLocation().split("?")[0];
  return (
    <nav aria-label="Main" className="flex flex-col gap-5">
      {NAV.map((g) => (
        <div key={g.group}>
          <p className="mb-1.5 px-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">{g.group}</p>
          <div className="flex flex-col gap-0.5">
            {g.items.map(({ to, label, icon: Icon, exact, highlight }: any) => {
              const active = exact ? loc === to : loc === to || loc.startsWith(to + "/");
              return (
                <Link key={to} to={to} onClick={onNavigate} aria-current={active ? "page" : undefined}
                  className={cn("group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all",
                    active ? "bg-gradient-to-r from-brand-500/20 to-transparent text-white"
                      : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-100")}>
                  {active && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-brand-400" />}
                  <Icon className={cn("h-[18px] w-[18px] shrink-0", active ? "text-brand-300" : "")} aria-hidden />
                  {label}
                  {highlight && <span className="ml-auto rounded bg-brand-500/20 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-brand-200">Live</span>}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </nav>
  );
}

/** Backend connection, engine state and model availability, refreshed every 15 s. */
function useSystemStatus() {
  const [st, setSt] = useState<{ connected: boolean | null; engine?: string; servable?: number; total?: number }>({ connected: null });
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const h = await fetch(API_BASE + "/health");
        if (!h.ok) throw new Error();
        const s = await compareApi.status();
        if (alive) setSt({ connected: true, engine: s.engine?.state, servable: s.engine?.models_servable, total: s.engine?.models_total });
      } catch {
        if (alive) setSt((p) => ({ ...p, connected: false }));
      }
    };
    tick();
    const id = setInterval(tick, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return st;
}

export function AppShell({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const sys = useSystemStatus();
  useEffect(() => setOpen(false), [loc]);
  const sidebar = (
    <div className="flex h-full flex-col border-r border-white/5 bg-[#060a14] px-3 py-4">
      <Link to="/app" className="mb-7 flex items-center gap-2.5 px-2">
        <Logo className="h-9 w-9" />
        <div className="leading-tight">
          <p className="text-sm font-semibold text-white">Liver Tumour Intelligence</p>
          <p className="text-[11px] text-slate-400">Research &amp; Model Benchmarking · V11</p>
        </div>
      </Link>
      <NavItems onNavigate={() => setOpen(false)} />
      <div className="mt-auto space-y-2 rounded-xl border border-white/5 bg-white/[0.02] p-3 text-[11px] leading-relaxed text-slate-400">
        <p className="flex items-center gap-1.5 font-medium text-slate-300"><LuCpu className="h-3.5 w-3.5" /> CPU inference · no GPU</p>
        <p>Quantum models run on an exact classical simulator — no quantum hardware.</p>
        <p className="text-slate-500">Research use only — not a medical device.</p>
      </div>
    </div>
  );
  const engineTone = sys.engine === "ready" ? "bg-emerald-400" : sys.engine === "failed" ? "bg-red-400" : "bg-amber-400";
  return (
    <div className="min-h-screen bg-surface">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-panel focus:px-3 focus:py-2">Skip to content</a>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 lg:block">{sidebar}</aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-72 animate-fade-in">{sidebar}</div>
        </div>
      )}
      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-2 border-b border-line bg-panel/70 px-4 backdrop-blur-xl sm:px-6">
          <button className="rounded-lg p-2 text-muted hover:bg-black/5 lg:hidden" onClick={() => setOpen(!open)}
            aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open}>
            {open ? <LuX className="h-5 w-5" /> : <LuMenu className="h-5 w-5" />}
          </button>
          <div className="hidden items-center gap-2 md:flex" aria-live="polite">
            <span className={cn("inline-flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs",
              sys.connected === false ? "text-red-400" : "text-muted")} title="Backend connection">
              {sys.connected === false ? <LuWifiOff className="h-3.5 w-3.5" /> : <LuWifi className="h-3.5 w-3.5 text-emerald-400" />}
              {sys.connected === null ? "Connecting…" : sys.connected ? "Backend connected" : "Backend unreachable"}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-muted" title="Model engine">
              <span className={cn("h-1.5 w-1.5 rounded-full", engineTone)} /> Engine {sys.engine || "—"}
            </span>
            {sys.total !== undefined && (
              <Link to="/app/models" className="inline-flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-muted hover:text-ink">
                <LuBoxes className="h-3.5 w-3.5" /> {sys.servable} / {sys.total} models servable
              </Link>
            )}
          </div>
          <div className="flex-1" />
          <ThemeToggle />
          <div className="hidden text-right text-xs leading-tight sm:block">
            <p className="font-medium text-ink">{user?.full_name}</p>
            <p className="text-muted">{user?.email} · {user?.role}</p>
          </div>
          <button onClick={async () => { await logout(); navigate("/login"); }} aria-label="Sign out" title="Sign out"
            className="ml-1 inline-flex items-center gap-1.5 rounded-lg px-2.5 py-2 text-sm text-muted hover:bg-black/5 hover:text-ink dark:hover:bg-white/5">
            <LuLogOut className="h-4 w-4" /> <span className="hidden sm:inline">Sign out</span>
          </button>
        </header>
        <main id="main" className="mx-auto w-full max-w-[1500px] px-4 py-6 sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({ title, description, actions, eyebrow }: { title: string; description?: ReactNode;
  actions?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-brand-500 dark:text-brand-300">{eyebrow}</p>}
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}
