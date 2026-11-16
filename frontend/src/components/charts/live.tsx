// Live comparison charts (SVG): family time bars, per-model execution timeline (Gantt), success/failure donut.
// Every value drawn is a measured value passed in; nothing is simulated.
import { useEffect, useRef, useState } from "react";
import { FAMILY_COLOR, FAMILY_LABEL, fmtMsShort, type Family } from "../../lib/compare";

function useWidth(fallback = 600): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((e) => setW(Math.max(260, Math.floor(e[0].contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Horizontal bars, one per family; `running` families show an animated bar with elapsed time. */
export function FamilyTimeBars({ rows, title, metric }: { title: string; metric: string;
  rows: { family: Family; value: number | null; running?: boolean; highlight?: "fastest" | "slowest" }[] }) {
  const max = Math.max(1, ...rows.map((r) => r.value ?? 0));
  return (
    <div role="img" aria-label={title} className="space-y-2.5">
      {rows.map((r) => (
        <div key={r.family} className="grid grid-cols-[110px_1fr_84px] items-center gap-3 text-sm">
          <span className="truncate font-medium text-ink">{FAMILY_LABEL[r.family]}</span>
          <div className="relative h-6 overflow-hidden rounded-md bg-white/[0.04] ring-1 ring-line">
            {r.value !== null && (
              <div className={r.running ? "bar-running h-full rounded-md" : "h-full rounded-md transition-all duration-500"}
                style={{ width: `${Math.max(1.5, (100 * r.value) / max)}%`, background: r.running ? undefined :
                  `linear-gradient(90deg, ${FAMILY_COLOR[r.family]}55, ${FAMILY_COLOR[r.family]})` }} />
            )}
            {r.highlight && (
              <span className={`absolute right-1.5 top-1/2 -translate-y-1/2 rounded px-1.5 text-[10px] font-semibold uppercase tracking-wide ${
                r.highlight === "fastest" ? "bg-emerald-500/20 text-emerald-300" : "bg-red-500/15 text-red-300"}`}>{r.highlight}</span>
            )}
          </div>
          <span className="text-right font-mono text-xs tabular-nums text-ink">{r.value === null ? "—" : fmtMsShort(r.value)}</span>
        </div>
      ))}
      <p className="text-[11px] text-muted">{metric}</p>
    </div>
  );
}

export interface TimelineRow { id: string; family: Family; start: number | null; end: number | null; status: string;
  label?: string }

/** Execution timeline: one row per executed model, x = ms since the family pipelines started. */
export function ExecutionTimeline({ rows, now, title }: { rows: TimelineRow[]; now?: number; title: string }) {
  const [ref, width] = useWidth();
  const rowH = 15, labelW = Math.min(230, Math.max(140, width * 0.28)), top = 22;
  const end = Math.max(1, now ?? 0, ...rows.map((r) => r.end ?? r.start ?? 0));
  const x = (v: number) => labelW + ((width - labelW - 12) * v) / end;
  const h = top + rows.length * rowH + 8;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * end);
  return (
    <div ref={ref} className="w-full overflow-hidden">
      <svg width={width} height={h} role="img" aria-label={title} className="text-muted">
        <title>{title}</title>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 4} y2={h - 4} stroke="currentColor" strokeOpacity={0.12} />
            <text x={x(t)} y={12} fontSize={10} textAnchor="middle" fill="currentColor">{fmtMsShort(t)}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const y = top + i * rowH;
          const s = r.start ?? 0, e = r.end ?? (r.status === "running" ? now ?? s : s);
          return (
            <g key={r.id}>
              <text x={labelW - 6} y={y + 10} fontSize={10} textAnchor="end" className="fill-ink">
                {r.id.length > 30 ? r.id.slice(0, 29) + "…" : r.id}</text>
              {r.start !== null && (
                <rect x={x(s)} y={y + 2} height={rowH - 5} rx={2} width={Math.max(2, x(e) - x(s))}
                  fill={r.status === "error" ? "#f87171" : FAMILY_COLOR[r.family]}
                  className={r.status === "running" ? "animate-pulse" : undefined} opacity={r.status === "running" ? 0.6 : 0.9}>
                  <title>{`${r.id} (${FAMILY_LABEL[r.family]}): ${fmtMsShort(s)} → ${fmtMsShort(e)}${r.label ? ` · ${r.label}` : ""}`}</title>
                </rect>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Donut of run outcomes (completed / skipped / failed). */
export function OutcomeDonut({ parts, title, size = 150 }: { title: string; size?: number;
  parts: { label: string; value: number; color: string }[] }) {
  const total = parts.reduce((a, p) => a + p.value, 0);
  const r = size / 2 - 10, c = size / 2, sw = 16;
  let acc = 0;
  return (
    <div className="flex items-center gap-4">
      <svg width={size} height={size} role="img" aria-label={title}>
        <title>{title}</title>
        <circle cx={c} cy={c} r={r} fill="none" stroke="currentColor" strokeOpacity={0.08} strokeWidth={sw} />
        {total > 0 && parts.filter((p) => p.value > 0).map((p) => {
          const frac = p.value / total;
          const a0 = acc * 2 * Math.PI - Math.PI / 2, a1 = (acc + frac) * 2 * Math.PI - Math.PI / 2;
          acc += frac;
          const large = frac > 0.5 ? 1 : 0;
          const d = frac >= 0.9999 ? `M ${c} ${c - r} A ${r} ${r} 0 1 1 ${c - 0.01} ${c - r}`
            : `M ${c + r * Math.cos(a0)} ${c + r * Math.sin(a0)} A ${r} ${r} 0 ${large} 1 ${c + r * Math.cos(a1)} ${c + r * Math.sin(a1)}`;
          return <path key={p.label} d={d} fill="none" stroke={p.color} strokeWidth={sw}><title>{`${p.label}: ${p.value}`}</title></path>;
        })}
        <text x={c} y={c - 2} textAnchor="middle" fontSize={22} fontWeight={600} className="fill-ink">{total}</text>
        <text x={c} y={c + 16} textAnchor="middle" fontSize={10} className="fill-muted">models</text>
      </svg>
      <ul className="space-y-1.5 text-sm">
        {parts.map((p) => (
          <li key={p.label} className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: p.color }} />
            <span className="text-muted">{p.label}</span><span className="ml-auto pl-3 font-mono tabular-nums text-ink">{p.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Small probability bar with the model's threshold and uncertainty band. */
export function ProbabilityBar({ p, thr, band = 0.1 }: { p: number; thr?: number | null; band?: number }) {
  const col = thr == null ? "#818cf8" : p >= thr + band ? "#fb7185" : p < thr - band ? "#2dd4bf" : "#fbbf24";
  return (
    <div className="relative h-1.5 w-full rounded-full bg-white/[0.06] ring-1 ring-line" aria-hidden>
      {thr != null && <div className="absolute inset-y-0 rounded bg-amber-400/15"
        style={{ left: `${Math.max(0, thr - band) * 100}%`, width: `${Math.min(1, band * 2) * 100}%` }} />}
      <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${p * 100}%`, background: col }} />
      {thr != null && <div className="absolute -top-0.5 h-2.5 w-px bg-white/70" style={{ left: `${thr * 100}%` }} />}
    </div>
  );
}
