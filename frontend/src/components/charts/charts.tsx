// SVG charts built on d3-scale / d3-shape (the role Recharts plays): responsive width, accessible titles and data
// tables for screen readers, light and dark themes. Every chart renders only data it is given — no placeholders.
import { scaleBand, scaleLinear } from "d3-scale";
import { line as d3line } from "d3-shape";
import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";

export const PALETTE = ["#14a08e", "#305c89", "#e0a526", "#c2410c", "#7c3aed", "#0ea5e9", "#64748b", "#db2777"];

function useWidth<T extends HTMLElement>(fallback = 600): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((e) => setW(Math.max(240, Math.floor(e[0].contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export interface BarDatum { label: string; value: number | null; color?: string; note?: string }

export function BarChart({ data, height = 240, domain, format = (v) => String(v), title, horizontal = false,
  threshold }: { data: BarDatum[]; height?: number; domain?: [number, number]; format?: (v: number) => string;
  title: string; horizontal?: boolean; threshold?: { value: number; label: string } }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const vals = data.map((d) => d.value ?? 0);
  const dom: [number, number] = domain || [Math.min(0, ...vals), Math.max(1e-9, ...vals) * 1.08];
  if (horizontal) {
    const rowH = 26;
    const h = Math.max(60, data.length * rowH + 30);
    const labelW = Math.min(260, Math.max(96, ...data.map((d) => Math.min(d.label.length, 34) * 7.1 + 14)));
    const x = scaleLinear().domain(dom).range([labelW, width - 44]).clamp(true);
    // bars grow from 0, or from the left edge of the axis when the domain does not include 0 (e.g. AUC 0.5–1)
    const base = Math.min(Math.max(0, dom[0]), dom[1]);
    return (
      <div ref={ref} className="w-full">
        <svg width={width} height={h} role="img" aria-label={title} className="text-muted">
          <title>{title}</title>
          {x.ticks(5).map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={4} y2={h - 22} stroke="currentColor" strokeOpacity={0.15} />
              <text x={x(t)} y={h - 6} textAnchor="middle" fontSize={11} fill="currentColor">{format(t)}</text>
            </g>
          ))}
          {threshold && <line x1={x(threshold.value)} x2={x(threshold.value)} y1={0} y2={h - 22} stroke="#c2410c"
            strokeDasharray="4 3"><title>{threshold.label}</title></line>}
          {data.map((d, i) => (
            <g key={d.label} transform={`translate(0, ${8 + i * rowH})`}>
              <text x={labelW - 8} y={13} textAnchor="end" fontSize={12} className="fill-ink">{d.label.length > 34 ? d.label.slice(0, 33) + "…" : d.label}</text>
              {d.value !== null && (
                <>
                  <rect x={x(Math.min(base, d.value))} y={2} height={rowH - 10} rx={3}
                    width={Math.max(1, Math.abs(x(d.value) - x(base)))} fill={d.color || PALETTE[0]}>
                    <title>{`${d.label}: ${format(d.value)}${d.note ? ` (${d.note})` : ""}`}</title>
                  </rect>
                  <text x={x(Math.max(base, d.value)) + 5} y={14} fontSize={11} fill="currentColor">{format(d.value)}</text>
                </>
              )}
              {d.value === null && <text x={labelW + 4} y={14} fontSize={11} fill="currentColor">n/a</text>}
            </g>
          ))}
        </svg>
        <DataTable title={title} rows={data.map((d) => [d.label, d.value === null ? "n/a" : format(d.value)])} />
      </div>
    );
  }
  const m = { t: 10, r: 10, b: 46, l: 44 };
  const x = scaleBand().domain(data.map((d) => d.label)).range([m.l, width - m.r]).padding(0.25);
  const y = scaleLinear().domain(dom).range([height - m.b, m.t]).clamp(true);
  const base = Math.min(Math.max(0, dom[0]), dom[1]);
  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={height} role="img" aria-label={title} className="text-muted">
        <title>{title}</title>
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={0.15} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="currentColor">{format(t)}</text>
          </g>
        ))}
        {data.map((d, i) => d.value !== null && (
          <rect key={d.label} x={x(d.label)} width={x.bandwidth()} y={y(Math.max(base, d.value))} rx={3}
            height={Math.max(1, Math.abs(y(d.value) - y(base)))} fill={d.color || PALETTE[i % PALETTE.length]}>
            <title>{`${d.label}: ${format(d.value)}`}</title>
          </rect>
        ))}
        {data.map((d) => (
          <text key={d.label} x={(x(d.label) || 0) + x.bandwidth() / 2} y={height - m.b + 16} textAnchor="middle"
            fontSize={11} fill="currentColor">{d.label.length > 14 ? d.label.slice(0, 13) + "…" : d.label}</text>
        ))}
      </svg>
      <DataTable title={title} rows={data.map((d) => [d.label, d.value === null ? "n/a" : format(d.value)])} />
    </div>
  );
}

export interface Series { name: string; points: { x: number; y: number | null }[]; color?: string }

export function LineChart({ series, height = 260, xLabel, yLabel, xDomain, yDomain, title, xFormat = (v) => String(v),
  yFormat = (v) => String(v) }: { series: Series[]; height?: number; xLabel?: string; yLabel?: string;
  xDomain?: [number, number]; yDomain?: [number, number]; title: string; xFormat?: (v: number) => string;
  yFormat?: (v: number) => string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const m = { t: 12, r: 14, b: 42, l: 52 };
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.map((p) => p.y).filter((v): v is number => v !== null));
  const x = scaleLinear().domain(xDomain || [Math.min(...xs), Math.max(...xs)]).range([m.l, width - m.r]);
  const y = scaleLinear().domain(yDomain || [Math.min(...ys), Math.max(...ys)]).nice().range([height - m.b, m.t]);
  const path = d3line<{ x: number; y: number | null }>().defined((p) => p.y !== null).x((p) => x(p.x)).y((p) => y(p.y as number));
  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={height} role="img" aria-label={title} className="text-muted">
        <title>{title}</title>
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={0.15} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="currentColor">{yFormat(t)}</text>
          </g>
        ))}
        {x.ticks(6).map((t) => (
          <text key={t} x={x(t)} y={height - m.b + 16} textAnchor="middle" fontSize={11} fill="currentColor">{xFormat(t)}</text>
        ))}
        {xLabel && <text x={(m.l + width - m.r) / 2} y={height - 6} textAnchor="middle" fontSize={11} fill="currentColor">{xLabel}</text>}
        {yLabel && <text transform={`translate(12, ${(height - m.b + m.t) / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill="currentColor">{yLabel}</text>}
        {series.map((s, i) => (
          <g key={s.name}>
            <path d={path(s.points) || ""} fill="none" stroke={s.color || PALETTE[i % PALETTE.length]} strokeWidth={2} />
            {s.points.map((p, j) => p.y !== null && (
              <circle key={j} cx={x(p.x)} cy={y(p.y)} r={3.2} fill={s.color || PALETTE[i % PALETTE.length]}>
                <title>{`${s.name}: ${xFormat(p.x)} → ${yFormat(p.y)}`}</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <Legend items={series.map((s, i) => ({ name: s.name, color: s.color || PALETTE[i % PALETTE.length] }))} />
    </div>
  );
}

export function Legend({ items }: { items: { name: string; color: string }[] }) {
  if (items.length < 2) return null;
  return (
    <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
      {items.map((i) => (
        <li key={i.name} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: i.color }} />{i.name}</li>
      ))}
    </ul>
  );
}

function DataTable({ title, rows }: { title: string; rows: ReactNode[][] }) {
  return (
    <table className="sr-only">
      <caption>{title}</caption>
      <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
    </table>
  );
}
