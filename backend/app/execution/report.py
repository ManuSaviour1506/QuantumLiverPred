"""PDF research report of one four-paradigm comparison run (reportlab). Not a clinical report."""
from __future__ import annotations

import base64
import io

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

from app.services.reports import _styles, _table

FAM = {"classical": "Classical ML", "deep_learning": "Deep Learning", "quantum": "Quantum ML", "hybrid": "Hybrid ML"}
FAM_COL = {"classical": "#3b82f6", "deep_learning": "#8b5cf6", "quantum": "#06b6d4", "hybrid": "#f59e0b"}
SHORT = {"Tumour detected": "Tumour", "No tumour detected": "No tumour", "Uncertain / unable to classify": "Uncertain"}


def _f(v, d=1):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))


def _bar(fams: dict) -> Drawing:
    items = [(f, s.get("inference_ms_mean")) for f, s in fams.items() if s.get("inference_ms_mean") is not None]
    d = Drawing(170 * mm, 45 * mm)
    if not items:
        d.add(String(0, 20, "no completed models", fontSize=8))
        return d
    c = HorizontalBarChart()
    c.x, c.y, c.width, c.height = 30 * mm, 5 * mm, 130 * mm, 36 * mm
    c.data = [[v for _, v in items]]
    c.categoryAxis.categoryNames = [FAM[f] for f, _ in items]
    c.categoryAxis.labels.fontSize = 7
    c.valueAxis.labels.fontSize = 7
    c.valueAxis.valueMin = 0
    c.bars[0].fillColor = colors.HexColor("#6366f1")
    for i, (f, _) in enumerate(items):
        c.bars[(0, i)].fillColor = colors.HexColor(FAM_COL[f])
    d.add(c)
    d.add(String(30 * mm, 42 * mm, "Mean per-model inference time (ms)", fontSize=7.5))
    return d


def comparison_pdf(run: dict, upload: dict, disclaimer: str) -> bytes:
    S = _styles()
    res = run["result"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=f"Comparison run {run['id']}")
    st = [Paragraph("Four-paradigm model comparison — research report", S["title"]),
          Paragraph(f"Run <b>{run['id']}</b> · {run['created_at']} · mode {run['mode']} · status {run['status']}",
                    S["small"]), Spacer(1, 4), Paragraph(disclaimer, S["warn"]), Spacer(1, 6)]
    pre = res.get("preprocessing") or {}
    an = upload.get("analysis") or {}
    img = an.get("preview_png_b64")
    info = [["Input", upload.get("input_name") or upload["id"]], ["Modality", upload.get("modality")],
            ["Source", upload.get("source") + (f" ({upload.get('sample_id')})" if upload.get("sample_id") else "")],
            ["Context slices", ", ".join(upload.get("context_slices") or []) or "none"],
            ["Quality control", (pre.get("quality") or {}).get("status")],
            ["Out-of-distribution", f"{(pre.get('ood') or {}).get('status')} (d2 ratio {(pre.get('ood') or {}).get('d2_ratio')})"],
            ["Liver gate", f"{(pre.get('liver_gate') or {}).get('n_liver_patches')} of "
                           f"{(pre.get('liver_gate') or {}).get('n_candidates')} patches"],
            ["U-Net liver check", f"{(pre.get('liver_check') or {}).get('status')} "
                                  f"({(pre.get('liver_check') or {}).get('unet_liver_px')} px)"],
            ["Stopped", pre.get("stopped") or "no"],
            ["Scheduling", (res.get("scheduling") or {}).get("description")],
            ["Hardware", f"{(res.get('hardware') or {}).get('cpu_count')} CPU · {(res.get('hardware') or {}).get('machine')} · "
                         "no GPU · quantum = CPU simulation"]]
    t = _table([[Paragraph(str(a), S["body"]), Paragraph(str(b), S["body"])] for a, b in info], [38 * mm, 92 * mm],
               header=False)
    if img:
        st.append(Table([[t, Image(io.BytesIO(base64.b64decode(img)), 48 * mm, 48 * mm)]], colWidths=[132 * mm, 50 * mm]))
    else:
        st.append(t)
    fams = res.get("families") or {}
    rk = res.get("ranking") or {}
    ag = res.get("agreement") or {}
    st += [Paragraph("Family summary", S["h2"]),
           _table([["Family", "Completed / attempted", "Failed", "Mean inference ms", "Family wall ms", "Fastest model",
                    "Majority call"]] +
                  [[FAM[f], f"{s.get('n_completed')}/{s.get('n_attempted')}", s.get("n_failed"),
                    _f(s.get("inference_ms_mean"), 2), _f(s.get("wall_ms")), s.get("fastest_model") or "—",
                    SHORT.get(s.get("combined_label"), "—")] for f, s in fams.items()],
                  [28 * mm, 30 * mm, 14 * mm, 26 * mm, 24 * mm, 40 * mm, 22 * mm]),
           Spacer(1, 4), _bar(fams),
           Paragraph(f"Fastest family: <b>{FAM.get(rk.get('fastest_family'), '—')}</b> · slowest: "
                     f"<b>{FAM.get(rk.get('slowest_family'), '—')}</b>. {rk.get('rule', '')}", S["body"]),
           Paragraph(f"Agreement: {ag.get('tumour')} tumour, {ag.get('no_tumour')} no tumour, {ag.get('uncertain')} "
                     f"uncertain, {ag.get('failed_or_skipped')} skipped / failed; agreement "
                     f"{_f(ag.get('agreement_pct'))} % of completed models. {ag.get('note', '')}", S["body"]),
           Paragraph(f"Quantum execution: {res.get('quantum_execution')}", S["small"]),
           PageBreak(), Paragraph("Per-model results", S["h2"])]
    rows = [["Model", "Family", "Status", "Call", "p", "thr", "Margin", "Feat. ms", "Infer. ms", "Cold"]]
    for m in res.get("models", []):
        tm = m.get("timings_ms") or {}
        rows.append([Paragraph(m["model_id"], S["small"]), FAM.get(m["family"], m["family"]), m["status"],
                     SHORT.get(m.get("label"), "—") if m["status"] == "ok" else "—", _f(m.get("probability"), 3),
                     _f(m.get("threshold"), 2), _f(m.get("confidence"), 2), _f(tm.get("features")),
                     _f(tm.get("inference"), 2), "yes" if m.get("cold_start") else ("no" if m["status"] == "ok" else "")])
    st.append(_table(rows, [46 * mm, 21 * mm, 20 * mm, 16 * mm, 11 * mm, 10 * mm, 12 * mm, 14 * mm, 15 * mm, 10 * mm]))
    skipped = [m for m in res.get("models", []) if m["status"] != "ok"]
    if skipped:
        st += [Paragraph("Models not executed or failed, with the reason", S["h2"]),
               _table([["Model", "Status", "Reason"]] +
                      [[Paragraph(m["model_id"], S["small"]), m["status"], Paragraph(str(m.get("reason") or ""), S["small"])]
                       for m in skipped], [50 * mm, 25 * mm, 105 * mm])]
    st += [Spacer(1, 6), Paragraph("Margin = distance of p from the model's own threshold scaled to [0, 1]; it is not a "
                                   "probability of being correct. p values of different models come from different "
                                   "calibration methods (V9: Platt on V9 validation scans; V10: Platt on inner "
                                   "out-of-fold predictions) and are not directly comparable. A single image cannot "
                                   "establish accuracy; see the benchmarking section of the platform.", S["small"])]
    doc.build(st)
    return buf.getvalue()
