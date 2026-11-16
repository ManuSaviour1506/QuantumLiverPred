"""PDF reports (reportlab): prediction report, model comparison report and experiment report.

Every report carries the research disclaimer and states the model's limitations; none is a clinical diagnosis.
"""
from __future__ import annotations

import io
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import DISCLAIMER, VERSION

NAVY = colors.HexColor("#0f2a44")
TEAL = colors.HexColor("#0d9488")
GREY = colors.HexColor("#64748b")


def _styles():
    ss = getSampleStyleSheet()
    return {"title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=17, spaceAfter=4),
            "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12, spaceBefore=8, spaceAfter=4),
            "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12),
            "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.5, leading=10, textColor=GREY),
            "warn": ParagraphStyle("w", parent=ss["BodyText"], fontSize=8.5, leading=11,
                                   textColor=colors.HexColor("#7c2d12"), backColor=colors.HexColor("#fff7ed"),
                                   borderColor=colors.HexColor("#fdba74"), borderWidth=0.6, borderPadding=6)}


def _table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                  ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5)]
    else:
        style += [("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8.5),
                  ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9"))]
    t.setStyle(TableStyle(style))
    return t


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(15 * mm, 10 * mm, f"Liver Tumour Intelligence Platform v{VERSION} — research use only, not a "
                                        f"diagnosis — page {doc.page}")
    canvas.restoreState()


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                             bottomMargin=16 * mm, title=title, author="Liver Tumour Intelligence Platform")


def prediction_report(p: dict, model: dict | None, images: dict[str, bytes]) -> bytes:
    s = _styles()
    buf = io.BytesIO()
    story = [Paragraph("Experimental prediction report", s["title"]),
             Paragraph("Research platform output — <b>not a clinical diagnosis</b>", s["body"]), Spacer(1, 4),
             Paragraph(DISCLAIMER, s["warn"]), Spacer(1, 6)]
    r = p.get("result") or {}
    rows = [["Prediction id", p["id"]], ["Model", f"{p.get('model_name') or p['model_id']} ({p['model_id']})"],
            ["Model version", _fmt(p.get("model_version"))], ["Execution mode", _fmt(p.get("execution_mode"))],
            ["Training dataset version", _fmt(p.get("dataset_version"))],
            ["Output", _fmt(p.get("label"))], ["Status / reason", f"{r.get('status')} {('— ' + r['reason']) if r.get('reason') else ''}"],
            ["Model probability (largest liver patch)", _fmt(p.get("probability"))],
            ["Operating threshold (band ± 0.10)", _fmt(p.get("threshold"))],
            ["Probability note", Paragraph(r.get("probability_note") or "—", s["body"])],
            ["Input", f"{_fmt(p.get('input_format'))}, {p.get('input_bytes') or 0} bytes, SHA-256 {p.get('input_sha256_prefix')}…"],
            ["Submitted / finished (UTC)", f"{p.get('created_at')} / {_fmt(p.get('finished_at'))}"],
            ["Processing time", f"{_fmt(p.get('processing_ms'), 1)} ms"]]
    story += [Paragraph("Result", s["h2"]), _table(rows, [62 * mm, 118 * mm], header=False)]
    d = r.get("details") or {}
    q = d.get("quality") or {}
    ood = d.get("ood") or {}
    gate = d.get("liver_gate") or {}
    story += [Paragraph("Input checks", s["h2"]), _table([
        ["Image quality", _fmt(q.get("status")) + ("" if not q.get("warnings") else f" (warnings: {', '.join(q['warnings'])})")],
        ["Out-of-distribution check", f"{_fmt(ood.get('status'))} (Mahalanobis ratio {_fmt(ood.get('d2_ratio'), 3)})"],
        ["Liver-tissue gate", f"{_fmt(gate.get('n_liver_patches'))} of {_fmt(gate.get('n_candidates'))} candidate patches"],
    ], [62 * mm, 118 * mm], header=False)]
    imgs = []
    for kind, label in (("preview", "Uploaded image (pre-processed preview)"),
                        ("patch_map", "Patch probabilities of this model (not a segmentation)"),
                        ("gradcam", "Grad-CAM (explanation, not localisation)")):
        if kind in images:
            imgs.append([Image(io.BytesIO(images[kind]), width=56 * mm, height=56 * mm), Paragraph(label, s["small"])])
    if imgs:
        story.append(Paragraph("Images", s["h2"]))
        story.append(_table([[c[0] for c in imgs], [c[1] for c in imgs]], [60 * mm] * len(imgs), header=False))
    if model:
        ev = ((model.get("metrics") or {}).get("evaluations") or [{}])[0]
        met = ev.get("metrics") or {}
        story += [Paragraph("Recorded model performance", s["h2"]),
                  Paragraph(f"{ev.get('name', '—')}; evaluation unit: {ev.get('evaluation_unit', '—')}.", s["body"]),
                  _table([["ROC-AUC", "Sensitivity", "Specificity", "F1", "ECE"],
                          [_fmt(met.get("auc", met.get("roc_auc", met.get("primary_mean_within_unit_auc")))),
                           _fmt(met.get("sensitivity")), _fmt(met.get("specificity")), _fmt(met.get("f1")),
                           _fmt(met.get("ece"))]])]
    story += [Paragraph("Limitations", s["h2"])]
    for t in ["The models discriminate tumour from tumour-free liver tissue in 96×96 patches; slice-level "
              "'No tumour detected' has not been validated (the training data contain no tumour-free liver slices).",
              "Training data: 1,745 LiTS-derived CT slices in reconstructed scan groups without patient identifiers; "
              "patient-level generalisation has not been established.",
              "Probabilities are model scores, not clinical confidence. Calibration was fitted on training data, but in the V10 "
              "nested evaluation of the radiomics models scores shifted between scan units (expected calibration error 0.10-0.23 on held-out "
              "units), so the value is not a reliable probability of tumour.",
              "Quantum models run on a classical NumPy simulator; no quantum hardware was used."]:
        story.append(Paragraph("• " + t, s["body"]))
    _doc(buf, "Prediction report").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def comparison_report(rows: list[dict], generated: str) -> bytes:
    s = _styles()
    buf = io.BytesIO()
    story = [Paragraph("Model comparison report", s["title"]), Paragraph(f"Generated {generated} UTC", s["small"]),
             Spacer(1, 4), Paragraph(DISCLAIMER, s["warn"]), Spacer(1, 6),
             Paragraph("Only models with saved evaluation results are listed. Results from different evaluations "
                       "(V9 fixed split vs V10 nested leave-one-unit-out) are not directly comparable.", s["body"])]
    head = ["Model", "Family", "Evaluation", "ROC-AUC", "Bal. acc.", "Sens.", "Spec.", "F1", "MCC"]
    body = [head]
    for r in rows:
        m = r.get("metrics") or {}
        body.append([Paragraph(r["name"], s["small"]), r["family"], Paragraph(r["evaluation"], s["small"]),
                     _fmt(r.get("roc_auc"), 3), _fmt(m.get("balanced_accuracy"), 3), _fmt(m.get("sensitivity"), 3),
                     _fmt(m.get("specificity"), 3), _fmt(m.get("f1"), 3), _fmt(m.get("mcc"), 3)])
    story.append(_table(body, [48 * mm, 18 * mm, 40 * mm, 14 * mm, 14 * mm, 12 * mm, 12 * mm, 11 * mm, 11 * mm]))
    _doc(buf, "Model comparison").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def experiment_report(exp: dict) -> bytes:
    s = _styles()
    buf = io.BytesIO()
    story = [Paragraph(f"Experiment report — {exp['name']}", s["title"]),
             Paragraph(f"{exp['kind']} · {exp['status']} · created {exp['created_at']}", s["small"]), Spacer(1, 4),
             Paragraph(DISCLAIMER, s["warn"]), Paragraph("Configuration", s["h2"]),
             Paragraph(json.dumps(exp.get("config") or {}, indent=1)[:3000].replace("\n", "<br/>"), s["small"]),
             Paragraph("Results", s["h2"])]
    res = exp.get("results") or {}
    summary = res.get("summary") if isinstance(res, dict) else None
    if isinstance(summary, list) and summary and isinstance(summary[0], dict):
        keys = list(summary[0].keys())[:7]
        story.append(_table([keys] + [[_fmt(r.get(k), 3) for k in keys] for r in summary[:40]]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(json.dumps(res, indent=1, default=str)[:6000].replace("\n", "<br/>"), s["small"]))
    if exp.get("error"):
        story += [Paragraph("Error", s["h2"]), Paragraph(exp["error"], s["body"])]
    _doc(buf, "Experiment report").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
