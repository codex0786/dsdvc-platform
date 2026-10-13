"""Executive deck (python-pptx) with native, editable charts.

Slides: title · executive summary · what drives the outcome · return by channel · recommended mix ·
expected impact · model accuracy · caveats & next experiments · appendix (data & method).
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from dsdvc import MODEL_VERSION

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
GREY = RGBColor(0xB5, 0xB4, 0xAE)
W, H = Inches(13.333), Inches(7.5)
FONT = "Arial"


def _apply_font(prs):
    """Set an explicit font on every run so decks render identically without the theme font installed."""
    for slide in prs.slides:
        for shp in slide.shapes:
            if shp.has_text_frame:
                for para in shp.text_frame.paragraphs:
                    for run in para.runs:
                        run.font.name = FONT
            if shp.has_table:
                for row in shp.table.rows:
                    for cell in row.cells:
                        for para in cell.text_frame.paragraphs:
                            for run in para.runs:
                                run.font.name = FONT
            if shp.has_chart:
                shp.chart.font.name = FONT


def _money(v: float, cur: str = "₹") -> str:
    """₹ amounts in lakh/crore (one consistent style for client decks); other currencies in K/M."""
    a = abs(v)
    if cur == "₹":
        s = f"{a / 1e7:.2f} Cr" if a >= 1e7 else (f"{a / 1e5:.2f} L" if a >= 1e4 else f"{a:,.0f}")
    else:
        s = f"{a / 1e6:.2f}M" if a >= 1e6 else (f"{a / 1e3:.1f}K" if a >= 1e4 else f"{a:,.0f}")
    return ("-" if v < 0 else "") + cur + s


ACRONYMS = {"ooh": "OOH", "tv": "TV", "yt": "YouTube", "youtube": "YouTube", "sem": "SEM", "ott": "OTT", "ctv": "CTV"}


def display_name(channel: str) -> str:
    c = channel.removeprefix("spend_")
    if c in ACRONYMS:
        return ACRONYMS[c]
    return " ".join(ACRONYMS.get(w, w.capitalize()) for w in c.split("_"))


def _slide(prs, title: str, kicker: str = ""):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), W - Inches(1.2), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(28), True, NAVY
    if kicker:
        p2 = tf.add_paragraph()
        p2.text = kicker
        p2.font.size, p2.font.color.rgb = Pt(14), MUTED
    return s


def _bullets(slide, items: list[str], left, top, width, height, size=18):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {it}"
        p.font.size, p.font.color.rgb = Pt(size), INK
        p.space_after = Pt(10)
    return tb


def _footer(slide, text: str):
    tb = slide.shapes.add_textbox(Inches(0.6), H - Inches(0.5), W - Inches(1.2), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size, p.font.color.rgb = Pt(10), MUTED


def _style_chart(chart, colors=(BLUE, ORANGE), legend=True):
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
    for i, s in enumerate(chart.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = colors[i % len(colors)]
    chart.font.size = Pt(12)


def _table(slide, df: pd.DataFrame, left, top, width, row_h=Inches(0.36), first_col_share: float = 0.34):
    rows, cols = len(df) + 1, len(df.columns)
    shp = slide.shapes.add_table(rows, cols, left, top, width, Emu(row_h * rows))
    tbl = shp.table
    first = int(width * first_col_share)
    tbl.columns[0].width = first
    for j in range(1, cols):
        tbl.columns[j].width = int((width - first) / (cols - 1))
    for j, c in enumerate(df.columns):
        cell = tbl.cell(0, j)
        cell.text = str(c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        para = cell.text_frame.paragraphs[0]
        para.font.color.rgb, para.font.bold, para.font.size = RGBColor(255, 255, 255), True, Pt(12)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(v)
            cell.text_frame.paragraphs[0].font.size = Pt(12)
            if j > 0:
                cell.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    return tbl


def executive_deck(*, client: str, plan, summary: pd.DataFrame, decomposition: pd.DataFrame, backtest, diagnostics: dict,
                   provenance: dict, horizon_weeks: int, outcome_label: str = "revenue", currency: str = "₹") -> bytes:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    t = plan.table.copy()
    chans = list(summary.channel)
    names = [display_name(c) for c in chans]
    foot = f"{client} · {MODEL_VERSION} · data {str(provenance.get('data_version', ''))[:10]} · {date.today().isoformat()}"

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, 0, 0, Inches(0.35), H)  # MSO_SHAPE.RECTANGLE
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.4), W - Inches(1.6), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = f"{client}"
    tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold, tf.paragraphs[0].font.color.rgb = Pt(40), True, NAVY
    p = tf.add_paragraph()
    p.text = "Marketing mix analysis & recommended budget plan"
    p.font.size, p.font.color.rgb = Pt(24), INK
    p = tf.add_paragraph()
    p.text = f"{date.today():%d %B %Y} · Estimates with 90% intervals · Recommendations require validation by experiment"
    p.font.size, p.font.color.rgb = Pt(14), MUTED

    # 2 executive summary
    movers = t.assign(delta=t.recommended_weekly - t.current_weekly).sort_values("delta")
    down, up = movers.iloc[0], movers.iloc[-1]
    best = summary.sort_values("mroi", ascending=False).iloc[0]
    s = _slide(prs, "Executive summary")
    bullets = [
        f"Shift budget toward {display_name(up.channel)} (+{_money(up.recommended_weekly - up.current_weekly, currency)}/week) and away from "
        f"{display_name(down.channel)} ({_money(down.recommended_weekly - down.current_weekly, currency)}/week), keeping the weekly budget at "
        f"{_money(plan.weekly_budget, currency)}.",
        f"Expected gain: {_money(plan.uplift, currency)} {outcome_label}/week (90% interval {_money(plan.uplift_low, currency)} "
        f"to {_money(plan.uplift_high, currency)}); {plan.p_better:.0%} probability the plan beats the current mix; "
        f"≈{_money(plan.uplift * horizon_weeks, currency)} over {horizon_weeks} weeks.",
        f"Highest return on the next {'rupee' if currency == '₹' else 'unit of spend'}: {display_name(best.channel)} (marginal ROI {best.mroi:.2f}, interval {best.mroi_low:.2f}–{best.mroi_high:.2f}).",
        f"Model accuracy on unseen weeks: {backtest.verdict.split(' — ')[0]} (MAPE {backtest.metrics['mape']:.1%})." if backtest
        else "Model accuracy has not been back-tested yet — treat as exploratory.",
        "Validate the largest shift with a geo-lift test before scaling.",
    ]
    _bullets(s, bullets, Inches(0.8), Inches(1.5), W - Inches(1.6), Inches(5))
    _footer(s, foot)

    # 3 drivers
    s = _slide(prs, f"What drives {outcome_label}", "Share of modelled outcome over the history (ESTIMATED)")
    dec = decomposition
    parts = {display_name(c): float(dec[c].sum()) for c in chans}
    parts = dict(sorted(parts.items(), key=lambda kv: kv[1]))
    parts["Base demand"] = float((dec["base"] + dec["trend"] + dec["seasonality"] + dec["controls"]).sum())
    cd = CategoryChartData()
    cd.categories = list(parts)
    tot = sum(max(v, 0) for v in parts.values()) or 1
    cd.add_series("Share", [round(100 * max(v, 0) / tot, 1) for v in parts.values()])
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.8), Inches(1.5), Inches(8), Inches(5.3), cd).chart
    _style_chart(ch, (BLUE,), legend=False)
    ch.plots[0].has_data_labels = True
    ch.plots[0].data_labels.number_format, ch.plots[0].data_labels.number_format_is_linked = '0.0"%"', False
    ch.plots[0].data_labels.font.size = Pt(13)
    ch.plots[0].gap_width = 60
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 100
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    _bullets(s, ["Base demand = what would happen with no paid media (brand, distribution, seasonality, price).",
                 "Media shares are incremental effects, not attributed clicks."], Inches(9.1), Inches(1.8), Inches(3.8), Inches(4), 14)
    _footer(s, foot)

    # 4 ROI
    s = _slide(prs, "Return by channel", "ROI over the history and marginal ROI at current spend (ESTIMATED, 90% intervals)")
    cd = CategoryChartData()
    cd.categories = names
    cd.add_series("ROI (average)", [float(x) for x in summary.roi])
    cd.add_series("Marginal ROI (next unit)", [float(x) for x in summary.mroi])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(7.4), Inches(5.3), cd).chart
    _style_chart(ch)
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xEC, 0xEC, 0xEA)
    tab = pd.DataFrame({"Channel": names, "ROI": [f"{r:.2f}" for r in summary.roi],
                        "90% interval": [f"{a:.2f}–{b:.2f}" for a, b in zip(summary.roi_low, summary.roi_high)],
                        "P(ROI>1)": [f"{p:.0%}" for p in summary.p_roi_above_1]})
    _table(s, tab, Inches(8.3), Inches(1.7), Inches(4.5))
    _footer(s, foot)

    # 5 recommended mix
    s = _slide(prs, "Recommended weekly budget mix", f"Same total budget ({_money(plan.weekly_budget, currency)}/week); RECOMMENDED")
    cd = CategoryChartData()
    cd.categories = [display_name(c) for c in t.channel]
    cd.add_series("Current", [float(x) for x in t.current_weekly])
    cd.add_series("Recommended", [float(x) for x in t.recommended_weekly])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(7.4), Inches(5.3), cd).chart
    _style_chart(ch, (GREY, BLUE))
    ch.value_axis.tick_labels.number_format, ch.value_axis.tick_labels.number_format_is_linked = '#,##0', False
    tab = pd.DataFrame({"Channel": [display_name(c) for c in t.channel], "Now/wk": [_money(v, currency) for v in t.current_weekly],
                        "Plan/wk": [_money(v, currency) for v in t.recommended_weekly],
                        "Change": [f"{c:+.0%}" for c in t.change_pct.fillna(0)]})
    _table(s, tab, Inches(8.3), Inches(1.7), Inches(4.6))
    _footer(s, foot)

    # 6 impact
    s = _slide(prs, "Expected impact", "Incremental vs the current mix at the same budget (RECOMMENDED; 90% interval)")
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.5), Inches(3.5))
    tf = tb.text_frame
    tf.paragraphs[0].text = f"+{_money(plan.uplift * horizon_weeks, currency)}"
    tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold, tf.paragraphs[0].font.color.rgb = Pt(66), True, BLUE
    for line, size in ((f"{outcome_label} over {horizon_weeks} weeks "
                        f"(range {_money(plan.uplift_low * horizon_weeks, currency)} to {_money(plan.uplift_high * horizon_weeks, currency)})", 22),
                       (f"{plan.p_better:.0%} probability of beating the current mix · {_money(plan.uplift, currency)} per week", 18)):
        p = tf.add_paragraph()
        p.text, p.font.size, p.font.color.rgb = line, Pt(size), INK
    _bullets(s, plan.notes or ["All channels stay within ±50% of their historical spend range."], Inches(0.8), Inches(5.0),
             Inches(11.5), Inches(1.8), 14)
    _footer(s, foot)

    # 7 accuracy
    s = _slide(prs, "How accurate is this model?", backtest.verdict if backtest else "Back-test not run")
    if backtest:
        ser = backtest.series.tail(52)
        cd = CategoryChartData()
        cd.categories = [d.strftime("%d %b %y") for d in ser.date]
        cd.add_series("Actual", [float(v) for v in ser.actual])
        cd.add_series("Predicted", [float(v) for v in ser.predicted])
        ch = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.6), Inches(1.5), Inches(8.4), Inches(5.3), cd).chart
        ch.has_legend = True
        ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
        for sr, colr in zip(ch.series, (MUTED, BLUE)):
            sr.format.line.color.rgb = colr
            sr.format.line.width = Pt(2.25)
            sr.smooth = False
        ch.category_axis.tick_labels.font.size = Pt(9)
        m = backtest.metrics
        _bullets(s, [f"The last {m['holdout_weeks']} weeks (right of the chart) were hidden from the model and predicted from spend alone.",
                     f"Weekly error (MAPE): {m['mape']:.1%}", f"Total error over the holdout: {m['total_error_pct']:+.1%}",
                     f"Actuals inside the 90% band: {m['interval_coverage_90']:.0%} of weeks"],
                 Inches(9.3), Inches(1.8), Inches(3.6), Inches(5), 14)
    _footer(s, foot)

    # 8 caveats & experiments
    s = _slide(prs, "Caveats and next experiments")
    weak = [w for w in diagnostics.get("warnings", [])][:4]
    exps = [f"Geo-lift test on {display_name(up.channel)}: hold out matched regions for 6–8 weeks to confirm the incremental return.",
            f"Pulse test on {display_name(down.channel)}: reduce spend in test regions to confirm the loss is smaller than modelled."]
    _bullets(s, (weak or ["No identifiability warnings from the data checks."]) + exps, Inches(0.8), Inches(1.4),
             W - Inches(1.6), Inches(5.5), 16)
    _footer(s, foot)

    # 9 appendix
    s = _slide(prs, "Appendix: data and method")
    _bullets(s, [f"Data: {provenance.get('filename', 'client upload')} · {provenance.get('rows', '')} weeks · validation "
                 f"{provenance.get('validation', '')} · SHA-256 {str(provenance.get('sha256', ''))[:16]}",
                 f"Authorisation: {provenance.get('attestation', 'not recorded')}",
                 "Model: Bayesian marketing mix model — geometric adstock (carryover), Hill saturation, trend, yearly seasonality, "
                 "price/promo controls; MAP estimate with Laplace posterior; lift tests used as calibration where supplied.",
                 f"Fit: R² {diagnostics.get('r2', float('nan')):.2f}; converged: {diagnostics.get('converged')}; "
                 f"lift tests used: {provenance.get('lift_tests', 0)}.",
                 "Limits: correlational model of historical variation; spend levels outside the observed range are extrapolations."],
             Inches(0.8), Inches(1.4), W - Inches(1.6), Inches(5.5), 14)
    _footer(s, foot)
    _apply_font(prs)
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
