"""Build the Markdown and PDF reports from the same computed results."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "checkpoint_research_report"
FIGURES = ROOT / "visualizations"
INK = colors.HexColor("#122630")
MUTED = colors.HexColor("#485C66")
TEAL = colors.HexColor("#086861")
WARM = colors.HexColor("#A14A26")
RULE = colors.HexColor("#D5E0E0")
TINT = colors.HexColor("#EDF5F3")
WIDTH = 464
TITLE = "Do financial headlines predict the S&P 500?"
SUBTITLE = "A trading-session alignment audit and chronological forecast test"


def style(name, size, leading, color=INK, font="Helvetica", before=0, after=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                          textColor=color, spaceBefore=before, spaceAfter=after)


STYLES = {
    "label": style("label", 9, 12, TEAL, "Helvetica-Bold", after=10),
    "title": style("title", 35, 38, INK, "Helvetica-Bold", after=12),
    "subtitle": style("subtitle", 14, 18, MUTED, after=21),
    "section": style("section", 25, 29, INK, "Helvetica-Bold", after=13),
    "subhead": style("subhead", 12, 15, TEAL, "Helvetica-Bold", before=12, after=6),
    "text": style("text", 10.5, 15, after=10),
    "lead": style("lead", 12, 17.5, INK, after=15),
    "caption": style("caption", 9, 12.5, MUTED, before=5, after=10),
    "cell": style("cell", 9.3, 12),
    "thead": style("thead", 9.3, 12, colors.white, "Helvetica-Bold"),
}


def label(drawing, x, y, text, color=MUTED, size=9, bold=False, anchor="start"):
    drawing.add(String(x, y, text, fillColor=color, fontSize=size,
                       fontName="Helvetica-Bold" if bold else "Helvetica", textAnchor=anchor))


def alignment_chart(audit):
    drawing = Drawing(WIDTH, 164)
    left, span = 151, 253
    drawing.add(Rect(0, 0, WIDTH, 164, fillColor=TINT, strokeColor=None))
    label(drawing, 16, 143, "Accuracy of the prior-return sign rule", TEAL, 11, True)
    for y, name, score, color in [
        (106, "Calendar rows", audit["calendar_lag_accuracy"], WARM),
        (61, "Trading sessions", audit["session_lag_accuracy"], TEAL),
    ]:
        label(drawing, 16, y + 4, name, INK, 10, True)
        drawing.add(Rect(left, y, span * score, 15, fillColor=color, strokeColor=None))
        label(drawing, left + span * score + 8, y + 4, f"{score:.1%}", color, 10, True)
    drawing.add(Line(left, 39, left + span, 39, strokeColor=RULE))
    for pct in (0, 25, 50, 75, 100):
        x = left + span * pct / 100
        drawing.add(Line(x, 35, x, 39, strokeColor=MUTED))
        label(drawing, x, 21, f"{pct}%", anchor="middle")
    return drawing


def uncertainty_chart(metrics, intervals):
    drawing = Drawing(WIDTH, 204)
    drawing.add(Rect(0, 0, WIDTH, 204, fillColor=TINT, strokeColor=None))
    label(drawing, 16, 182, "Change in test error when news is added", TEAL, 11, True)
    for y, name, estimate, bounds, unit in metrics:
        low, high = intervals[name]
        minimum, maximum = bounds
        left, right = 153, 319

        def position(value):
            return left + (right - left) * (value - minimum) / (maximum - minimum)

        zero = position(0)
        drawing.add(Line(left, y, right, y, strokeColor=RULE, strokeWidth=1))
        drawing.add(Line(zero, y - 10, zero, y + 10, strokeColor=MUTED, strokeWidth=1))
        drawing.add(Line(position(low), y, position(high), y, strokeColor=TEAL, strokeWidth=2))
        drawing.add(Circle(position(estimate), y, 3.5, fillColor=TEAL, strokeColor=None))
        label(drawing, 16, y - 3, unit, INK, 9.5, True)
        label(drawing, 450, y - 3, f"[{low:+.4f}, {high:+.4f}]" if name == "brier"
              else f"[{low:+.2f}, {high:+.2f}]", MUTED, 9, anchor="end")
        label(drawing, zero, y - 22, "0", size=8, anchor="middle")
    label(drawing, 16, 15, "Negative values favor the model with news. Each row has its own scale.", size=8.5)
    return drawing


class Report:
    def __init__(self):
        self.pdf = []
        self.markdown = []

    def paragraph(self, text, kind="text"):
        self.pdf.append(Paragraph(escape(text), STYLES[kind]))
        prefix = {"title": "# ", "section": "## ", "subhead": "### "}.get(kind, "")
        self.markdown.extend((prefix + text, ""))

    def table(self, headers, rows, widths):
        cells = [[Paragraph(escape(cell), STYLES["thead"]) for cell in headers]]
        cells.extend([Paragraph(escape(cell), STYLES["cell"]) for cell in row] for row in rows)
        table = Table(cells, colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        self.pdf.extend((table, Spacer(1, 8)))
        self.markdown.append("| " + " | ".join(headers) + " |")
        self.markdown.append("| " + " | ".join("---" for _ in headers) + " |")
        self.markdown.extend("| " + " | ".join(row) + " |" for row in rows)
        self.markdown.append("")

    def chart(self, drawing, name, alt, caption):
        renderSVG.drawToFile(drawing, str(FIGURES / name))
        self.pdf.append(drawing)
        self.markdown.extend((f"![{alt}](visualizations/{name})", ""))
        self.paragraph(caption, "caption")

    def page(self):
        self.pdf.append(PageBreak())

    def save(self):
        Path(f"{REPORT}.md").write_text("\n".join(self.markdown), encoding="utf-8")
        document = SimpleDocTemplate(f"{REPORT}.pdf", pagesize=(576, 756), leftMargin=50,
                                     rightMargin=50, topMargin=51, bottomMargin=48,
                                     title=TITLE, subject=SUBTITLE)
        document.build(self.pdf, onFirstPage=page_footer, onLaterPages=page_footer)


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.line(56, 38, 520, 38)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(56, 24, "FINANCIAL NEWS AND DAILY INDEX RETURNS")
    canvas.drawRightString(520, 24, str(document.page))
    canvas.restoreState()


def write_report(results):
    audit, assoc = results["audit"], results["associations"]
    train, test = results["training"], results["holdout"]
    direction, volatility = results["direction"], results["volatility"]
    brier_change = direction["market_news_up_probability"]["brier"] - direction["market_up_probability"]["brier"]
    mae_change = volatility["market_news_abs_return"]["mae"] - volatility["market_abs_return"]["mae"]
    rmse_change = volatility["market_news_abs_return"]["rmse"] - volatility["market_abs_return"]["rmse"]
    interval = results["bootstrap"]["intervals"]
    bp_interval = {key: [value * 10000 for value in limits] if key != "brier" else limits
                   for key, limits in interval.items()}

    report = Report()
    report.paragraph("RESEARCH REPORT", "label")
    report.paragraph(TITLE, "title")
    report.paragraph(SUBTITLE, "subtitle")
    report.paragraph("Abstract", "subhead")
    report.paragraph(
        f"Across {audit['headline_count']:,} headline scores, we tested whether news improves forecasts "
        "of the S&P 500's next close-to-close return. The original calendar join repeated the target "
        f"session on {audit['reuse_fraction']:.1%} of rows. A prior-return sign rule scored "
        f"{audit['calendar_lag_accuracy']:.1%} on the calendar rows and {audit['session_lag_accuracy']:.1%} "
        "after grouping by session. In a chronological test with a conservative news cutoff, adding "
        "headline features worsened directional probability forecasts and left mean absolute error "
        "for return magnitude practically unchanged.", "lead")
    report.paragraph("Data and unit of observation", "subhead")
    report.paragraph(
        f"The supplied aggregate covers {audit['publication_days']:,} publication dates from "
        f"{results['publication_range'][0]} to {results['publication_range'][1]}. Each date maps to the "
        f"first subsequent market session recorded in the source, yielding {audit['sessions']:,} unique "
        f"target sessions from {results['session_range'][0]} to {results['session_range'][1]}. "
        "We evaluate forecasts ending in 2023. The later target is counted only in the source audit.")
    report.paragraph(
        "The input contains headline count, mean sentiment, population standard deviation of sentiment, "
        "and the S&P 500 close-to-close return assigned to each publication date. The archived scoring "
        "script uses VADER with a custom financial term list. The submitted scored-headline sample is "
        "available for inspection. All reported results start from the bundled daily aggregate.")
    report.table(
        ["Question", "Measure"],
        [("Does tone track index returns?", "Pearson correlation with the preceding and assigned sessions."),
         ("Does news improve directional forecasts?", "Brier score, ROC-AUC and classification accuracy."),
         ("Does news improve return-magnitude forecasts?", "MAE and RMSE for absolute close-to-close returns.")],
        [214, 250])

    report.page()
    report.paragraph("01   DATA ALIGNMENT", "label")
    report.paragraph("How the target leaked", "section")
    report.paragraph(
        "The source maps Friday, Saturday and Sunday to Monday whenever Monday is the next trading day. "
        "On the weekend rows, shifting the return down by one publication row can reveal Monday's return "
        f"before it occurs. We counted {audit['reused_targets']:,} rows whose assigned target session "
        "matched the preceding row's target session.")
    report.table(
        ["Headline publication date", "Assigned target session", "Assigned return"],
        [(row["published_date"], row["market_date"], f'{row["market_return"]:+.3%}') for row in audit["example"]],
        [179, 161, 124])
    report.chart(
        alignment_chart(audit), "alignment_audit.svg", "Prior-return rule accuracy before and after alignment",
        f"Figure 1. Both samples use target dates in 2022-2023. Calendar accuracy weights "
        f"{audit['calendar_test_rows']:,} publication rows. Session accuracy gives equal weight to "
        f"{audit['session_test_rows']:,} sessions. The calendar rule correctly classifies "
        f"{audit['calendar_test_reused_correct']:,} of {audit['calendar_test_reused']:,} rows that "
        "repeat the preceding target session.")
    report.paragraph("Pooling and descriptive associations", "subhead")
    report.paragraph(
        "We pool publication dates by assigned session and weight headline scores by article count. "
        "Dispersion is reconstructed from the pooled first and second moments, including differences "
        "in average sentiment between the dates pooled into a session.")
    report.paragraph(
        f"Through 2023, average tone had a Pearson correlation of {assoc['tone_previous']:+.3f} with the "
        f"preceding session's return and {assoc['tone_next']:+.3f} with the assigned session's return. "
        f"Dispersion and assigned absolute return had a correlation of {assoc['dispersion_target']:+.3f} "
        f"overall and {assoc['dispersion_holdout']:+.3f} in the test period. These descriptive pairings "
        "use date-aligned news, which can include articles published after the preceding close.")

    report.page()
    report.paragraph("02   FORECAST DESIGN AND RESULTS", "label")
    report.paragraph("What the model could know", "section")
    report.paragraph(
        "Each forecast uses information available at the close of the preceding recorded session. "
        "Headlines are admitted only if their recorded publication date is strictly earlier than that "
        "cutoff session. For a normal Friday-close to Monday-close target, the newest eligible headline "
        "is dated Thursday. This buffer excludes Friday and weekend news because the input has dates "
        "without dependable intraday publication times.")
    report.paragraph(
        f"After a return-history warm-up, we fit once on {train['sessions']:,} sessions "
        f"({train['first']} to {train['last']}) and test on {test['sessions']:,} sessions "
        f"({test['first']} to {test['last']}). In the test period, {test['up_fraction']:.1%} of "
        "returns were positive. We standardize predictors using training data and keep the fitted models "
        "fixed while updating features with newly observed market history.")
    report.paragraph(
        "The market-only model uses the preceding return, the mean of the last 5 returns and mean "
        "absolute returns over 5 and 20 sessions. The same model with news adds buffered mean sentiment, "
        "dispersion and log headline count. Both models use linear least squares. Direction probabilities "
        "are clipped to [0, 1], with 0.5 as the class threshold. Absolute-return forecasts are floored at zero.")
    report.paragraph("Directional forecasts", "subhead")
    model_names = ["Constant baseline", "Market history", "Market history + news"]
    report.table(
        ["Model", "Accuracy", "ROC-AUC", "Brier score"],
        [(model, f"{data['accuracy']:.2%}", f"{data['auc']:.3f}", f"{data['brier']:.4f}")
         for model, data in zip(model_names, direction.values())],
        [204, 81, 81, 98])
    report.paragraph(
        "The constant probability is the training share of up sessions. Lower Brier scores indicate "
        "better probability forecasts. Both fitted models score worse than this baseline.", "caption")
    report.paragraph("Absolute-return forecasts", "subhead")
    report.table(
        ["Model", "MAE (pp)", "RMSE (pp)"],
        [(model, f"{100 * data['mae']:.4f}", f"{100 * data['rmse']:.4f}")
         for model, data in zip(model_names, volatility.values())],
        [238, 113, 113])
    report.paragraph(
        "The constant forecast is the training median absolute return. Mean absolute error (MAE) and "
        "root mean square error (RMSE) are in percentage points of index return. Market history reduces "
        "both. Adding news leaves MAE almost identical.", "caption")

    report.page()
    report.paragraph("03   UNCERTAINTY AND REPRODUCIBILITY", "label")
    report.paragraph("What the evidence supports", "section")
    report.paragraph(
        f"Adding news raises the Brier score by {brier_change:.4f}. The increase also appears in "
        "each test year. Absolute-return error changes little, with the MAE difference changing "
        "sign between the two years. The correlation of buffered disagreement with absolute return "
        f"falls to {assoc['buffered_dispersion_holdout']:+.3f} on the test sessions.")
    metrics = [
        (150, "brier", brier_change, (-0.004, 0.012), "Brier score"),
        (105, "mae", mae_change * 10000, (-1.1, 1), "MAE (bp)"),
        (60, "rmse", rmse_change * 10000, (-1.1, 1), "RMSE (bp)"),
    ]
    report.chart(
        uncertainty_chart(metrics, bp_interval), "forecast_uncertainty.svg",
        "Paired bootstrap intervals for changes in forecast errors when news is added",
        f"Figure 2. News model minus market-only model, with paired 95% percentile intervals from "
        f"{results['bootstrap']['resamples']:,} moving-block resamples of "
        f"{results['bootstrap']['block_sessions']} consecutive test sessions. One basis point (bp) is "
        "0.01 percentage point. Intervals describe test-period uncertainty conditional on the fitted models.")
    report.paragraph("Scope of the conclusion", "subhead")
    report.paragraph(
        "The results apply to this headline corpus, daily index targets and these linear models. Earlier "
        "project iterations examined the same years, so this is a retrospective test rather than an "
        "independent confirmation. The cutoff sacrifices recent news to respect the available dates. "
        "Absolute return is a daily magnitude proxy. Some headlines concern individual companies, and "
        "the scoring extension has no independent human-labeled validation in this submission.")
    report.paragraph("Reproduce the results", "subhead")
    report.paragraph(
        "From the project root, install requirements.txt and run python scripts/news_research.py, then "
        "python scripts/generate_pdf_report.py. The first command records predictions and dates in "
        "datasets/holdout_predictions.csv and computed metrics and the input hash in "
        "datasets/research_results.json. The second renders both reports and figures from those metrics. "
        "The full raw headline archive and original index file are unavailable in this submission.")
    report.save()


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    results = json.loads((ROOT / "datasets" / "research_results.json").read_text(encoding="utf-8"))
    write_report(results)
