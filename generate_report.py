"""Build the required short PDF report from the generated experiment results."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "term_paper_report.pdf"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_report() -> None:
    data = pd.read_csv(RESULTS / "policy_results.csv")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=21))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=10.5))
    doc = SimpleDocTemplate(str(REPORT), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    story = []

    story.append(Paragraph("CSE-307 Operating Systems", styles["TitleCenter"]))
    story.append(Paragraph("Learning-Augmented Page Replacement: Classical Algorithms Meet Adaptive Prediction", styles["Heading2"]))
    story.append(Paragraph("Student: Md. Mahdi Israk Mahir &nbsp;&nbsp; Student ID: 202414016 &nbsp;&nbsp; Track 1", styles["Small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Problem framing", styles["Heading2"]))
    story.append(Paragraph(
        "Page replacement decides which resident page should leave memory when a new page is referenced. "
        "FIFO and LRU are inexpensive online policies, while Optimal (Belady's) is an offline upper-bound "
        "that knows future references. This paper studies whether a small learned policy can use current-state "
        "signals to make useful eviction decisions when the access pattern changes after deployment.", styles["BodyText"]))

    story.append(Paragraph("2. Implementation and learned component", styles["Heading2"]))
    story.append(Paragraph(
        "The implementation uses four frames and ten possible pages. FIFO maintains arrival order; LRU tracks "
        "the most recent reference position; and Optimal evicts the resident page with the farthest next use. "
        "The learned policy is a shallow decision tree trained on a separate synthetic trace. For every resident "
        "candidate, it receives only information available at the current miss: page age, access frequency, "
        "recency rank, and page identifier. Training labels identify the candidate that Optimal would evict on "
        "the training trace. Future references are never supplied as runtime features.", styles["BodyText"]))

    story.append(Paragraph("3. Experimental setup", styles["Heading2"]))
    story.append(Paragraph(
        "The evaluation trace contains 800 references and is generated with a fixed seed (307). The first 400 "
        "references are locality-heavy: pages 0–4 are favored and occasional sequential accesses are inserted. "
        "The second 400 references switch to a random/bursty pattern over pages 0–9. Each policy is evaluated "
        "separately in both phases using page-fault count and hit ratio.", styles["BodyText"]))

    story.append(Paragraph("4. Results", styles["Heading2"]))
    table_data = [["Policy", "Locality faults", "Locality hit", "Shift faults", "Shift hit"]]
    for policy in ["FIFO", "LRU", "Optimal", "Learned"]:
        rows = data[data["policy"] == policy].set_index("phase")
        table_data.append([policy, int(rows.loc["locality", "faults"]), pct(rows.loc["locality", "hit_ratio"]),
                           int(rows.loc["random_bursty", "faults"]), pct(rows.loc["random_bursty", "hit_ratio"])])
    table = Table(table_data, colWidths=[1.05 * inch, 1.15 * inch, 1.05 * inch, 1.05 * inch, 1.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#EAF2F8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))
    story.append(Image(str(RESULTS / "hit_ratio_by_phase.png"), width=5.7 * inch, height=3.55 * inch))

    story.append(PageBreak())
    story.append(Paragraph("5. Analysis of the workload shift", styles["Heading2"]))
    story.append(Paragraph(
        "The workload change hurts all policies because the second phase has a larger active set and less "
        "predictable reuse. LRU falls from 84.0% to 53.8% hit ratio, while FIFO falls from 79.8% to 53.0%. "
        "Optimal remains strongest (90.0% to 70.3%) because it has future knowledge and therefore shows the "
        "best achievable reference point for this trace.", styles["BodyText"]))
    story.append(Paragraph(
        "The learned policy reaches 86.5% in the locality phase, close to Optimal and better than both FIFO and "
        "LRU. However, it falls to 44.3% after the shift, the largest absolute degradation in this experiment. "
        "The tree learned relationships from a separate trace whose features were dominated by stable reuse. "
        "When the trace becomes random and bursty, age and frequency become less reliable predictors of the "
        "next use. This result is useful: learning does not automatically guarantee robustness under drift.", styles["BodyText"]))
    story.append(Paragraph(
        "A limitation is that the workload is synthetic and the classifier is intentionally small. A stronger "
        "adaptive system could detect phase changes online, retrain periodically, use a larger history window, "
        "or combine the learned score with an LRU fallback. Those extensions are outside the base Track 1 scope.", styles["BodyText"]))

    story.append(Paragraph("6. Conclusion", styles["Heading2"]))
    story.append(Paragraph(
        "The classical implementations provide understandable baselines and Optimal supplies an upper bound. "
        "The decision tree can improve eviction choices when the workload resembles training conditions, but the "
        "shift test shows why operating-system heuristics must be evaluated under distribution change. The main "
        "lesson is that a learned component should be monitored and adapted rather than trusted as a permanently "
        "correct replacement for a classical policy.", styles["BodyText"]))

    story.append(Paragraph("Reproducibility", styles["Heading2"]))
    story.append(Paragraph(
        "From the repository root, install the packages in requirements.txt and run "
        "<b>python track1_page_replacement.py</b>. The command regenerates the trace, CSV table, and charts. "
        "The report is generated with <b>python generate_report.py</b>.", styles["Small"]))
    doc.build(story)


if __name__ == "__main__":
    build_report()
    print(REPORT)
