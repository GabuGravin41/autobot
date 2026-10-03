"""
Generate the visual-first pitch deck (16:9, 13.333 x 7.5 in) with python-pptx.

Layout grammar (every figure slide uses the same grid):
    header band   : category tag + short title (left), hero metric (right)
    figure stage  : embedded high-res figure, ~60-65% of the canvas
    chip row      : two KPI badges + one 1-line takeaway chip
All explanatory prose lives in the speaker notes, not on the slides.

Run:  python papers/autonomous_ai_systems/generate_deck.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
DECK_PATH = HERE / "Autonomous_AI_Systems_Presentation.pptx"

# Palette
NAVY = RGBColor(15, 23, 42)
SLATE = RGBColor(30, 41, 59)
SLATE_EDGE = RGBColor(51, 65, 85)
WHITE = RGBColor(248, 250, 252)
MUTED = RGBColor(148, 163, 184)
CYAN = RGBColor(6, 182, 212)
EMERALD = RGBColor(16, 185, 129)
AMBER = RGBColor(245, 158, 11)
BLUE = RGBColor(59, 130, 246)
RED = RGBColor(239, 68, 68)

FONT = "Calibri"

# Grid (inches)
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.5
CONTENT_W = SLIDE_W - 2 * MARGIN
FIGURE_BOX = (MARGIN, 1.35, CONTENT_W, 5.12)
CHIP_Y, CHIP_H = 6.62, 0.56
KPI_W, GAP = 2.7, 0.2


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------
def fill_frame(tf, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, inset=0.0):
    """lines: list of (text, size_pt, color, bold). One paragraph per line."""
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_top = tf.margin_bottom = 0
    tf.margin_left = tf.margin_right = Inches(inset)
    for i, (text, size, color, bold) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color


def add_text(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_frame(box.text_frame, lines, align, anchor)
    return box


def add_shape(slide, kind, x, y, w, h, fill, line=None, line_w=1.25, radius=None):
    shp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    if radius is not None:
        shp.adjustments[0] = radius
    shp.shadow.inherit = False
    return shp


def new_slide(prs, notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_tag(slide, x, y, text, color):
    w = 0.09 * len(text) + 0.4
    pill = add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, 0.3, SLATE, radius=0.5)
    fill_frame(pill.text_frame, [(text, 10, color, True)], PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


def add_header(slide, tag, title, color, hero=None):
    add_tag(slide, MARGIN, 0.35, tag, color)
    add_text(slide, MARGIN, 0.72, 8.4, 0.55, [(title, 28, WHITE, True)])
    if hero:
        value, label = hero
        add_text(slide, 9.0, 0.28, MARGIN + CONTENT_W - 9.0, 1.0,
                 [(value, 36, color, True), (label, 11, MUTED, False)],
                 align=PP_ALIGN.RIGHT)


def add_figure(slide, name, crop_top=0.0, crop_bottom=0.0, box=FIGURE_BOX):
    """Fit a figure (after cropping its baked-in title) inside box, centered."""
    path = FIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing figure: {path}")
    pic = slide.shapes.add_picture(str(path), 0, 0)
    ratio = pic.width / (pic.height * (1 - crop_top - crop_bottom))
    bx, by, bw, bh = box
    w, h = (bh * ratio, bh) if bw / bh > ratio else (bw, bw / ratio)
    pic.left, pic.top = Inches(bx + (bw - w) / 2), Inches(by + (bh - h) / 2)
    pic.width, pic.height = Inches(w), Inches(h)
    pic.crop_top, pic.crop_bottom = crop_top, crop_bottom
    return pic


def add_kpi(slide, x, y, w, h, value, label, color, value_pt=16, label_pt=10):
    card = add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, SLATE,
                     line=color, radius=0.2)
    fill_frame(card.text_frame, [(value, value_pt, color, True), (label, label_pt, MUTED, False)],
               PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, inset=0.1)
    return card


def add_takeaway(slide, x, y, w, h, text, color=EMERALD, size=15):
    chip = add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, SLATE,
                     line=color, line_w=2, radius=0.3)
    fill_frame(chip.text_frame, [("→  " + text, size, WHITE, True)],
               PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, inset=0.15)
    return chip


def add_chip_row(slide, kpis, takeaway, takeaway_color=EMERALD):
    x = MARGIN
    for value, label, color in kpis:
        add_kpi(slide, x, CHIP_Y, KPI_W, CHIP_H, value, label, color)
        x += KPI_W + GAP
    add_takeaway(slide, x, CHIP_Y, MARGIN + CONTENT_W - x, CHIP_H, takeaway, takeaway_color)


def figure_slide(prs, tag, title, color, hero, figure, crop, kpis, takeaway, notes):
    s = new_slide(prs, notes)
    add_header(s, tag, title, color, hero)
    add_figure(s, figure, *crop)
    add_chip_row(s, kpis, takeaway)
    return s


# ----------------------------------------------------------------------------
# Slides
# ----------------------------------------------------------------------------
def slide_title(prs):
    s = new_slide(prs, (
        "Long-horizon autonomy fails quietly. This talk covers the architecture, the two "
        "failure modes that cost us the most, verified results, and the economics of a "
        "12-hour unattended AI engineering agent."))
    add_tag(s, MARGIN, 0.9, "AUTOBOT SYSTEMS RESEARCH", CYAN)
    add_text(s, MARGIN, 1.4, 7.6, 2.0, [
        ("Long-Horizon Autonomy", 44, WHITE, True),
        ("Fails Quietly", 44, CYAN, True),
    ])
    add_text(s, MARGIN, 3.35, 7.6, 0.9, [
        ("Architecture · Failure modes · Economics", 18, WHITE, False),
        ("Dalton Omondi", 13, MUTED, False),
    ])

    # Horizon ladder: circle size grows with the unattended horizon
    baseline = 4.1
    ladder = [(8.2, 0.85, "2 m", "Chat", AMBER, 16),
              (9.3, 1.3, "30 m", "Agent chains", BLUE, 20),
              (10.85, 1.95, "12 h", "Autobot", EMERALD, 30)]
    (x1, d1, *_), (x3, d3, *_) = ladder[0], ladder[-1]
    conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                  Inches(x1 + d1 / 2), Inches(baseline - d1 / 2),
                                  Inches(x3 + d3 / 2), Inches(baseline - d3 / 2))
    conn.line.color.rgb = SLATE_EDGE
    conn.line.width = Pt(2)
    for x, d, val, label, col, pt in ladder:
        c = add_shape(s, MSO_SHAPE.OVAL, x, baseline - d, d, d, SLATE, line=col, line_w=3)
        fill_frame(c.text_frame, [(val, pt, col, True)], PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        add_text(s, x - 0.3, baseline + 0.12, d + 0.6, 0.3, [(label, 11, MUTED, False)],
                 align=PP_ALIGN.CENTER)

    stats = [("12+ h", "unattended horizon", EMERALD),
             ("6", "concurrent domains", CYAN),
             ("$5", "marginal cost per run", BLUE)]
    w = (CONTENT_W - 2 * 0.3) / 3
    for i, (val, lbl, col) in enumerate(stats):
        add_kpi(s, MARGIN + i * (w + 0.3), 5.0, w, 1.6, val, lbl, col, value_pt=44, label_pt=14)


def slide_enterprise(prs):
    s = new_slide(prs, (
        "Project SEQUOY: dynamically generated HL7 FHIR R4 transaction bundles (Encounter, "
        "Condition, Patient) mapped to KenyaEMR / OpenMRS CIEL concepts (CIEL 113054 "
        "pre-eclampsia, CIEL 230 PPH); 10 Kenyan hospital procurement dossiers drafted.\n"
        "Kenya Camera AI: 50.9 ms average CPU inference with YOLOX-Nano ONNX; 9-12 concurrent "
        "camera feeds on a $150 Intel N100 mini PC; 7 smoke tests pass in 1.55 s with no GPU."))
    add_header(s, "REAL-WORLD TRANSLATION", "From leaderboards to production", CYAN)

    col_w = (CONTENT_W - 0.4) / 2
    tile_w = (col_w - 2 * GAP) / 3
    columns = [
        ("Project SEQUOY", "Clinical EHR · Kenya", CYAN,
         [("FHIR R4", "transaction bundles"), ("CIEL", "KenyaEMR / OpenMRS concepts"),
          ("10", "hospital dossiers")],
         "Standards-native from day one"),
        ("Kenya Camera AI", "Edge vision · CPU only", BLUE,
         [("50.9 ms", "CPU inference"), ("9–12", "feeds per box"),
          ("$150", "Intel N100, no GPU")],
         "7/7 smoke tests pass in 1.55 s"),
    ]
    for c, (name, sub, col, tiles, takeaway) in enumerate(columns):
        x0 = MARGIN + c * (col_w + 0.4)
        add_text(s, x0, 1.5, col_w, 0.7, [(name, 22, col, True), (sub, 12, MUTED, False)])
        for i, (val, lbl) in enumerate(tiles):
            add_kpi(s, x0 + i * (tile_w + GAP), 2.4, tile_w, 3.3, val, lbl, col,
                    value_pt=30, label_pt=12)
        add_takeaway(s, x0, 6.0, col_w, 0.6, takeaway, col, size=14)


def slide_laws(prs):
    s = new_slide(prs, (
        "Law 1 - Paranoid liveness: silent deadlocks outnumbered loud crashes roughly 5:1, so "
        "watchdog timers and scheduled interrupts are mandatory.\n"
        "Law 2 - Pre-dispatch budget proofs: budgets compound multiplicatively across task "
        "splits; prove worst-case runtime <= the hard limit before launching.\n"
        "Law 3 - Physical boundary projections: statistical models drift into non-physical "
        "space; clamp at every hop and monitor clip hit-rates."))
    add_header(s, "PRINCIPLES", "Three laws of long-horizon autonomy", CYAN)

    laws = [("5 : 1", "silent stalls vs loud crashes", "Paranoid liveness",
             "Watchdogs + scheduled interrupts", RED),
            ("≤ 12 h", "worst case, proven pre-launch", "Budget proofs",
             "Prove T_worst ≤ limit, then dispatch", AMBER),
            ("Every hop", "invariants checked in-loop", "Boundary projections",
             "Clamp per hop; log clip hit-rate", EMERALD)]
    w = (CONTENT_W - 2 * 0.3) / 3
    for i, (metric, metric_lbl, name, rule, col) in enumerate(laws):
        x = MARGIN + i * (w + 0.3)
        add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.55, w, 5.1, SLATE, line=col,
                  line_w=2, radius=0.06)
        num = add_shape(s, MSO_SHAPE.OVAL, x + w / 2 - 0.4, 1.85, 0.8, 0.8, col)
        fill_frame(num.text_frame, [(str(i + 1), 24, NAVY, True)], PP_ALIGN.CENTER,
                   MSO_ANCHOR.MIDDLE)
        add_text(s, x + 0.2, 2.95, w - 0.4, 1.4,
                 [(metric, 44, col, True), (metric_lbl, 12, MUTED, False)],
                 align=PP_ALIGN.CENTER)
        add_text(s, x + 0.2, 4.75, w - 0.4, 1.4,
                 [(name, 20, WHITE, True), (rule, 13, MUTED, False)],
                 align=PP_ALIGN.CENTER)


def slide_roadmap(prs):
    s = new_slide(prs, (
        "Measure: ablate the outer loop to quantify MTTD and MTTR, record stall-hours avoided, "
        "and log clip-activation telemetry.\n"
        "Generalize: replace bespoke scripts with a declarative job registry, a per-hop "
        "invariant library, and a pre-dispatch budget verifier CLI.\n"
        "Prove: publish telemetry logs, reproducible commit snapshots, and run continuous "
        "multi-day autonomous swarms."))
    add_header(s, "ROADMAP", "Next: measure, generalize, prove", CYAN)

    stages = [("MEASURE", CYAN, ["MTTD / MTTR ablation", "Stall-hours avoided",
                                 "Clip-rate telemetry"]),
              ("GENERALIZE", EMERALD, ["Declarative job registry", "Invariant library",
                                       "Budget-verifier CLI"]),
              ("PROVE", BLUE, ["Open telemetry logs", "Reproducible snapshots",
                               "Multi-day swarms"])]
    w = (CONTENT_W - 2 * 0.3) / 3
    for i, (name, col, items) in enumerate(stages):
        x = MARGIN + i * (w + 0.3)
        kind = MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON
        arrow = add_shape(s, kind, x, 1.6, w, 1.1, col)
        fill_frame(arrow.text_frame, [(name, 22, NAVY, True)], PP_ALIGN.CENTER,
                   MSO_ANCHOR.MIDDLE)
        for j, item in enumerate(items):
            chip = add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x + 0.25, 3.05 + j * 0.9,
                             w - 0.5, 0.7, SLATE, line=col, radius=0.3)
            fill_frame(chip.text_frame, [(item, 15, WHITE, True)], PP_ALIGN.CENTER,
                       MSO_ANCHOR.MIDDLE)
    add_text(s, MARGIN, 6.0, CONTENT_W, 0.7,
             [("Autonomy is an operations problem.", 28, CYAN, True)],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def build_visual_deck():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    slide_title(prs)

    figure_slide(
        prs, "THE PROBLEM", "The horizon gap", AMBER, ("360×", "longer than a chat turn"),
        "fig_horizon_gap.png", (0.10, 0.06),
        [("12–48 h", "Autobot horizon", EMERALD), ("Silent", "dominant failure mode", AMBER)],
        "Beyond 30 min, operations decide success",
        "Short horizons (2 min): model reasoning dominates and a human is the watchdog. "
        "Long horizons (12 h+): operations dominate - budgets, liveness, silent sleep and "
        "drift. Single-turn benchmarks mask these long-horizon failure modes entirely.")

    figure_slide(
        prs, "SYSTEM DESIGN", "Dual-loop supervision", BLUE, ("2 m / 30 m", "inner / outer cadence"),
        "fig_supervision_architecture.png", (0.085, 0.0),
        [("Python", "deterministic inner loop", CYAN), ("LLM", "scheduled outer reconciler", BLUE)],
        "Cheap loop notices. Expensive loop thinks.",
        "Erlang / Kubernetes lineage. Fast inner loop (2 min): deterministic Python supervisor "
        "that polls workers, checks health and extracts tracebacks. Slow outer loop (30 min): "
        "scheduled LLM reconciler that audits state, diagnoses crashes and manages midnight-UTC "
        "quotas. Together they stop silent overnight stalls.")

    figure_slide(
        prs, "FAILURE AUTOPSY 1 · SWE-BENCH", "Budgets compound multiplicatively", RED,
        ("18.4 h", "projected vs 12 h hard kill"),
        "fig2_swe_runtime_math.png", (0.08, 0.0),
        [("8.0 → 3.5 min", "per-task agent cap", AMBER), ("≤ 8.5 h", "calibrated runtime", EMERALD)],
        "Prove T_worst ≤ limit before dispatch",
        "Exp 6/7: 8.0 min budget x 120 tasks = 16 h, plus 2.4 h overhead = 18.4 h, which hits "
        "the 12 h SIGKILL. SWE-bench agents fail on ~85% of tasks and burn the full cap each "
        "time. Exp 8: 3.5 min budget with 2048 thinking tokens keeps total runtime <= 8.5 h.")

    figure_slide(
        prs, "FAILURE AUTOPSY 2 · CLIMATE EMULATION", "Autoregressive boundary drift", AMBER,
        ("Hour 2.1", "AssertionError at hop 8"),
        "fig_boundary_drift.png", (0.08, 0.0),
        [("185k", "rows rolled out", CYAN), ("0.000 m", "tree height reached", RED)],
        "Enforce invariants every hop — not at export",
        "At hour 2.1 an AssertionError fired: tree height reached 0.000 m at an arid test site. "
        "The invariant was only checked at export after 185k rows, wasting two hours of compute. "
        "Exp 7 clamps per hop and passes; report clip hit-rates so clamping never hides drift.")

    figure_slide(
        prs, "VERIFIED RESULTS", "One agent, five live leaderboards", EMERALD,
        ("Top 4.88%", "S6E9 · #131 / 2,683"),
        "fig_multidomain_standings.png", (0.08, 0.0),
        [("Top 11.8%", "Biohub · #460 / 3,906", CYAN), ("Top 14.5%", "UMUD · #45 / 310", BLUE)],
        "Five concurrent campaigns under one supervisor",
        "Playground S6E9: rank #131 of 2,683 (top 4.88%). Biohub cell tracking: #460 of 3,906 "
        "(top 11.8%). UMUD muscle architecture: #45 of 310 (top 14.5%). Soil granulometry and "
        "traffic flow were held concurrently by the same unattended agent.")

    figure_slide(
        prs, "ECONOMICS", "Cost of a 12-hour campaign", EMERALD,
        ("864×", "cheaper than a 3-engineer team"),
        "fig1_cost_comparison.png", (0.08, 0.0),
        [("$5.00", "marginal cost per run", EMERALD), ("~$35", "fully loaded (4× L4)", BLUE)],
        "Engineer-hours become supervision-minutes",
        "Marginal cost: $5.00 per 12 h run ($0.29 electricity + $4.71 inference). Fully loaded "
        "~$35 including amortized 4x L4 GPU compute and tooling, versus $4,320 for three senior "
        "engineers over the same 12 hours.")

    slide_enterprise(prs)
    slide_laws(prs)
    slide_roadmap(prs)

    try:
        prs.save(str(DECK_PATH))
        out = DECK_PATH
    except PermissionError:
        # The deck is usually locked because it is open in PowerPoint.
        out = DECK_PATH.with_name(DECK_PATH.stem + "_new.pptx")
        prs.save(str(out))
        print(f"WARNING: {DECK_PATH.name} is locked (open in PowerPoint?); saved copy instead.")
    print(f"Visual-first deck compiled: {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build_visual_deck()
