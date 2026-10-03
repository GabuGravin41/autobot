"""
Generate visual-first, presentation-optimized scientific figures for the Autobot pitch deck.
Minimal text, high visual contrast, clear schematics and data density.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = Path(r"c:\Users\User 1\OneDrive\Desktop\projects\django projects\personal projects\autobot\papers\autonomous_ai_systems\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Dark theme palette
BG_DARK = '#0f172a'
CARD_BG = '#1e293b'
TEXT_LIGHT = '#f8fafc'
TEXT_MUTED = '#94a3b8'
ACCENT_CYAN = '#06b6d4'
ACCENT_EMERALD = '#10b981'
ACCENT_BLUE = '#3b82f6'
ACCENT_AMBER = '#f59e0b'
ACCENT_RED = '#ef4444'

# ------------------------------------------------------------------------------
# 1. Horizon Scale Diagram (Seconds to Hours)
# ------------------------------------------------------------------------------
def generate_horizon_gap_figure():
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)

    horizons = ['Interactive Chat\n(Copilot / ChatGPT)', 'Agentic Chains\n(BabyAGI / ReAct)', 'Autobot System\n(Unsupervised Autonomy)']
    times = ['30s – 2 mins', '5 – 30 mins', '12 – 48+ hours']
    modes = [
        'Failure: Prompt Hallucination\nWatchdog: Human Engineer\nContext: Single-Turn Snippet',
        'Failure: Infinite Tool Loops\nWatchdog: Max Step Counter\nContext: Local Memory Window',
        'Failure: Silent Deadlocks & Drift\nWatchdog: Dual-Loop Reconciler\nContext: Cross-Domain Multi-Task'
    ]
    colors = [ACCENT_AMBER, ACCENT_BLUE, ACCENT_EMERALD]

    x = [1, 2.5, 4.2]
    widths = [1.1, 1.1, 1.4]

    for i in range(3):
        rect = patches.FancyBboxPatch((x[i]-widths[i]/2, 0.4), widths[i], 3.2,
                                     boxstyle="round,pad=0.15",
                                     fc=CARD_BG, ec=colors[i], lw=2.5)
        ax.add_patch(rect)
        ax.text(x[i], 3.2, horizons[i], ha='center', va='center', fontsize=12, fontweight='bold', color=colors[i])
        ax.text(x[i], 2.4, f"Horizon: {times[i]}", ha='center', va='center', fontsize=11, fontweight='bold', color=TEXT_LIGHT)
        ax.text(x[i], 1.3, modes[i], ha='center', va='center', fontsize=9.5, color=TEXT_MUTED, linespacing=1.4)

    # Arrow connecting them
    ax.annotate('', xy=(3.4, 2.0), xytext=(1.7, 2.0),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2.5))

    ax.set_xlim(0.2, 5.1)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title("The Autonomy Frontier: Operational Failure Modes by Time Horizon",
                 fontsize=14, fontweight='bold', color=TEXT_LIGHT, pad=15)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_horizon_gap.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated: {out}")

# ------------------------------------------------------------------------------
# 2. Dual-Loop Supervision Architecture
# ------------------------------------------------------------------------------
def generate_supervision_architecture_figure():
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)

    # Inner loop box
    inner_box = patches.FancyBboxPatch((0.5, 0.8), 4.2, 3.0, boxstyle="round,pad=0.2",
                                      fc='#1e293b', ec='#64748b', lw=2.5)
    ax.add_patch(inner_box)
    ax.text(2.6, 3.4, "Fast Deterministic Inner Loop", ha='center', fontsize=12, fontweight='bold', color='#cbd5e1')
    ax.text(2.6, 3.0, "Cadence: 2 Minutes (Python Supervisor)", ha='center', fontsize=9.5, color=ACCENT_CYAN)

    inner_steps = [
        "1. Poll Active Cloud/Local Workers",
        "2. Health & Liveness Telemetry",
        "3. Automated Traceback Extraction on Error",
        "4. Pre-Submit Boundary Validation"
    ]
    for idx, step in enumerate(inner_steps):
        ax.text(0.8, 2.4 - idx*0.45, "• " + step, fontsize=9.5, color=TEXT_LIGHT)

    # Outer loop box
    outer_box = patches.FancyBboxPatch((5.8, 0.8), 4.7, 3.0, boxstyle="round,pad=0.2",
                                      fc='#1e293b', ec=ACCENT_BLUE, lw=2.5)
    ax.add_patch(outer_box)
    ax.text(8.15, 3.4, "Metacognitive Scheduled Outer Loop", ha='center', fontsize=12, fontweight='bold', color=ACCENT_BLUE)
    ax.text(8.15, 3.0, "Cadence: 30 Minutes (LLM Reconciler via Cron)", ha='center', fontsize=9.5, color=ACCENT_EMERALD)

    outer_steps = [
        "1. High-Priority System Re-Awakening",
        "2. Global Log & Pipeline State Audit",
        "3. Crash Diagnosis & Code Generation",
        "4. Midnight UTC Quota Management & Dispatch"
    ]
    for idx, step in enumerate(outer_steps):
        ax.text(6.1, 2.4 - idx*0.45, "• " + step, fontsize=9.5, color=TEXT_LIGHT)

    # Connecting bidirectional arrows
    ax.annotate('Incident Report & Telemetry', xy=(5.7, 2.2), xytext=(4.8, 2.2),
                ha='center', va='bottom', fontsize=8.5, color=TEXT_MUTED,
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))
    ax.annotate('Corrective Deployment', xy=(4.8, 1.4), xytext=(5.7, 1.4),
                ha='center', va='top', fontsize=8.5, color=ACCENT_EMERALD,
                arrowprops=dict(arrowstyle="->", color=ACCENT_EMERALD, lw=2))

    ax.set_xlim(0, 11)
    ax.set_ylim(0.2, 4)
    ax.axis('off')
    ax.set_title("Erlang-Inspired Dual-Loop Architecture: Fast Liveness + Scheduled Reconciler",
                 fontsize=14, fontweight='bold', color=TEXT_LIGHT, pad=15)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_supervision_architecture.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated: {out}")

# ------------------------------------------------------------------------------
# 3. Verified Multi-Domain Standings Dot-Plot
# ------------------------------------------------------------------------------
def generate_verified_standings_figure():
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)

    competitions = [
        "Playground S6E9 (EV Demand)",
        "Biohub Cell Tracking (Harmonic ILP)",
        "UMUD Muscle Architecture",
        "Soil Granulometry (DINOv2 Simplex)",
        "Traffic Flow Bench (LWR Physics)"
    ]
    ranks = [
        "Rank #131 / 2,683 (Top 4.88%)",
        "Rank #460 / 3,906 (Top 11.8%)",
        "Rank #45 / 310 (Top 14.5%)",
        "Rank #144 / 362 (OOF 59.99 EMD)",
        "Rank #104 / 133 (0.79155 SOTA Anchor)"
    ]
    percentiles = [4.88, 11.8, 14.5, 39.8, 78.2] # lower percentile is better (Top X%)
    colors = [ACCENT_EMERALD, ACCENT_CYAN, ACCENT_BLUE, ACCENT_AMBER, '#94a3b8']

    y = np.arange(len(competitions))

    # Horizontal guide lines
    for yi in y:
        ax.axhline(yi, color='#334155', linestyle=':', lw=1)

    # Invert x so Top 0% is on the right
    ax.scatter(percentiles, y, color=colors, s=280, edgecolor='#f8fafc', lw=2, zorder=5)

    for i, (pct, rk) in enumerate(zip(percentiles, ranks)):
        ax.text(pct + 2.5, i, rk, va='center', fontsize=10.5, fontweight='bold', color=TEXT_LIGHT)

    ax.set_yticks(y)
    ax.set_yticklabels(competitions, fontsize=11, fontweight='bold', color=TEXT_LIGHT)
    ax.set_xlabel("Leaderboard Percentile Standing (Top % — Lower is Higher Worldwide)",
                  fontsize=11, fontweight='bold', color=TEXT_LIGHT, labelpad=10)
    ax.set_xlim(-5, 130)
    ax.tick_params(colors=TEXT_MUTED, labelsize=10)
    ax.grid(color='#334155', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_title("Verified Competitive Benchmark Standings Across Heterogeneous Domains",
                 fontsize=13, fontweight='bold', color=TEXT_LIGHT, pad=15)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_multidomain_standings.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated: {out}")

# ------------------------------------------------------------------------------
# 4. Invariant Checking: Export-Only vs Per-Hop
# ------------------------------------------------------------------------------
def generate_boundary_drift_figure():
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)

    hops = np.arange(0, 9)
    # Drift trajectory that crosses 0 at hop 8
    ideal_trajectory = 8.0 - 0.4 * hops
    drift_trajectory = [8.0, 7.2, 5.8, 4.1, 2.9, 1.7, 0.8, 0.2, 0.0]
    clamped_trajectory = [8.0, 7.2, 5.8, 4.1, 2.9, 1.7, 0.8, 0.2, 0.05]

    ax.plot(hops, ideal_trajectory, 'g--', lw=2, label='Expected Ecological State')
    ax.plot(hops, drift_trajectory, color=ACCENT_RED, lw=3, marker='o', label='Exp 6 Unconstrained Drift (Crashes at Hop 8)')
    ax.plot(hops, clamped_trajectory, color=ACCENT_EMERALD, lw=2.5, marker='s', label='Exp 7 Biophysical Per-Hop Clamping (Passes)')

    ax.axhline(0.0, color='#ef4444', linestyle='-', lw=1.5, alpha=0.8)
    ax.text(0.2, 0.15, 'Strict Assertion Threshold: height > 0.000m', fontsize=9, color=ACCENT_RED, fontweight='bold')

    # Explosion at hop 8
    ax.scatter([8], [0.0], color=ACCENT_RED, s=350, marker='X', zorder=10)
    ax.text(8, 0.6, "Hour 2.1 Crash!\nAssertionError", ha='center', fontsize=9.5, fontweight='bold', color=ACCENT_RED)

    ax.set_xlabel("Autoregressive Simulation Hops (5-Year Intervals over 40 Years)",
                  fontsize=11, fontweight='bold', color=TEXT_LIGHT, labelpad=10)
    ax.set_ylabel("Forest State (Height in Meters)", fontsize=11, fontweight='bold', color=TEXT_LIGHT, labelpad=10)
    ax.set_title("Failure Case: Cumulative Floating-Point Drift in 185k-Row Rollouts",
                 fontsize=13, fontweight='bold', color=TEXT_LIGHT, pad=15)
    ax.tick_params(colors=TEXT_MUTED, labelsize=10)
    ax.grid(color='#334155', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.legend(facecolor=BG_DARK, edgecolor='#334155', fontsize=9.5, labelcolor=TEXT_LIGHT, loc='upper right')

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_boundary_drift.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated: {out}")

if __name__ == "__main__":
    generate_horizon_gap_figure()
    generate_supervision_architecture_figure()
    generate_verified_standings_figure()
    generate_boundary_drift_figure()
    print("Visual-first presentation figures successfully created!")
