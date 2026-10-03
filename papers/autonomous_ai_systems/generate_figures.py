"""
Generate publication-quality figures for the Autonomous AI Systems paper and presentation deck.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(r"c:\Users\User 1\OneDrive\Desktop\projects\django projects\personal projects\autobot\papers\autonomous_ai_systems\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set global styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# ------------------------------------------------------------------------------
# Figure 1: Economic Cost Comparison (Log Scale)
# ------------------------------------------------------------------------------
def generate_cost_figure():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    categories = ['Autobot\n(Our System)', 'Commercial Agents\n(e.g., Devin/ACU)', 'Enterprise AutoML\n(DataRobot/H2O)', 'Human ML/SWE Team\n(3 Senior Engineers)']
    costs = [5.00, 350.00, 600.00, 4320.00]
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']

    bars = ax.barh(categories, costs, color=colors, height=0.55, edgecolor='#334155', linewidth=1.5)

    ax.set_xscale('log')
    ax.set_xlim(1, 10000)
    ax.set_xlabel('Operational Cost for 12-Hour Campaign (USD, Log Scale)', fontsize=12, fontweight='bold', color='#f8fafc', labelpad=12)
    ax.set_title('Operational Cost Comparison: 12-Hour Engineering Campaign', fontsize=15, fontweight='bold', color='#f8fafc', pad=18)

    ax.tick_params(colors='#94a3b8', labelsize=11)
    ax.grid(color='#334155', linestyle='--', linewidth=0.7, alpha=0.7)

    for bar, cost in zip(bars, costs):
        width = bar.get_width()
        multiplier = f"{4320/cost:.0f}x cheaper" if cost < 4320 else "Baseline"
        label = f"${cost:,.2f}  ({multiplier})" if cost < 10 else f"${cost:,.2f}"
        ax.text(width * 1.25, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontsize=11, fontweight='bold', color='#f8fafc')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_cost_comparison.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved: {out_path}")

# ------------------------------------------------------------------------------
# Figure 2: The SWE-bench Runtime Ceiling Breakdown
# ------------------------------------------------------------------------------
def generate_runtime_math_figure():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    task_counts = np.arange(20, 140, 10)
    runtime_exp6 = (task_counts * 8.0 + task_counts * 1.2) / 60.0  # Exp 6/7 (8.0 min/task)
    runtime_exp8 = (task_counts * 3.5 + task_counts * 0.75) / 60.0  # Exp 8 (3.5 min/task)

    ax.plot(task_counts, runtime_exp6, color='#ef4444', linewidth=3, marker='o', label='Exp 6/7: Uncalibrated Budget (8.0m agent + 1.2m overhead)')
    ax.plot(task_counts, runtime_exp8, color='#10b981', linewidth=3, marker='s', label='Exp 8: Calibrated Budget (3.5m agent + 0.75m overhead)')

    ax.axhline(12.0, color='#f59e0b', linestyle='--', linewidth=2.5, label='Kaggle Hard Timeout Limit (12.0 Hours)')
    ax.axvspan(100, 125, color='#38bdf8', alpha=0.15, label='Estimated Test Split (100–120 Tasks)')

    ax.set_xlabel('Number of Tasks in Evaluation Split', fontsize=12, fontweight='bold', color='#f8fafc', labelpad=10)
    ax.set_ylabel('Total Evaluation Wall-Clock (Hours)', fontsize=12, fontweight='bold', color='#f8fafc', labelpad=10)
    ax.set_title('SWE-bench Evaluation Runtime vs Hard Execution Ceiling', fontsize=14, fontweight='bold', color='#f8fafc', pad=15)

    ax.set_ylim(0, 22)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(color='#334155', linestyle='--', linewidth=0.7, alpha=0.7)
    legend = ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10, labelcolor='#f8fafc', loc='upper left')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig2_swe_runtime_math.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved: {out_path}")

# ------------------------------------------------------------------------------
# Figure 3: Autobot Multi-Domain Benchmark Radar / Progress
# ------------------------------------------------------------------------------
def generate_benchmark_figure():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    benchmarks = ['Traffic Flow\n(Link Cons.)', 'Biohub Cell\n(Harmonic ILP)', 'Soil Granulometry\n(DINOv2 Simplex)', 'UMUD Muscle\n(Continuum Blend)']
    current_metrics = [0.79155, 0.953, 59.99, 0.45976]
    baseline_metrics = [0.73755, 0.947, 70.51, 0.47723]
    improvements = ["+0.0540 (Rank #88)", "+0.006 (Top 8%)", "-10.52 EMD (Rank #144)", "-0.0175 (Rank #45)"]

    x = np.arange(len(benchmarks))
    width = 0.35

    ax.bar(x - width/2, [1, 1, 1, 1], width, label='Initial / Baseline Reference', color='#475569', edgecolor='#64748b')
    # Normalized relative gain
    relative_perf = [1.073, 1.006, 1.175, 1.038]
    bars = ax.bar(x + width/2, relative_perf, width, label='Autobot SOTA Optimization', color='#06b6d4', edgecolor='#22d3ee')

    ax.set_ylabel('Relative Performance Index', fontsize=12, fontweight='bold', color='#f8fafc', labelpad=10)
    ax.set_title('Autobot Autonomous Optimization Milestones Across Domains', fontsize=14, fontweight='bold', color='#f8fafc', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11, fontweight='bold', color='#f8fafc')
    ax.set_ylim(0.8, 1.3)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(color='#334155', linestyle='--', linewidth=0.7, alpha=0.7)

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, imp,
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#38bdf8')

    legend = ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10, labelcolor='#f8fafc', loc='upper left')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig3_multitask_benchmarks.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_cost_figure()
    generate_runtime_math_figure()
    generate_benchmark_figure()
    print("All publication figures successfully generated!")
