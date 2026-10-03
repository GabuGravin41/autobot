"""
Dispatch Competition Diagnostic Report to Claude Opus 5.5
Competition: IEEE Big Data Cup - Traffic Flow Prediction
Goal: Consolidate strategies to advance from 0.80939 to >= 0.88382 (Top 10)
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

PROMPT = """
You are Claude Opus 5.5, acting as the Lead Transportation AI & Spatiotemporal Physics Architect for Autobot.
We need your highest-level hydrodynamic traffic theory, graph neural network, and gradient boosting insights to formulate our breakthrough strategy for the IEEE Big Data Traffic Flow Challenge.

=== COMPETITION CONTEXT & LEADERBOARD TARGET ===
- Competition: 2026 IEEE Big Data Cup - Traffic Flow Benchmark
- Metric: Flow Correlation / Explained Variance / R2 / WAPE on spatiotemporal traffic network during incident shockwaves.
- Total Teams: 201
- Leaderboard Distribution:
  * Rank 1: 0.92406
  * Rank 3: 0.91665
  * Rank 5: 0.90043
  * Rank 10: 0.88382
  * CURRENT AUTOBOT STATUS: Score 0.80939 (Exp 11, Rank 86 / 201).
- GOAL: Advance from 0.80939 into the Top 10 (>= 0.88382).

=== EMPIRICAL PROGRESSION & WHAT WE HAVE LEARNED ===
1. Exp 1 (LB 0.53880): Physical LWR + NNLS Demand Reconstruction Baseline.
   - Pure kinematic wave model with static O-D matrix. Under-predicted congestion queue spillbacks.

2. Exp 4-6 (LB 0.665 -> 0.690): Spatiotemporal Shockwaves & Delayed Onset Dynamics.
   - Identified that bottleneck queues don't form instantaneously; queue propagation has a physical delay onset (T >= 20-25 min) as vehicle density reaches critical density k_c.

3. Exp 7-8 (LB 0.737 -> 0.79155): Monotonic Surface & Physical Link Conservation.
   - Enforced conservation of vehicles: inflow - outflow = d(accumulation)/dt across interconnected link nodes.

4. Exp 11 (LB 0.80939) [CURRENT PB]: Physics-Consistent Link Conservation + Calibrated LightGBM Booster.
   - Extracted physics residuals and trained a LightGBM booster on link topology features, upstream sensor lags, and flow ratios.
   - Reached 0.80939, but still trailing the 0.88 - 0.92 leaders!

=== WHAT ARE THE TOP 10 LEADERS (0.88 - 0.92) DOING DIFFERENTLY? ===
1. Graph Topology & Spatial Adjacency: Are leaders using Spatio-Temporal Graph Convolutional Networks (ST-GCN / DCRNN / ASTGCN) or spatial diffusion along directed traffic flow?
2. Upstream/Downstream Cross-Correlation & Wave Speed: Physical shockwaves travel backward at characteristic speed w ~ -15 to -20 km/h. How can time-lagged cross-correlation features between consecutive sensor stations capture this wave?
3. Capacity Drop & Hysteresis: At bottleneck breakdowns, discharge flow drops by 10-15% (capacity drop phenomenon). Does linear/LGBM model fail because it misses the non-linear catastrophe bifurcations of fundamental diagrams?
4. Multi-Scale Temporal Features: Peak hours vs off-peak, day-of-week periodicities, and rolling window standard deviation of speeds/densities.

=== YOUR MISSION ===
Do NOT write code yet. Provide a master-level technical and transport-modeling consultation:
1. Deconstruct the Gap: What is the fundamental difference between our 0.809 physics+LGBM pipeline and the 0.883+ Top 10 systems?
2. High-Alpha Architectural Breakthroughs:
   - Avenue 1: Spatiotemporal Wave Alignment (Directional upstream/downstream lag matrices).
   - Avenue 2: Physics-Informed Neural Network (PINN) or Graph Diffusion vs Pure Boosting.
   - Avenue 3: Target Transformation & Residual Decomposition (Predicting free-flow ratio v/v_free or density accumulation instead of raw count).
3. Recommended High-Conviction Experiment Specification: Detail the exact feature engineering pipeline, spatial graph formulation, loss function, and model architecture to bridge from 0.809 to >= 0.885.
"""

def main():
    print("Pushing IEEE Traffic Flow Diagnostic Report to Claude Opus 5.5...")
    res = claude_code_bridge.run_headless(
        PROMPT,
        cwd=str(REPO_ROOT),
        permission_mode="plan",
        timeout=300
    )
    if res.get("ok"):
        data = res.get("data")
        result = data.get("result", "") if isinstance(data, dict) else str(data)
        out_file = REPO_ROOT / "competitions" / "ieee_traffic_flow" / "OPUS_5_5_STRATEGY_REPORT.md"
        out_file.write_text(result, encoding="utf-8")
        print(f"SUCCESS: Report saved to {out_file}")
        print("\n" + "="*80)
        print("CLAUDE OPUS 5.5 IEEE TRAFFIC STRATEGY PREVIEW:")
        print("="*80)
        print(result[:2000] + "\n...")
    else:
        print("Execution failed:", res.get("error"))

if __name__ == "__main__":
    main()
