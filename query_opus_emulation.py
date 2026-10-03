"""
Dispatch Competition Diagnostic Report to Claude Opus 5.5
Competition: IEEE Big Data Cup - AI Emulation Challenge
Goal: Consolidate strategies to advance from 0.205 to <= 0.072 (Top 10)
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

PROMPT = """
You are Claude Opus 5.5, acting as the Lead Earth System Modeling & Biophysical Machine Learning Scientist for Autobot.
We need your highest-level climate emulation, transformer sequence modeling, and biophysical dynamics reasoning for the IEEE Big Data AI Emulation Challenge.

=== COMPETITION CONTEXT & LEADERBOARD TARGET ===
- Competition: IEEE Big Data Cup 2026 - AI Emulation Challenge
- Goal: Emulate multi-decadal global terrestrial carbon dynamics (Net Primary Productivity, Heterotrophic Respiration, Soil Organic Carbon, Biomass Pools) across diverse biomes under SSP climate scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5).
- Metric: Normalized RMSE / Relative Error across multi-decadal temporal rollouts (LOWER is better!).
- Total Teams: 97
- Leaderboard Distribution:
  * Rank 1: 0.046
  * Rank 3: 0.055
  * Rank 5: 0.062
  * Rank 10: 0.072
  * Rank 15: 0.078
  * CURRENT AUTOBOT STATUS: Score 0.205 (Exp 10, Rank 74 / 97).
- TARGET: Advance from 0.205 into the Top 10 (<= 0.072).

=== EMPIRICAL PROGRESSION & WHAT WE HAVE LEARNED ===
1. Exp 1 (LB 0.274): LightGBM Baseline with Climate Drivers.
   - Used raw precipitation, 2m temperature, downward solar radiation, vapor pressure deficit.
   - Severe error accumulation when predicting decadal horizons.

2. Exp 4-5 (LB 0.270 -> 0.222): Autoregressive Rollout & State Transition Function.
   - Stepping at smaller time intervals (STEP=10 years) reduced compound drift.

3. Exp 9 (LB 0.207): Biophysical Clamped Linear Rollout + Warming Anomalies.
   - Normalizing climate variables by historical baseline (computing climate anomalies Delta_T, Delta_P) prevented out-of-distribution saturation.

4. Exp 10 (LB 0.205) [CURRENT PB]: Deep Carbon Sequence Emulator.
   - Neural temporal architecture, achieving 0.205 on public test set.
   - But Top 10 teams are at 0.046 - 0.072 (a factor of 3x-4x lower error!).

=== WHAT ARE THE TOP 10 LEADERS (0.046 - 0.072) DOING? ===
1. Carbon Mass Conservation & Inflow/Outflow ODE Formulation:
   - Terrestrial carbon obeys dC_pool/dt = Input_Flux (NPP) - Decay_Rate * C_pool.
   - Leaders likely integrate a physics-guided neural ODE / state-space model where carbon cannot be spontaneously created or destroyed.
2. Cross-Site Global Embeddings & Biome Clustering:
   - Sites share soil taxonomy, vegetation functional types (PFT), and climate envelopes.
   - Spatial cross-attention or graph message passing across ecologically similar sites.
3. Multi-Decadal Cumulative Temperature / Drought Indices:
   - SPEI (Standardized Precipitation-Evapotranspiration Index), growing degree days (GDD), soil moisture memory.
4. Sequence-to-Sequence Direct Multi-Horizon vs Autoregressive:
   - Autoregressive rollouts suffer from compounding exposure bias. Does a direct Seq2Seq or Temporal Fusion Transformer (TFT) with teacher forcing or non-autoregressive decoding avoid this?

=== YOUR MISSION ===
Do NOT write code yet. Provide a master-level technical and climate-modeling consultation:
1. Deconstruct the 0.205 vs 0.072 Gap: Why is pure sequence/boosting drifting to 0.205 while leaders achieve 0.046 - 0.072? What is the missing structural inductive bias?
2. High-Alpha Architectural Breakthroughs:
   - Avenue 1: Physics-Guided Mass Balance Conservation (Constrained ODE solver vs unconstrained regression).
   - Avenue 2: Climate Anomaly & Bioclimatic Feature Engineering (Biome clustering, drought stress indices).
   - Avenue 3: Model Architecture (Temporal Fusion Transformer, State-Space Mamba/S4, or Ensembled GBDT with exact ODE integration).
3. Recommended High-Conviction Experiment Specification: Detail the exact model architecture, training objective, feature representation, and inference rollout procedure.
"""

def main():
    print("Pushing IEEE AI Emulation Diagnostic Report to Claude Opus 5.5...")
    res = claude_code_bridge.run_headless(
        PROMPT,
        cwd=str(REPO_ROOT),
        permission_mode="plan",
        timeout=300
    )
    if res.get("ok"):
        data = res.get("data")
        result = data.get("result", "") if isinstance(data, dict) else str(data)
        out_file = REPO_ROOT / "competitions" / "ieee_ai_emulation" / "OPUS_5_5_STRATEGY_REPORT.md"
        out_file.write_text(result, encoding="utf-8")
        print(f"SUCCESS: Report saved to {out_file}")
        print("\n" + "="*80)
        print("CLAUDE OPUS 5.5 IEEE EMULATION STRATEGY PREVIEW:")
        print("="*80)
        print(result[:2000] + "\n...")
    else:
        print("Execution failed:", res.get("error"))

if __name__ == "__main__":
    main()
