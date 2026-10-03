"""
Dispatch Competition Diagnostic Report to Claude Opus 5.5
Competition: Biohub - Cell Tracking During Development
Goal: Consolidate strategies to break 0.954 tie wall and reach >= 0.958 (Top 3% Silver)
"""

import sys
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

PROMPT = """
You are Claude Opus 5.5, acting as the Principal Bio-Imaging & Quantitative Tracking Architect for Autobot.
We need your highest-level mathematical, biological, and algorithmic reasoning to formulate our final breakthrough strategy.

=== COMPETITION CONTEXT & URGENCY ===
- Competition: Kaggle Biohub - Cell Tracking During Development (4D zebrafish embryonic light-sheet microscopy).
- Deadline: CLOSES IN ~24 HOURS!
- Total Teams: 3,955
- Leaderboard Distribution:
  * Top 10 (Gold): 0.964 - 0.969
  * Top 3% (Deep Silver): 0.957 - 0.958 (Rank 118)
  * Top 5% (Silver Cutoff): 0.955 (Rank 197)
  * Top 10% (Bronze Cutoff): 0.953 (Rank 395)
  * CURRENT AUTOBOT STATUS: Score 0.954 (Rank 276 - 386).
- CRITICAL VULNERABILITY: 110 teams are currently TIED at exactly 0.954! If 10 more teams submit or private shakeup occurs, our 0.954 will be shoved completely out of Bronze! We MUST break this tie cluster and reach >= 0.958 for a distinctive, unshakeable Silver medal.

=== WHAT WE HAVE TRIED & PRECISE EMPIRICAL LESSONS ===
1. Exp 1 (LB 0.947): Baseline DeepCenter 3D U-Net + Harmonic Probability ILP.
   - Dual-seed temporal forward/backward harmonic association.
   - ILP edge penaltyDelta = -p2 + division_weight - appearance_weight.

2. Exp 4 (LB 0.953): Harmonic V3 + V1284 Sub-Voxel Neural Regression Head.
   - Added v1284_head continuous centroid refinement (sub-pixel shift).
   - Fast-ship mode (16-min execution, BIOHUB_VALIDATOR_ENABLE=0).
   - Jumped from 0.947 to 0.953!

3. Exp 5 (LB 0.949) [FAILURE]: Loosened Division Thresholds.
   - Loosened SAFE_DIV_MAX_UM from 9.0 to 10.0 um.
   - Lowered SAFE_DIV_DIVERGE_UM from 2.25 to 1.75 um.
   - Lowered DEEPCENTER_SAFE_DIV_THRESHOLD from 0.25 to 0.15.
   - RESULT: Dropped to 0.949. Why: Spurious, hallucinated divisions fragmented real biological tracks and corrupted the lineage tree.

4. Exp 6 (LB 0.954) [CURRENT CHAMPION]: Global ILP Division Optimization.
   - Mathematical proof: In tracksdata/solvers/_ilp_solver.py, division_weight was 1.2 in public baseline. Because edge probability p2 <= 1.0, an ILP fork was mathematically IMPOSSIBLE (p2 >= division_weight never met). Public notebooks had ZERO ILP divisions; all divisions came from post-processing!
   - We set BIOHUB_ILP_DIVISION_WEIGHT = 0.78, unlocking true ILP divisions.
   - Result: 0.954 (Our PB, but caught in the 110-team tie).

5. Exp 7 (LB 0.930) [FAILURE]: Post-ILP Harmonic Geometry Filter.
   - Tried BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER = 1.
   - RESULT: Severe drop to 0.930. Pruned too aggressively and destroyed valid mitotic events.

6. Exp 8 (LB 0.912) [FAILURE]: Iterative Flow Tracking.
   - Tried multi-hop iterative optical flow tracking across time steps.
   - RESULT: Dropped to 0.912 due to error accumulation across large cell migrations.

=== AVAILABLE ASSETS & CONSTRAINTS ===
- Pretrained weights mounted in Kaggle datasets:
  * `pilkwang/biohub-tracking-support-pack-50ep-v1` (UNet 50-epoch weights, tracking wheels)
  * `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`
  * `pilkwang/biohub-temporal-unet3d-seed314159-v1`
  * `anvithpothula/biohub-v1284-head-s075` (Sub-voxel regression head)
- Evaluation Metric: Tracking Metric (TRA score combining detection F1 + edge association + division recall).
- Runtime Budget: Kaggle GPU (T4 / P100), offline (enable_internet: false), <= 9 hours (our current pipeline runs in ~25-35 minutes).

=== YOUR MISSION ===
Do NOT write code yet. Provide a master-level technical and mathematical consultation:
1. Leaderboard Tie Deconstruction: Why are 110 teams stuck at 0.954? What exact component in the public code is the bottleneck holding everyone at 0.954?
2. High-Alpha Breakthrough Strategies: Propose 3 distinct, high-conviction avenues that can push the score from 0.954 past 0.958 - 0.962:
   - Avenue A: Precision Centroid Detection & Spatial Thresholding (Can we ensemble detection masks or use test-time augmentations without timeout?).
   - Avenue B: Mathematical ILP Graph Formulation & Cost Matrix Calibration (Appearance, disappearance, transition weights, tracklet gap closing).
   - Avenue C: Morphological / Lineage Post-Processing (Bilateral cytokinesis sister angle/volume constraints, short-track rescue vs false-positive leaf pruning).
3. Risk & Shakeup Assessment: Which of these strategies are robust to the private test set, and which risk private LB overfitting?
4. Concrete Recommended Experiment: Synthesize into a single high-conviction experiment specification with exact parameter values and architecture choices.
"""

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

def main():
    print("Pushing Biohub Diagnostic Report to Claude Opus 5.5 via claude_code_bridge...")
    res = claude_code_bridge.run_headless(
        PROMPT,
        cwd=str(REPO_ROOT),
        permission_mode="plan",
        timeout=300
    )
    if res.get("ok"):
        data = res.get("data")
        result = data.get("result", "") if isinstance(data, dict) else str(data)
        out_file = REPO_ROOT / "competitions" / "biohub_cell_tracking" / "OPUS_5_5_STRATEGY_REPORT.md"
        out_file.write_text(result, encoding="utf-8")
        print(f"SUCCESS: Report saved to {out_file}")
        print("\n" + "="*80)
        print("CLAUDE OPUS 5.5 BIOHUB STRATEGY PREVIEW:")
        print("="*80)
        print(result[:2000] + "\n...")
    else:
        print("Execution failed:", res.get("error"))

if __name__ == "__main__":
    main()
