"""
Delegate Biohub SOTA Architecture to Claude Code (Opus 5.5) via Autobot Bridge.
Focus: Overcome the 0.953 ceiling and push past 0.959 into Silver/Gold territory.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

def main():
    prompt = (
        "You are Claude Code (Opus 5.5), acting as the Lead Bio-Imaging & Tracking Optimization Scientist for Autobot.\n\n"
        "=== MISSION ===\n"
        "In Kaggle 'Biohub - Cell Tracking During Development', our current public score is 0.953 (Exp 4, Rank #525/3935).\n"
        "Leaderboard Analysis:\n"
        "  - Total Teams: 3,935\n"
        "  - Top 10 (Gold): 0.969\n"
        "  - Top 5% (Silver): 0.955 (Rank 196)\n"
        "  - Top 10% (Bronze): 0.953 (Rank 393)\n"
        "  - Score 0.959 jumps straight to Rank 69 (Top 1.7% - solid Silver, knocking on Gold)!\n"
        "Deadline is in 3 days. We have 4 submissions remaining today, and 5 more tomorrow.\n\n"
        "=== CRITICAL MATHEMATICAL & EMPIRICAL DISCOVERIES ===\n"
        "1. THE ILP DIVISION WEIGHT BREAKTHROUGH (Roger's Proof):\n"
        "   - In tracksdata/solvers/_ilp_solver.py, the ILP objective cost for an outgoing fork is:\n"
        "     Delta = -p2 + division_weight - appearance_weight\n"
        "   - Because appearance_weight = 0.0, the ILP creates a mitotic fork if and only if: p2 >= division_weight.\n"
        "   - In the public notebooks (and our Exp 4/5), BIOHUB_ILP_DIVISION_WEIGHT was set to 1.2.\n"
        "   - Since edge probabilities p2 <= 1.0 ALWAYS, ANY division_weight >= 1.0 MAKES AN ILP FORK ARITHMETICALLY IMPOSSIBLE!\n"
        "   - Thus, previous runs had ZERO divisions solved globally by the ILP solver; all divisions came only from ad-hoc post-processing.\n"
        "   - FIX: Setting BIOHUB_ILP_DIVISION_WEIGHT to 0.75 - 0.82 enables the ILP solver to discover true biological mitotic divisions globally.\n\n"
        "2. EXP 4 (0.953) vs EXP 5 (0.949) ROOT CAUSE:\n"
        "   - Exp 4 used SAFE_DIV_MAX_UM=9.0, SAFE_DIV_DIVERGE_UM=2.25, and DEEPCENTER_SAFE_DIV_THRESHOLD=0.25 -> 0.953 LB.\n"
        "   - Exp 5 loosened SAFE_DIV_MAX_UM to 10.0, lowered DIVERGE_UM to 1.75, and dropped DEEPCENTER_SAFE_DIV_THRESHOLD to 0.15.\n"
        "     This allowed false-positive divisions to hallucinate, fragmenting tracks and dropping score to 0.949.\n"
        "   - CONCLUSION: Mitosis requires strict geometric conservation. Daughters must diverge (>= 2.25 um) and have bilateral sister symmetry (<= 0.60).\n\n"
        "3. FAST-SHIP RUNTIME HARDENING:\n"
        "   - Set BIOHUB_VALIDATOR_ENABLE = '0' so the submission does not run the 90-minute offline validator sweep on hidden test sets.\n"
        "   - Set BIOHUB_ILP_TIMEOUT_S = '1200' and BIOHUB_REPAIR_DEADLINE_S = '27000' to guarantee zero timeouts during private rerun.\n\n"
        "=== YOUR TASK ===\n"
        "Create Experiment 6: 'competitions/biohub_cell_tracking/exp6_harmonic_multihop_sota/':\n"
        "1. Create `competitions/biohub_cell_tracking/exp6_harmonic_multihop_sota/kernel-metadata.json`:\n"
        "   - id: 'daltongabrielomondi/autobot-biohub-exp6-harmonic-multihop-sota'\n"
        "   - title: 'Autobot Biohub Exp6 Harmonic MultiHop SOTA'\n"
        "   - code_file: 'harmonic_multihop_sota.ipynb'\n"
        "   - language: 'python'\n"
        "   - kernel_type: 'notebook'\n"
        "   - is_private: 'true'\n"
        "   - enable_gpu: 'true'\n"
        "   - enable_internet: 'false'\n"
        "   - competition_sources: ['biohub-cell-tracking-during-development']\n"
        "   - dataset_sources: [\n"
        "       'pilkwang/biohub-tracking-support-pack-50ep-v1',\n"
        "       'pilkwang/biohub-deepcenter-unet3d-center-prior-v1',\n"
        "       'pilkwang/biohub-temporal-unet3d-seed314159-v1',\n"
        "       'anvithpothula/biohub-v1284-head-s075'\n"
        "     ]\n"
        "2. Build `competitions/biohub_cell_tracking/exp6_harmonic_multihop_sota/harmonic_multihop_sota.ipynb`:\n"
        "   - Start from the verified 0.953 codebase `competitions/biohub_cell_tracking/exp4_harmonic_subvoxel_sota/harmonic_fusion_sota.ipynb`.\n"
        "   - Apply the ILP division fix: BIOHUB_ILP_DIVISION_WEIGHT = '0.78' (empirically unlocks high-confidence global solver divisions).\n"
        "   - Apply calibrated cytokinesis constraints: SAFE_DIV_MAX_UM = '9.0', SAFE_DIV_SISTER_MAX_UM = '14.0', SAFE_DIV_SISTER_SYMMETRY_TAU = '0.6', SAFE_DIV_DIVERGE_UM = '2.25', DEEPCENTER_SAFE_DIV_THRESHOLD = '0.25'.\n"
        "   - Preserve sub-voxel continuous refinement head (v1284_head.pt) for spatial accuracy.\n"
        "   - Retain motion relinking (tight=5.5 um, learned_bonus=1.0) and gap closing (gap=3, step=5.0 um).\n"
        "   - Ensure BIOHUB_VALIDATOR_ENABLE = '0' for ultra-fast, robust 16-minute execution with zero timeouts.\n"
        "3. Verify that the generated notebook JSON is 100% valid and write a clear, quantified design log in:\n"
        "   `competitions/biohub_cell_tracking/exp6_harmonic_multihop_sota/DESIGN_REPORT.md`.\n\n"
        "Execute this task thoroughly and confirm when the notebook and metadata are staged and ready for Kaggle dispatch."
    )

    print("Delegating Biohub Exp 6 Architecture & Implementation to Claude Code (Opus 5.5)...")
    res = claude_code_bridge.run_headless(
        prompt,
        cwd=str(REPO_ROOT),
        permission_mode="acceptEdits",
        timeout=600
    )

    if res.get("ok"):
        data = res.get("data")
        result_text = data.get("result", "") if isinstance(data, dict) else str(data)
        out_log = Path("competitions/biohub_cell_tracking/CLAUDE_BIOHUB_EXP6_REPORT.md")
        out_log.write_text(result_text, encoding="utf-8")
        print(f"SUCCESS: Claude Code completed the task. Log written to {out_log}")
        print("\n" + "=" * 60)
        print("CLAUDE SUMMARY:")
        print("=" * 60)
        print(result_text[:1500] + "\n...")
    else:
        print("Claude execution failed:", res.get("error"))

if __name__ == "__main__":
    main()
