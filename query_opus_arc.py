"""
Dispatch Competition Diagnostic Report to Claude Opus 5.5
Competition: ARC Prize 2026 - ARC-AGI-3 Challenge
Goal: Consolidate strategies to advance from 4.04% to >= 7.0% - 9.0% (Solid Silver / Top 10)
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

PROMPT = """
You are Claude Opus 5.5, acting as the Principal AGI & Reasoning Architect for Autobot.
We need your highest-level neural-symbolic, test-time compute, and search orchestration reasoning for the ARC Prize 2026 (ARC-AGI-3 Challenge).

=== COMPETITION CONTEXT & LEADERBOARD TOPOLOGY ===
- Competition: ARC Prize 2026 - ARC-AGI-3 (Abstraction and Reasoning Corpus)
- Prize Pool: $850,000 USD | Format: Code Competition on Kaggle
- Hardware: Dedicated Nvidia RTX Pro 6000 Ada (48GB VRAM), up to 9-12 hours per submission.
- Total Teams: 3,377
- Leaderboard Distribution:
  * Rank 1: 27.29% (Tufa Labs)
  * Rank 10: 9.96% (rellik13)
  * Rank 12: 8.62% (Keith Tyser)
  * Top 5% (Silver): ~5.5% - 6.5%
  * Top 10% (Bronze Cutoff): 3.65% (Rank 337)
  * CURRENT AUTOBOT STATUS: Score 4.04% (Exp 3, Ref 56610419).
- GOAL: Advance from 4.04% into >= 7.0% - 9.0% (Securing high Silver / knocking on Top 10).

=== OUR EMPIRICAL PROGRESSION & ENGINEERING STACK ===
1. Exp 1: Baseline Duck Qwen-27B FP8. Encountered initialization and memory errors.
2. Exp 2 (LB 1.54%): Duck Qwen-27B on RTX 6000 Ada.
   - Identified severe agent loop defects:
     * Context window flooding with high-res base64 grid images (ate 26% of context).
     * Memory wipeout on game_over retries (erased learned world model physics).
     * HUD thrashing (misinterpreted countdown bar changes as puzzle progress).
     * ACTION7 (undo) was advertised but unmapped.
3. Exp 3 (LB 4.04%) [CURRENT CHAMPION]: RadixArk Qwen 3.8 FlashNext NVFP4 MTP + AgentFix SOTA.
   - Mounted `keithtyser/qwen3-8-flash-next-nvfp4` with 3-token NextN MTP speculative decoding (`TAAF_VLLM_MTP_TOKENS: 3`), providing 3-4x throughput acceleration.
   - Implemented AgentFix suite:
     * AGENTFIX_IMAGES: stripped redundant history images, bounding token overhead.
     * AGENTFIX_MEMORY: persistent world model across retries.
     * Expanded analyzer timeout to 1200s (20 mins per puzzle domain).
   - Scored 4.04% (surpassing the 3.65% Top 10% milestone).

=== RESEARCH ARTIFACTS IN CODEBASE ===
- `research_sota/keithtyser_nvfp4/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb` (8.62% benchmark)
- `research_sota/scottlegrand/taaf-flashnext-sheetu12b-0922.ipynb` (Sheetu 12B / TAAF architecture)
- Environment interaction: communicates via local Kaggle Arcade gateway (`http://gateway:8001/`).

=== YOUR MISSION ===
Do NOT write code yet. Provide a master-level technical and neuro-symbolic reasoning consultation:
1. Deconstruct the 4.04% vs 8.62% / 9.96% Gap:
   - What are Keith Tyser and Scott Le Grand doing in `taaf-flashnext-sheetu12b` and `duck-qwen3-8-flash-next-nvfp4` that we are missing in Exp 3?
   - Is it search strategy (MCTS / branch-and-bound / beam search with rollouts vs greedy single-trajectory)?
   - Is it prompt formulation (symbolic coordinate diffs, grid color-frequency invariance, goal state projection)?
   - Is it action verification (testing candidate actions against learned world model rules before committing to the arcade environment)?
2. High-Alpha Architectural Breakthroughs:
   - Avenue 1: Test-Time Compute Allocation (How to distribute the 9-12 hour budget: fast-exit simple puzzles in 2 minutes to allocate 45+ minutes to solvable multi-room environments).
   - Avenue 2: Symbolic Code Execution & Transformation Primitives (Can the LLM synthesize Python/NumPy grid transforms or cellular automata rules as tools?).
   - Avenue 3: Memory & Multi-Attempt Search (Best-of-N trajectory sampling, state rollback, dead-end pruning).
3. Recommended High-Conviction Experiment Specification for Exp 4:
   - Detail the exact configuration, prompt patches, search parameters, and model integration to take Autobot from 4.04% to >= 7.5% - 9.0%.
"""

def main():
    print("Pushing ARC-AGI-3 Diagnostic Report to Claude Opus 5.5...")
    res = claude_code_bridge.run_headless(
        PROMPT,
        cwd=str(REPO_ROOT),
        permission_mode="plan",
        timeout=300
    )
    if res.get("ok"):
        data = res.get("data")
        result = data.get("result", "") if isinstance(data, dict) else str(data)
        out_file = REPO_ROOT / "competitions" / "arc_prize_2026" / "OPUS_5_5_STRATEGY_REPORT.md"
        out_file.write_text(result, encoding="utf-8")
        print(f"SUCCESS: Report saved to {out_file}")
        print("\n" + "="*80)
        print("CLAUDE OPUS 5.5 ARC-AGI-3 STRATEGY PREVIEW:")
        print("="*80)
        print(result[:2000] + "\n...")
    else:
        print("Execution failed:", res.get("error"))

if __name__ == "__main__":
    main()
