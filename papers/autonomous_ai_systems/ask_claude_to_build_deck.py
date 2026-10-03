"""
Delegate presentation deck generation directly to Claude Code (Opus 5.5).
Claude will write papers/autonomous_ai_systems/generate_deck.py and compile the PPTX.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

def main():
    prompt = (
        "You are Claude Code (Opus 5.5), acting as the lead Presentation Architect for Autobot.\n"
        "The user specifically instructed: 'the ppt should be less words and more figures something like "
        "scientific presentations or pitches so that it is easy to read, and present and not too cluttered with prose'.\n\n"
        "Your task:\n"
        "Write or rewrite `papers/autonomous_ai_systems/generate_deck.py` using `python-pptx` to compile "
        "`papers/autonomous_ai_systems/Autonomous_AI_Systems_Presentation.pptx`.\n\n"
        "Design Rules:\n"
        "1. VISUAL-FIRST: Every slide must be dominated by a graphic, chart, or visual schematic (occupying 65-75% of canvas).\n"
        "2. MINIMAL PROSE: Use bold KPI badges, large metric callouts, and punchy 1-line takeaway chips. ZERO walls of text.\n"
        "3. AESTHETICS: 16:9 widescreen (13.333 x 7.5 in), dark navy theme (RGB 15, 23, 42), slate cards (RGB 30, 41, 59), "
        "cyan (RGB 6, 182, 212), emerald (RGB 16, 185, 129), amber (RGB 245, 158, 11), blue (RGB 59, 130, 246).\n"
        "4. EMBED EXISTING HIGH-RES FIGURES from `papers/autonomous_ai_systems/figures/`:\n"
        "   - `fig_horizon_gap.png` (Slide 2: Horizon scale)\n"
        "   - `fig_supervision_architecture.png` (Slide 3: Dual-loop architecture)\n"
        "   - `fig2_swe_runtime_math.png` (Slide 4: SWE budget compounding)\n"
        "   - `fig_boundary_drift.png` (Slide 5: Autoregressive boundary drift)\n"
        "   - `fig_multidomain_standings.png` (Slide 6: Verified benchmark standings)\n"
        "   - `fig1_cost_comparison.png` (Slide 7: Economic comparison)\n"
        "5. Compile the presentation and ensure `python papers/autonomous_ai_systems/generate_deck.py` runs with 0 errors.\n"
        "6. Provide a concise summary of the visual deck architecture you implemented."
    )

    print("Delegating visual deck generation directly to Claude Code (Opus 5.5)...")
    res = claude_code_bridge.run_headless(prompt, cwd=str(REPO_ROOT), permission_mode="acceptEdits", timeout=600)

    if res.get("ok"):
        data = res.get("data")
        result_text = data.get("result", "") if isinstance(data, dict) else str(data)
        out_log = Path("papers/autonomous_ai_systems/CLAUDE_DECK_GENERATION_REPORT.md")
        out_log.write_text(result_text, encoding="utf-8")
        print(f"SUCCESS: Claude Code completed the task. Log written to {out_log}")
        print("\n" + "=" * 60)
        print("CLAUDE SUMMARY:")
        print("=" * 60)
        print(result_text[:1200] + "\n...")
    else:
        print("Claude execution failed:", res.get("error"))

if __name__ == "__main__":
    main()
