"""
Invoke Claude Code (Opus 5.5) via Autobot's built-in bridge to conduct a comprehensive
peer-review of the paper and design the presentation slide architecture.
"""

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

def main():
    paper_path = Path("papers/autonomous_ai_systems/AUTONOMOUS_AI_SYSTEMS_LONG_HORIZON_ENGINEERING_PAPER.md")
    if not paper_path.exists():
        print(f"Error: {paper_path} not found.")
        sys.exit(1)

    paper_content = paper_path.read_text(encoding="utf-8")

    prompt = (
        "You are serving as an elite Senior Peer Reviewer and Presentation Architect for our new systems research paper titled:\n"
        "'Autonomous AI Systems for Long-Horizon Engineering: Architecture, Failure Modes, and Economic Viability'.\n\n"
        "Here is the complete paper text:\n"
        "==============================================================================\n"
        + paper_content + "\n"
        "==============================================================================\n\n"
        "Please execute a rigorous, peer-level review and presentation design:\n\n"
        "1. METHODOLOGICAL & THEORETICAL REVIEW:\n"
        "   - Critique the framing of the 'Dual-Loop Reactive Engine' (inner supervisor vs outer metacognitive loop).\n"
        "   - Verify the mathematical logic of the SWE-bench budget breakdown (8.0m vs 3.5m per task on a 120-task split).\n"
        "   - Evaluate the failure autopsy of the earth system autoregressive boundary violation and the proposed self-healing loop.\n"
        "   - Assess the realism and rigor of the economic analysis comparing Autobot ($5/day) to Devin ($350/day), AutoML ($600/day), and human engineering teams ($4,320/day).\n"
        "   - What are the sharpest potential reviewer objections, and how should we preemptively address them in the manuscript?\n\n"
        "2. PRESENTATION DECK ARCHITECTURE & SLIDE-BY-SLIDE DESIGN:\n"
        "   - Provide a slide-by-slide architectural recommendation for a 10-slide keynote presentation deck.\n"
        "   - For each slide, provide:\n"
        "     a) Exact Slide Title & Subtitle\n"
        "     b) Core Visual Concept / Diagram Layout\n"
        "     c) Key High-Impact Bullet Points (quantified and concise)\n"
        "     d) Professional Executive Speaker Notes for the presenter\n\n"
        "3. CONCRETE REVISIONS & FUTURE ROADMAP:\n"
        "   - Specific textual or structural refinements you recommend adding to the paper right now to maximize impact and academic rigor.\n\n"
        "Deliver a comprehensive, publication-grade critique and presentation plan."
    )

    print("Dispatching paper to Claude Code (Opus 5.5) via Autobot bridge...")
    res = claude_code_bridge.run_headless(prompt, timeout=600)

    if res.get("ok") and isinstance(res.get("data"), dict):
        result_text = res["data"].get("result", "")
        out_file = Path("papers/autonomous_ai_systems/CLAUDE_PEER_REVIEW_AND_PRESENTATION_PLAN.md")
        out_file.write_text(result_text, encoding="utf-8")
        print(f"SUCCESS: Claude Opus 5.5 review written to {out_file} ({len(result_text):,} bytes)")
        print("\n" + "=" * 60)
        print("CLAUDE OPUS 5.5 PEER REVIEW EXCERPT:")
        print("=" * 60)
        print(result_text[:1200] + "\n...")
    else:
        print("Claude invocation failed:", res.get("error"))

if __name__ == "__main__":
    main()
