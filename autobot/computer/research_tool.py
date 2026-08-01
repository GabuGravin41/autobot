"""
Research Tool — Structured research pipeline for Autobot.

Two modes:
  1. Built-in (always available): LLM-powered research plan + browser guidance.
     Returns a structured research brief the agent uses to guide its browser work.

  2. AutoResearchClaw (optional): if `researchclaw` CLI is installed and
     Python >= 3.11 is available, delegates to the full end-to-end pipeline
     that produces a conference-ready LaTeX paper.

Usage (agent computer_call):
    computer.research.plan(topic="gene annotation in E. coli")
    computer.research.deep(topic="transformer architectures for protein folding")
    computer.research.status()   # check if AutoResearchClaw run is still going
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class Research:
    """
    Research pipeline tool.

    Methods:
        plan(topic)   — Quick research brief: sources to check, search terms, approach
        deep(topic)   — Full research run (uses AutoResearchClaw if available)
        status()      — Check if a background deep run is still active
    """

    def __init__(self) -> None:
        self._last_run_dir: Path | None = None
        self._proc: subprocess.Popen | None = None

    # ── Quick plan (always available) ────────────────────────────────────────

    def plan(self, topic: str) -> str:
        """
        Generate a quick research brief for the given topic.

        Returns a structured plan: search queries, key sources to visit,
        approach for collecting and synthesising information.
        Uses the agent's LLM — no extra setup needed.

        Args:
            topic: The research question or subject to investigate.

        Returns:
            A structured research brief as text.
        """
        # Called from asyncio.to_thread() so we're in a worker thread — safe to asyncio.run()
        try:
            return asyncio.run(self._build_plan(topic))
        except Exception as e:
            logger.error(f"Research plan failed: {e}")
            return self._fallback_plan(topic)

    async def _build_plan(self, topic: str) -> str:
        """Build research plan using the LLM if available."""
        try:
            from autobot.agent.runner import AgentRunner
            # Get the LLM client from the running runner if available
            runner = AgentRunner._current  # type: ignore[attr-defined]
            if runner and hasattr(runner, "llm_client"):
                client = runner.llm_client
                model = runner.model
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a research planning assistant. Be concise and actionable."},
                        {"role": "user", "content": (
                            f"Create a structured research brief for: {topic}\n\n"
                            "Include:\n"
                            "1. 5 specific Google/Scholar search queries\n"
                            "2. Key websites/databases to visit (with URLs)\n"
                            "3. Information to extract from each source\n"
                            "4. How to synthesise findings into a report\n"
                            "Keep each section short and actionable."
                        )},
                    ],
                    max_tokens=600,
                )
                return response.choices[0].message.content or self._fallback_plan(topic)
        except Exception:
            pass
        return self._fallback_plan(topic)

    def _fallback_plan(self, topic: str) -> str:
        """Template-based plan when LLM is unavailable."""
        safe = topic.replace('"', "").strip()
        return f"""RESEARCH BRIEF: {topic}

SEARCH QUERIES (use in Google/Google Scholar):
  1. "{safe}" overview
  2. "{safe}" recent advances 2024 2025
  3. "{safe}" methodology review
  4. "{safe}" site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov
  5. "{safe}" best practices tools software

KEY SOURCES TO VISIT:
  - https://scholar.google.com  (academic papers)
  - https://arxiv.org           (preprints)
  - https://pubmed.ncbi.nlm.nih.gov  (biomedical)
  - https://www.semanticscholar.org  (AI-powered search)
  - https://en.wikipedia.org    (background context)

EXTRACTION APPROACH:
  For each source: note author, year, key findings, methodology, limitations.
  Collect 5-10 high-quality sources before synthesising.

SYNTHESIS:
  Write sections: Background → Key Methods → Current State of the Art → Gaps → Conclusion
  Cite sources inline. Save draft to clipboard with computer.clipboard.copy().

To run a full automated pipeline (produces LaTeX paper), use: computer.research.deep(topic)"""

    # ── Deep run (AutoResearchClaw if available) ──────────────────────────────

    def deep(self, topic: str, output_dir: str | None = None) -> str:
        """
        Run a full end-to-end research pipeline for the given topic.

        If AutoResearchClaw is installed (pip install from github.com/aiming-lab/AutoResearchClaw),
        delegates to it for a full literature review + experiment + LaTeX paper output.
        Otherwise falls back to a detailed research plan with browser instructions.

        Args:
            topic:      The research question or hypothesis.
            output_dir: Where to save outputs (default: runs/research/<timestamp>).

        Returns:
            Status message with output location or research plan.
        """
        if _autoresearchclaw_available():
            return self._run_autoresearchclaw(topic, output_dir)
        return (
            f"AutoResearchClaw not installed — running built-in plan mode.\n\n"
            + self._fallback_plan(topic)
            + "\n\nTo install the full pipeline:\n"
            "  git clone https://github.com/aiming-lab/AutoResearchClaw\n"
            "  cd AutoResearchClaw && pip install -e .\n"
            "(Requires Python 3.11+)"
        )

    def _run_autoresearchclaw(self, topic: str, output_dir: str | None) -> str:
        """Launch AutoResearchClaw CLI as a background subprocess."""
        import tempfile
        import time

        out_dir = Path(output_dir) if output_dir else (
            Path("runs") / "research" / f"rc_{int(time.time())}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        self._last_run_dir = out_dir

        cmd = [
            sys.executable, "-m", "researchclaw", "run",
            "--topic", topic,
            "--output-dir", str(out_dir),
            "--auto-approve",
        ]
        log_file = open(out_dir / "rc.log", "w")
        self._proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True
        )
        logger.info(f"🔬 AutoResearchClaw started (PID {self._proc.pid}), output: {out_dir}")
        return (
            f"AutoResearchClaw started in background (PID {self._proc.pid}).\n"
            f"Output directory: {out_dir}\n"
            f"Log: {out_dir}/rc.log\n"
            f"Check progress with: computer.research.status()\n"
            f"Results will appear in: {out_dir}/deliverables/"
        )

    def status(self) -> str:
        """
        Check the status of a running AutoResearchClaw deep research job.

        Returns:
            Status string: running / completed / not started, with output path.
        """
        if self._proc is None:
            return "No deep research job running. Use computer.research.deep(topic) to start one."

        rc = self._proc.poll()
        if rc is None:
            return f"AutoResearchClaw running (PID {self._proc.pid}). Output: {self._last_run_dir}"

        deliverables = self._last_run_dir / "deliverables" if self._last_run_dir else None
        if deliverables and deliverables.exists():
            papers = list(deliverables.glob("*.tex")) + list(deliverables.glob("*.md"))
            files = ", ".join(p.name for p in papers[:5])
            return f"AutoResearchClaw finished (exit {rc}). Files: {files} in {deliverables}"
        return f"AutoResearchClaw finished (exit {rc}). Check: {self._last_run_dir}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _autoresearchclaw_available() -> bool:
    """Check if the researchclaw CLI is importable."""
    try:
        import importlib.util
        return importlib.util.find_spec("researchclaw") is not None
    except Exception:
        return False
