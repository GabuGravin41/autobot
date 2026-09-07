"""
Trace Logger -- Offline teacher dataset writer.

Tesla V12 lesson applied:
  Tesla moved the old modular stack out of the live runtime and into an
  offline labeling pipeline.  The heuristic modules became TEACHERS that
  auto-generate rich ground-truth training targets from fleet video,
  using past+future frames and unlimited compute.

  TraceLogger is Autobot version of this idea.  Every agent step is
  serialized to JSONL: perception context, the LLM proxy assessment
  (the agent predicted understanding), the actions taken, and the
  verifier verdict.  These traces are the raw dataset for future
  fine-tuning of faster, lighter local models.

  The traces are written to runs/<timestamp>/traces.jsonl.
  Screenshots are excluded by default (AUTOBOT_TRACE_VISION=1 to include).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autobot.agent.models import AgentOutput, ActionResult
    from autobot.agent.step_verifier import VerificationResult

logger = logging.getLogger(__name__)

_TRACE_VISION = os.getenv("AUTOBOT_TRACE_VISION", "0").strip() == "1"


class TraceLogger:
    """
    Writes per-step execution traces to a JSONL file.

    Each line is a self-contained JSON record describing one step:
      - step_number, goal
      - proxy_assessment (the LLM predicted scene understanding)
      - actions_taken and their results
      - verification_result (StepVerifier verdict)
      - url_before / url_after
      - screenshot_b64 (only if AUTOBOT_TRACE_VISION=1)
      - timestamp

    On finalize(), a summary record is appended with overall success/failure.
    """

    def __init__(self, run_dir: Optional[Path] = None) -> None:
        if run_dir is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path.cwd() / "runs" / ts
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self.run_dir / "traces.jsonl"
        self._started_at = time.time()
        logger.info(f"TraceLogger: writing to {self._trace_path}")

    def log_step(
        self,
        step_number: int,
        goal: str,
        agent_output: "AgentOutput",
        action_results: "list[ActionResult]",
        verification_result: "VerificationResult | None",
        url_before: str,
        url_after: str,
        screenshot_b64: str = "",
    ) -> None:
        """Append one step record to the JSONL trace file."""
        try:
            pa = agent_output.proxy_assessment
            record: dict[str, Any] = {
                "ts": time.time(),
                "step": step_number,
                "goal": goal,
                "proxy_assessment": {
                    "target_element": pa.target_element if pa else "",
                    "expected_state_delta": pa.expected_state_delta if pa else "",
                    "active_window_focus": pa.active_window_focus if pa else "",
                    "risk_label": pa.risk_label if pa else "",
                } if pa else None,
                "thinking_snippet": agent_output.thinking[:300] if agent_output.thinking else "",
                "next_goal": agent_output.next_goal,
                "actions": [
                    {
                        "name": r.action_name,
                        "success": r.success,
                        "error": r.error,
                        "content_snippet": (r.extracted_content or "")[:200],
                    }
                    for r in action_results
                ],
                "verification": {
                    "label": verification_result.label if verification_result else "NONE",
                    "passed": verification_result.passed if verification_result else None,
                    "reason": verification_result.reason if verification_result else "",
                } if verification_result else None,
                "url_before": url_before,
                "url_after": url_after,
            }
            if _TRACE_VISION and screenshot_b64:
                record["screenshot_b64"] = screenshot_b64
            self._append(record)
        except Exception as e:
            logger.warning(f"TraceLogger.log_step failed (run still proceeding): {e}")

    def finalize(self, success: bool, result_text: str = "") -> None:
        """Append a summary record marking the end of the run."""
        try:
            self._append({
                "ts": time.time(),
                "type": "run_summary",
                "success": success,
                "duration_seconds": round(time.time() - self._started_at, 1),
                "result_snippet": result_text[:400],
            })
            logger.info(
                f"TraceLogger: run finalized (success={success}) -> {self._trace_path}"
            )
        except Exception as e:
            logger.warning(f"TraceLogger.finalize failed: {e}")

    def _append(self, record: dict) -> None:
        with self._trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
