import json
import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JudgeOutput(BaseModel):
    success: bool
    reasoning: str


class JudgeAgent:
    """
    Evaluates whether an autonomous agent successfully completed its goal.
    
    Takes the original goal, the agent's final output, and a summary
    of the steps taken to make an impartial decision.
    """

    def __init__(self, llm_client: Any, model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.model = model

    def _fast_heuristic_check(
        self, goal: str, result_text: str
    ) -> JudgeOutput | None:
        """
        Instant heuristic pre-check — avoids LLM for obvious cases.
        Returns a JudgeOutput if the verdict is clear, None if LLM is needed.

        Every failure signal here matches the EXACT, STRUCTURAL wording this
        codebase actually produces on a real failure (verified against
        runner.py's exception handler and loop.py's _summarize_history/
        LLMUnavailableError), not a bare word that could appear anywhere.

        The previous version matched loose substrings like "error:",
        "timed out", "unable to", and "could not complete" ANYWHERE in the
        text — including inside the agent's own free-form summary of what it
        did. A run that researched and wrote a paper discussing, say,
        measurement error or a switching threshold would get silently
        marked as FAILED before the LLM ever saw it, for a task that had
        nothing to do with the agent failing at anything. Precision matters
        more than recall here: the LLM judge exists specifically to handle
        exactly the ambiguous cases this heuristic used to guess wrong on;
        this pre-check should only fire when the wording is nearly
        impossible to produce except by the real failure path it names.
        """
        stripped = result_text.strip()
        result_lower = stripped.lower()

        # AgentRunner.run()'s except block: `f"Error: {e}\n\nTraceback:\n{tb}"`
        # — always the first characters of the string on that path, never
        # merely mentioned mid-sentence.
        if result_lower.startswith("error:"):
            return JudgeOutput(success=False, reasoning="Heuristic: result begins with an Error: wrapper")

        # Python's own fixed-format stack trace header — for this to appear
        # coincidentally in prose would be extraordinary.
        if "traceback (most recent call last)" in result_lower:
            return JudgeOutput(success=False, reasoning="Heuristic: result contains a Python traceback")

        # loop.py's _summarize_history(): "Agent ran N steps without calling
        # 'done'." — the loop hit max_steps without the agent ever finishing.
        if "without calling 'done'" in result_lower:
            return JudgeOutput(success=False, reasoning="Heuristic: agent hit max steps without completing")

        # loop.py's LLMUnavailableError message.
        if "failed" in result_lower and "times in a row" in result_lower and "llm" in result_lower:
            return JudgeOutput(success=False, reasoning="Heuristic: LLM was unavailable for the run")

        # Judge's own error wrapper, only at the start — guards against
        # re-evaluation loops, not a bare mention of the word "judge".
        if result_lower.startswith("judge error:"):
            return JudgeOutput(success=False, reasoning="Heuristic: a prior judge call itself errored")

        # A prior judge already confirmed success (re-evaluation shouldn't
        # happen, but handle it rather than paying for a second LLM call).
        if "judge verification: success" in result_lower:
            return JudgeOutput(success=True, reasoning="Prior judge verification confirmed success")

        # Genuinely empty output — not "shorter than some arbitrary length",
        # since a real short success ("Sent the email.", "Fixed.") is
        # entirely plausible and shouldn't be second-guessed by a length
        # threshold with nothing to do with correctness.
        if len(stripped) < 3:
            return JudgeOutput(success=False, reasoning="Heuristic: result is empty or near-empty")

        # Explicit, high-confidence success phrasing.
        if any(s in result_lower for s in ["successfully completed", "task complete", "done(success=true"]):
            return JudgeOutput(success=True, reasoning="Heuristic: result contains explicit success confirmation")

        return None  # Ambiguous — let the LLM judge decide.

    async def evaluate(self, goal: str, result_text: str, history_summary: str) -> JudgeOutput:
        """Evaluate the agent's performance."""
        # Fast path — skip LLM for obvious outcomes
        heuristic = self._fast_heuristic_check(goal, result_text)
        if heuristic is not None:
            logger.debug(f"Judge heuristic shortcut: success={heuristic.success}")
            return heuristic

        prompt = (
            "You are an impartial Judge Agent evaluating if an autonomous browser agent "
            "successfully completed its task.\n\n"
            f"Original Goal:\n{goal}\n\n"
            f"Agent's Execution History (for context):\n{history_summary}\n\n"
            f"Agent's Final Result:\n{result_text}\n\n"
            "Evaluate if the original goal was met. You must output completely valid JSON matching this schema:\n"
            '{\n  "success": boolean,\n  "reasoning": "string explaining why"\n}'
        )

        try:
            # Consolidate into single user message for local LLM compatibility
            full_content = f"SYSTEM: You are an impartial Judge Agent.\n\n{prompt}"
            
            args = {
                "model": self.model,
                "messages": [{"role": "user", "content": full_content}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }

            async def _internal_call(current_args: dict) -> str:
                try:
                    # async
                    resp = await self.llm_client.chat.completions.create(**current_args)
                    return str(resp.choices[0].message.content)
                except TypeError:
                    # sync fallback
                    import asyncio
                    resp = await asyncio.to_thread(
                        self.llm_client.chat.completions.create,
                        **current_args
                    )
                    return str(resp.choices[0].message.content)

            try:
                text = await _internal_call(args)
            except Exception as e:
                error_msg = str(e).lower()
                if "400" in error_msg or "response_format" in error_msg or "json_object" in error_msg:
                    logger.warning(f"Judge model {self.model} failed with JSON mode. Retrying without JSON mode...")
                    args.pop("response_format", None)
                    text = await _internal_call(args)
                else:
                    raise e

            text = text.strip()
            
            # Extract JSON using robust fallback strategy
            data = None
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if not json_match:
                    json_match = re.search(r"(\{.*?\})", text, re.DOTALL)
                
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
            
            if data is None:
                raise ValueError("Could not parse JSON from Judge output")

            return JudgeOutput(
                success=bool(data.get("success", False)),
                reasoning=str(data.get("reasoning", "No reasoning provided.")),
            )
        except Exception as e:
            logger.error(f"Judge Agent failed to evaluate: {e}")
            return JudgeOutput(success=False, reasoning=f"Judge error: {str(e)}")
