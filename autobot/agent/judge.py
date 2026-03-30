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
        """
        result_lower = result_text.lower()

        # Clear failure signals
        failure_signals = [
            "error:", "exception:", "traceback", "task failed",
            "could not complete", "unable to", "timed out",
            "judge error:", "circuit breaker", "max steps reached",
        ]
        for sig in failure_signals:
            if sig in result_lower:
                return JudgeOutput(
                    success=False,
                    reasoning=f"Heuristic: result contains failure signal '{sig}'"
                )

        # Clear success signals paired with "done" or "success"
        if "judge verification: success" in result_lower:
            # Prior judge already confirmed success (re-evaluation shouldn't happen, but handle it)
            return JudgeOutput(success=True, reasoning="Prior judge verification confirmed success")

        # Very short results with no task content are suspicious
        if len(result_text.strip()) < 20 and "done" not in result_lower:
            return JudgeOutput(
                success=False,
                reasoning="Heuristic: result is too short to indicate task completion"
            )

        # If result contains explicit success markers, trust them
        if any(s in result_lower for s in ["successfully completed", "task complete", "done(success=true"]):
            return JudgeOutput(
                success=True,
                reasoning="Heuristic: result contains explicit success confirmation"
            )

        return None  # Proceed to LLM evaluation

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
            args = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
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
