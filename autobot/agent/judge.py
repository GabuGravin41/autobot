import json
import logging
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

    async def evaluate(self, goal: str, result_text: str, history_summary: str) -> JudgeOutput:
        """Evaluate the agent's performance."""
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
            # support both sync and async clients
            try:
                # async
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content
            except TypeError:
                import asyncio
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content

            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

            data = json.loads(text)
            return JudgeOutput(
                success=bool(data.get("success", False)),
                reasoning=str(data.get("reasoning", "No reasoning provided.")),
            )
        except Exception as e:
            logger.error(f"Judge Agent failed to evaluate: {e}")
            return JudgeOutput(success=False, reasoning=f"Judge error: {str(e)}")
