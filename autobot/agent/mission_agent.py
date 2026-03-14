"""
Mission Agent — High-level strategist that coordinates complex multi-step missions.

The MissionAgent uses a MissionManager to track progress through a series
of Objectives. For each objective, it spins up an AgentLoop to perform
the actual browser/OS interactions.

This enables Autobot to handle ambitious, long-running tasks like:
- Kaggle competitions (download → code → submit → iterate)
- Gene annotation (load files → search NCBI → paste results → repeat)
- Building websites (open VS Code → write code → test in browser → deploy)
- Email monitoring (open Gmail → scan emails → summarize priorities)
"""
from __future__ import annotations

import logging
import json
import os
from typing import Any, Callable, List, Optional

from autobot.agent.loop import AgentLoop
from autobot.agent.mission import Mission, MissionManager, MissionStatus, Objective, ObjectiveStatus

logger = logging.getLogger(__name__)


class MissionAgent:
    """
    High-level agent that plans and executes multi-objective missions.

    For simple single-action tasks, use AgentLoop directly.
    For complex multi-phase tasks, MissionAgent breaks them into objectives
    and executes each with its own AgentLoop instance, passing mission context.
    """

    def __init__(
        self,
        page: Any,
        llm_client: Any,
        mission_goal: str,
        model: str = "gpt-4o",
        max_steps_per_objective: int = 50,
        log_callback: Callable[[str], None] | None = None,
    ):
        self.page = page
        self.llm_client = llm_client
        self.mission_goal = mission_goal
        self.model = model
        self.max_steps_per_objective = int(
            os.getenv("AUTOBOT_STEPS_PER_OBJECTIVE", str(max_steps_per_objective))
        )
        self.log = log_callback or (lambda msg: logger.info(msg))

        # Initial empty mission — will be planned in run()
        self.mission = Mission(id="mission_1", goal=mission_goal)
        self.manager = MissionManager(self.mission)

        # Track the current agent loop for status reporting
        self.current_agent_loop: AgentLoop | None = None
        self._run_dir = None

    async def run(self) -> str:
        """
        Execute the mission: Plan → Loop(Execute Objective) → Complete.
        """
        self.log(f"🚀 Starting Mission: {self.mission_goal}")

        # 1. PLAN
        await self._plan_mission()

        if self.mission.status == MissionStatus.FAILED:
            return f"Mission planning failed: {self.mission.mission_log[-1]}"

        self.log(f"📋 Mission planned with {len(self.mission.objectives)} objectives:")
        for i, obj in enumerate(self.mission.objectives, 1):
            self.log(f"  {i}. {obj.description}")

        # 2. EXECUTE each objective
        failed_count = 0
        while self.mission.status == MissionStatus.EXECUTING:
            objective = self.manager.get_current_objective()
            if not objective:
                self.mission.status = MissionStatus.COMPLETED
                break

            obj_index = self.mission.objectives.index(objective) + 1
            total = len(self.mission.objectives)
            self.log(f"🎯 Objective {obj_index}/{total}: {objective.description}")
            objective.status = ObjectiveStatus.IN_PROGRESS

            # Build mission context so the agent knows the bigger picture
            custom_instructions = self._build_context_for_objective(objective)

            # Run AgentLoop for this objective
            agent = AgentLoop(
                page=self.page,
                llm_client=self.llm_client,
                goal=objective.description,
                model=self.model,
                max_steps=self.max_steps_per_objective,
                custom_instructions=custom_instructions,
            )
            agent._run_dir = self._run_dir
            self.current_agent_loop = agent

            try:
                result = await agent.run()
            except Exception as e:
                result = f"Error: {e}"
                logger.error(f"Objective failed with exception: {e}")

            # 3. EVALUATE OBJECTIVE — be generous, don't fail on partial success
            is_failure = (
                result.lower().startswith("error:")
                or "impossible" in result.lower()
                or ("fail" in result.lower() and "success" not in result.lower())
            )

            if is_failure:
                self.manager.fail_current_objective(result)
                failed_count += 1
                self.log(f"  ❌ Objective failed: {result[:150]}")

                # Don't abort entire mission on one failure — skip and continue
                # unless more than half the objectives have failed
                if failed_count > len(self.mission.objectives) // 2:
                    self.log("  ⚠️ Too many failures — aborting mission")
                    break
                else:
                    self.log("  ⏭️ Skipping to next objective...")
                    # Mark as failed but continue (reset mission status)
                    self.mission.status = MissionStatus.EXECUTING
            else:
                self.manager.complete_current_objective(result)
                self.log(f"  ✅ Objective completed: {result[:150]}")

        # 4. FINAL SUMMARY
        summary = self._get_mission_summary()
        completed = sum(1 for o in self.mission.objectives if o.status == ObjectiveStatus.COMPLETED)
        total = len(self.mission.objectives)

        if completed == total:
            self.log(f"🏆 Mission Complete! All {total} objectives achieved.")
            return f"Mission Success! ({completed}/{total} objectives)\n\n{summary}"
        elif completed > 0:
            self.log(f"⚠️ Mission Partial Success: {completed}/{total} objectives completed.")
            return f"Mission Partial Success ({completed}/{total} objectives)\n\n{summary}"
        else:
            self.log(f"❌ Mission Failed: 0/{total} objectives completed.")
            return f"Mission Failed (0/{total} objectives)\n\n{summary}"

    def _build_context_for_objective(self, current_obj: Objective) -> str:
        """Build context string so the agent knows about the broader mission."""
        lines = [
            f"MISSION: {self.mission_goal}",
            f"CURRENT OBJECTIVE: {current_obj.description}",
            "",
            "MISSION PROGRESS:",
        ]

        for obj in self.mission.objectives:
            if obj.id == current_obj.id:
                lines.append(f"  → [CURRENT] {obj.description}")
            elif obj.status == ObjectiveStatus.COMPLETED:
                lines.append(f"  ✅ [DONE] {obj.description}: {obj.result[:100] if obj.result else ''}")
            elif obj.status == ObjectiveStatus.FAILED:
                lines.append(f"  ❌ [FAILED] {obj.description}: {obj.result[:80] if obj.result else ''}")
            else:
                lines.append(f"  ⬜ [PENDING] {obj.description}")

        lines.append("")
        lines.append("Focus ONLY on the current objective. Use findings from completed objectives if relevant.")
        return "\n".join(lines)

    async def _plan_mission(self):
        """Use LLM to break the mission goal into objectives."""
        logger.debug("Planning mission objectives...")

        prompt = f"""You are a Mission Strategist for a desktop automation agent called Autobot.

Autobot can:
- Control the browser (navigate, click, type, scroll, manage tabs)
- Control any desktop application (VS Code, terminal, file manager, etc.) via Alt+Tab and mouse/keyboard
- Copy/paste between applications using the clipboard
- Upload and download files
- Interact with AI chatbots (ChatGPT, Grok, Claude) to generate code or get information
- Read screenshots to understand what's on screen

Break the following mission into 3-7 sequential objectives. Each objective should be:
- A concrete, verifiable action (not vague)
- Achievable in ~20-50 agent steps
- Ordered logically (dependencies flow forward)

Mission: {self.mission_goal}

Respond with ONLY a JSON object like:
{{"objectives": [{{"id": "obj1", "description": "Navigate to kaggle.com and log in using saved credentials"}}, ...]}}"""

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content or ""

            # Parse JSON (handle markdown code blocks)
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)

            objectives_data = data if isinstance(data, list) else data.get("objectives", [])
            if not objectives_data:
                raise ValueError("No objectives returned from planner")

            for obj_data in objectives_data:
                if isinstance(obj_data, str):
                    obj_data = {"id": f"obj{len(self.mission.objectives)+1}", "description": obj_data}
                if "id" not in obj_data:
                    obj_data["id"] = f"obj{len(self.mission.objectives)+1}"
                self.mission.objectives.append(Objective(**obj_data))

            self.mission.status = MissionStatus.EXECUTING
            self.mission.add_log(f"Mission planned with {len(self.mission.objectives)} objectives.")

        except Exception as e:
            logger.error(f"LLM planning failed: {e}. Using fallback single-objective plan.")
            # Fallback: treat entire goal as a single objective
            self.mission.objectives = [
                Objective(id="obj1", description=self.mission_goal)
            ]
            self.mission.status = MissionStatus.EXECUTING
            self.mission.add_log(f"Fallback plan: single objective (planning failed: {e})")

    def _get_mission_summary(self) -> str:
        """Returns a text summary of all objectives and their results."""
        summary = []
        for obj in self.mission.objectives:
            if obj.status == ObjectiveStatus.COMPLETED:
                icon = "✅"
            elif obj.status == ObjectiveStatus.FAILED:
                icon = "❌"
            elif obj.status == ObjectiveStatus.IN_PROGRESS:
                icon = "🔄"
            else:
                icon = "⬜"
            line = f"{icon} {obj.description}"
            if obj.result:
                line += f"\n   → {obj.result[:200]}"
            summary.append(line)
        return "\n".join(summary)

    def get_status(self) -> dict:
        """Get mission status for the dashboard API."""
        return {
            "mission_goal": self.mission_goal,
            "mission_status": self.mission.status.value,
            "objectives": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "status": obj.status.value,
                    "result": obj.result[:200] if obj.result else None,
                }
                for obj in self.mission.objectives
            ],
            "current_objective": (
                self.manager.get_current_objective().description
                if self.manager.get_current_objective()
                else None
            ),
            "log": self.mission.mission_log[-10:],
        }
