"""
Mission Agent — High-level strategist that coordinates complex multi-step missions.

The MissionAgent uses a MissionManager to track progress through a series
of Objectives. For each objective, it spins up an AgentLoop to perform
the actual browser/OS interactions.
"""
from __future__ import annotations

import logging
import json
from typing import Any, List, Optional

from autobot.agent.loop import AgentLoop
from autobot.agent.mission import Mission, MissionManager, MissionStatus, Objective, ObjectiveStatus

logger = logging.getLogger(__name__)

class MissionAgent:
    """
    High-level agent that plans and executes multi-objective missions.
    """

    def __init__(
        self,
        page: Any,
        llm_client: Any,
        mission_goal: str,
        model: str = "gpt-4o",
    ):
        self.page = page
        self.llm_client = llm_client
        self.mission_goal = mission_goal
        self.model = model
        
        # Initial empty mission — will be planned in run()
        self.mission = Mission(id="mission_1", goal=mission_goal)
        self.manager = MissionManager(self.mission)

    async def run(self) -> str:
        """
        Execute the mission: Plan → Loop(Execute Objective) → Complete.
        """
        logger.info(f"🚀 Starting Mission: {self.mission_goal}")
        
        # 1. PLAN
        await self._plan_mission()
        
        # 2. EXECUTE
        while self.mission.status == MissionStatus.EXECUTING:
            objective = self.manager.get_current_objective()
            if not objective:
                self.mission.status = MissionStatus.COMPLETED
                break
                
            logger.info(f"🎯 Current Objective: {objective.description}")
            objective.status = ObjectiveStatus.IN_PROGRESS
            
            # Run AgentLoop for this objective
            # We pass the mission context to the agent so it knows the bigger picture
            custom_instructions = f"Current Mission: {self.mission_goal}\nMission Progress: {self._get_mission_summary()}"
            
            agent = AgentLoop(
                page=self.page,
                llm_client=self.llm_client,
                goal=objective.description,
                model=self.model,
                custom_instructions=custom_instructions,
            )
            
            result = await agent.run()
            
            # 3. EVALUATE OBJECTIVE
            if "fail" in result.lower() or "error" in result.lower():
                self.manager.fail_current_objective(result)
            else:
                self.manager.complete_current_objective(result)

        if self.mission.status == MissionStatus.COMPLETED:
            return f"Mission Success! Summary: {self._get_mission_summary()}"
        else:
            return f"Mission Failed. Reason: {self.mission.mission_log[-1]}"

    async def _plan_mission(self):
        """Use LLM to break the mission goal into objectives."""
        logger.debug("Planning mission objectives...")
        
        prompt = f"""You are a Mission Strategist. Break the following high-level mission into 3-5 sequential objectives.
Each objective must be a clear, actionable task for a browser/OS automation agent.

Mission Goal: {self.mission_goal}

Respond with a JSON list of objectives:
[{{"id": "obj1", "description": "Download the dataset from Kaggle"}}, ...]
"""
        try:
            # Simple LLM call for planning
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            
            objectives_data = data.get("objectives", [])
            for obj_data in objectives_data:
                self.mission.objectives.append(Objective(**obj_data))
                
            self.mission.status = MissionStatus.EXECUTING
            self.mission.add_log(f"Mission planned with {len(self.mission.objectives)} objectives.")
            
        except Exception as e:
            logger.error(f"Failed to plan mission: {e}")
            self.mission.status = MissionStatus.FAILED
            self.mission.add_log(f"Planning failed: {e}")

    def _get_mission_summary(self) -> str:
        """Returns a text summary of completed objectives."""
        summary = []
        for obj in self.mission.objectives:
            status_icon = "[OK]" if obj.status == ObjectiveStatus.COMPLETED else "[..]" if obj.status == ObjectiveStatus.PENDING else "[EX]" if obj.status == ObjectiveStatus.IN_PROGRESS else "[!!]"
            summary.append(f"{status_icon} {obj.description}")
        return "\n".join(summary)
