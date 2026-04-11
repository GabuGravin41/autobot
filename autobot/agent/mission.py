"""
Mission Models — Hierarchical planning and tracking for complex agent tasks.

A Mission consists of multiple Objectives.
An Objective is a high-level goal that the AgentLoop works to complete.
The MissionManager coordinates the transition between objectives.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class ObjectiveStatus(enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Objective(BaseModel):
    """A high-level objective within a Mission."""
    id: str
    description: str
    # Planner-set fields (from 2-stage analysis)
    success_criteria: Optional[str] = None   # "Done when: X is visible / Y is downloaded"
    max_steps: Optional[int] = None          # Planner's step budget estimate
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class MissionStatus(enum.Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

class Mission(BaseModel):
    """A complete mission consisting of sequential objectives."""
    id: str
    goal: str
    objectives: List[Objective] = []
    status: MissionStatus = MissionStatus.PLANNING
    mission_log: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_log(self, message: str):
        self.mission_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.updated_at = datetime.now()

class MissionManager:
    """Manages the lifecycle of a Mission and its objectives."""
    
    def __init__(self, mission: Mission):
        self.mission = mission

    def get_current_objective(self) -> Optional[Objective]:
        """Returns the first pending or in-progress objective."""
        for obj in self.mission.objectives:
            if obj.status in (ObjectiveStatus.PENDING, ObjectiveStatus.IN_PROGRESS):
                return obj
        return None

    def complete_current_objective(self, result: str):
        """Marks the current objective as completed and logs the result."""
        obj = self.get_current_objective()
        if obj:
            obj.status = ObjectiveStatus.COMPLETED
            obj.result = result
            obj.completed_at = datetime.now()
            self.mission.add_log(f"Completed objective '{obj.description}': {result}")
            
            # Check if mission is complete
            if not self.get_current_objective():
                self.mission.status = MissionStatus.COMPLETED
                self.mission.add_log("Mission completed.")

    def fail_current_objective(self, reason: str):
        """Marks the current objective as failed."""
        obj = self.get_current_objective()
        if obj:
            obj.status = ObjectiveStatus.FAILED
            obj.result = reason
            obj.completed_at = datetime.now()
            self.mission.status = MissionStatus.FAILED
            self.mission.add_log(f"Objective failed: {obj.description}. Reason: {reason}")
