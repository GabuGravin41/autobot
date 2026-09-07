"""
Autobot Learning Module — Online Reinforcement Learning from task outcomes.

The learning pipeline tracks (state, action, reward) tuples across runs and
learns which tool strategies work best in which contexts. Over time this gives
the agent a persistent performance advantage: it remembers that DOM clicks
work better than coordinate guessing on form-heavy pages, that navigating away
is faster than clicking when the wrong page is loaded, etc.

Components:
    ExperienceStore   — SQLite-backed storage for (state, action, outcome) tuples
    RewardComputer    — Converts step outcomes into scalar reward signals
    PolicyMemory      — Maps (context, action_type) → success rate / preference
    RLController      — Integrates everything; called by AgentLoop each step
"""
from autobot.learning.experience_store import ExperienceStore
from autobot.learning.reward_computer import RewardComputer
from autobot.learning.policy_memory import PolicyMemory
from autobot.learning.rl_controller import RLController, rl_controller

__all__ = [
    "ExperienceStore",
    "RewardComputer",
    "PolicyMemory",
    "RLController",
    "rl_controller",
]
