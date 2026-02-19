from __future__ import annotations

import json

from .engine import TaskStep, WorkflowPlan
from .workflows import (
    console_fix_assist_workflow,
    open_target_workflow,
    research_paper_workflow,
    simple_search_workflow,
    website_builder_workflow,
)


def build_plan_from_text(task: str) -> WorkflowPlan:
    text = task.strip()
    lower = text.lower()

    if lower.startswith("search "):
        return simple_search_workflow(text[7:].strip())

    if lower.startswith("open "):
        return open_target_workflow(text[5:].strip())

    if lower.startswith("run "):
        return WorkflowPlan(
            name="run_command",
            description="Run an OS command.",
            steps=[TaskStep(action="run_command", args={"command": text[4:]}, description=f"Run command: {text[4:]}")],
        )

    if lower == "list adapters":
        return WorkflowPlan(
            name="list_adapters",
            description="Show all adapter action libraries.",
            steps=[TaskStep(action="adapter_list_actions", save_as="adapter_library", description="Load adapter library")],
        )

    if lower.startswith("adapter "):
        payload = text[8:].strip()
        # Format: adapter <adapter_name> <action_name> {"json":"params"}
        parts = payload.split(" ", 2)
        if len(parts) < 2:
            raise ValueError("Adapter command format: adapter <name> <action> [json_params]")
        adapter_name = parts[0].strip()
        adapter_action = parts[1].strip()
        params = {}
        if len(parts) == 3 and parts[2].strip():
            try:
                params = json.loads(parts[2].strip())
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid adapter params JSON: {error}") from error
        return WorkflowPlan(
            name="adapter_call",
            description=f"Run adapter action {adapter_name}.{adapter_action}",
            steps=[
                TaskStep(
                    action="adapter_call",
                    args={
                        "adapter": adapter_name,
                        "adapter_action": adapter_action,
                        "params": params,
                        "confirmed": False,
                    },
                    description=f"Adapter call: {adapter_name}.{adapter_action}",
                )
            ],
        )

    if lower.startswith("open path "):
        return WorkflowPlan(
            name="open_path",
            description="Open local file/folder path.",
            steps=[
                TaskStep(
                    action="open_path",
                    args={"path": text[10:].strip()},
                    description=f"Open path: {text[10:].strip()}",
                )
            ],
        )

    if lower.startswith("switch window"):
        return WorkflowPlan(
            name="switch_window",
            description="Switch active desktop window.",
            steps=[TaskStep(action="desktop_switch_window", description="Switch window using Alt+Tab")],
        )

    if lower.startswith("type "):
        return WorkflowPlan(
            name="desktop_type",
            description="Type text into active app.",
            steps=[TaskStep(action="desktop_type", args={"text": text[5:]}, description="Type into focused window")],
        )

    if lower.startswith("wait "):
        seconds = _parse_seconds(text[5:])
        return WorkflowPlan(
            name="wait",
            description=f"Wait {seconds} seconds.",
            steps=[TaskStep(action="wait", args={"seconds": seconds}, description=f"Wait {seconds} seconds")],
        )

    if "build a website about" in lower:
        topic = _extract_after_phrase(text, "build a website about")
        return website_builder_workflow(topic)

    if "write a paper on" in lower:
        topic = _extract_after_phrase(text, "write a paper on")
        return research_paper_workflow(topic)

    if "fix console" in lower or "console errors" in lower:
        return console_fix_assist_workflow()

    return WorkflowPlan(
        name="generic_assistant",
        description="Fallback flow for generic tasks.",
        steps=[
            TaskStep(action="log", args={"message": f"Task received: {text}"}, description="Log incoming task"),
            TaskStep(
                action="log",
                args={
                    "message": (
                        "Use explicit commands for best results: search <query>, open <target>, run <command>, wait <seconds>."
                    )
                },
                description="Show supported command formats",
            ),
        ],
    )


def _extract_after_phrase(text: str, phrase: str) -> str:
    lower = text.lower()
    idx = lower.find(phrase)
    if idx < 0:
        return ""
    return text[idx + len(phrase) :].strip(" .:")


def _parse_seconds(value: str) -> float:
    cleaned = value.strip().lower().replace("seconds", "").replace("second", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 1.0
