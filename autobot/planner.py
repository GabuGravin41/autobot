from __future__ import annotations

import json

from .engine import TaskStep, WorkflowPlan
from .workflows import (
    console_fix_assist_workflow,
    open_target_workflow,
    research_paper_workflow,
    simple_search_workflow,
    tool_call_stress_workflow,
    website_builder_workflow,
)


def build_plan_from_text(task: str) -> WorkflowPlan:
    text = task.strip()
    lower = text.lower()

    if lower.startswith("search "):
        return simple_search_workflow(text[7:].strip())

    if lower.startswith("open "):
        return open_target_workflow(text[5:].strip())

    if lower == "run benchmarks":
        return WorkflowPlan(
            name="benchmark_run",
            description="Run internal benchmark suite.",
            steps=[TaskStep(action="benchmark_run", save_as="benchmark_results", description="Execute benchmark suite")],
        )

    if lower.startswith("run tool stress "):
        payload = text[16:].strip()
        # Format: run tool stress <phone>|<docs_existing_url>|<download_check_path>|<message>
        parts = [item.strip() for item in payload.split("|")]
        while len(parts) < 4:
            parts.append("")
        return tool_call_stress_workflow(
            whatsapp_phone=parts[0],
            docs_existing_url=parts[1],
            download_check_path=parts[2],
            outgoing_message=parts[3] or "Autobot tool-calling test message",
        )

    if lower.startswith("run "):
        return WorkflowPlan(
            name="run_command",
            description="Run an OS command.",
            steps=[TaskStep(action="run_command", args={"command": text[4:]}, description=f"Run command: {text[4:]}")],
        )

    if lower == "browser mode":
        return WorkflowPlan(
            name="browser_mode_status",
            description="Show active browser mode status.",
            steps=[TaskStep(action="browser_mode_status", save_as="browser_mode_status", description="Load browser mode status")],
        )

    if lower == "list adapters":
        return WorkflowPlan(
            name="list_adapters",
            description="Show all adapter action libraries.",
            steps=[TaskStep(action="adapter_list_actions", save_as="adapter_library", description="Load adapter library")],
        )

    if lower == "adapter telemetry":
        return WorkflowPlan(
            name="adapter_telemetry",
            description="Show adapter telemetry and selector metrics.",
            steps=[
                TaskStep(
                    action="adapter_get_telemetry",
                    save_as="adapter_telemetry",
                    description="Load adapter telemetry",
                )
            ],
        )

    if lower.startswith("adapter policy "):
        profile = text[15:].strip().lower()
        return WorkflowPlan(
            name="adapter_policy",
            description=f"Set adapter policy profile to {profile}",
            steps=[
                TaskStep(
                    action="adapter_set_policy",
                    args={"profile": profile},
                    description=f"Set adapter policy to {profile}",
                )
            ],
        )

    if lower.startswith("adapter prepare "):
        payload = text[16:].strip()
        # Format: adapter prepare <adapter_name> <action_name> {"json":"params"}
        parts = payload.split(" ", 2)
        if len(parts) < 2:
            raise ValueError("Adapter prepare format: adapter prepare <name> <action> [json_params]")
        adapter_name = parts[0].strip()
        adapter_action = parts[1].strip()
        params = {}
        if len(parts) == 3 and parts[2].strip():
            try:
                params = json.loads(parts[2].strip())
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid adapter params JSON: {error}") from error
        return WorkflowPlan(
            name="adapter_prepare_sensitive",
            description=f"Prepare sensitive adapter action {adapter_name}.{adapter_action}",
            steps=[
                TaskStep(
                    action="adapter_prepare_sensitive",
                    args={"adapter": adapter_name, "adapter_action": adapter_action, "params": params},
                    save_as="sensitive_token_payload",
                    description=f"Prepare token for {adapter_name}.{adapter_action}",
                ),
                TaskStep(
                    action="log",
                    args={"message": "Sensitive action prepared. Use adapter confirm <token> to execute."},
                    description="Show confirm instructions",
                ),
            ],
        )

    if lower.startswith("adapter confirm "):
        token = text[16:].strip()
        return WorkflowPlan(
            name="adapter_confirm_sensitive",
            description="Confirm and execute prepared sensitive adapter action.",
            steps=[
                TaskStep(
                    action="adapter_confirm_sensitive",
                    args={"token": token},
                    description="Confirm sensitive action token",
                )
            ],
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
