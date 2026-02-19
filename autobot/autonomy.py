from __future__ import annotations

from dataclasses import dataclass

from .engine import AutomationEngine, ExecutionResult, TaskStep
from .llm_brain import LLMBrain


@dataclass
class AutonomousConfig:
    max_loops: int = 5
    max_steps_per_loop: int = 5
    diagnostics_command: str = ""
    target_url: str = ""
    allow_desktop_actions: bool = False
    allow_sensitive_adapter_actions: bool = False


class AutonomousRunner:
    def __init__(self, engine: AutomationEngine, logger=None) -> None:
        self.engine = engine
        self.logger = logger or (lambda _msg: None)
        self.brain = LLMBrain(logger=self.logger)
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True
        self.engine.cancel()

    def run(self, goal: str, config: AutonomousConfig) -> ExecutionResult:
        self._cancel_requested = False
        self.engine.state["autonomy_goal"] = goal
        self.engine.state["autonomy_loops"] = 0

        loop_result = ExecutionResult(success=True, completed_steps=0, total_steps=0, state=self.engine.state)
        for loop_index in range(1, config.max_loops + 1):
            if self._cancel_requested:
                self.logger("Autonomous mode cancelled by user.")
                break

            self.engine.state["autonomy_loops"] = loop_index
            self.logger(f"========== AUTONOMY LOOP {loop_index}/{config.max_loops} ==========")
            diagnostics_steps = self._diagnostics_steps(config)
            loop_result = self.engine.run_steps(
                steps=diagnostics_steps,
                plan_name=f"autonomy_diagnostics_{loop_index}",
                plan_description="Capture current system diagnostics.",
                close_on_finish=False,
            )
            if not loop_result.success:
                self.logger("Diagnostics plan failed, stopping autonomous mode.")
                return loop_result

            if self._completion_reached():
                self.logger("No active errors detected. Goal loop marked complete.")
                self.engine.close()
                return ExecutionResult(
                    success=True,
                    completed_steps=loop_index,
                    total_steps=config.max_loops,
                    state=dict(self.engine.state),
                )

            decision = self.brain.decide_next_steps(
                goal=goal,
                state=self.engine.state,
                allowed_actions=_allowed_actions(config.allow_desktop_actions),
                max_steps=config.max_steps_per_loop,
            )
            decision.steps = _sanitize_decision_steps(decision.steps, config.allow_sensitive_adapter_actions)
            self.logger(f"Brain decision: done={decision.done} | {decision.reason}")
            if decision.done:
                self.engine.close()
                return ExecutionResult(
                    success=True,
                    completed_steps=loop_index,
                    total_steps=config.max_loops,
                    state=dict(self.engine.state),
                )

            if not decision.steps:
                self.logger("Brain returned no executable steps. Stopping loop.")
                self.engine.close()
                return ExecutionResult(
                    success=False,
                    completed_steps=loop_index,
                    total_steps=config.max_loops,
                    state=dict(self.engine.state),
                )

            loop_result = self.engine.run_steps(
                steps=decision.steps,
                plan_name=f"autonomy_actions_{loop_index}",
                plan_description="Execute LLM-selected high-level actions.",
                close_on_finish=False,
            )
            if not loop_result.success:
                self.logger("Action plan failed, stopping autonomous mode.")
                self.engine.close()
                return loop_result

        self.engine.close()
        return ExecutionResult(
            success=False,
            completed_steps=config.max_loops,
            total_steps=config.max_loops,
            state=dict(self.engine.state),
        )

    def _diagnostics_steps(self, config: AutonomousConfig) -> list[TaskStep]:
        steps: list[TaskStep] = []
        if config.target_url:
            steps.append(
                TaskStep(
                    action="open_url",
                    args={"url": config.target_url},
                    description="Open target URL for diagnostics",
                    retries=1,
                    continue_on_error=True,
                )
            )
            steps.append(TaskStep(action="wait", args={"seconds": 1.5}, description="Wait for page load"))
            steps.append(
                TaskStep(
                    action="browser_read_console_errors",
                    description="Capture console errors",
                    save_as="console_errors",
                    continue_on_error=True,
                )
            )

        if config.diagnostics_command:
            steps.append(
                TaskStep(
                    action="run_command",
                    args={"command": config.diagnostics_command, "timeout_seconds": 180},
                    description=f"Run diagnostics command: {config.diagnostics_command}",
                    save_as="last_test_output",
                    continue_on_error=True,
                )
            )
        return steps

    def _completion_reached(self) -> bool:
        console_errors = str(self.engine.state.get("console_errors", "")).strip()
        last_exit = self.engine.state.get("last_command_exit_code")
        if console_errors:
            return False
        if last_exit is None:
            return False
        return int(last_exit) == 0


def _allowed_actions(allow_desktop_actions: bool) -> list[str]:
    core = [
        "log",
        "wait",
        "adapter_list_actions",
        "adapter_call",
        "open_url",
        "search_google",
        "browser_fill",
        "browser_click",
        "browser_press",
        "browser_read_text",
        "browser_read_console_errors",
        "run_command",
        "open_vscode",
        "open_app",
        "open_path",
        "clipboard_get",
        "clipboard_set",
    ]
    if allow_desktop_actions:
        core.extend(["desktop_type", "desktop_hotkey", "desktop_move", "desktop_click", "desktop_switch_window"])
    return core


def _sanitize_decision_steps(steps: list[TaskStep], allow_sensitive: bool) -> list[TaskStep]:
    sanitized: list[TaskStep] = []
    for step in steps:
        if step.action == "adapter_call":
            args = dict(step.args)
            if not allow_sensitive:
                args["confirmed"] = False
            step = TaskStep(
                action=step.action,
                args=args,
                save_as=step.save_as,
                description=step.description,
                condition=step.condition,
                retries=step.retries,
                retry_delay_seconds=step.retry_delay_seconds,
                continue_on_error=step.continue_on_error,
            )
        sanitized.append(step)
    return sanitized
