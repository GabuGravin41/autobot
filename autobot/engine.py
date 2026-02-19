from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .adapters.base import AdapterConfirmationError
from .adapters.manager import AdapterManager
from .browser_agent import BrowserController

try:
    import pyautogui
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None


LogFn = Callable[[str], None]


@dataclass
class TaskStep:
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    save_as: str | None = None
    description: str = ""
    condition: str | None = None
    retries: int = 0
    retry_delay_seconds: float = 1.0
    continue_on_error: bool = False


@dataclass
class WorkflowPlan:
    name: str
    description: str
    steps: list[TaskStep]


@dataclass
class ExecutionResult:
    success: bool
    completed_steps: int
    total_steps: int
    state: dict[str, Any]


class ActionLimiter:
    def __init__(self, logger: LogFn, max_per_minute: int = 45, min_interval_s: float = 0.08) -> None:
        self.logger = logger
        self.max_per_minute = max_per_minute
        self.min_interval_s = min_interval_s
        self._recent: list[float] = []
        self._last_action_ts = 0.0

    def before_action(self, action_name: str) -> None:
        now = time.time()
        if self._last_action_ts > 0:
            since_last = now - self._last_action_ts
            if since_last < self.min_interval_s:
                delay = self.min_interval_s - since_last
                time.sleep(delay)
                now = time.time()

        one_minute_ago = now - 60.0
        self._recent = [item for item in self._recent if item >= one_minute_ago]
        if len(self._recent) >= self.max_per_minute:
            delay = max(0.2, self._recent[0] + 60.0 - now)
            self.logger(f"Rate limiter active before '{action_name}', sleeping {delay:.2f}s.")
            time.sleep(delay)
            now = time.time()
            one_minute_ago = now - 60.0
            self._recent = [item for item in self._recent if item >= one_minute_ago]

        self._recent.append(now)
        self._last_action_ts = now


class AutomationEngine:
    def __init__(self, logger: LogFn | None = None) -> None:
        self.logger = logger or (lambda _msg: None)
        self.browser = BrowserController()
        self.adapters = AdapterManager(browser=self.browser, logger=self.logger)
        self.state: dict[str, Any] = {}
        self._cancel_requested = False
        self._limiter = ActionLimiter(logger=self.logger)

    def cancel(self) -> None:
        self._cancel_requested = True
        self.logger("Cancellation requested. Stopping at next safe checkpoint.")

    def close(self) -> None:
        self.browser.close()

    def get_adapter_library(self) -> dict[str, dict[str, dict[str, Any]]]:
        return self.adapters.list_adapters()

    def run_plan(self, plan: WorkflowPlan) -> ExecutionResult:
        return self.run_steps(
            steps=plan.steps,
            plan_name=plan.name,
            plan_description=plan.description,
            close_on_finish=True,
        )

    def run_steps(
        self,
        steps: list[TaskStep],
        plan_name: str = "dynamic_plan",
        plan_description: str = "",
        close_on_finish: bool = False,
    ) -> ExecutionResult:
        self._cancel_requested = False
        self.logger(f"Running workflow: {plan_name}")
        if plan_description:
            self.logger(plan_description)
        total = len(steps)
        completed = 0
        run_started_at = datetime.now(timezone.utc)
        step_logs: list[dict[str, Any]] = []
        run_success = False
        history_written = False

        try:
            for idx, step in enumerate(steps, start=1):
                if self._cancel_requested:
                    self.logger("Execution cancelled by user.")
                    run_path = self._write_run_history(
                        plan_name=plan_name,
                        plan_description=plan_description,
                        started_at=run_started_at,
                        finished_at=datetime.now(timezone.utc),
                        success=False,
                        completed_steps=completed,
                        total_steps=total,
                        step_logs=step_logs,
                    )
                    self.state["last_run_history_path"] = run_path
                    history_written = True
                    return ExecutionResult(
                        success=False,
                        completed_steps=completed,
                        total_steps=total,
                        state=dict(self.state),
                    )

                if not self._condition_allows(step):
                    self.logger(f"[{idx}/{total}] Skipped: {step.description or step.action} (condition=false)")
                    step_logs.append(
                        {
                            "index": idx,
                            "action": step.action,
                            "description": step.description,
                            "status": "skipped",
                            "condition": step.condition,
                        }
                    )
                    completed += 1
                    continue

                self.logger(f"[{idx}/{total}] {step.description or step.action}")
                attempts = max(1, step.retries + 1)
                value: Any = None
                last_error: Exception | None = None
                step_started_at = datetime.now(timezone.utc)
                step_log: dict[str, Any] = {
                    "index": idx,
                    "action": step.action,
                    "description": step.description,
                    "condition": step.condition,
                    "attempts_allowed": attempts,
                    "attempts_used": 0,
                    "status": "running",
                    "args": _json_safe(_render_vars(step.args, self.state)),
                    "started_at": step_started_at.isoformat(),
                }
                for attempt in range(1, attempts + 1):
                    try:
                        rendered_args = _render_vars(step.args, self.state)
                        self._limiter.before_action(step.action)
                        value = self._execute_action(step.action, rendered_args)
                        step_log["attempts_used"] = attempt
                        if step.save_as:
                            self.state[step.save_as] = value
                            self.logger(f"Saved output as '{step.save_as}'.")
                        last_error = None
                        break
                    except Exception as error:  # noqa: BLE001
                        step_log["attempts_used"] = attempt
                        last_error = error
                        if attempt < attempts:
                            self.logger(
                                f"Step failed (attempt {attempt}/{attempts}) -> retry in {step.retry_delay_seconds}s: {error}"
                            )
                            time.sleep(step.retry_delay_seconds)
                        elif step.continue_on_error:
                            self.logger(f"Step failed and marked continue_on_error: {error}")
                            self.state["last_error"] = str(error)
                            step_log["status"] = "failed_continue"
                            step_log["error"] = str(error)
                            break
                        else:
                            raise

                if last_error and not step.continue_on_error:
                    step_log["status"] = "failed"
                    step_log["error"] = str(last_error)
                    step_log["finished_at"] = datetime.now(timezone.utc).isoformat()
                    step_logs.append(step_log)
                    raise last_error

                if "status" not in step_log or step_log["status"] == "running":
                    step_log["status"] = "ok"
                step_log["result"] = _json_safe(value)
                step_log["finished_at"] = datetime.now(timezone.utc).isoformat()
                step_logs.append(step_log)
                completed += 1

            self.logger("Workflow completed successfully.")
            run_success = True
            run_path = self._write_run_history(
                plan_name=plan_name,
                plan_description=plan_description,
                started_at=run_started_at,
                finished_at=datetime.now(timezone.utc),
                success=True,
                completed_steps=completed,
                total_steps=total,
                step_logs=step_logs,
            )
            self.state["last_run_history_path"] = run_path
            history_written = True
            return ExecutionResult(
                success=True,
                completed_steps=completed,
                total_steps=total,
                state=dict(self.state),
            )
        except Exception as error:
            self.state["last_error"] = str(error)
            raise
        finally:
            if not history_written:
                run_path = self._write_run_history(
                    plan_name=plan_name,
                    plan_description=plan_description,
                    started_at=run_started_at,
                    finished_at=datetime.now(timezone.utc),
                    success=run_success,
                    completed_steps=completed,
                    total_steps=total,
                    step_logs=step_logs,
                )
                self.state["last_run_history_path"] = run_path
            if close_on_finish:
                self.close()

    def _execute_action(self, action: str, args: dict[str, Any]) -> Any:
        if action == "open_url":
            message = self.browser.goto(str(args["url"]))
            self.logger(message)
            return message

        if action == "browser_mode_status":
            data = self.browser.mode_status()
            self.logger(f"Browser mode status: active={data.get('active_mode')} configured={data.get('configured_mode')}")
            return data

        if action == "benchmark_run":
            from .benchmark import run_benchmarks

            data = run_benchmarks(logger=self.logger)
            self.logger(f"Benchmark suite completed: {len(data)} cases.")
            return data

        if action == "adapter_list_actions":
            data = self.adapters.list_adapters()
            self.logger("Loaded adapter action libraries.")
            return data

        if action == "adapter_set_policy":
            profile = str(args.get("profile", "balanced")).strip()
            value = self.adapters.set_policy(profile)
            self.state["adapter_policy_profile"] = value
            self.logger(f"Adapter policy set to: {value}")
            return value

        if action == "adapter_prepare_sensitive":
            adapter_name = str(args.get("adapter", "")).strip()
            adapter_action = str(args.get("adapter_action", "")).strip()
            params = args.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("adapter_prepare_sensitive expects args.params as an object.")
            data = self.adapters.prepare_sensitive_action(adapter_name=adapter_name, action=adapter_action, params=params)
            self.state["last_sensitive_prepare"] = data
            self.logger(f"Prepared sensitive action token for: {adapter_name}.{adapter_action}")
            return data

        if action == "adapter_confirm_sensitive":
            token = str(args.get("token", "")).strip()
            if not token:
                raise ValueError("adapter_confirm_sensitive requires token.")
            result = self.adapters.confirm_sensitive_action(token)
            self.logger("Confirmed and executed sensitive adapter action.")
            return result

        if action == "adapter_get_telemetry":
            data = self.adapters.telemetry()
            self.logger("Loaded adapter telemetry.")
            return data

        if action == "adapter_call":
            adapter_name = str(args.get("adapter", "")).strip()
            adapter_action = str(args.get("adapter_action", "")).strip()
            params = args.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("adapter_call expects args.params as an object.")
            confirmed = bool(args.get("confirmed", False))
            try:
                result = self.adapters.call(
                    adapter_name=adapter_name,
                    action=adapter_action,
                    params=params,
                    confirmed=confirmed,
                )
            except AdapterConfirmationError as error:
                self.state["last_error"] = str(error)
                raise
            self.logger(f"Adapter call completed: {adapter_name}.{adapter_action}")
            return result

        if action == "search_google":
            message = self.browser.search(str(args["query"]))
            self.logger(message)
            return message

        if action == "browser_fill":
            message = self.browser.fill(
                selector=str(args["selector"]),
                text=str(args["text"]),
                timeout_ms=int(args.get("timeout_ms", 10000)),
            )
            self.logger(message)
            return message

        if action == "browser_click":
            message = self.browser.click(
                selector=str(args["selector"]),
                timeout_ms=int(args.get("timeout_ms", 10000)),
            )
            self.logger(message)
            return message

        if action == "browser_press":
            message = self.browser.press(str(args["key"]))
            self.logger(message)
            return message

        if action == "browser_read_text":
            text = self.browser.read_text(
                selector=str(args["selector"]),
                timeout_ms=int(args.get("timeout_ms", 10000)),
            )
            self.logger(f"Captured text from {args['selector']}.")
            return text

        if action == "browser_read_console_errors":
            errors = self.browser.read_console_errors()
            self.logger(f"Captured {len(errors)} console-like error entries.")
            return "\n".join(errors)

        if action == "open_vscode":
            subprocess.Popen(["code"])
            self.logger("Opened VS Code.")
            return "Opened VS Code."

        if action == "open_app":
            command = str(args["command"])
            subprocess.Popen(command, shell=True)
            self.logger(f"Started app command: {command}")
            return command

        if action == "open_path":
            target_path = str(args["path"])
            path_obj = Path(target_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Path does not exist: {target_path}")
            if hasattr(os, "startfile"):
                os.startfile(path_obj)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", target_path])
            self.logger(f"Opened path: {target_path}")
            return target_path

        if action == "run_command":
            command = str(args["command"])
            timeout = int(args.get("timeout_seconds", 120))
            result = subprocess.run(command, capture_output=True, text=True, shell=True, timeout=timeout, check=False)
            output = (result.stdout or result.stderr or "").strip()
            self.state["last_command_exit_code"] = result.returncode
            self.state["last_command_output"] = output
            self.logger(f"Command exit code: {result.returncode}")
            if output:
                self.logger(output[:5000])
            return output

        if action == "wait":
            seconds = float(args.get("seconds", 1))
            time.sleep(seconds)
            self.logger(f"Waited {seconds:.1f}s.")
            return seconds

        if action == "clipboard_set":
            text = str(args.get("text", ""))
            _set_clipboard(text)
            self.logger("Clipboard updated.")
            return text

        if action == "clipboard_get":
            text = _get_clipboard()
            self.logger("Clipboard captured.")
            return text

        if action == "desktop_type":
            _require_pyautogui()
            text = str(args.get("text", ""))
            interval = float(args.get("interval", 0.02))
            pyautogui.write(text, interval=interval)
            self.logger("Typed text through desktop keyboard automation.")
            return text

        if action == "desktop_hotkey":
            _require_pyautogui()
            keys = args.get("keys", [])
            if not isinstance(keys, list) or not keys:
                raise ValueError("desktop_hotkey requires 'keys' list.")
            pyautogui.hotkey(*[str(key) for key in keys])
            self.logger(f"Sent hotkey: {' + '.join([str(key) for key in keys])}")
            return keys

        if action == "desktop_click":
            _require_pyautogui()
            x = int(args["x"])
            y = int(args["y"])
            button = str(args.get("button", "left"))
            pyautogui.click(x=x, y=y, button=button)
            self.logger(f"Clicked desktop at ({x}, {y}) using {button} button.")
            return {"x": x, "y": y, "button": button}

        if action == "desktop_move":
            _require_pyautogui()
            x = int(args["x"])
            y = int(args["y"])
            duration = float(args.get("duration", 0.2))
            pyautogui.moveTo(x, y, duration=duration)
            self.logger(f"Moved cursor to ({x}, {y}).")
            return {"x": x, "y": y}

        if action == "desktop_switch_window":
            _require_pyautogui()
            pyautogui.hotkey("alt", "tab")
            self.logger("Switched to next window (Alt+Tab).")
            return "alt+tab"

        if action == "desktop_press":
            _require_pyautogui()
            key = str(args.get("key", "enter"))
            pyautogui.press(key)
            self.logger(f"Pressed desktop key: {key}")
            return key

        if action == "log":
            message = str(args.get("message", ""))
            self.logger(message)
            return message

        raise ValueError(f"Unknown action '{action}'")

    def _condition_allows(self, step: TaskStep) -> bool:
        if not step.condition:
            return True
        expression = _render_vars(step.condition, self.state)
        return _evaluate_condition(str(expression), self.state)

    def _write_run_history(
        self,
        plan_name: str,
        plan_description: str,
        started_at: datetime,
        finished_at: datetime,
        success: bool,
        completed_steps: int,
        total_steps: int,
        step_logs: list[dict[str, Any]],
    ) -> str:
        runs_dir = Path.cwd() / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        path = runs_dir / f"{stamp}_{plan_name}.json"
        payload = {
            "plan_name": plan_name,
            "plan_description": plan_description,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "success": success,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "state_snapshot": _json_safe(dict(self.state)),
            "adapter_telemetry": _json_safe(self.adapters.telemetry()),
            "steps": step_logs,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger(f"Run history written: {path}")
        return str(path)


def _render_vars(value: Any, state: dict[str, Any]) -> Any:
    if isinstance(value, str):
        rendered = value
        for key, state_value in state.items():
            rendered = rendered.replace("{" + key + "}", str(state_value))
        return rendered
    if isinstance(value, list):
        return [_render_vars(item, state) for item in value]
    if isinstance(value, dict):
        return {k: _render_vars(v, state) for k, v in value.items()}
    return value


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _evaluate_condition(expression: str, state: dict[str, Any]) -> bool:
    text = expression.strip()
    if not text:
        return True
    if text.lower() in {"true", "1", "yes"}:
        return True
    if text.lower() in {"false", "0", "no"}:
        return False

    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_locals = {"state": state}
    try:
        result = eval(text, safe_globals, safe_locals)  # noqa: S307
    except Exception:
        return False
    return bool(result)


def _set_clipboard(text: str) -> None:
    command = "$v = @'\n" + text + "\n'@; Set-Clipboard -Value $v"
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )


def _get_clipboard() -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout or "").strip()


def _require_pyautogui() -> None:
    if pyautogui is None:
        raise RuntimeError("Desktop actions require pyautogui. Install it with: pip install pyautogui")
