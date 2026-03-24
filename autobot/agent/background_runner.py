"""
Background Task Runner — Non-visual tasks that run in parallel with visual AgentLoop.

Key differences from AgentLoop:
- No screenshot, no DOM snapshot, no ScreenLock acquired
- Only non-visual tool calls are allowed: terminal, files, clipboard, research, kaggle, vault
- Mouse, keyboard, and display calls are silently blocked
- Can run fully in parallel with a foreground visual task

Best for:
  - Running long training jobs: computer.terminal.start("python train.py")
  - Monitoring APIs and checking output periodically
  - File processing, data analysis, Kaggle submission uploads
  - Background research while the user's browser session stays uninterrupted

Usage:
    runner = BackgroundTaskRunner.from_env(log_callback=log)
    result = await runner.run("train the model and wait for it to finish")
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from autobot.computer.computer import Computer

logger = logging.getLogger(__name__)

# Modules that require screen access — blocked in background mode
_BLOCKED_MODULES = frozenset({"mouse", "keyboard", "display"})

# Short system prompt for background tasks (no vision section — saves ~3,000 tokens/step)
_BACKGROUND_SYSTEM_PROMPT = """
You are a background automation agent. You have NO screen access and cannot see, click, or type.
Your only tools are terminal commands, file operations, clipboard, research, kaggle, and vault.

Available tools (call via computer_call action):
- computer.terminal.run(cmd, cwd=".")  — run a shell command and return its output
- computer.terminal.start(cmd, cwd=".") — start a long-running command, returns PID
- computer.terminal.output(pid)         — read output from a background process
- computer.terminal.running(pid)        — check if process is still running
- computer.terminal.wait(pid, timeout)  — wait for process to finish
- computer.files.list(path)             — list files in a directory
- computer.files.read(path)             — read a file's contents
- computer.files.write(path, content)   — write content to a file
- computer.clipboard.get()              — read current clipboard text
- computer.clipboard.set(text)          — write text to clipboard
- computer.vault.get(name)              — retrieve a stored secret
- computer.kaggle.submit(...)           — submit to a Kaggle competition

Rules:
1. Use terminal.start() for long-running commands (training, downloads, installs).
2. Poll terminal.running(pid) + terminal.output(pid) to monitor progress.
3. Use done action when the task is complete.
4. Record key findings in the memory field.

Respond in JSON:
{
  "thinking": "<brief plan>",
  "next_goal": "<one action to do now>",
  "memory": "<key facts to remember>",
  "action": [{"computer_call": {"call": "computer.terminal.run(\"echo hello\")"}}]
}
""".strip()


class BackgroundTaskRunner:
    """
    Runs non-visual LLM-driven tasks without ScreenLock or screenshot overhead.

    Safe to run in parallel with a foreground AgentLoop — it never touches
    the mouse, keyboard, or display.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gemini-2.0-flash",
        max_steps: int = 50,
        log_callback: Callable[[str], None] | None = None,
        task_id: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.log = log_callback or (lambda msg: logger.info(msg))
        self.task_id = task_id or f"bg-{id(self)}"
        self.computer = Computer()
        self.step_number = 0
        self.status: str = "idle"
        self.result: str = ""
        self._history: list[dict] = []   # lightweight text history (no screenshots)

    @classmethod
    def from_env(
        cls,
        log_callback: Callable[[str], None] | None = None,
        task_id: str | None = None,
    ) -> "BackgroundTaskRunner":
        from autobot.agent.runner import _create_llm_client
        llm_client = _create_llm_client()
        model = os.getenv("AUTOBOT_LLM_MODEL", "gemini-2.0-flash")
        return cls(
            llm_client=llm_client,
            model=model,
            max_steps=int(os.getenv("AUTOBOT_BG_MAX_STEPS", "50")),
            log_callback=log_callback,
            task_id=task_id,
        )

    async def run(self, goal: str) -> str:
        """
        Run a background task end-to-end without acquiring ScreenLock.

        Returns:
            Result text from the agent's done action, or a timeout message.
        """
        self.status = "running"
        self.log(f"[BG {self.task_id}] Starting background task: {goal}")

        for step in range(self.max_steps):
            self.step_number = step
            self.log(f"[BG {self.task_id}] Step {step + 1}/{self.max_steps}")

            try:
                output = await self._call_llm(goal)
            except Exception as e:
                self.log(f"[BG {self.task_id}] LLM error: {e}")
                await asyncio.sleep(2.0)
                continue

            if output is None:
                self.log(f"[BG {self.task_id}] Empty LLM response — skipping step")
                continue

            # Execute actions (non-visual only)
            for action_dict in output.get("action", []):
                if "done" in action_dict:
                    done_text = action_dict["done"].get("text", "Task completed.")
                    self.log(f"[BG {self.task_id}] Done: {done_text}")
                    self.status = "done"
                    self.result = done_text
                    return done_text

                if "computer_call" in action_dict:
                    call_str = action_dict["computer_call"].get("call", "")
                    result = await self._dispatch(call_str)
                    self.log(f"[BG {self.task_id}] {call_str[:80]} → {str(result)[:120]}")

                    # Record in history for next step context
                    self._history.append({
                        "step": step,
                        "goal": output.get("next_goal", ""),
                        "call": call_str,
                        "result": str(result)[:300],
                    })
                    # Keep last 15 steps in history
                    if len(self._history) > 15:
                        self._history = self._history[-15:]

            await asyncio.sleep(0.5)

        self.status = "done"
        self.result = f"Background task reached max steps ({self.max_steps}) without explicit completion."
        self.log(f"[BG {self.task_id}] {self.result}")
        return self.result

    async def _call_llm(self, goal: str) -> dict | None:
        """Call the LLM with current task context. Returns parsed JSON dict or None."""
        history_text = "\n".join(
            f"Step {h['step']}: {h['goal']}\n  → {h['call']}\n  Result: {h['result']}"
            for h in self._history[-10:]
        ) or "(no history yet)"

        user_content = (
            f"Goal: {goal}\n\n"
            f"Recent history:\n{history_text}\n\n"
            f"What is your next action?"
        )

        messages = [
            {"role": "system", "content": _BACKGROUND_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        args: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = await self.llm_client.chat.completions.create(**args)
            if not resp.choices:
                return None
            content = resp.choices[0].message.content
            if not content or not content.strip():
                return None
            return json.loads(content)
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.warning(f"[BG] LLM call failed: {e}")
            return None

    async def _dispatch(self, call_str: str) -> Any:
        """
        Safely dispatch a computer.* call, blocking screen-touching modules.
        Uses the same AST-safe dispatch pattern as AgentLoop.
        """
        import ast
        import re

        call_str = call_str.strip()
        if not call_str.startswith("computer."):
            return f"Error: call must start with 'computer.': {call_str}"

        pattern = r"^computer\.(\w+)\.(\w+)\((.*)\)$"
        match = re.match(pattern, call_str, re.DOTALL)
        if not match:
            return f"Error: cannot parse call: {call_str}"

        module_name, method_name, args_str = match.groups()

        # Block screen-touching modules in background mode
        if module_name in _BLOCKED_MODULES:
            return (
                f"[BLOCKED] '{module_name}' is not available in background mode. "
                "Use terminal, files, clipboard, research, kaggle, or vault instead."
            )

        module = getattr(self.computer, module_name, None)
        if module is None:
            return f"Error: unknown module '{module_name}'"

        method = getattr(module, method_name, None)
        if method is None:
            return f"Error: unknown method 'computer.{module_name}.{method_name}'"

        # Parse arguments safely via AST
        args: list = []
        kwargs: dict = {}
        if args_str.strip():
            try:
                tree = ast.parse(f"_f({args_str})", mode="eval")
                call_node = tree.body
                for node in call_node.args:
                    args.append(ast.literal_eval(node))
                for kw in call_node.keywords:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
            except Exception as e:
                return f"Error parsing arguments: {e}"

        try:
            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = await asyncio.to_thread(method, *args, **kwargs)
            return result
        except Exception as e:
            return f"Error executing {call_str}: {e}"
