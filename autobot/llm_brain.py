from __future__ import annotations

import json
import os
from urllib import request
from dataclasses import dataclass
from typing import Any

from .engine import TaskStep

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


@dataclass
class BrainDecision:
    done: bool
    reason: str
    steps: list[TaskStep]


@dataclass
class PlanDraft:
    title: str
    summary: str
    steps: list[TaskStep]


class LLMBrain:
    def __init__(self, logger=None) -> None:
        self.logger = logger or (lambda _msg: None)
        self.model_name = os.getenv("AUTOBOT_LLM_MODEL", "gemini-1.5-flash")
        self.provider = os.getenv("AUTOBOT_LLM_PROVIDER", "gemini").strip().lower()
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
        self.openai_base_url = os.getenv("AUTOBOT_OPENAI_BASE_URL", "https://api.x.ai/v1")
        self.enabled = False
        self._model = None
        if self.provider == "gemini" and genai is not None and bool(self.api_key):
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self.enabled = True
        elif self.provider == "openai_compat" and bool(self.openai_api_key):
            self.enabled = True

    def decide_next_steps(
        self,
        goal: str,
        state: dict[str, Any],
        allowed_actions: list[str],
        max_steps: int = 4,
    ) -> BrainDecision:
        if not self.enabled:
            return self._fallback_decision(goal, state)

    def generate_plan_draft(self, user_prompt: str, allowed_actions: list[str], max_steps: int = 10) -> PlanDraft:
        if not self.enabled:
            return PlanDraft(
                title="Fallback plan",
                summary="LLM not configured; using simple manual fallback.",
                steps=[
                    TaskStep(action="log", args={"message": f"Task received: {user_prompt}"}, description="Log user prompt"),
                    TaskStep(
                        action="log",
                        args={"message": "Configure GROK/Gemini API key to generate autonomous tool plans."},
                        description="Prompt for LLM setup",
                    ),
                ],
            )

        prompt = self._build_draft_prompt(user_prompt=user_prompt, allowed_actions=allowed_actions, max_steps=max_steps)
        try:
            if self.provider == "gemini":
                if self._model is None:
                    raise RuntimeError("Gemini model not initialized.")
                response = self._model.generate_content(prompt)
                text = (response.text or "").strip()
            else:
                text = self._call_openai_compatible(prompt)
            return _parse_plan_draft(text=text, allowed_actions=set(allowed_actions), max_steps=max_steps)
        except Exception as error:  # noqa: BLE001
            self.logger(f"Plan draft generation failed: {error}")
            return PlanDraft(
                title="Fallback plan",
                summary=f"Plan generation failed: {error}",
                steps=[TaskStep(action="log", args={"message": f"Plan generation failed: {error}"}, description="Log planner error")],
            )

        prompt = self._build_prompt(goal=goal, state=state, allowed_actions=allowed_actions, max_steps=max_steps)
        try:
            if self.provider == "gemini":
                if self._model is None:
                    return self._fallback_decision(goal, state)
                response = self._model.generate_content(prompt)
                text = (response.text or "").strip()
            else:
                text = self._call_openai_compatible(prompt)
            decision = _parse_decision(text=text, allowed_actions=set(allowed_actions), max_steps=max_steps)
            return decision
        except Exception as error:  # noqa: BLE001
            self.logger(f"LLM planning failed, fallback mode active: {error}")
            return self._fallback_decision(goal, state)

    def _fallback_decision(self, goal: str, state: dict[str, Any]) -> BrainDecision:
        last_errors = str(state.get("console_errors", "")).strip()
        test_output = str(state.get("last_test_output", "")).strip()
        if not last_errors and not test_output:
            return BrainDecision(done=True, reason="No obvious errors detected.", steps=[])

        prompt = (
            f"Goal: {goal}\n\n"
            "Please fix these issues:\n"
            f"{last_errors}\n\n{test_output}\n\n"
            "Suggest a patch."
        )
        return BrainDecision(
            done=False,
            reason="Fallback: copying diagnostic prompt to clipboard for external assistant.",
            steps=[
                TaskStep(action="clipboard_set", args={"text": prompt}, description="Copy diagnostics prompt to clipboard"),
                TaskStep(
                    action="log",
                    args={"message": "Prompt copied. Paste into CASA/Grok/DeepSeek, apply fix, then rerun loop."},
                    description="Log manual handoff",
                ),
            ],
        )

    def _build_prompt(self, goal: str, state: dict[str, Any], allowed_actions: list[str], max_steps: int) -> str:
        compact_state = {
            "last_command_exit_code": state.get("last_command_exit_code"),
            "last_test_output": _trim_text(str(state.get("last_test_output", "")), 5000),
            "console_errors": _trim_text(str(state.get("console_errors", "")), 5000),
            "last_error": _trim_text(str(state.get("last_error", "")), 2000),
        }
        schema = {
            "done": "boolean",
            "reason": "string",
            "steps": [
                {
                    "action": "one of allowed actions",
                    "args": "object with action arguments; for adapter_call use adapter, adapter_action, params, confirmed",
                    "description": "short text",
                    "save_as": "optional state key",
                    "retries": "optional int",
                    "continue_on_error": "optional bool",
                }
            ],
        }
        return (
            "You are the planning brain for a local automation agent.\n"
            "Return ONLY strict JSON.\n"
            f"Goal: {goal}\n"
            f"Allowed actions: {allowed_actions}\n"
            f"Max steps: {max_steps}\n"
            f"Current state: {json.dumps(compact_state)}\n"
            f"Output schema: {json.dumps(schema)}\n"
            "Rules:\n"
            "- Use high-level tool calls, keep steps short.\n"
            "- Prefer diagnostics and safe actions.\n"
            "- If enough evidence suggests task is complete, set done=true.\n"
            "- Never use destructive commands.\n"
        )

    def _build_draft_prompt(self, user_prompt: str, allowed_actions: list[str], max_steps: int) -> str:
        schema = {
            "title": "short plan title",
            "summary": "one paragraph summary",
            "steps": [
                {
                    "action": "one of allowed actions",
                    "args": "object with action arguments",
                    "description": "step description",
                    "save_as": "optional",
                    "retries": "optional int",
                    "continue_on_error": "optional bool",
                }
            ],
        }
        return (
            "You are an automation planner for a local autonomous computer controller.\n"
            "Return ONLY strict JSON.\n"
            f"User prompt: {user_prompt}\n"
            f"Allowed actions: {allowed_actions}\n"
            f"Max steps: {max_steps}\n"
            f"Output schema: {json.dumps(schema)}\n"
            "Rules:\n"
            "- Produce safe, high-level action steps.\n"
            "- Use explicit action arguments.\n"
            "- Keep steps deterministic and practical.\n"
            "- Avoid destructive actions.\n"
        )

    def _call_openai_compatible(self, prompt: str) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY/XAI_API_KEY is missing.")
        endpoint = self.openai_base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            },
        )
        with request.urlopen(req, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned by openai_compat provider.")
        message = choices[0].get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError("Empty content returned by openai_compat provider.")
        return content


def _parse_decision(text: str, allowed_actions: set[str], max_steps: int) -> BrainDecision:
    payload = _extract_json(text)
    data = json.loads(payload)
    done = bool(data.get("done", False))
    reason = str(data.get("reason", "")).strip() or "No reason provided."
    raw_steps = data.get("steps", [])
    steps: list[TaskStep] = []

    if isinstance(raw_steps, list):
        for item in raw_steps[:max_steps]:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip()
            if not action or action not in allowed_actions:
                continue
            args = item.get("args", {})
            if not isinstance(args, dict):
                args = {}
            steps.append(
                TaskStep(
                    action=action,
                    args=args,
                    description=str(item.get("description", "")).strip(),
                    save_as=_none_if_empty(item.get("save_as")),
                    retries=int(item.get("retries", 0)),
                    continue_on_error=bool(item.get("continue_on_error", False)),
                )
            )

    return BrainDecision(done=done, reason=reason, steps=steps)


def _parse_plan_draft(text: str, allowed_actions: set[str], max_steps: int) -> PlanDraft:
    payload = _extract_json(text)
    data = json.loads(payload)
    title = str(data.get("title", "AI Plan")).strip() or "AI Plan"
    summary = str(data.get("summary", "")).strip() or "No summary provided."
    raw_steps = data.get("steps", [])
    steps: list[TaskStep] = []
    if isinstance(raw_steps, list):
        for item in raw_steps[:max_steps]:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip()
            if not action or action not in allowed_actions:
                continue
            args = item.get("args", {})
            if not isinstance(args, dict):
                args = {}
            steps.append(
                TaskStep(
                    action=action,
                    args=args,
                    description=str(item.get("description", "")).strip(),
                    save_as=_none_if_empty(item.get("save_as")),
                    retries=int(item.get("retries", 0)),
                    continue_on_error=bool(item.get("continue_on_error", False)),
                )
            )
    return PlanDraft(title=title, summary=summary, steps=steps)


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return cleaned[start : end + 1]


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _none_if_empty(value: Any) -> str | None:
    text = str(value).strip()
    return text or None
