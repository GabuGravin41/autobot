from __future__ import annotations

import json
import os
import sys
from urllib import request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from typing import Any

from .engine import TaskStep


def _get_genai():
    """Lazy import to avoid FutureWarning when using Grok (openai_compat)."""
    try:
        import google.generativeai as genai  # noqa: PLC0415
        return genai
    except ImportError:  # pragma: no cover - optional dependency
        return None


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
        self.model_name = os.getenv("AUTOBOT_LLM_MODEL", "")
        raw = os.getenv("AUTOBOT_LLM_PROVIDER", "openrouter").strip().lower()
        self.provider = "openai_compat" if raw == "openai_compact" else raw
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self.openai_base_url = os.getenv("AUTOBOT_OPENAI_BASE_URL", "https://api.x.ai/v1")
        
        if self.provider == "openrouter":
            self.openai_base_url = "https://openrouter.ai/api/v1"
            # Prioritize OpenRouter specific key if available
            self.openai_api_key = os.getenv("OPENROUTER_API_KEY") or self.openai_api_key
            if not self.model_name:
                # DeepSeek V3.1 free on OpenRouter (recommended default)
                self.model_name = "deepseek/deepseek-chat-v3.1:free"

        if self.provider == "gemini" and not self.model_name:
            self.model_name = "gemini-1.5-flash"
        self.enabled = False
        self._model = None
        if self.provider == "gemini" and bool(self.api_key):
            _genai = _get_genai()
            if _genai is not None:
                _genai.configure(api_key=self.api_key)
                self._model = _genai.GenerativeModel(self.model_name)
                self.enabled = True
        elif (self.provider == "openai_compat" or self.provider == "openrouter") and bool(self.openai_api_key):
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

    def generate_plan_draft(
        self,
        user_prompt: str,
        allowed_actions: list[str],
        max_steps: int = 10,
        tool_catalog: str | None = None,
    ) -> PlanDraft:
        if not self.enabled:
            return PlanDraft(
                title="Fallback plan",
                summary="LLM not configured; using simple manual fallback.",
                steps=[
                    TaskStep(action="log", args={"message": f"Task received: {user_prompt}"}, description="Log user prompt"),
                    TaskStep(
                        action="log",
                        args={"message": "Set OPENROUTER_API_KEY and AUTOBOT_LLM_PROVIDER=openrouter for DeepSeek V3.1 (free), or configure Gemini/Grok."},
                        description="Prompt for LLM setup",
                    ),
                ],
            )

        prompt = self._build_draft_prompt(
            user_prompt=user_prompt,
            allowed_actions=allowed_actions,
            max_steps=max_steps,
            tool_catalog=tool_catalog,
        )
        try:
            if self.provider == "gemini":
                if self._model is None:
                    raise RuntimeError("Gemini model not initialized.")
                response = self._model.generate_content(prompt)
                text = (response.text or "").strip()
            else:
                try:
                    text = self._call_openai_compatible(prompt)
                except RuntimeError as e:
                    if "403" in str(e) and "x.ai" in self.openai_base_url:
                        self.logger("403 with current model; retrying with grok-2")
                        text = self._call_openai_compatible(prompt, model_override="grok-2")
                    else:
                        raise
            return _parse_plan_draft(text=text, allowed_actions=set(allowed_actions), max_steps=max_steps)
        except Exception as error:  # noqa: BLE001
            err_msg = str(error).strip()
            self.logger(f"Plan draft generation failed: {err_msg}")
            print(f"Autobot LLM error: {err_msg}", file=sys.stderr)
            sys.stderr.flush()
            return PlanDraft(
                title="Fallback plan",
                summary=f"Plan generation failed: {err_msg}",
                steps=[TaskStep(action="log", args={"message": f"Plan generation failed: {err_msg}"}, description="Log planner error")],
            )

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
        # Rich state feedback so the AI can iterate: use run result, clipboard, errors, and what was saved.
        compact_state = {
            "last_command_exit_code": state.get("last_command_exit_code"),
            "last_command_output": _trim_text(str(state.get("last_command_output", "")), 4000),
            "last_test_output": _trim_text(str(state.get("last_test_output", "")), 5000),
            "console_errors": _trim_text(str(state.get("console_errors", "")), 5000),
            "last_error": _trim_text(str(state.get("last_error", "")), 2000),
            "autonomy_loops": state.get("autonomy_loops"),
            "run_dir": state.get("run_dir"),
            "last_run_history_path": state.get("last_run_history_path"),
            "current_url": _trim_text(str(state.get("current_url", "")), 500),
            "last_screenshot_path": state.get("last_screenshot_path"),
            "last_notify_message": state.get("last_notify_message"),
        }
        # Include short previews of saved state keys so the AI knows what data is available (e.g. latex_text, doc_text).
        saved_keys = [k for k, v in state.items() if k not in (
            "run_dir", "last_run_history_path", "adapter_telemetry", "telemetry",
            "autonomy_goal", "autonomy_loops"
        ) and v is not None and str(v).strip()]
        if saved_keys:
            compact_state["saved_state_keys"] = saved_keys
            for key in ["latex_text", "doc_text", "console_errors"]:
                if key in state and state[key]:
                    s = str(state[key])
                    compact_state[f"{key}_preview"] = _trim_text(s, 800)
        schema = {
            "done": "boolean",
            "reason": "string",
            "steps": [
                {
                    "id": "optional unique string identifier for this step",
                    "depends_on": "optional list of step ids that must complete before this step",
                    "target_node": "optional string ID (e.g. AnyDesk node alias) if this step must run remotely",
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
            "You are the planning brain for a local automation agent. Drive the system like a human would: be patient, check if pages have loaded, use search, and read the UI state before acting.\n"
            "Return ONLY strict JSON.\n"
            f"Goal: {goal}\n"
            f"Allowed actions: {allowed_actions}\n"
            f"Max steps: {max_steps}\n"
            f"Current state: {json.dumps(compact_state)}\n"
            f"Output schema: {json.dumps(schema)}\n"
            "Rules:\n"
            "- MODES ARE STRATEGIC CAPABILITIES: Use 'browser_set_mode' to adapt to the site's defenses.\n"
            "  - 'devtools': High precision. Required for CSS selectors, 'browser_click', 'browser_fill', and 'browser_get_content' (copying).\n"
            "  - 'human_profile': High stealth. Use for sites with bot detection. Disable selector tools. Use 'desktop_type' and 'desktop_press'.\n"
            "- DISTRIBUTED COMPUTING: Set 'target_node' to an alias or node ID if a specific sub-agent or remote machine should execute the task. Leave null for local.\n"
            "- DATA EXTRACTION (COPYING): Use 'browser_get_content' in 'devtools' mode to capture page text. Then save it using 'state_set'.\n"
            "- NAVIGATION: Use 'browser_click_text' for robust navigation without CSS selectors. It works in 'devtools' mode.\n"
            "- AUTONOMY: If a site blocks you, switch mode and try an alternative tool. Do NOT ask the user for help unless you are blocked by a CAPTCHA.\n"
            "- CAPTCHAS: Use 'request_human_help' only if you see a CAPTCHA or 'Verify you are human' screen.\n"
            "- WHEN last_error IS SET: The previous step failed. Adapt: suggest 'wait' (longer, e.g. 30–120s), a different action, or request_human_help. Do not repeat the same failing step without a change (e.g. wait first, or try another selector/tool).\n"
            "- RELIABILITY OVER SPEED: Sites and LLMs (Grok, ChatGPT) often need 60–120s to respond. Prefer adding a 'wait' step rather than assuming the page is ready.\n"
        )

    def _build_draft_prompt(
        self,
        user_prompt: str,
        allowed_actions: list[str],
        max_steps: int,
        tool_catalog: str | None = None,
    ) -> str:
        schema = {
            "title": "short plan title",
            "summary": "one paragraph summary",
            "steps": [
                {
                    "id": "optional unique string identifier for this step",
                    "depends_on": "optional list of step ids that must complete before this step",
                    "action": "one of allowed actions",
                    "args": "object with action arguments; for adapter_call use adapter, adapter_action, params (object), confirmed (bool)",
                    "description": "step description",
                    "save_as": "optional",
                    "retries": "optional int",
                    "continue_on_error": "optional bool",
                }
            ],
        }
        catalog_block = ""
        if tool_catalog:
            catalog_block = f"\nAvailable tools (use these to fulfill the user request):\n{tool_catalog}\n"
        return (
            "You are a strict JSON-only automation planner. "
            "Convert the user request into a sequence of atomic tool calls.\n\n"
            "### RULES:\n"
            "1. JSON FORMAT: Output ONLY valid, parsable JSON. No markdown backticks. No conversational filler.\n"
            "2. ESCAPING: Be extremely careful with special characters. All quotes inside strings MUST be escaped (e.g. \\\"text\\\"). All newlines must be literal \\n.\n"
            "3. GRANULARITY AND PARALLELISM: Break goals down. If tasks can be done in parallel, use 'id' and 'depends_on' properties to create a graph of plans (DAG)."
            "   Example: Step 1 sets 'id':'A'. Step 2 sets 'id':'B'. Step 3 sets 'depends_on':['A', 'B'].\n"
            "4. DISTRIBUTED EXECUTION: If processing should be given to a remote worker (AnyDesk sub-agent), set 'target_node' on the step.\n"
            "5. DATA FLOW: Use 'browser_get_content' to capture large blocks of text. Use 'state_set' to save it. Use '{key}' in later steps to paste it.\n"
            "6. FIDELITY: Use the exact details provided by the user in your prompts/messages.\n"
            "7. MODES: Use 'browser_set_mode' to 'devtools' before any extraction or clicking.\n\n"
            f"User request: {user_prompt}\n"
            f"Allowed action names: {allowed_actions}\n"
            f"Max steps: {max_steps}\n"
            f"{catalog_block}"
            f"Output schema: {json.dumps(schema)}\n"
        )

    def _call_openai_compatible(self, prompt: str, model_override: str | None = None) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY/XAI_API_KEY is missing.")
        model = model_override or self.model_name
        messages = [
            {"role": "system", "content": "Return only strict JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            from openai import OpenAI
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/autobot-ai",
                    "X-Title": "Autobot Jarvis",
                }
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url.rstrip("/"),
                default_headers=extra_headers
            )
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            if not completion.choices:
                raise RuntimeError("No choices returned by openai_compat provider.")
            content = (completion.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Empty content returned by openai_compat provider.")
            # Strip reasoning if it exists (DeepSeek R1 <think> tags)
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()
            return content
        except ImportError:
            pass

        endpoint = self.openai_base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        data = json.dumps(body).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
            "User-Agent": "Autobot/1.0",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/autobot-ai"
            headers["X-Title"] = "Autobot Jarvis"

        req = request.Request(
            endpoint,
            data=data,
            method="POST",
            headers=headers
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            body_bytes = e.read() if e.fp else b""
            try:
                body_str = body_bytes.decode("utf-8", errors="replace")
            except Exception:
                body_str = ""
            hint = ""
            if e.code == 403:
                hint = " (Check API key and model name; for Grok try AUTOBOT_LLM_MODEL=grok-2 or grok-4-1-fast-reasoning)"
            raise RuntimeError(
                f"HTTP {e.code} {e.reason}{hint}. Response: {body_str[:500]}"
            ) from e
        except URLError as e:
            raise RuntimeError(f"Request failed: {e.reason}") from e

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned by openai_compat provider.")
        message = choices[0].get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError("Empty content returned by openai_compat provider.")
            
        # Strip reasoning tags if using DeepSeek R1
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()
            
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
                    id=_none_if_empty(item.get("id")),
                    depends_on=item.get("depends_on", []),
                    target_node=_none_if_empty(item.get("target_node")),
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
                    id=_none_if_empty(item.get("id")),
                    depends_on=item.get("depends_on", []),
                    target_node=_none_if_empty(item.get("target_node")),
                )
            )
    return PlanDraft(title=title, summary=summary, steps=steps)


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    # Handle the extremely common case where LLM wraps in ```json ... ```
    if "```" in cleaned:
        # Split by backticks and find the segment that looks like JSON or follows the 'json' tag
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
            if part.startswith("{") and "}" in part:
                # Nested case or partial capture
                start_raw = part.find("{")
                end_raw = part.rfind("}")
                return part[start_raw : end_raw + 1]
    
    start_main = cleaned.find("{")
    end_main = cleaned.rfind("}")
    if start_main == -1 or end_main == -1 or end_main <= start_main:
        # Use a simpler way to truncate for the error message
        preview = (text + "...") if len(text) > 200 else text
        if len(preview) > 200:
            preview = preview[:200]
        raise ValueError(f"No JSON object found in LLM response: {preview}")
    return cleaned[start_main : end_main + 1]


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _none_if_empty(value: Any) -> str | None:
    text = str(value).strip()
    return text or None
