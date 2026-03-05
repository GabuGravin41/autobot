"""
autobot/llm_brain.py — Multi-agent LLM brain for Autobot.

Agent roles:
  - decide_next_steps()   : Executor agent — decides next concrete tool-call steps
  - decompose_goal()      : Planner agent  — breaks a goal into ordered phases
  - verify_progress()     : Verifier agent — checks whether a phase/goal is complete
  - summarize_page()      : Context agent  — distills raw page text into usable state
  - generate_plan_draft() : Draft planner  — converts natural-language prompt → WorkflowPlan
"""
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


# ── Data classes ──────────────────────────────────────────────────────────────

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


@dataclass
class GoalPhasesPlan:
    """Output of the Planner agent — ordered list of phase descriptions."""
    phases: list[str]
    reasoning: str


@dataclass
class VerifierResult:
    """Output of the Verifier agent."""
    goal_done: bool
    phase_complete: bool
    feedback: str


# ── LLMBrain ─────────────────────────────────────────────────────────────────

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
            self.openai_api_key = os.getenv("OPENROUTER_API_KEY") or self.openai_api_key
            if not self.model_name:
                self.model_name = "google/gemini-2.0-flash-001"

        if self.provider == "gemini" and not self.model_name:
            self.model_name = "gemini-2.0-flash-001"

        self.enabled = False
        self._model = None
        if self.provider == "gemini" and bool(self.api_key):
            _genai = _get_genai()
            if _genai is not None:
                _genai.configure(api_key=self.api_key)
                self._model = _genai.GenerativeModel(self.model_name)
                self.enabled = True
        elif (self.provider in ("openai_compat", "openrouter")) and bool(self.openai_api_key):
            self.enabled = True

    # ── Executor agent ────────────────────────────────────────────────────────

    def decide_next_steps(
        self,
        goal: str,
        state: dict[str, Any],
        allowed_actions: list[str],
        max_steps: int = 4,
    ) -> BrainDecision:
        """Executor agent: given the current goal + state, decide what to do next."""
        if not self.enabled:
            return self._fallback_decision(goal, state)
        prompt = self._build_prompt(goal=goal, state=state, allowed_actions=allowed_actions, max_steps=max_steps)
        try:
            text = self._call_llm(prompt)
            return _parse_decision_with_retry(
                text=text,
                allowed_actions=set(allowed_actions),
                max_steps=max_steps,
                retry_fn=lambda corrective: self._call_llm(corrective),
            )
        except Exception as error:  # noqa: BLE001
            self.logger(f"LLM executor failed, fallback mode: {error}")
            return self._fallback_decision(goal, state)

    # ── Planner agent ─────────────────────────────────────────────────────────

    def decompose_goal(self, goal: str, context: str = "") -> GoalPhasesPlan:
        """Planner agent: break a high-level goal into ordered, concrete phases."""
        if not self.enabled:
            return GoalPhasesPlan(phases=[goal], reasoning="LLM not configured; treating goal as single phase.")

        prompt = (
            "You are a strategic planner for an autonomous AI agent that controls a computer.\n"
            "Break the following goal into clear, ordered phases. Each phase should be a concrete, \n"
            "verifiable milestone (e.g. 'Read competition rules and understand the dataset', "
            "'Write initial solution code', 'Submit code and read score', 'Improve code based on score').\n\n"
            f"GOAL: {goal}\n"
            + (f"CONTEXT (current page / environment): {context}\n" if context else "")
            + "\nReturn ONLY strict JSON in this format:\n"
            '{"phases": ["Phase 1 description", "Phase 2 description", ...], "reasoning": "brief explanation"}\n'
            "Rules:\n"
            "- 2 to 7 phases maximum\n"
            "- Each phase must be a single sentence, action-oriented (start with a verb)\n"
            "- Phases must be sequential and each builds on the last\n"
        )
        try:
            text = self._call_llm(prompt)
            payload = _extract_json(text)
            data = json.loads(payload)
            phases = [str(p).strip() for p in data.get("phases", []) if str(p).strip()]
            reasoning = str(data.get("reasoning", "")).strip()
            if not phases:
                phases = [goal]
            return GoalPhasesPlan(phases=phases, reasoning=reasoning)
        except Exception as error:  # noqa: BLE001
            self.logger(f"Planner agent failed: {error}; using single-phase fallback")
            return GoalPhasesPlan(phases=[goal], reasoning=f"Planner error: {error}")

    # ── Verifier agent ────────────────────────────────────────────────────────

    def verify_progress(
        self,
        goal: str,
        current_phase: str,
        state: dict[str, Any],
    ) -> VerifierResult:
        """Verifier agent: check if the current phase is complete and if the overall goal is done."""
        if not self.enabled:
            return VerifierResult(goal_done=False, phase_complete=False, feedback="LLM not configured.")

        compact_state = {
            "current_url": _trim_text(str(state.get("current_url", "")), 300),
            "last_command_exit_code": state.get("last_command_exit_code"),
            "last_command_output": _trim_text(str(state.get("last_command_output", "")), 2000),
            "console_errors": _trim_text(str(state.get("console_errors", "")), 1000),
            "last_error": _trim_text(str(state.get("last_error", "")), 1000),
            "page_text_summary": _trim_text(str(state.get("page_text_summary", "")), 2000),
            "autonomy_loops": state.get("autonomy_loops"),
            "last_score": state.get("last_score"),
            "last_output": _trim_text(str(state.get("last_output", "")), 1500),
        }
        # Include any saved state keys that have meaningful values
        extra_keys = [k for k in state if k not in compact_state and state[k] is not None and str(state[k]).strip()]
        if extra_keys:
            compact_state["other_saved_keys"] = extra_keys[:10]

        prompt = (
            "You are a progress verifier for an autonomous AI agent.\n"
            "Review the current state and decide:\n"
            "1. Is the current phase COMPLETE? (Has the agent done what the phase requires?)\n"
            "2. Is the overall GOAL DONE? (Is the final objective fully achieved?)\n\n"
            f"OVERALL GOAL: {goal}\n"
            f"CURRENT PHASE: {current_phase}\n"
            f"CURRENT STATE: {json.dumps(compact_state)}\n\n"
            "Return ONLY strict JSON:\n"
            '{"goal_done": bool, "phase_complete": bool, "feedback": "one sentence explanation"}\n'
            "Be lenient on phase completion (if the agent took meaningful steps, phase is complete).\n"
            "Be strict on goal_done (only true if the final objective is clearly satisfied).\n"
        )
        try:
            text = self._call_llm(prompt)
            payload = _extract_json(text)
            data = json.loads(payload)
            return VerifierResult(
                goal_done=bool(data.get("goal_done", False)),
                phase_complete=bool(data.get("phase_complete", False)),
                feedback=str(data.get("feedback", "")).strip() or "No feedback.",
            )
        except Exception as error:  # noqa: BLE001
            self.logger(f"Verifier agent failed: {error}")
            return VerifierResult(goal_done=False, phase_complete=False, feedback=f"Verifier error: {error}")

    # ── Context agent ─────────────────────────────────────────────────────────

    def summarize_page(self, url: str, title: str, raw_text: str) -> str:
        """Context agent: summarize page content into a concise description for the brain."""
        if not self.enabled or not raw_text.strip():
            return f"URL: {url} | Title: {title}"

        text_preview = _trim_text(raw_text.strip(), 6000)
        prompt = (
            "You are summarizing a web page for an AI automation agent.\n"
            "Extract the key information: page type, main content, important forms/buttons/links, and any data relevant for automation.\n"
            "Be concise (3-6 sentences max). Focus on what an agent would need to act on this page.\n\n"
            f"URL: {url}\n"
            f"TITLE: {title}\n"
            f"PAGE TEXT (partial):\n{text_preview}\n\n"
            "Return a plain text summary (not JSON)."
        )
        try:
            summary = self._call_llm(prompt)
            return summary.strip()[:2000]
        except Exception as error:  # noqa: BLE001
            self.logger(f"Context agent (summarize_page) failed: {error}")
            return f"URL: {url} | Title: {title} | [Summary failed: {error}]"

    # ── Plan draft (for UI / chat) ────────────────────────────────────────────

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
                        args={"message": "Set OPENROUTER_API_KEY and AUTOBOT_LLM_PROVIDER=openrouter for Gemini Flash (free), or configure another provider."},
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
            text = self._call_llm(prompt)
            return _parse_plan_draft_with_retry(
                text=text,
                allowed_actions=set(allowed_actions),
                max_steps=max_steps,
                retry_fn=lambda corrective: self._call_llm(corrective),
            )
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, model_override: str | None = None) -> str:
        """Unified LLM call — routes to the configured provider."""
        if self.provider == "gemini":
            if self._model is None:
                raise RuntimeError("Gemini model not initialized.")
            response = self._model.generate_content(prompt)
            return (response.text or "").strip()
        return self._call_openai_compatible(prompt, model_override=model_override)

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
                TaskStep(action="clipboard_set", args={"text": prompt}, description="Copy diagnostics to clipboard"),
                TaskStep(
                    action="log",
                    args={"message": "Prompt copied. Paste into Grok/DeepSeek, apply fix, then rerun loop."},
                    description="Log manual handoff",
                ),
            ],
        )

    def _build_prompt(self, goal: str, state: dict[str, Any], allowed_actions: list[str], max_steps: int) -> str:
        compact_state = {
            "last_command_exit_code": state.get("last_command_exit_code"),
            "last_command_output": _trim_text(str(state.get("last_command_output", "")), 4000),
            "last_test_output": _trim_text(str(state.get("last_test_output", "")), 5000),
            "console_errors": _trim_text(str(state.get("console_errors", "")), 5000),
            "last_error": _trim_text(str(state.get("last_error", "")), 2000),
            "autonomy_loops": state.get("autonomy_loops"),
            "current_phase": state.get("current_phase"),
            "current_phase_index": state.get("current_phase_index"),
            "phase_plan": state.get("phase_plan"),
            "run_dir": state.get("run_dir"),
            "current_url": _trim_text(str(state.get("current_url", "")), 500),
            "page_text_summary": _trim_text(str(state.get("page_text_summary", "")), 3000),
            "last_screenshot_path": state.get("last_screenshot_path"),
            "last_notify_message": state.get("last_notify_message"),
            "last_score": state.get("last_score"),
        }
        saved_keys = [k for k, v in state.items() if k not in (
            "run_dir", "last_run_history_path", "adapter_telemetry", "telemetry",
            "autonomy_goal", "autonomy_loops", "phase_plan",
        ) and v is not None and str(v).strip()]
        if saved_keys:
            compact_state["saved_state_keys"] = saved_keys
            for key in ["latex_text", "doc_text", "console_errors", "last_output"]:
                if key in state and state[key]:
                    compact_state[f"{key}_preview"] = _trim_text(str(state[key]), 800)

        schema = {
            "done": "boolean — true ONLY if the overall goal is fully achieved",
            "reason": "string — brief explanation of what you're doing and why",
            "steps": [
                {
                    "id": "optional unique step id",
                    "depends_on": "optional list of step ids that must complete first",
                    "target_node": "optional remote node alias for distributed execution",
                    "action": "one of allowed actions",
                    "args": "object with action arguments",
                    "description": "short human-readable description",
                    "save_as": "optional state key to store result",
                    "retries": "optional int",
                    "continue_on_error": "optional bool",
                }
            ],
        }
        return (
            "You are the executor brain for a local automation agent that controls a real computer.\n"
            "Think like a patient, methodical human. Check pages have loaded. Read the UI state before acting.\n"
            "Return ONLY strict JSON — no markdown, no explanation outside the JSON.\n\n"
            f"GOAL: {goal}\n"
            f"CURRENT PHASE: {state.get('current_phase', 'N/A')}\n"
            f"Allowed actions: {allowed_actions}\n"
            f"Max steps this call: {max_steps}\n"
            f"Current state: {json.dumps(compact_state)}\n"
            f"Output schema: {json.dumps(schema)}\n\n"
            "RULES:\n"
            "- Use 'browser_set_mode devtools' before any CSS selector operations (click, fill, read_text).\n"
            "- Use 'browser_set_mode human_profile' for bot-protected sites.\n"
            "- Use 'open_url' + 'wait' before trying to interact — pages need time.\n"
            "- On failure (last_error set): ADAPT. Try waiting longer (30-120s), a different selector, or request_human_help for CAPTCHAs.\n"
            "- Use 'page_text_summary' in state for context about what's on screen.\n"
            "- Never repeat a failing step unchanged. Always change something.\n"
            "- 'done': only set to true when the OVERALL GOAL is complete, not just one step.\n"
            "- Use 'knowledge_set' and 'knowledge_get' to persist data across runs (e.g. counters, mission progress).\n"
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
                    "id": "optional unique step id",
                    "depends_on": "optional list of step ids",
                    "action": "one of allowed actions",
                    "args": "object with action arguments",
                    "description": "step description",
                    "save_as": "optional state key",
                    "retries": "optional int",
                    "continue_on_error": "optional bool",
                }
            ],
        }
        catalog_block = ""
        if tool_catalog:
            catalog_block = f"\nAvailable tools:\n{tool_catalog}\n"
        return (
            "You are a strict JSON-only automation planner. "
            "Convert the user request into a sequence of atomic tool calls.\n\n"
            "### RULES:\n"
            "1. OUTPUT: Return ONLY valid parsable JSON. No markdown. No extra text.\n"
            "2. ESCAPING: Escape all quotes inside strings. Use \\n for newlines.\n"
            "3. PARALLELISM: Use 'id' and 'depends_on' to create a step dependency graph (DAG) for parallel steps.\n"
            "4. DISTRIBUTED: Set 'target_node' for steps that must run on a remote machine.\n"
            "5. DATA FLOW: Use 'browser_get_content' to capture text. Use 'save_as' to store it. Reference with '{key}'.\n"
            "6. FIDELITY: Use exact details from the user request.\n"
            "7. MODES: Use 'browser_set_mode' to 'devtools' before any click/fill/read.\n"
            "8. WAIT: Add 'wait' steps after navigation (sites need time to load).\n\n"
            f"User request: {user_prompt}\n"
            f"Allowed actions: {allowed_actions}\n"
            f"Max steps: {max_steps}\n"
            f"{catalog_block}"
            f"Output schema: {json.dumps(schema)}\n"
        )

    def _call_openai_compatible(self, prompt: str, model_override: str | None = None) -> str:
        if not self.openai_api_key:
            raise RuntimeError("No API key configured (OPENROUTER_API_KEY / OPENAI_API_KEY / XAI_API_KEY).")
        model = model_override or self.model_name
        messages = [
            {"role": "system", "content": "Return only strict JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            from openai import OpenAI
            extra_headers: dict[str, str] = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/autobot-ai",
                    "X-Title": "Autobot",
                }
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url.rstrip("/"),
                default_headers=extra_headers,
            )
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            if not completion.choices:
                raise RuntimeError("No choices returned by provider.")
            content = (completion.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Empty content returned by provider.")
            return _strip_reasoning_tags(content)
        except ImportError:
            pass

        # Fallback: raw HTTP
        endpoint = self.openai_base_url.rstrip("/") + "/chat/completions"
        body = {"model": model, "messages": messages, "temperature": 0.2}
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
            "User-Agent": "Autobot/1.0",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/autobot-ai"
            headers["X-Title"] = "Autobot"

        req = request.Request(endpoint, data=data, method="POST", headers=headers)
        try:
            with request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body_bytes = e.read() if e.fp else b""
            body_str = body_bytes.decode("utf-8", errors="replace")
            hint = " (Check API key and model name)" if e.code == 403 else ""
            raise RuntimeError(f"HTTP {e.code} {e.reason}{hint}. Body: {body_str[:500]}") from e
        except URLError as e:
            raise RuntimeError(f"Request failed: {e.reason}") from e

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned by provider.")
        content = str(choices[0].get("message", {}).get("content", "")).strip()
        if not content:
            raise RuntimeError("Empty content returned by provider.")
        return _strip_reasoning_tags(content)


# ── JSON parse helpers ────────────────────────────────────────────────────────

def _parse_decision_with_retry(
    text: str,
    allowed_actions: set[str],
    max_steps: int,
    retry_fn: Any,
) -> BrainDecision:
    """Parse a BrainDecision from LLM output; retry once with a corrective prompt if parsing fails."""
    try:
        return _parse_decision(text, allowed_actions, max_steps)
    except (ValueError, json.JSONDecodeError, KeyError) as first_err:
        corrective = (
            "Your previous response was not valid JSON. "
            "Return ONLY a valid JSON object, no markdown, no explanation:\n"
            f'{{"done": false, "reason": "...", "steps": [{{"action": "log", "args": {{"message": "retry"}}}}]}}\n\n'
            f"Previous invalid response: {text[:500]}"
        )
        try:
            corrected_text = retry_fn(corrective)
            return _parse_decision(corrected_text, allowed_actions, max_steps)
        except Exception:
            # Final fallback: return a safe do-nothing decision
            return BrainDecision(
                done=False,
                reason=f"JSON parse failed twice: {first_err}",
                steps=[TaskStep(action="log", args={"message": f"Planner JSON error: {first_err}"}, description="Log parse error")],
            )


def _parse_plan_draft_with_retry(
    text: str,
    allowed_actions: set[str],
    max_steps: int,
    retry_fn: Any,
) -> PlanDraft:
    """Parse a PlanDraft from LLM output; retry once with a corrective prompt if parsing fails."""
    try:
        return _parse_plan_draft(text, allowed_actions, max_steps)
    except (ValueError, json.JSONDecodeError, KeyError) as first_err:
        corrective = (
            "Your previous response was not valid JSON. "
            'Return ONLY a valid JSON object like: {"title": "...", "summary": "...", "steps": [...]}\n\n'
            f"Previous invalid response: {text[:500]}"
        )
        try:
            corrected_text = retry_fn(corrective)
            return _parse_plan_draft(corrected_text, allowed_actions, max_steps)
        except Exception:
            return PlanDraft(
                title="Parse Error",
                summary=f"Plan parse failed twice: {first_err}",
                steps=[TaskStep(action="log", args={"message": f"Plan parse error: {first_err}"}, description="Log parse error")],
            )


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
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
            if part.startswith("{") and "}" in part:
                start_raw = part.find("{")
                end_raw = part.rfind("}")
                return part[start_raw : end_raw + 1]

    start_main = cleaned.find("{")
    end_main = cleaned.rfind("}")
    if start_main == -1 or end_main == -1 or end_main <= start_main:
        preview = cleaned[:200] + ("..." if len(cleaned) > 200 else "")
        raise ValueError(f"No JSON object found in LLM response: {preview}")
    return cleaned[start_main : end_main + 1]


def _strip_reasoning_tags(content: str) -> str:
    """Remove DeepSeek R1 / other model reasoning blocks."""
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _none_if_empty(value: Any) -> str | None:
    text = str(value).strip()
    return text or None
