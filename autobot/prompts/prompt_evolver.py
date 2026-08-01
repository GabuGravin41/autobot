"""
Prompt Evolver — Dynamic prompt evolution based on task progress.

The PromptEvolver tracks the agent's trajectory toward its goal and
injects course corrections when the agent drifts off track.

Key insight from n8n / workflow automation patterns:
  Every step is a decision node. If the agent is on track → let it run.
  If it's deviating → inject a targeted correction, not a generic retry message.

The evolver provides three types of dynamic injections:

  1. TRAJECTORY ALERT — "You have been on this page for 5 steps with no progress.
                         You need to navigate to X."

  2. STRATEGY SHIFT   — "Your current approach (mouse clicks) has failed 3 times.
                         Switch to DOM-based interaction."

  3. PHASE REMINDER   — "You are in Phase 2/3. Phase 1 (done): found LeetCode problem.
                         Phase 2 (current): write the solution.
                         Phase 3 (next): submit and verify."

These injections are small XML blocks inserted into the step prompt,
NOT rewrites of the system prompt. They are step-specific and expire
once the situation changes.

Integration:
    # In loop.py, before building step prompt:
    evolution_hint = self._evolver.get_evolution_hint(
        step_number=self.step_number,
        url=current_url,
        recent_actions=[...],
        goal=self.goal,
        consecutive_failures=self._consecutive_failures,
        history=self.history[-5:],
    )
    # Pass to StepPromptBuilder as custom_instructions override
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvolutionHint:
    """A dynamic prompt injection for the current step."""
    type: str          # "trajectory_alert" | "strategy_shift" | "phase_reminder" | "none"
    text: str          # The XML block to inject
    urgency: str       # "high" | "medium" | "low"
    reason: str        # Why this hint was generated (for logging)


NO_HINT = EvolutionHint(type="none", text="", urgency="low", reason="on track")


@dataclass
class GoalPhase:
    """A phase in a multi-phase task."""
    number: int
    description: str
    status: str = "pending"   # pending | in_progress | done | failed
    completion_signal: str = ""  # URL pattern or keyword that marks this done


class PromptEvolver:
    """
    Tracks agent trajectory and generates dynamic prompt corrections.

    State machine:
      - ON_TRACK    → no injection
      - DRIFTING    → trajectory alert
      - STUCK       → strategy shift
      - OFF_COURSE  → strong correction + phase reminder
    """

    # Thresholds
    DRIFT_URL_THRESHOLD    = 4    # Same URL for N steps without progress → alert
    STUCK_FAIL_THRESHOLD   = 3    # N consecutive failures → strategy shift
    LOOP_GOAL_THRESHOLD    = 3    # Same micro-goal N times → loop alert
    STRATEGY_WINDOW        = 5    # Look back N steps for strategy patterns

    def __init__(self, goal: str = "") -> None:
        # URL visit tracking for trajectory analysis
        self._url_step_counts: dict[str, int] = {}
        self._current_url = ""
        self._current_url_entry_step = 0

        # Goal tracking for loop detection
        self._goal_history: list[str] = []

        # Phase tracking for multi-phase tasks
        self._phases: list[GoalPhase] = []
        self._current_phase: int = 0
        self._phases_parsed = False

        # Strategy tracking
        self._strategy_history: list[str] = []  # action_tool per step
        self._last_strategy_shift_step = -999

        # Step tracking
        self._last_hint_step = -1
        self._last_hint_type = "none"

        # Auto-detect phases from goal text if provided
        if goal:
            self._auto_detect_phases(goal)

    # ── Main entry point ──────────────────────────────────────────────────────

    def get_evolution_hint(
        self,
        step_number: int,
        url: str,
        goal: str,
        recent_actions: list[dict],   # list of {action_name, success, error}
        consecutive_failures: int,
        history: list[Any] | None = None,
        max_steps: int | None = None,  # total step budget (for budget awareness)
    ) -> EvolutionHint:
        """
        Analyze current state and return an evolution hint (or NO_HINT if on track).

        Called before each LLM call. Must be fast (pure in-memory computation).
        """
        self._update_tracking(step_number, url, goal, recent_actions)

        # Check conditions in priority order (most urgent first)

        # 0. Budget warning — agent is running out of steps
        if max_steps and max_steps > 0:
            steps_remaining = max_steps - step_number
            budget_pct = steps_remaining / max_steps
            if steps_remaining <= 5:
                return EvolutionHint(
                    type="budget_critical",
                    text=(
                        f"<budget_alert urgency='critical'>\n"
                        f"🚨 CRITICAL: Only {steps_remaining} step(s) remaining (of {max_steps} total).\n"
                        f"You MUST call done() NOW with whatever you have accomplished.\n"
                        f"Do NOT attempt any new actions. Summarize results and finish.\n"
                        f"</budget_alert>"
                    ),
                    urgency="high",
                    reason=f"only {steps_remaining} steps left",
                )
            elif budget_pct <= 0.20 and steps_remaining <= 15:
                return EvolutionHint(
                    type="budget_warning",
                    text=(
                        f"<budget_warning urgency='high'>\n"
                        f"⏰ BUDGET WARNING: {steps_remaining} steps remaining ({int(budget_pct*100)}% left).\n"
                        f"Prioritize completing the most important parts of the task.\n"
                        f"Avoid exploring new directions — focus on finishing what's already in progress.\n"
                        f"If you can't complete everything, complete the most valuable parts and call done().\n"
                        f"</budget_warning>"
                    ),
                    urgency="medium",
                    reason=f"{steps_remaining} steps remaining ({int(budget_pct*100)}%)",
                )

        # 1. Strategy shift — agent keeps using the same failing tool
        if consecutive_failures >= self.STUCK_FAIL_THRESHOLD:
            if step_number - self._last_strategy_shift_step >= 3:  # don't spam
                hint = self._strategy_shift_hint(recent_actions, step_number, goal)
                if hint:
                    self._last_strategy_shift_step = step_number
                    return hint

        # 2. Trajectory alert — agent stuck on wrong page
        stuck_steps = self._url_stuck_steps(step_number, url)
        if stuck_steps >= self.DRIFT_URL_THRESHOLD:
            hint = self._trajectory_alert_hint(url, stuck_steps, goal, step_number)
            if hint and step_number != self._last_hint_step:
                return hint

        # 3. Goal loop — micro-goal repeating
        loop_count = self._goal_loop_count(goal)
        if loop_count >= self.LOOP_GOAL_THRESHOLD:
            hint = self._loop_correction_hint(goal, loop_count, step_number)
            if hint:
                return hint

        # 4. Phase reminder — multi-phase task, remind of current phase
        if self._phases and step_number % 8 == 0 and step_number > 0:
            hint = self._phase_reminder_hint(step_number)
            if hint:
                return hint

        return NO_HINT

    # ── Hint generators ───────────────────────────────────────────────────────

    def _strategy_shift_hint(
        self,
        recent_actions: list[dict],
        step_number: int,
        goal: str,
    ) -> EvolutionHint | None:
        """Generate a strategy shift when the same failing tool keeps being used."""
        if not recent_actions:
            return None

        # Find the dominant failing tool
        failing_tools = [
            a.get("action_name", "") for a in recent_actions[-5:]
            if not a.get("success", True)
        ]
        if not failing_tools:
            return None

        from collections import Counter
        most_common_fail = Counter(failing_tools).most_common(1)[0][0]

        # Suggest an alternative based on what's failing
        alternative = _suggest_alternative(most_common_fail)

        text = (
            f"<strategy_correction urgency='high'>\n"
            f"⚠️  STRATEGY SHIFT REQUIRED\n"
            f"The current approach ({most_common_fail}) has failed "
            f"{len(failing_tools)} times in the last 5 steps.\n"
            f"You MUST try a different approach:\n"
            f"{alternative}\n"
            f"Do NOT repeat the same action type that has been failing.\n"
            f"</strategy_correction>"
        )
        return EvolutionHint(
            type="strategy_shift",
            text=text,
            urgency="high",
            reason=f"{most_common_fail} failed {len(failing_tools)} times",
        )

    def _trajectory_alert_hint(
        self,
        url: str,
        stuck_steps: int,
        goal: str,
        step_number: int,
    ) -> EvolutionHint | None:
        """Generate an alert when the agent is stuck on the same URL."""
        # Don't fire on blank/loading pages
        if url in ("about:blank", "", "chrome://newtab/"):
            return None

        # Only alert if this page seems irrelevant to the goal
        if _url_relevant_to_goal(url, goal):
            return None  # On the right page — stuck but not lost

        text = (
            f"<trajectory_alert urgency='medium'>\n"
            f"📍 TRAJECTORY CHECK: You have been on {_format_url(url)} "
            f"for {stuck_steps} steps without apparent progress.\n"
            f"Your goal is: {goal[:100]}\n"
            f"Ask yourself: Is this page relevant to the goal?\n"
            f"If NOT → open a new tab (Ctrl+T) and navigate to your target directly.\n"
            f"If YES → identify what is blocking you and handle that obstacle first.\n"
            f"</trajectory_alert>"
        )
        return EvolutionHint(
            type="trajectory_alert",
            text=text,
            urgency="medium",
            reason=f"stuck on {_format_url(url)} for {stuck_steps} steps",
        )

    def _loop_correction_hint(
        self,
        goal: str,
        count: int,
        step_number: int,
    ) -> EvolutionHint | None:
        """Generate a correction when the agent's micro-goal keeps repeating."""
        # Deduplicate — only fire once per 5 steps
        if self._last_hint_type == "loop_correction" and step_number - self._last_hint_step < 5:
            return None

        # Extract the repeating goal
        repeating_goal = self._most_repeated_goal()
        self._last_hint_step = step_number
        self._last_hint_type = "loop_correction"

        text = (
            f"<loop_correction urgency='high'>\n"
            f"🔄 LOOP DETECTED: You have been attempting \"{repeating_goal[:80]}\" "
            f"{count} times.\n"
            f"This approach is NOT working. You MUST:\n"
            f"1. Stop repeating the same action.\n"
            f"2. Think about WHY it's failing — read any error messages carefully.\n"
            f"3. Try a completely different approach (different tool, different element, "
            f"or navigate away and come back).\n"
            f"4. If you cannot make progress, call done(success=False) and report the obstacle.\n"
            f"</loop_correction>"
        )
        return EvolutionHint(
            type="loop_correction",
            text=text,
            urgency="high",
            reason=f"goal repeated {count} times: {repeating_goal[:40]}",
        )

    def _phase_reminder_hint(self, step_number: int) -> EvolutionHint | None:
        """Inject a phase reminder for multi-phase tasks."""
        if not self._phases or self._current_phase >= len(self._phases):
            return None

        current = self._phases[self._current_phase]
        done_phases = [p for p in self._phases if p.status == "done"]
        pending_phases = [p for p in self._phases if p.status == "pending"]

        phase_lines = []
        for p in self._phases:
            icon = {"done": "✅", "in_progress": "🔄", "pending": "⏳", "failed": "❌"}.get(p.status, "⏳")
            phase_lines.append(f"  {icon} Phase {p.number}: {p.description[:60]}")

        text = (
            f"<phase_reminder>\n"
            f"📋 Multi-phase task progress:\n"
            + "\n".join(phase_lines) +
            f"\n→ Currently executing Phase {current.number}: {current.description[:80]}\n"
            f"Complete this phase before moving to the next.\n"
            f"</phase_reminder>"
        )
        return EvolutionHint(
            type="phase_reminder",
            text=text,
            urgency="low",
            reason=f"phase {current.number} in progress",
        )

    # ── Phase management ──────────────────────────────────────────────────────

    def _auto_detect_phases(self, goal: str) -> None:
        """
        Auto-detect phases from goal text without needing explicit set_phases() call.

        Detects patterns like:
          "Step 1: do X. Step 2: do Y. Step 3: do Z."
          "Phase 1 - Search. Phase 2 - Extract. Phase 3 - Submit."
          "First, do X. Then do Y. Finally do Z."
          "1) search 2) extract 3) report"
        """
        if self._phases_parsed:
            return

        # Pattern 1: numbered steps/phases with explicit labels
        numbered = re.findall(
            r'(?:step|phase|part)\s*(\d+)[:\-\.]?\s*([^.!?;\n]{10,80})',
            goal,
            re.IGNORECASE,
        )
        if len(numbered) >= 2:
            phases_detected = []
            for num_str, desc in numbered[:5]:
                phases_detected.append({
                    "description": desc.strip(),
                    "completion_signal": "",
                })
            self.set_phases(phases_detected)
            return

        # Pattern 2: bare numbered list "1) ... 2) ... 3) ..."
        bare_numbered = re.findall(r'\b(\d+)[.)]\s+([^.!?;\n]{10,80})', goal)
        if len(bare_numbered) >= 2:
            phases_detected = []
            for _, desc in bare_numbered[:5]:
                phases_detected.append({
                    "description": desc.strip(),
                    "completion_signal": "",
                })
            self.set_phases(phases_detected)
            return

        # Pattern 3: connective phrases split into phases
        connective_re = re.compile(
            r'\b(first|second|third|fourth|then|after that|and then|next|finally)\b',
            re.IGNORECASE,
        )
        parts = [p.strip() for p in connective_re.split(goal) if p.strip() and len(p.strip()) > 15]
        # Filter out the connective words themselves
        connectives = {"first", "second", "third", "fourth", "then", "after that",
                       "and then", "next", "finally"}
        parts = [p for p in parts if p.lower() not in connectives]
        if len(parts) >= 3:
            phases_detected = [{"description": p[:80], "completion_signal": ""} for p in parts[:5]]
            self.set_phases(phases_detected)

    def set_phases(self, phases: list[dict]) -> None:
        """Set explicit phases for a multi-phase task."""
        self._phases = [
            GoalPhase(
                number=i + 1,
                description=p.get("description", ""),
                completion_signal=p.get("completion_signal", ""),
            )
            for i, p in enumerate(phases)
        ]
        if self._phases:
            self._phases[0].status = "in_progress"
        self._phases_parsed = True

    def advance_phase(self, url: str = "") -> bool:
        """
        Try to advance to the next phase.
        Returns True if phase was advanced.
        """
        if not self._phases or self._current_phase >= len(self._phases):
            return False

        current = self._phases[self._current_phase]
        # Check completion signal
        if current.completion_signal and url:
            if current.completion_signal.lower() not in url.lower():
                return False  # Not there yet

        current.status = "done"
        self._current_phase += 1
        if self._current_phase < len(self._phases):
            self._phases[self._current_phase].status = "in_progress"
        return True

    # ── Internal tracking helpers ─────────────────────────────────────────────

    def _update_tracking(
        self,
        step_number: int,
        url: str,
        goal: str,
        recent_actions: list[dict],
    ) -> None:
        """Update internal tracking state."""
        # URL tracking
        if url != self._current_url:
            prev_url = self._current_url
            self._current_url = url
            self._current_url_entry_step = step_number
            self._url_step_counts[url] = self._url_step_counts.get(url, 0) + 1
            # Auto-advance phase when URL changes (check completion signal)
            if self._phases and prev_url != url:
                self.advance_phase(url)

        # Goal tracking
        if goal:
            self._goal_history.append(goal)
            self._goal_history = self._goal_history[-20:]  # keep last 20

        # Strategy tracking
        for action in recent_actions[-1:]:  # only the latest action
            tool = action.get("action_name", "unknown")
            self._strategy_history.append(tool)
            self._strategy_history = self._strategy_history[-20:]

    def _url_stuck_steps(self, step_number: int, url: str) -> int:
        """Number of steps spent on the current URL."""
        return step_number - self._current_url_entry_step

    def _goal_loop_count(self, current_goal: str) -> int:
        """Count how many recent steps had the same micro-goal."""
        if not current_goal or not self._goal_history:
            return 0
        normalised = _normalise_goal(current_goal)
        return sum(
            1 for g in self._goal_history[-10:]
            if _normalise_goal(g) == normalised
        )

    def _most_repeated_goal(self) -> str:
        """Find the most repeated goal in recent history."""
        if not self._goal_history:
            return ""
        from collections import Counter
        counts = Counter(_normalise_goal(g) for g in self._goal_history[-10:])
        return counts.most_common(1)[0][0] if counts else ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalise_goal(goal: str) -> str:
    """Normalise a goal string for comparison (lowercase, strip whitespace)."""
    return re.sub(r"\s+", " ", goal.lower().strip())[:100]


def _format_url(url: str) -> str:
    """Format a URL for display (domain only)."""
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"/.*", "", url)
    return url[:40]


def _url_relevant_to_goal(url: str, goal: str) -> bool:
    """Check if the URL seems relevant to the task goal."""
    goal_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", goal.lower()))
    stop = {"search", "find", "open", "navigate", "click", "help", "with", "that", "this"}
    goal_words -= stop
    url_lower = url.lower()
    return any(w in url_lower for w in goal_words)


def _suggest_alternative(failing_tool: str) -> str:
    """Suggest an alternative approach for a failing tool."""
    alternatives = {
        "computer_call": (
            "• If using mouse clicks: try DOM-based click (dom.click with element index)\n"
            "• If using keyboard: check that the target element has focus first\n"
            "• Try navigate() to reload or go to the target URL directly"
        ),
        "mouse.click": (
            "• Try dom.click using the element index from the DOM snapshot\n"
            "• Verify coordinates — use the DOM snapshot to find the right element\n"
            "• Try scrolling to make the element visible before clicking"
        ),
        "navigate": (
            "• Check that the URL is correct (no typos, correct protocol)\n"
            "• Try opening in a new tab (Ctrl+T)\n"
            "• If page won't load, try a different URL or check network connection"
        ),
        "keyboard.type": (
            "• Click/focus the target input field before typing\n"
            "• Try dom.input with the element index from the DOM snapshot\n"
            "• Clear any existing text first with Ctrl+A then Delete"
        ),
        "dom.click": (
            "• Use the DOM snapshot to find the correct element index\n"
            "• Try coordinate-based click as fallback\n"
            "• The element may be in a modal or iframe — check the DOM snapshot carefully"
        ),
    }
    # Try to match the failing tool to a known alternative
    for key, suggestion in alternatives.items():
        if key in failing_tool:
            return suggestion
    return (
        "• Try a completely different tool/approach\n"
        "• Navigate away and come back\n"
        "• If stuck, call done(success=False) with a clear error description"
    )


# Module-level singleton
prompt_evolver = PromptEvolver()
