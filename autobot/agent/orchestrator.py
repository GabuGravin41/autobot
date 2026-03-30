"""
Orchestrator — Master multi-agent coordinator for Autobot.

The Orchestrator sits above the AgentLoop and is responsible for:

  1. TASK ANALYSIS       — Decompose the goal into sub-tasks
  2. AGENT ROUTING       — Route sub-tasks to the right specialist
  3. INFORMATION FLOW    — Synchronize context between agents
  4. ERROR RECOVERY      — Re-route when a sub-agent fails
  5. RESULT SYNTHESIS    — Merge partial results into a final answer

Specialist agents:
  WebNavigator     — Browser navigation, form filling, clicking
  CodeExecutor     — Terminal, VS Code, code editing
  DataExtractor    — Scraping, data collection, research
  FormFiller       — Targeted form input workflows
  FileManager      — Upload/download, file system, OS operations

The Orchestrator does NOT replace the AgentLoop — it wraps around it.
For simple tasks it's transparent (just passes through to AgentLoop).
For complex multi-phase tasks it orchestrates multiple specialized loops.

Architecture:
                   ┌─────────────────┐
    Goal ──────►   │   Orchestrator  │  ◄── World State
                   └────────┬────────┘
                            │ task decomposition
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        WebNavigator   CodeExecutor  DataExtractor
          (loop)         (loop)         (loop)
              │             │             │
              └─────────────┴─────────────┘
                            │
                    Result Synthesis
                            │
                        Final Answer
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Agent type classification ─────────────────────────────────────────────────

class AgentType(str, Enum):
    WEB_NAVIGATOR  = "web_navigator"
    CODE_EXECUTOR  = "code_executor"
    DATA_EXTRACTOR = "data_extractor"
    FORM_FILLER    = "form_filler"
    FILE_MANAGER   = "file_manager"
    RESEARCHER     = "researcher"     # Deep research / multi-source information gathering
    GENERAL        = "general"


@dataclass
class SubTask:
    """A single decomposed sub-task to be executed by a specialist."""
    task_id: str
    description: str
    agent_type: AgentType
    priority: int = 0           # lower = execute first
    depends_on: list[str] = field(default_factory=list)  # task_ids
    context: dict = field(default_factory=dict)          # shared info from prior tasks
    result: str = ""
    success: bool | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration(self) -> float:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 1)
        return 0.0


@dataclass
class OrchestrationPlan:
    """The Orchestrator's decomposition of the overall goal."""
    goal: str
    tasks: list[SubTask]
    reasoning: str = ""
    complexity: str = "simple"   # simple | moderate | complex


# ── Task type classifier ──────────────────────────────────────────────────────

class TaskClassifier:
    """
    Classifies a task description into the appropriate agent type
    using keyword heuristics (no LLM call needed — fast and deterministic).
    """

    _WEB_KEYWORDS = {
        "navigate", "browser", "website", "url", "click", "search",
        "login", "form", "submit", "scroll", "open", "go to", "visit",
        "download", "upload to website", "fill in", "leetcode", "kaggle",
        "github", "google", "youtube", "amazon",
    }
    _CODE_KEYWORDS = {
        "code", "script", "python", "javascript", "run", "execute", "terminal",
        "command", "bash", "shell", "vs code", "editor", "compile", "debug",
        "install", "pip", "npm", "git", "commit", "push",
    }
    _DATA_KEYWORDS = {
        "scrape", "extract", "collect", "data", "table", "csv", "json",
        "gather", "summarize",
    }
    _RESEARCH_KEYWORDS = {
        "research", "find information", "look up", "investigate", "compare",
        "analyze", "study", "what is", "how does", "explain", "overview",
        "pros and cons", "best practices", "examples of",
    }
    _FORM_KEYWORDS = {
        "fill", "form", "input", "type", "enter", "submit", "register",
        "sign up", "apply", "contact", "survey",
    }
    _FILE_KEYWORDS = {
        "file", "folder", "directory", "copy", "move", "rename", "delete",
        "upload", "download file", "open file", "save",
    }

    @classmethod
    def classify(cls, description: str) -> AgentType:
        desc = description.lower()

        # Score each type
        scores = {
            AgentType.CODE_EXECUTOR:  sum(1 for k in cls._CODE_KEYWORDS if k in desc),
            AgentType.DATA_EXTRACTOR: sum(1 for k in cls._DATA_KEYWORDS if k in desc),
            AgentType.FORM_FILLER:    sum(1 for k in cls._FORM_KEYWORDS if k in desc),
            AgentType.FILE_MANAGER:   sum(1 for k in cls._FILE_KEYWORDS if k in desc),
            AgentType.WEB_NAVIGATOR:  sum(1 for k in cls._WEB_KEYWORDS if k in desc),
            AgentType.RESEARCHER:     sum(1 for k in cls._RESEARCH_KEYWORDS if k in desc),
        }

        best_type = max(scores, key=lambda t: scores[t])
        if scores[best_type] == 0:
            return AgentType.GENERAL
        return best_type

    @classmethod
    def is_complex(cls, goal: str) -> bool:
        """True if the goal has multiple distinct phases or requires multiple tools."""
        goal_lower = goal.lower()

        # Explicit sequencing connectives
        connectives = [
            "then", "after that", "and then", "next", "finally",
            "step 1", "step 2", "step 3", "phase", "first", "second", "third",
        ]
        connective_score = sum(1 for c in connectives if c in goal_lower)
        if connective_score >= 2:
            return True

        # Multi-tool: requires both browser AND code/terminal
        if cls.needs_code_and_browser(goal):
            return True

        # Explicit multiple objectives: numbered list "1) ... 2) ... 3) ..."
        numbered = re.findall(r'\b\d+[.)]\s+\w', goal)
        if len(numbered) >= 2:
            return True

        # Long goals with many distinct actions (heuristic: >120 chars and multiple verbs)
        if len(goal) > 120:
            action_verbs = [
                "search", "find", "open", "navigate", "click", "download",
                "upload", "extract", "collect", "write", "run", "execute",
                "compile", "submit", "analyze", "compare",
            ]
            verb_count = sum(1 for v in action_verbs if v in goal_lower)
            if verb_count >= 3:
                return True

        return False

    @classmethod
    def needs_code_and_browser(cls, goal: str) -> bool:
        """True if the goal genuinely requires both browser and code execution.

        Requires an explicit execution verb (run, execute, compile, etc.) alongside
        browser keywords — prevents false positives when user just mentions a language
        name in a search context (e.g. "search for python tutorials").
        """
        goal_lower = goal.lower()
        # Execution verbs: explicitly running/writing code (not just mentioning a language)
        _EXEC_KEYWORDS = {
            "run", "execute", "compile", "debug", "install", "pip", "npm",
            "git commit", "git push", "bash", "shell", "terminal", "script",
            "write a", "write the", "vs code", "code editor",
        }
        has_exec = any(k in goal_lower for k in _EXEC_KEYWORDS)
        has_web = any(k in goal_lower for k in cls._WEB_KEYWORDS)
        return has_exec and has_web


# ── Custom instructions per agent type ───────────────────────────────────────

_AGENT_INSTRUCTIONS: dict[AgentType, str] = {
    AgentType.WEB_NAVIGATOR: """
You are the Web Navigator specialist. Your job is to navigate websites,
click elements, fill forms, and extract information from web pages.

Prioritize DOM-based interactions (dom.click, dom.input) over coordinate-based
mouse clicks. Always check the DOM snapshot for exact element indices before clicking.
If a page hasn't loaded, wait and retry rather than clicking blindly.
""".strip(),

    AgentType.CODE_EXECUTOR: """
You are the Code Executor specialist. Your job is to write and run code,
manage terminals, use VS Code, and execute scripts.

Use computer.terminal.run() for shell commands. For VS Code, use Alt+Tab to
switch to it, then keyboard shortcuts (Ctrl+P to open files, etc.).
Always check command output before proceeding to the next step.
""".strip(),

    AgentType.DATA_EXTRACTOR: """
You are the Data Extractor specialist. Your job is to collect, scrape, and
organize data from web pages and files.

Use the DOM snapshot to identify data tables and list elements. Extract
text using dom.click to select and Ctrl+C to copy. For large datasets,
look for pagination and handle each page systematically.
""".strip(),

    AgentType.FORM_FILLER: """
You are the Form Filler specialist. Your job is to accurately fill out forms.

Always use DOM element indices for form fields — never guess coordinates.
Clear existing values before typing new ones. After filling each field,
verify the value was entered correctly before moving to the next.
Submit only when all required fields are complete.
""".strip(),

    AgentType.FILE_MANAGER: """
You are the File Manager specialist. Your job is to handle file operations.

Use computer.files tools for file system operations. For file dialogs
(open/save), use the OS file picker via keyboard shortcuts.
Always confirm file paths before overwriting existing files.
""".strip(),

    AgentType.RESEARCHER: """
You are the Researcher specialist. Your job is to find, gather, and synthesize
information from multiple sources efficiently.

Strategy:
1. Start with a broad search (Google or DuckDuckGo) to identify the best sources.
2. Open the top 2-3 results in separate tabs.
3. Extract the key information from each source using the DOM snapshot.
4. Synthesize what you found — don't just quote; identify patterns and insights.
5. When you have enough information, write a clear summary and call done().

Prefer official docs, Wikipedia, and authoritative sources over SEO-farm content.
Use Ctrl+F to find specific terms on long pages rather than scrolling blindly.
""".strip(),

    AgentType.GENERAL: "",
}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Master multi-agent coordinator.

    For simple single-phase tasks: transparent pass-through (no overhead).
    For complex multi-phase tasks: decompose → route → synchronize → synthesize.
    """

    def __init__(
        self,
        page: Any,
        llm_client: Any,
        model: str,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.page = page
        self.llm_client = llm_client
        self.model = model
        self.log = log_callback or (lambda msg: logger.info(msg))

        # Active sub-tasks and their results
        self._plan: OrchestrationPlan | None = None
        self._completed_tasks: list[SubTask] = []
        self._shared_context: dict[str, Any] = {}

        # Message bus for inter-agent communication
        from autobot.agent.message_bus import message_bus
        self._bus = message_bus
        self._bus.register("orchestrator")

    async def run(self, goal: str, max_steps_per_task: int = 50) -> str:
        """
        Execute a goal using the multi-agent system.

        Simple goals → single AgentLoop (no decomposition overhead).
        Complex goals → decompose into sub-tasks and route to specialists.
        """
        self.log(f"🎭 Orchestrator analyzing: {goal[:80]}")

        # Classify complexity
        if not TaskClassifier.is_complex(goal):
            self.log("📋 Simple task — using single-agent direct execution")
            return await self._run_single(goal, max_steps_per_task)

        # Complex task — decompose and orchestrate
        self.log("🔀 Complex task — decomposing into sub-tasks")
        plan = await self._decompose(goal)
        self.log(
            f"📊 Plan: {len(plan.tasks)} tasks | "
            f"complexity={plan.complexity} | {plan.reasoning[:80]}"
        )
        self._plan = plan
        return await self._execute_plan(plan, max_steps_per_task)

    async def _run_single(self, goal: str, max_steps: int) -> str:
        """Run a simple task directly with AgentLoop."""
        from autobot.agent.loop import AgentLoop
        agent_type = TaskClassifier.classify(goal)
        custom_instructions = _AGENT_INSTRUCTIONS.get(agent_type, "")

        loop = AgentLoop(
            page=self.page,
            llm_client=self.llm_client,
            goal=goal,
            model=self.model,
            max_steps=max_steps,
            custom_instructions=custom_instructions or None,
        )
        return await loop.run()

    async def _decompose(self, goal: str) -> OrchestrationPlan:
        """
        Decompose a complex goal into ordered sub-tasks.

        Uses LLM for decomposition when available; falls back to heuristic splitting.
        """
        try:
            return await self._llm_decompose(goal)
        except Exception as e:
            logger.warning(f"LLM decomposition failed ({e}), using heuristic")
            return self._heuristic_decompose(goal)

    async def _llm_decompose(self, goal: str) -> OrchestrationPlan:
        """Use the LLM to decompose the goal into structured sub-tasks."""
        prompt = f"""You are a task planning agent. Decompose this goal into sequential sub-tasks.

GOAL: {goal}

Return a JSON object:
{{
  "reasoning": "Why this decomposition makes sense",
  "complexity": "simple|moderate|complex",
  "tasks": [
    {{
      "task_id": "t1",
      "description": "Specific sub-task description",
      "agent_type": "web_navigator|code_executor|data_extractor|form_filler|file_manager|general",
      "priority": 0,
      "depends_on": []
    }}
  ]
}}

Rules:
- Maximum 5 sub-tasks
- Each sub-task should be independently executable
- agent_type must match the work: code_executor for coding, web_navigator for browsing
- depends_on lists task_ids that must complete before this task starts
- Return ONLY valid JSON, no markdown"""

        resp = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()

        # Extract JSON from response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in LLM response")

        data = json.loads(json_match.group(0))
        tasks = []
        for i, t in enumerate(data.get("tasks", [])):
            tasks.append(SubTask(
                task_id=t.get("task_id", f"t{i+1}"),
                description=t.get("description", ""),
                agent_type=AgentType(t.get("agent_type", "general")),
                priority=t.get("priority", i),
                depends_on=t.get("depends_on", []),
            ))

        return OrchestrationPlan(
            goal=goal,
            tasks=tasks,
            reasoning=data.get("reasoning", ""),
            complexity=data.get("complexity", "moderate"),
        )

    def _heuristic_decompose(self, goal: str) -> OrchestrationPlan:
        """
        Heuristic fallback: split goal by connectives.
        No LLM needed — pure string parsing.
        """
        import uuid
        connectives = re.compile(
            r'\b(then|after that|and then|next|finally|step \d+)\b',
            re.IGNORECASE
        )
        parts = [p.strip() for p in connectives.split(goal) if p.strip() and len(p.strip()) > 10]
        if not parts:
            parts = [goal]

        tasks = []
        for i, part in enumerate(parts[:5]):
            agent_type = TaskClassifier.classify(part)
            tasks.append(SubTask(
                task_id=f"t{i+1}",
                description=part,
                agent_type=agent_type,
                priority=i,
                depends_on=[f"t{i}"] if i > 0 else [],
            ))

        return OrchestrationPlan(
            goal=goal,
            tasks=tasks,
            reasoning="Heuristic decomposition by task connectives",
            complexity="moderate",
        )

    async def _execute_plan(
        self, plan: OrchestrationPlan, max_steps_per_task: int
    ) -> str:
        """
        Execute all sub-tasks respecting dependencies.

        Independent tasks (no unmet dependencies) run in PARALLEL via asyncio.gather().
        Tasks with dependencies run after all their dependencies complete.
        Each task's results are injected into the context for subsequent tasks.
        """
        from autobot.agent.loop import AgentLoop

        results: dict[str, str] = {}
        completed_ids: set[str] = set()
        remaining = list(sorted(plan.tasks, key=lambda t: t.priority))

        while remaining:
            # Find all tasks whose dependencies are already satisfied
            ready = [
                t for t in remaining
                if all(dep in completed_ids for dep in t.depends_on)
            ]
            if not ready:
                # Dependency deadlock — run the next task anyway to avoid hanging
                ready = [remaining[0]]

            if len(ready) > 1:
                self.log(
                    f"⚡ Parallel execution: {len(ready)} independent tasks "
                    f"({', '.join(t.task_id for t in ready)})"
                )

            # Remove ready tasks from remaining
            for t in ready:
                remaining.remove(t)

            async def _run_task(task: SubTask) -> tuple[str, str]:
                """Run a single sub-task and return (task_id, result)."""
                self.log(
                    f"\n🤖 [{task.agent_type.value}] Task {task.task_id}: "
                    f"{task.description[:80]}"
                )
                enriched_goal = self._enrich_goal(task, results)
                custom_instructions = _AGENT_INSTRUCTIONS.get(task.agent_type, "")
                if results:
                    context_str = "\n".join(
                        f"[{tid}] {r[:200]}" for tid, r in results.items()
                        if tid in task.depends_on or not task.depends_on
                    )
                    if context_str:
                        prefix = f"{custom_instructions}\n\n" if custom_instructions else ""
                        custom_instructions = f"{prefix}PRIOR TASK RESULTS (use as context):\n{context_str}"

                task.started_at = time.time()
                try:
                    loop = AgentLoop(
                        page=self.page,
                        llm_client=self.llm_client,
                        goal=enriched_goal,
                        model=self.model,
                        max_steps=max_steps_per_task,
                        custom_instructions=custom_instructions or None,
                    )
                    result = await loop.run()
                    task.result = result
                    task.success = True
                    task.finished_at = time.time()
                    self._completed_tasks.append(task)
                    self._shared_context[f"task_{task.task_id}"] = result
                    self.log(
                        f"✅ Task {task.task_id} complete "
                        f"({task.duration:.1f}s): {result[:100]}"
                    )
                    return task.task_id, result
                except Exception as e:
                    task.success = False
                    task.result = f"Failed: {e}"
                    task.finished_at = time.time()
                    self.log(f"❌ Task {task.task_id} failed: {e}")
                    return task.task_id, f"[FAILED] {e}"

            # Run ready tasks in parallel
            batch_results = await asyncio.gather(*[_run_task(t) for t in ready])
            for task_id, result in batch_results:
                results[task_id] = result
                completed_ids.add(task_id)

        return self._synthesize_results(plan, results)

    def _enrich_goal(self, task: SubTask, prior_results: dict[str, str]) -> str:
        """Add context from prior tasks to a sub-task's goal."""
        if not prior_results:
            return task.description

        # Inject relevant prior results into the goal
        relevant = {
            tid: r for tid, r in prior_results.items()
            if tid in task.depends_on
        }
        if not relevant:
            return task.description

        context_lines = [f"Previous step result: {r[:300]}" for r in relevant.values()]
        return f"{task.description}\n\nContext:\n" + "\n".join(context_lines)

    def _synthesize_results(
        self, plan: OrchestrationPlan, results: dict[str, str]
    ) -> str:
        """Merge all sub-task results into a final answer."""
        successful = [t for t in plan.tasks if t.success]
        failed = [t for t in plan.tasks if t.success is False]

        if not results:
            return "No tasks completed."

        # If only one task, return its result directly
        if len(plan.tasks) == 1:
            return list(results.values())[0]

        lines = [f"Mission complete: {plan.goal[:80]}"]
        lines.append(f"Completed {len(successful)}/{len(plan.tasks)} tasks.\n")

        for task in plan.tasks:
            icon = "✅" if task.success else "❌"
            lines.append(
                f"{icon} [{task.agent_type.value}] {task.description[:60]}\n"
                f"   Result: {results.get(task.task_id, 'no result')[:200]}"
            )

        if failed:
            lines.append(
                f"\nNote: {len(failed)} task(s) failed — "
                "completed tasks may still be usable."
            )

        return "\n".join(lines)

    def get_plan_status(self) -> dict:
        """Return current orchestration plan status for the dashboard."""
        if not self._plan:
            return {}
        return {
            "goal": self._plan.goal,
            "complexity": self._plan.complexity,
            "total_tasks": len(self._plan.tasks),
            "completed_tasks": len(self._completed_tasks),
            "tasks": [
                {
                    "id": t.task_id,
                    "description": t.description[:80],
                    "agent_type": t.agent_type.value,
                    "success": t.success,
                    "duration": t.duration,
                    "result_preview": t.result[:100] if t.result else "",
                }
                for t in self._plan.tasks
            ],
        }
