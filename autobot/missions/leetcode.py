"""
LeetCode Mission — Solve unsolved LeetCode problems using multi-AI consultation.

Strategy:
  1. Open LeetCode, find the first N unsolved problems on the user's account.
  2. For each problem:
     a. Read the problem description carefully.
     b. Open browser tabs for multiple AI assistants (Claude, Grok, DeepSeek).
     c. Paste the full problem to each AI, ask for a solution.
     d. Collect the solutions, pick the most agreed-upon approach.
     e. Paste the winning solution into LeetCode's editor.
     f. Run tests first — if failing, feed errors back to the AI and ask for a fix.
     g. Submit when tests pass. Retry up to 3 times per problem.
     h. Record result (SOLVED/FAILED) and move to the next.
  3. Track accuracy and report at the end.

This mission runs Autobot's full visual agent loop — it uses the real browser,
real AI chat interfaces (no API keys needed), and real LeetCode submission.

Usage (API):
    POST /api/mission/leetcode
    {"num_problems": 5, "language": "python3"}

Usage (direct):
    mission = LeetCodeMission.from_env()
    await mission.run(num_problems=5)
"""
from __future__ import annotations

import os
from typing import Callable, Any


# ── LeetCode AI consultation goal prompt ──────────────────────────────────────

def build_leetcode_goal(num_problems: int = 5, language: str = "python3") -> str:
    """
    Build a detailed goal string for Autobot's AgentLoop.

    This prompt encodes the full multi-AI LeetCode solving strategy so the
    agent knows exactly what to do at every stage without hardcoded logic.
    """
    return f"""
Solve the next {num_problems} unsolved LeetCode problems on my account using the multi-AI strategy below.
Code language: {language}

═══════════════════════════════════════════════════
STRATEGY: MULTI-AI CONSULTATION
═══════════════════════════════════════════════════

You are going to orchestrate multiple AI assistants (Claude, Grok, DeepSeek) to solve
LeetCode problems by copy-pasting between browser tabs. You do NOT need to solve the
problems yourself — your job is to manage the process.

BROWSER TAB LAYOUT (set this up once at the start):
  Tab 1 → https://leetcode.com/problemset/  (LeetCode problems list)
  Tab 2 → https://claude.ai/new             (Claude)
  Tab 3 → https://grok.com                  (Grok)
  Tab 4 → https://chat.deepseek.com         (DeepSeek)

═══════════════════════════════════════════════════
STEP-BY-STEP WORKFLOW
═══════════════════════════════════════════════════

PHASE 1 — SETUP (do once):
  1. Open LeetCode at https://leetcode.com/problemset/
  2. Verify you are logged in (look for your username in the top right). If not, go to https://leetcode.com/accounts/login/ and log in.
  3. On the problems page, filter by Status = "Not Started" to see unsolved problems.
  4. Open tabs for Claude, Grok, DeepSeek (Ctrl+T for each, type the URL).
  5. Log into any AI chat that requires it (they may already be logged in from your browser session).

PHASE 2 — PER PROBLEM LOOP (repeat {num_problems} times):
  A. SELECT THE PROBLEM
     - Switch back to Tab 1 (LeetCode).
     - Click the first "Not Started" problem in the list.
     - Wait for the problem page to load fully.
     - Read the full problem: title, description, examples, constraints.
     - Copy the problem text to clipboard (select all visible problem text with Ctrl+A or drag-select, then Ctrl+C).
     - Save to memory: "current_problem = [title]"

  B. CONSULT AI ASSISTANTS
     For each AI tab (Claude → Grok → DeepSeek):
       - Switch to that AI tab.
       - Click the chat input box.
       - Clear any previous conversation (look for "New Chat" or "+" button).
       - Type this prompt (replace the brackets):
         "Solve this LeetCode problem in {language}. Give ONLY the solution code, no explanation:

         [PASTE PROBLEM TEXT HERE]"
       - Use computer.clipboard.paste() to paste the problem text after "Solve this..."
       - Press Enter and wait for the complete response (wait up to 30 seconds).
       - When the AI stops generating (response is complete), copy the code block it generated.
       - Save the solution: "claude_solution" / "grok_solution" / "deepseek_solution"

  C. CHOOSE THE BEST SOLUTION
     - Compare the three solutions:
       * If all three agree (same approach/structure) → use Claude's version.
       * If two agree → use the agreed-upon one.
       * If all three differ → use the Claude solution (it tends to be cleanest).
     - Copy the chosen solution to clipboard.

  D. SUBMIT TO LEETCODE
     - Switch back to Tab 1 (LeetCode problem tab).
     - Click the code editor area to focus it.
     - Select all existing code (Ctrl+A) and delete it.
     - Paste the solution (Ctrl+V).
     - Verify the language dropdown shows "{language}" — change it if needed.
     - Click "Run" button to test against sample test cases first.
     - Wait up to 20 seconds for test results.

  E. HANDLE TEST RESULTS
     If tests PASS:
       - Click "Submit" button.
       - Wait up to 30 seconds for submission result.
       - If ACCEPTED: record "SOLVED: [title]" in memory. Move to next problem.
       - If WRONG ANSWER or TIME LIMIT EXCEEDED: go to step F.

     If tests FAIL:
       - Screenshot the error and failing test case.
       - Go to step F (fix with AI).

  F. AI-ASSISTED FIX (up to 3 attempts per problem)
     - Note the error: wrong answer, time limit, runtime error.
     - Switch to Claude tab.
     - In the same conversation, type:
       "The solution failed with this error: [describe error]
       Failing test case: [input] → expected [expected] but got [actual]
       Please fix the {language} code."
     - Wait for Claude's fix.
     - Copy the fixed code.
     - Return to LeetCode, replace the code, run tests again.
     - If passes: submit. If fails after 3 attempts: record "FAILED: [title]" and move on.

PHASE 3 — REPORT:
  After all {num_problems} problems, report:
  - Total solved / total attempted
  - Accuracy percentage
  - List of solved and failed problems
  - Average attempts per problem

═══════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════
1. NEVER skip a problem — if it fails after 3 attempts, mark it and continue.
2. Always verify the AI's code is complete before submitting (no "..." truncations).
3. If an AI chat is rate-limited or unavailable, skip it and use the other two.
4. If LeetCode shows a CAPTCHA, take a screenshot and pause (call done with success=False).
5. Use memory field to track: current problem title, attempt number, results so far.
6. If a tab is loading, use the wait action (up to 30 seconds for AI responses).
7. ALWAYS test with "Run" before hitting "Submit" — it costs no submission attempts.

TRACK THIS IN MEMORY AT EVERY STEP:
  solved_count = [number]
  failed_count = [number]
  current_problem = [title]
  attempt_number = [1/2/3]
  problems_done = [comma-separated list]

START NOW: Set up the browser tabs, then begin with problem 1.
""".strip()


# ── Mission class ─────────────────────────────────────────────────────────────

class LeetCodeMission:
    """
    Runs the LeetCode multi-AI solving mission via Autobot's AgentRunner.

    This is a thin wrapper that builds the goal string and hands it to AgentRunner.
    The agent does all the actual work — navigating, copy-pasting, submitting.
    """

    def __init__(
        self,
        agent_runner: Any,
        num_problems: int = 5,
        language: str = "python3",
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.agent_runner = agent_runner
        self.num_problems = num_problems
        self.language = language
        self.log = log_callback or print

    @classmethod
    def from_env(
        cls,
        num_problems: int = 5,
        language: str = "python3",
        log_callback: Callable[[str], None] | None = None,
    ) -> "LeetCodeMission":
        from autobot.agent.runner import AgentRunner
        runner = AgentRunner.from_env(log_callback=log_callback)
        # Use a generous step budget — solving N problems × ~20 steps each
        runner.max_steps = max(200, num_problems * 40)
        return cls(
            agent_runner=runner,
            num_problems=num_problems,
            language=language,
            log_callback=log_callback,
        )

    async def run(self) -> str:
        goal = build_leetcode_goal(
            num_problems=self.num_problems,
            language=self.language,
        )
        self.log(f"Starting LeetCode Mission: solve {self.num_problems} problems in {self.language}")
        self.log(f"Strategy: multi-AI consultation (Claude + Grok + DeepSeek)")
        result = await self.agent_runner.run(
            goal=goal,
            max_steps=self.agent_runner.max_steps,
        )
        return result
