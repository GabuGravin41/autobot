# Autobot — Briefing for Antigravity

This is a handoff doc so Antigravity Pro can pick up work on this project with
native access to Dalton's machine. Read this once, then use the pointers at
the bottom to go deeper on any specific piece.

## What Autobot is (domain)

Autobot is a ~10-month-old local desktop/browser automation agent. It runs on
Dalton's own Windows machine and controls it directly — clicking, typing,
reading windows via UI Automation, driving Chrome through a companion
extension (not CDP — that was retired), and shelling out to real CLIs/APIs
for anything that has one (Kaggle's REST API, Claude Code's headless CLI).
It is NOT a chatbot and NOT a cloud agent — its whole reason to exist is
"a sovereign agent that can operate a real computer autonomously," closer in
spirit to a very capable ops/QA engineer than to a chat assistant.

Core loop: `autobot "<task>"` → `AgentRunner.from_env()` builds an LLM client
(OpenRouter → Anthropic → Gemini → OpenAI, picked from `.env`) → `CoreLoop`
feeds it a screen-state description each step (UIA element tree, `[N]`
indices) → the LLM returns one JSON action (`click`, `type_into`,
`browser_click`, `run_shell`, `computer_call`, `done`, etc.) → Autobot
executes it → repeat. Every action is classified by risk tier
(SAFE / CAUTION / IRREVERSIBLE) in `autobot/agent/approval.py`, which gates
what runs silently vs. what needs the user's live approval, and that gating
behaves differently depending on `AUTOBOT_APPROVAL_MODE` /
`AUTOBOT_UNATTENDED`.

## What we're currently working on

**Immediate goal**: prove Autobot can autonomously compete in real Kaggle
competitions (both frontier and community ones), where the division of
labor is deliberate:
- **Autobot's job**: operate the computer — sequence steps, verify each one
  actually happened before starting the next, poll for completion, hand
  real output (errors, scores, leaderboard position) to the next step.
- **Claude Code's job** (via `computer.claude_code.run(...)`, a real headless
  CLI call, not UI automation of a chat window): the actual thinking —
  strategy, code generation, interpreting results. Autobot is told
  explicitly in its system prompt not to second-guess or out-think this;
  when something about strategy is ambiguous, it asks Claude Code rather
  than deciding unilaterally.

This was built on top of "Round 6," a recently-shipped feature set:
Kaggle-specific unattended-autonomy approval policy, Antigravity CLI
integration (yes — Autobot can already drive Antigravity the same way it
drives Claude Code, via a real CLI bridge, not screenshots), and a
multi-project orchestrator. All of that shipped, was adversarially reviewed
(two real bugs found and fixed), tested (164 tests passing), and pushed to
Dalton's machine.

**Where we are right now, specifically**: walking through first-time Kaggle
credential setup end-to-end on Dalton's real machine, and in the process
discovered the installed `kaggle` package (2.2.4, kagglesdk-backed) has
drifted from what `autobot/computer/kaggle_tool.py` was written against.
Two real breaks, confirmed live:

1. `api.competitions_list()` now returns a wrapper object
   (`ApiListCompetitionsResponse`), not a plain list — the list is at
   `.competitions`. **Fixed already** (see `kaggle_tool.py`'s
   `list_competitions()` — defensive `getattr(response, "competitions",
   response)` so it works against old and new SDK versions).
2. `api.competition_view_leaderboard(competition)` **no longer exists** on
   the installed `KaggleApi` class at all — raises `AttributeError`. Python
   suggests `competition_leaderboard_cli` as a fuzzy-match replacement, but
   that name and its "_cli" suffix strongly suggest it may return
   CLI-formatted/printed output rather than clean structured data — **this
   is unverified and is the next concrete thing to check** (see Task list
   below).

## How Antigravity can help right now

Antigravity has native device access, which is more direct than the
copy-paste-diagnostic-output loop this has been going through with Dalton in
chat. The single highest-value thing Antigravity can do immediately:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# find every leaderboard-related method on the real installed class
print([m for m in dir(api) if "leaderboard" in m.lower()])

# inspect the actual candidate before wiring it in
help(api.competition_leaderboard_cli)
result = api.competition_leaderboard_cli("titanic")
print(type(result))
print(result if not hasattr(result, "__len__") else result[:3])
```

Run this in the project's Python environment (wherever `pip install kaggle`
landed — check with `python -m pip show kaggle`), on Dalton's real
authenticated `kaggle.json`. The goal is to confirm: (a) the correct
current replacement method name, (b) its return shape (does it give
`teamName`/`rank`/`score`-style structured data, or pre-formatted text that
would need parsing?), before touching `get_leaderboard()` in
`autobot/computer/kaggle_tool.py`. This project's standing rule is
"verify before fixing, don't guess" — don't wire in
`competition_leaderboard_cli` on the strength of the AttributeError's fuzzy
suggestion alone.

Repo root on Dalton's machine: the folder containing `autobot/`, `tests/`,
`ROADMAP.md`, `README.md`, `DESIGN_PHILOSOPHY.md`.

## Key files to know

- `autobot/agent/runner.py` — `AgentRunner`, LLM client construction.
- `autobot/agent/core_loop.py` (referenced, not detailed here) — the
  perceive → decide → act loop.
- `autobot/agent/approval.py` — risk-tier classifier (SAFE/CAUTION/
  IRREVERSIBLE) and unattended-mode gating.
- `autobot/computer/kaggle_tool.py` — Kaggle API wrapper; current focus.
- `autobot/integrations/antigravity_bridge.py` — how Autobot itself drives
  Antigravity's CLI (`agy`) — worth reading so Antigravity understands it's
  both a *tool Autobot calls* and now also *a collaborator on the codebase*.
- `autobot/integrations/claude_code_bridge.py` — same pattern, for Claude
  Code's headless CLI.
- `autobot/diagnostics.py` — `autobot --doctor`, the preflight check
  (deps, Chrome, Kaggle creds, Claude Code CLI, Antigravity CLI, unattended
  mode). Run this first on any fresh check-in — it's the fastest way to see
  what's actually configured on Dalton's machine right now.
- `autobot/prompts/system_prompt.md` — the actual instructions given to the
  LLM driving Autobot each step, including the Kaggle+Claude Code+Overleaf
  workflow section.
- `ROADMAP.md` — full history including the detailed Round 6 write-up and
  adversarial-review findings.
- `DESIGN_PHILOSOPHY.md` — the project's standing design principles (e.g.
  "other AI tools get driven through their CLI/API, never through
  screenshots of their chat window" — this is why Kaggle/Claude
  Code/Antigravity are all wired as real API/CLI calls instead of browser
  automation).

## Ground rules worth inheriting

- Never guess an API's current shape — verify live before writing a fix.
- Never touch `submit()` (real competition submission) casually — it's
  IRREVERSIBLE-tier by design and always requires live human approval,
  in every mode, no exceptions.
- Kernel iteration (`pull_kernel`/`push_kernel`/`kernel_status`/
  `kernel_output`) is deliberately SAFE-tier — free to iterate on a
  notebook without approval friction.
- Every fix should land with a test that would have caught the bug —
  `tests/test_kaggle_tool.py` currently has zero coverage for
  `list_competitions()`/`get_leaderboard()`, which is exactly why this drift
  went unnoticed until a live run hit it.
- Dalton's live Kaggle API token was accidentally pasted into chat earlier
  this session and should be treated as compromised — flag if it looks like
  it hasn't been regenerated yet at kaggle.com/settings.
