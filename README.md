# Autobot

> **New (Sep 2026): the butler.** Autobot now runs as an always-on task manager that hands work to
> Claude Code / Antigravity, checks results itself, and asks you only for decisions. Start with
> **[BUTLER.md](BUTLER.md)**; Gmail setup is in **[GMAIL_SETUP.md](GMAIL_SETUP.md)**.


Autobot is a local autonomous OS co-pilot and desktop automation controller designed to execute complex workflows directly on your laptop setup.

## 🚀 Autonomous OS Co-Pilot Architecture

> This section described an earlier CDP/Playwright-centric design (a
> `perception/` + `actuation/` + `governance/` package split). That
> architecture was retired in Sep 2026 — see `DESIGN_PHILOSOPHY.md` for why.
> Those packages only exist under `_archive/` now; nothing in the live
> `autobot/` package imports them.

What actually runs today is **CoreLoop** (`autobot/agent/core_loop.py`), a
single observe → decide → act → verify → distill loop:

1. **Perception, split by surface, not unified into one snapshot type:**
   native desktop apps via UIAutomation (`autobot/computer/window.py`), web
   page content via the Chrome extension's DOM bridge
   (`autobot/browser/extension_bridge.py`, no CDP), and vision/screenshot as
   the deliberate last resort. CoreLoop's `_observe()` tells the model which
   source applies each step rather than leaving it to guess.
2. **Actuation** through the same split: `computer.window.click/type` for
   native UI, `browser_text/browser_list/browser_click/browser_type/browser_paste`
   for web content, `computer.mouse`/`computer.keyboard`/`computer.terminal`
   for the rest of the machine, and direct API/CLI tools
   (`computer.claude_code`, `computer.kaggle`) where one exists instead of
   automating a UI.
3. **Governance & Permission Dial** (`autobot/agent/approval.py`): risk-tiered
   (SAFE / CAUTION / DANGER / IRREVERSIBLE) gating with 3 modes:
   - 🛡️ `strict`: nothing risky runs without a live approval.
   - ⚡ `balanced` (default): safe reads/navigation auto-proceed, risky
     actions pause for approval.
   - 🚀 `trusted`: clicks and shell commands don't stop you.
   IRREVERSIBLE actions are hard-gated in every mode, no exceptions.
4. **Skill distillation** (`autobot/knowledge/skill_distiller.py`) replaces
   the old "Reflection" step: a successful run's action path is saved and
   re-injected as context next time a similar goal comes in.

**Product direction and long-term vision:** see **[AUTOBOT_MISSION.md](AUTOBOT_MISSION.md)** in the project root.

**Other docs in this repo:**
- **[AUTOBOT_MISSION.md](AUTOBOT_MISSION.md)** — mission, architecture, and the autonomous-computing manifesto.
- **[DESIGN_PHILOSOPHY.md](DESIGN_PHILOSOPHY.md)** — the "no blind actions" rule and other execution guidelines the code is expected to follow.
- **[ROADMAP.md](ROADMAP.md)** — ground-truth log of what's actually wired up and verified working vs. what's built but not yet reachable from a real run. Read this if `--doctor` is green but a feature still doesn't do anything — the code may exist but not be called from anywhere yet.
- **[USE_CASES.md](USE_CASES.md)** — target end-to-end flows (portfolio builder, LeetCode grinding, Kaggle competitions) the system is being built toward. Aspirational, not all implemented yet.
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — future packaging/deployment plan (hosted frontend + downloadable local agent). Not built yet.

## Current capabilities

- Uses a persistent Chrome automation profile directory (bootstrapped from your Chrome profile on first run when possible).
- Includes autonomous multi-loop mode:
  - diagnose -> plan -> execute -> retest
  - loop limits, per-loop step caps, and cancellation
  - conditional step execution, retries, continue-on-error semantics
  - structured run logs at `runs/<timestamp>_<plan>.json`
- Runs one-shot commands:
  - `search <query>`
  - `open <url|target>`
  - `run <os command>`
  - `browser mode`
  - `run benchmarks`
  - `run tool stress <phone>|<docs_existing_url>|<download_check_path>|<message>`
  - `open path <local path>`
  - `switch window`
  - `type <text>`
  - `wait <seconds>`
  - `list adapters`
  - `adapter <name> <action> <json_params>`
  - `adapter telemetry`
  - `adapter policy <strict|balanced|trusted>`
  - `adapter prepare <name> <action> <json_params>`
  - `adapter confirm <token>`
- Runs preset workflows:
  - `website_builder`
  - `research_paper`
  - `console_fix_assist`
- Includes stateful app adapters:
  - `whatsapp_web`
  - `instagram_web`
  - `overleaf_web`
  - `google_docs_web`
  - `grok_web`
  - `vscode_desktop`
- Adapter actions are explicit and per-site, with confirmation gates for sensitive operations such as message send and PDF download.
- UI includes an adapter panel with action docs and a required checkbox for sensitive actions.
- Adapter reliability layer includes session health checks, selector fallback configs, and action/selector telemetry.
- Human-mode adapter navigation maps are stored in `autobot/adapters/human_nav/*.json`.
- Sensitive control now supports two-step flow in strict mode:
  - prepare action -> receive token -> confirm token to execute
- Supports browser actions in engine:
  - open/search/click/fill/press/read text/read console errors
- Supports system actions in engine:
  - open VS Code, run shell command, clipboard set/get
- Supports optional desktop actions (`pyautogui`):
  - type text, send hotkeys, move cursor, click coordinates
  - switch active window, press single key

## 🚀 Quick Start (Shipping Mode)

If you just want to run Autobot as a finished product:

1. **Clone & Setup:**
   ```bash
   git clone https://github.com/GabuGravin41/autobot.git
   cd autobot
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -e .
   autobot --setup
   ```

2. **Build the dashboard UI (one-time, or after pulling frontend changes):**
   ```bash
   cd frontend && npm install && npm run build && cd ..
   ```
   Skip this and `autobot --server` still runs, but `/` serves a bare fallback page instead of the real dashboard.

3. **Verify the install:**
   ```bash
   autobot --doctor
   ```
   Pure stdlib, no LLM calls, costs nothing. Checks Python version, dependencies,
   Chrome, your LLM API key, and more — fix every `[FAIL]` it reports before
   going further. `[WARN]` items reduce capability but won't block a run.

4. **Run the Dashboard:**
   ```bash
   autobot --server
   ```
   *Access the command center at http://127.0.0.1:8000.*

5. **Try a Goal:**
   Enter a goal in the dashboard or extension: *"Go to Kaggle, list the top 5 active competitions, and save them to a file."*

## 🛠️ Installation & Setup (Dev Mode)

For developers who want to modify the code:

1. **Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Setup Tools:**
   ```bash
   autobot --setup
   autobot --doctor   # verify the install before running anything
   ```

3. **Run Dev Servers:**
   - **Backend:** `python -m autobot.main` (Port 8000)
   - **Frontend:** `cd frontend && npm install && npm run dev` (Port 3000)

## Environment variables

- `AUTOBOT_CHROME_USER_DATA_DIR` (optional)
- `AUTOBOT_CHROME_PROFILE_DIR` (optional, default: `Default`)
- `AUTOBOT_CHROME_EXECUTABLE` (optional)
- `AUTOBOT_CHROME_SOURCE_USER_DATA_DIR` (optional, default: local Chrome user-data root for bootstrap)
- `AUTOBOT_CHROME_SOURCE_PROFILE_DIR` (optional, default: same as `AUTOBOT_CHROME_PROFILE_DIR`)
- `AUTOBOT_CHROME_LAUNCH_TIMEOUT_MS` (optional, default: `15000`)
- `AUTOBOT_BROWSER_MODE` (optional; only `human_profile` is supported; default: `human_profile`)
- `AUTOBOT_OPEN_NEW_TAB` (optional: `1` or `0`; default: `1`) — In human_profile, when opening a URL, open it in a **new tab** (leave current tab open) instead of reusing the same tab. When navigating to the **same site** again in a chain (e.g. WhatsApp home then open chat), the same tab is reused; when switching to a different site, a new tab is opened. Set to `0` to always reuse the current tab.
- **Load waits (human_profile, seconds):** Optional patience after opening slow sites. Set to `0` to skip. Defaults: `AUTOBOT_WHATSAPP_LOAD_WAIT` = 8, `AUTOBOT_WHATSAPP_CHAT_LOAD_WAIT` = 5, `AUTOBOT_OVERLEAF_LOAD_WAIT` = 5, `AUTOBOT_GROK_LOAD_WAIT` = 4, `AUTOBOT_GOOGLE_DOCS_LOAD_WAIT` = 4.
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` (optional, enables LLM planner in autonomous mode)
- `AUTOBOT_LLM_PROVIDER` (optional: `gemini` or `openai_compat`)
- `OPENAI_API_KEY` or `XAI_API_KEY` (for `openai_compat`, including Grok-compatible endpoints)
- `AUTOBOT_OPENAI_BASE_URL` (optional, default: `https://api.x.ai/v1`)
- `AUTOBOT_LLM_MODEL` (optional; default: `gemini-1.5-flash` for Gemini; for Grok use e.g. `grok-2` or the model name from your X.AI account)

**Using Grok as the LLM (AI Planner):** Set `AUTOBOT_LLM_PROVIDER=openai_compat`, `XAI_API_KEY=<your key>`, and `AUTOBOT_LLM_MODEL=grok-2` or `grok-4-1-fast-reasoning` (see [x.ai docs](https://docs.x.ai/docs/guides/chat-completions)). If you get HTTP 403, the model name may be invalid—try `grok-2`. Get your API key from [x.ai Console](https://console.x.ai/team/default/api-keys).

### Controlling token spend

| Variable | Values | Effect |
| :--- | :--- | :--- |
| `AUTOBOT_VISION_MODE` | `always` / `auto` / `never` | `auto` (default) sends a screenshot only on the first step, when the DOM is too sparse to act on, or after a failed action. `always` costs roughly 1-2k extra tokens *per step*. `never` is text-only and cheapest, but blind to canvas/image-only UIs. |
| `AUTOBOT_APPROVAL_MODE` | `strict` / `balanced` / `trusted` | How often the agent pauses for permission. IRREVERSIBLE actions (deletion, payments, credentials, sending under your identity) always pause, in every mode. |
| `AUTOBOT_STEPS_PER_OBJECTIVE` | integer | Step budget per mission objective. |
| `AUTOBOT_SKIP_JUDGE` | `1` to enable | Skips the Judge Agent's separate LLM call at the end of every run (roughly doubles token spend on top of the task itself). Falls back to the run's own `done(success=...)` signal — a real cost/rigor tradeoff, not free: the Judge exists specifically to catch a run that believed it succeeded when it didn't. |

The largest saving isn't a setting: repeated tasks get distilled into **learned skills** (`autobot/knowledge/skills/`) and replayed instead of re-reasoned. `autobot --doctor` reports how many you've accumulated.

## Run folder layout

Every run is stored as a **folder** with a human-readable name so you can see what it is at a glance:

- **Folder name:** `plan_YYYY-MM-DD_HH-MM-SS` (e.g. `tool_call_stress_2026-02-20_10-56-21`). No cryptic timestamps or long numbers.
- **Inside the folder:**
  - **`about.txt`** – Short summary: plan name, started/finished time, success, steps. Read this first.
  - **`history.json`** – Full step log, state, adapter telemetry.
  - **`artifacts.json`** – Message sent, PDF path, doc/LaTeX previews (when applicable).
  - **`screenshots/`** – Screenshots from key steps.
  - **`console.log`** – Console output (when run from CLI).

To **organize old runs** (single JSON files from before this layout), run once:

```bash
python -m autobot.organize_runs       # dry run: shows what would be done
python -m autobot.organize_runs --do # move each runs/*.json into a named folder + about.txt
```

After that, every run is a folder with at least `history.json` and `about.txt`; newer runs also have artifacts, screenshots, and console log.

## Running the tool-call stress test

Each run creates a **run folder** (see "Run folder layout" above), e.g. `runs/tool_call_stress_2026-02-20_10-56-21/`.

**From the UI:** Set browser mode to **human_profile**, choose preset **tool_call_stress**, and in Topic use:  
`<phone>|<docs_url>|<download_path>|<message>`  
Example: `+27930793632858|https://docs.google.com/document/d/ID/edit|C:/Users/.../Downloads/out.pdf|Test message`  
Then run. When finished, the log shows **Run folder:** with the path to open.

**From the CLI (e.g. when you’re away):**

```bash
python -m autobot.run_stress "+1234|https://docs.../edit|C:/path/to/download.pdf|Your message"
# or with env:
set AUTOBOT_STRESS_TOPIC=+1234|https://...|C:/path/to/pdf|Message
python -m autobot.run_stress
```

With empty topic or `|||message`, WhatsApp and file-download steps are skipped; the rest of the chain still runs and screenshots are captured so you can confirm behavior when you return.

## Code iteration and Autonomous mode (iterate until it works)

To get the system to **iterate with the AI until tests pass** (or until a goal is met):

1. **Code iteration preset**  
   - In the UI, choose preset **code_iteration** and in Topic enter the test command (e.g. `pytest`, `npm test`, `python -m pytest tests/`).  
   - Or in the quick task box: **run code iteration** or **code iteration pytest**.  
   - This runs the command, captures output, opens Grok, and puts the failure output in the clipboard so you (or the next step) can paste and get a fix. Run the preset again after applying the fix to repeat.

2. **Autonomous mode (AI decides next steps from state)**  
   - Set **Goal** to something like “Make the tests in my project pass” or “Fix the failing build.”  
   - Set **Diagnostics command** to the command that checks success (e.g. `pytest`, `npm test`).  
   - Optionally set **Target URL** to your app (e.g. `http://localhost:3000`) so the engine can open it and capture console errors.  
   - Run **Autonomous mode**. Each loop: the engine runs the diagnostics command, captures exit code and output, passes that **state** to the AI (OpenRouter/DeepSeek by default). The AI returns the next steps (e.g. open Grok, set clipboard with a fix request, run a command). The engine executes them and loops until the goal is met or max loops are reached.  
   - So: **functionality first**—the AI uses real state (test output, errors, clipboard) to decide what to do next, and you can iterate until the “crazy” goal is achieved.

## Browser control without CDP (extension DOM bridge)

CoreLoop's primary perception is UIAutomation, not CDP — but UIAutomation
only reliably sees Chrome's toolbar and tabs, not the content of the page
itself. For actual page content, Autobot talks to the page through the
**Autobot Chrome extension** (`extension/`), which is already installed in
your real, logged-in Chrome and can read/click the DOM directly through
permissions Chrome already granted it — no debug port, no isolated
profile, none of the SingletonLock/profile-corruption issues CDP launch
caused on Windows.

**One-time setup:**
1. `chrome://extensions` → enable **Developer mode** → **Load unpacked** →
   select the `extension/` folder. (Already loaded? Click its reload icon
   to pick up updates.)
2. Click the Autobot toolbar icon once on any normal page to confirm the
   content script is alive there.
3. Start the backend (`autobot --server`, or `python -m autobot.main`).

**Try it directly** (no LLM needed — proves the bridge itself works):
```bash
python demo_extension_bridge.py          # status + read the focused tab + list its clickable elements
python demo_extension_bridge.py list     # numbered list of visible clickable elements
python demo_extension_bridge.py click 3  # click element [3] from the last `list`
```

**From the agent loop:** CoreLoop exposes this as four actions —
`browser_text`, `browser_list`, `browser_click`, `browser_type` — documented
in `autobot/prompts/system_prompt.md`. They act on whichever tab is
currently focused in your real Chrome.

The bridge is a simple HTTP-polling round trip (backend queues a command →
extension's background.js polls for it every ~1s → runs it in the active
tab's content script → posts the result back) — see
`autobot/browser/extension_bridge.py` for the implementation and
`tests/test_extension_bridge.py` for offline tests of the queue mechanics.

## Kaggle + Claude Code + Antigravity + Overleaf

Four different mechanisms for four different levels of API access —
picked deliberately, not automated-by-default:

- **Kaggle** — a real API. `autobot/computer/kaggle_tool.py`'s `Kaggle`
  class wraps `kaggle.api.kaggle_api_extended.KaggleApi` directly (the same
  library the official `kaggle` CLI itself is built on). Reachable from the
  agent loop via the existing `computer_call` action:
  `computer.kaggle.pull_kernel(kernel, path)`, `.kernel_status(kernel)`,
  `.push_kernel(path)`, `.kernel_output(kernel, path)` — all new tonight,
  alongside the pre-existing `.list_competitions()`, `.download_data()`,
  `.submit()`, `.get_leaderboard()`. Tests: `tests/test_kaggle_tool.py`
  (kernel methods, mocked KaggleApi — no real Kaggle account touched).
  Risk tiers are deliberately split: `pull_kernel`/`push_kernel`/
  `kernel_status`/`kernel_output` are SAFE-tier — create, edit, run, and
  read a notebook as many times as you want, no approval pause in any
  mode. `computer.kaggle.submit(...)` (a real competition submission) is
  the opposite: IRREVERSIBLE-tier in `autobot/agent/approval.py`, always
  requires a live "Allow" click, in every approval mode including
  `trusted`. Tests for the split: `tests/test_approval_new_patterns.py`
  and `tests/test_core_loop.py::TestKaggleRiskSeparation`.
  **Keeps working while you're away from the computer**: set
  `AUTOBOT_UNATTENDED=1` and CAUTION/DANGER actions (including Kaggle
  kernel iteration) proceed automatically instead of pausing on a prompt
  nobody's there to answer — logged and desktop-notified so you can review
  what happened when you're back. `submit()` is the one exception: it
  stays IRREVERSIBLE and never auto-proceeds, unattended or not — see
  `approval.py`'s "Unattended mode" docstring and
  `tests/test_unattended_approval.py`.
- **Claude Code** — a real headless CLI mode. `autobot/integrations/
  claude_code_bridge.py` wraps `claude -p --output-format json` as a
  subprocess (no shell=True — argv only, so nothing in a prompt can be
  interpreted as shell syntax); `autobot/computer/claude_code_tool.py`'s
  thin `ClaudeCode` class exposes it the same way as Kaggle:
  `computer.claude_code.run(prompt, cwd=..., permission_mode=...)`.
  `permission_mode` defaults to `"plan"` (read-only); `"acceptEdits"` /
  `"bypassPermissions"` are DANGER-tier and gated accordingly. Tests:
  `tests/test_claude_code_bridge.py` (subprocess/argv contract, mocked —
  no real `claude` CLI invoked) and `tests/test_claude_code_tool.py` (the
  wrapper class).
- **Antigravity** — Google Antigravity's own headless CLI (`agy`), the
  same shape of integration as Claude Code above: `autobot/integrations/
  antigravity_bridge.py` wraps `agy -p --output-format json` as a
  subprocess (no shell=True), `autobot/computer/antigravity_tool.py`
  exposes `computer.antigravity.run(prompt, cwd=..., skip_permissions=...,
  model=..., effort=..., agent=...)`. `skip_permissions` defaults to
  `False` (agy respects its own scoped allowlist); `True`
  (`--dangerously-skip-permissions`) is DANGER-tier, mirroring Claude
  Code's write-capable modes. Tests: `tests/test_antigravity_bridge.py`,
  `tests/test_antigravity_tool.py`, and the DANGER-pattern tests in
  `tests/test_approval_new_patterns.py`.
- **Overleaf** — no public write API, so this is the one leg that still
  needs the browser DOM bridge above. `browser_paste` (new tonight,
  alongside `browser_text`/`browser_list`/`browser_click`/`browser_type`)
  dispatches a synthetic `paste` event rather than writing `value`/
  `textContent` directly, because Overleaf's editor (CodeMirror) manages
  its own document state and a direct write doesn't reliably register —
  paste events are what these editors actually listen for. **This is
  unproven against the real, live Overleaf editor** — it hasn't been run
  against an actual document yet. Try it supervised first; if the paste
  event doesn't register, the documented fallback is a real OS-level paste
  (`computer.clipboard.set(text)` then `computer.keyboard.press("ctrl+v")`
  after `browser_click` focuses the editor).

A typical goal for the agent loop: pull a kernel, hand its code to Claude
Code, push the result back, poll until the run completes, pull the output,
have Claude Code draft a paper section from it, then paste that into
Overleaf. See the "Kaggle + Claude Code + Overleaf Workflow" section of
`autobot/prompts/system_prompt.md` for the exact sequence the agent is
told to follow, and why each step is checked before the next one starts.

## Multi-project orchestrator (Claude Code / Antigravity check-ins)

For juggling several AI-assisted coding projects at once — Claude Code in
VS Code, more than one Antigravity project, whatever else uses one of the
two backends above — without re-explaining each one's goal every time you
switch context:

- **`autobot/knowledge/project_registry.py`** — register a project once
  with `registry.register(name, working_dir, backend, intent)`. `intent`
  is stored exactly as you wrote or pasted it — never summarized or
  rewritten by Autobot — because the whole point is Autobot remembering
  what *you* said you wanted, not its own paraphrase of it. `backend` is
  `"claude_code"` or `"antigravity"` (validated — an unrecognized backend
  raises immediately rather than silently producing a project the
  check-in loop can never reach later). One JSON file per project under
  `autobot/knowledge/projects/`, same storage pattern as the skill
  library. `add_intent_note(name, note)` appends something new you told
  it about a project without overwriting the original intent. Tests:
  `tests/test_project_registry.py` (23 tests, real temp-directory
  fixtures).
- **`autobot/agent/orchestrator_checkin.py`** — `OrchestratorCheckIn().
  check_in_on("project-name")` asks that project's AI backend for a
  status update (what's done, in progress, blocked, and what needs a
  decision from you) and records the summary back to the registry.
  `check_in_on_all()` sweeps every tracked project; one project's backend
  being unreachable doesn't stop the others from reporting in. Each
  check-in resumes the project's prior Claude Code session / Antigravity
  conversation by id, so the AI has context from earlier check-ins
  instead of a cold start every time. Tests:
  `tests/test_orchestrator_checkin.py` (24 tests).

**Deliberately read-only for now.** This asks each project's AI "what's
your status," never "go do X" — sending an actual follow-up instruction on
your behalf while you're not watching is a bigger decision (real file
writes, real token/compute spend in someone else's project, unattended)
than reading a status report, and it hasn't gotten the same kind of
explicit go-ahead the Kaggle unattended-mode question above did. The
check-in prompt itself explicitly tells the backend not to start new
work in response. Extending this to actual autonomous prompting is the
natural next step, but on purpose not this one — see
`orchestrator_checkin.py`'s module docstring for the full reasoning.

Also fixed tonight, found while wiring this in: `computer_call` actions
(the only way `computer.kaggle.*`, `computer.claude_code.*`,
`computer.research.*`, `computer.vault.*`, and `computer.files.*` are
reachable — none of them have dedicated action names) were silently
failing on every single call — `CoreLoop._dispatch()` called the async
`dispatch_computer_call()` without awaiting it, which doesn't run it at
all. Regression test: `tests/test_core_loop.py::TestComputerCallDispatch`.

## Notes

- If launch fails on default Chrome profile restrictions, let Autobot use a dedicated automation profile directory.
- First launch may require one-time sign-in in the automation profile if cookies cannot be copied.
- In `human_profile` mode, URL/search and keyboard flows use your real Chrome session; DOM-selector actions and devtools automation are disabled in this build.
- Human-mode safety guard blocks typing if Autobot/Cursor window appears focused, to prevent runaway self-trigger loops.
- **Load waits (patience):** After adapter actions that open slow-loading pages (e.g. WhatsApp, Overleaf), the engine waits a few seconds before the next step. Defaults are in code (e.g. 8s after WhatsApp open_home, 6s after open_chat). Override by placing a `load_waits.json` next to `engine.py` (same format: `{"adapter": {"action": seconds}}`).
- AI Planner Chat panel allows prompt -> plan preview -> execute workflow (uses configured provider or safe fallback).
- End-to-end tool-calling stress workflow is available as preset `tool_call_stress`.
- Desktop actions are intentionally explicit and coordinate-based to keep behavior predictable.
- This is a foundation for larger autonomous loops; extend workflows in `autobot/workflows.py`.
- Autonomous mode blocks obvious bulk-messaging intents by default. Use consent-based, explicit tasks only.
- Login behavior:
  - Adapters include a `attempt_google_continue_login` action to click "Continue with Google" when present.
  - If not present, rely on profile-saved password autofill or ask user to intervene.
- WhatsApp Web (human mode): The adapter opens WhatsApp home first, waits for load, then opens the chat by phone. Phone numbers are normalized to digits only. If opening by direct link is unreliable, use search instead by passing `use_search: true` in the adapter params for `open_chat` (e.g. in a workflow or UI adapter call). Keep the Chrome window focused (or click it) before running so the new tab is visible.
