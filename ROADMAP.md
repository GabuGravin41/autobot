# Autobot Roadmap

This document exists because the project's actual failure mode, twice now,
has been the same one: real capability gets built, then never wired to
anything that runs, and the next round of work builds more unconnected
capability on top instead of finishing the connection. This roadmap is
kept in the repo (not just chat history) so that doesn't happen a third
time — check an item off only when it is actually reachable from a real
run, not when the file exists.

## The vision, reframed

The goal is **not** an AI operating system. It's a general **computer-use
orchestrator**: something that can operate arbitrary software the way a
human does — browser, native desktop apps (Artemis, VESTA, Excel, DICOM
viewers, DAWs), and other AI tools (Claude Code, ChatGPT, Grok) — under a
permission model the user controls, and that gets *cheaper* over time on
repeated tasks instead of re-reasoning from scratch every run.

Three separable capabilities, not one monolith:
1. **General computer control** — not just browser DOM clicking.
2. **A real permission model** — from "ask me everything" to "full
   autonomy," with certain categories that stay hard-gated no matter what.
3. **A skill library** — successful runs get distilled into reusable,
   cheap-to-replay procedures instead of being re-derived every time.

## Status as of this document

### Done and verified wired (this round of fixes)
- Core `AgentLoop` can actually run — its DOM extraction import was dead
  (`autobot/dom/extraction.py` deleted, never unimported) since March 2026;
  every real entry point was crashing before a single browser action fired.
- Click/fill execution unified onto the CDP path (`computer/browser.py`)
  instead of two independently-indexed systems that could silently drift
  apart.
- `MissionAgent` objective decomposition is now actually reachable from
  `AgentRunner` for multi-phase goals, instead of forcing everything
  through one flat step budget.
- `ApprovalGuard` now has a non-bypassable **IRREVERSIBLE** tier (deletion,
  financial transactions, credential entry, sending/publishing under the
  user's identity) that no approval mode — including `trusted` — can skip,
  and it is now actually called from `AgentLoop._execute_actions` instead
  of sitting unused.
- The dashboard's `/api/human_input` endpoint was hardcoded to always
  return "nothing pending," which meant a real approval request would
  register and then silently time out with no way to ever click Allow.
  Now backed by `human_gate.get_pending()`/`respond()` with a real
  `POST /api/human_input/respond` endpoint.
- Removed a cookie/session-file-copying fallback in `browser/launcher.py`
  that directly violated `DESIGN_PHILOSOPHY.md`'s own rule against
  manipulating locked Chrome profile databases.

### Round 2 — the "computer use" unlock (done, with tests)
- **`uiautomation` was a hard crash, not an optional feature.**
  `computer/computer.py` imported it unconditionally on Windows while
  `requirements.txt` had it commented out. Since `AgentLoop` constructs
  `Computer()`, a fresh install on Windows killed the entire agent at
  import. Now optional with a clear degraded-mode warning, and declared
  properly with a `sys_platform == "win32"` marker.
- **Generic `computer_call` action (roadmap #2) — the actual unlock.**
  The mechanism was documented across 600+ lines of
  `prompts/system_prompt_full.md`, handled by `lesson_extractor.py` and
  `experience_store.py`, and dispatched by `background_runner.py` — but
  the field was missing from `ActionModel`, so pydantic silently dropped
  it and the loop executed "unknown action". The agent could *see* every
  OS tool in its catalog and invoke none of them. Fixed, and the active
  `system_prompt.md` now documents the calling syntax (it previously
  injected `{tool_catalog}` without ever saying how to call it).
- **One shared dispatcher.** `computer/dispatch.py` is now the single
  AST-safe implementation used by both the foreground loop and
  `background_runner` (whose docstring already claimed to share one with
  `AgentLoop` that never existed). Calls are parsed structurally, never
  `eval()`d; arguments must be literals; private/dunder names rejected.
  18 behavioral tests including injection and traversal attempts.
- **Unknown actions now teach the model.** An unrecognized action key used
  to produce a bare "Unknown action: unknown" — no signal, so the LLM
  would emit the same bad action again. It now names the invalid keys,
  lists the valid ones, and shows the `computer_call` syntax.
- **Native app perception (roadmap #3).** When a non-browser window has
  focus, its UIA element tree is extracted into a `<native_window_state>`
  prompt section, with an explicit warning that its `[N]` indices are a
  separate index space from browser DOM indices. Extraction goes through
  `computer.window` (not a fresh `NativeExtractionService`) so the indices
  the agent reads are the same ones `computer.window.click(N)` resolves —
  the same index-drift class of bug that was fixed for the browser.
- **Skill distillation write path (roadmap #1).** `save_skill()` was never
  called from anywhere: the skill library could never populate, so
  `get_skill_prompt_context()` always returned empty and every run
  re-derived from scratch at full cost. `distill_from_run()` now records
  the successful path on `done(success=True)`, turns failed actions into
  "lessons learned", bumps `success_count` on repeat, and keeps the
  shortest known path. 15 behavioral tests covering the full
  distill → save → find → inject loop.

### Built, but still not wired to a real run
- **Domain helpers** — audited. `bio_synthesizer.py`/`dicom_synthesizer.py`/
  `materials_synthesizer.py` had a real code-injection vulnerability, now
  fixed and tested (44 tests). `overleaf_helper.py`'s `OverleafHelper` is a
  complete, already-built solution to the CodeMirror paste-reliability risk
  — resilient multi-selector clicking, clipboard-based LaTeX injection with
  a keyboard fallback, recompile handling. Still zero callers anywhere.
  Deliberately not wired in yet — real failure data from a live run beats
  speculative wiring for a risk that hasn't been confirmed to occur.
- ~~`system_prompt_full.md`~~ — resolved. It documented a stale vision-
  first, coordinate-guessing-primary architecture that this project's
  actual CDP/DOM-index-first design has superseded; merging it in would
  have reintroduced the exact blind-action anti-pattern already fixed
  twice. Deleted. Pulled the one genuinely good idea (a situational-
  awareness checklist: where am I / is this relevant / are there obstacles)
  into the active `system_prompt.md`, correctly reframed for DOM-index
  clicking as primary rather than vision+coordinates.

### Round 9 (Sep 2026) — the butler: from a clicking agent to a manager of workers

**Why the pivot.** A review of the actual run logs showed the last time CoreLoop ran on its own
(Sep 7, DeepSeek V4 Flash, an Overleaf task) it looped for 12 steps and did nothing, while all nine
competitions after that were driven by Antigravity using Autobot's Kaggle tools as a library. The
useful part of Autobot was its tools and guardrails; the missing part was a manager. Dalton's
own workflow (think, prompt a capable tool, switch, judge, prompt again) is the spec.

**Built (autobot/butler/, autobot/llm/, autobot/integrations/{cli_exec,gmail}.py):**
- A daemon that owns time (`autobot butler run`): SQLite state in ~/.autobot, single instance,
  crash-safe resume, bounded concurrency, clean shutdown that kills its worker processes.
- A per-task state machine (playbook.py): start -> propose checks -> work -> verify -> finalize,
  with Kaggle rounds that wait on kernels for free, escalation, rate-limit cooldowns, and an
  inbox for questions/approvals. No model can declare a task done; checks.py decides.
- Lanes: coding, document (LaTeX), learning (modules), research (web search), kaggle, email.
  The Kaggle post-mortem lessons are standing instructions in lanes.py.
- Workers: Claude Code and Antigravity headless, with structured JSON reports and hard denials
  (push, submit, recursive delete). Works on subscriptions; no API key needed.
- Gmail via the official API: sync -> digest -> drafts; sending only on approval. The triage
  worker has no shell and no web. Live test: it flagged a prompt-injection email as phishing.
- Inbox via CLI and a token-protected phone page (/butler).
- Optional manager model with a fallback chain (claude_cli, agy_cli, Groq, Cerebras, OpenRouter,
  Ollama, Gemini) and schema-constrained output.

**Bugs found and fixed on the way (each reproduced first):**
- `_normalize_status` read every real Kaggle status as "error" (kagglesdk's str() contains
  "failureMessage"). This one bug explains the "liveness false positives" in five competition
  logs and the ledger staleness in two.
- Capacity check-then-push race across processes -> cross-process file lock; ledger writes now
  lock-reload-save with unique temp names.
- A malformed action ended CoreLoop runs as "Task complete" (success=True).
- Failed runs were saved as "proven" skills and injected into later runs.
- CoreLoop showed the model only its last step (cause of the Sep 7 loop) -> history window,
  repeat blocking.
- Windows: npm `.cmd` shims not found by subprocess; prompts truncated at newlines by cmd.exe;
  agy's own 5-minute print timeout; kaggle log download crashing on cp1252; kaggle sys.exit()
  killing long-running processes; state written relative to the current folder.
- /api/health was hardcoded to "OK".

**Verified live (not just unit tests):** a real Claude Code worker fixed a bug end to end; the
real daemon ran a research task (live web search, 38 links) and a LaTeX learning module
concurrently; the email lane handled a real model with an injection email. 390 tests pass.

**Still to do:** a Windows run of `autobot butler smoke` on Dalton's machine; Gmail connected
for real; a week of unattended running with the failure log reviewed before adding the next
lane (desktop-app automation last).

### Round 8 (Sep 2026) — multi-project orchestration, and closing the gap it exposed

Triggered directly by Dalton's request: "also ability to run multiple
projects like openclaw ... make any updates you think are necessary to the
codebase." Two things came out of actually acting on that, not just the
one that was asked for.

**What was asked for — built and wired end to end:**
- `autobot/agent/orchestrator_dispatch.py` (new) — the write-capable half
  of the multi-project orchestrator. `dispatch_on()` sends one real
  instruction to one tracked project's AI backend (Claude Code or
  Antigravity, resuming its prior session); `dispatch_on_all()` runs many
  projects **concurrently**, bounded by an `asyncio.Semaphore`
  (`AUTOBOT_ORCHESTRATOR_MAX_CONCURRENT`, default 3) — OpenClaw's actual
  pattern (per Dalton's own `THINKING_AND_DECISIONS.md` notes) is stateless
  turns over a persistent on-disk workspace plus a global concurrency
  throttle, not unbounded parallel dispatch. One project's crash/timeout
  never affects another's — proven with real timing-based tests
  (`test_projects_actually_run_concurrently_not_sequentially`,
  `test_semaphore_bounds_actual_concurrency`), not just mocked call counts.
- Kept as a **separate module** from Round 6's `orchestrator_checkin.py`
  rather than merged into it — that module's docstring explicitly deferred
  write-capable dispatch pending "an explicit, considered decision," the
  same way Kaggle's unattended-autonomy question got one earlier this
  project. Dalton's request IS that decision; the separation means the
  already-safe read-only check-in path was never put at risk making it.
- **`autobot/cli.py`** — `--register-project` / `--project-dir` /
  `--project-backend` / `--project-intent`, `--list-projects`,
  `--check-in [NAME]`, `--dispatch NAME INSTRUCTION`, `--dispatch-all FILE`.
  Write access stays opt-in and per-project (a `--dispatch-all` JSON file
  can carry `permission_modes`/`skip_permissions` overrides keyed by
  project name); the default for every project not explicitly overridden
  is still "plan" / read-only, inherited from the bridge functions rather
  than reinvented here.
- Kaggle's own account-wide hardware caps enforced as a real semaphore
  (`autobot/computer/kaggle_watchdog.py`: `KAGGLE_GPU_SLOT_LIMIT = 2`,
  `KAGGLE_CPU_SLOT_LIMIT = 4`, `check_capacity()`), wired into
  `push_kernel()` — necessary specifically *because* multiple
  Kaggle-competition projects can now be dispatched concurrently through
  the orchestrator, and Kaggle's kernel slots are a real, hard,
  account-wide limit that concurrent dispatch could otherwise blow through
  without anything ever calling the real API to find out.

**What "make any updates you think are necessary" surfaced, unprompted:**
- Grepped for callers of Round 6's `project_registry.py` and
  `orchestrator_checkin.py` before building on top of them, on the "verify
  before building" discipline this project has used all along — found
  **zero** production callers anywhere in `cli.py` or `web/app.py`. Both
  modules were fully built and fully tested in Round 6, and had been
  sitting in the exact "Built, but still not wired to a real run" state
  this roadmap exists to catch, for an entire round, silently. This CLI
  wiring is that connection finally getting made — for both the Round 6
  code and the new Round 8 dispatch module at once.
- `autobot/knowledge/project_registry.py`'s `_save()` was a plain
  `path.write_text()` — flagged in Round 6 as safe only because that
  round's only caller was a genuinely sequential for-loop. Round 8's
  `dispatch_on_all()` is exactly the concurrent caller that assumption was
  waiting for (multiple `asyncio` tasks calling `record_check_in()` /
  `update_session()` at once). Fixed to the same atomic
  temp-file-then-rename pattern already used by
  `kaggle_watchdog.py`'s `KaggleJobLedger.save()`.
- `autobot/integrations/claude_code_bridge.py` was missing the
  nonexistent-`cwd` guard that `antigravity_bridge.py` already had (flagged
  in Round 6 as "real but out of scope for that round's diff"). A stale or
  moved `working_dir` on a `TrackedProject` is a real, now-live risk the
  moment `orchestrator_dispatch.py` can dispatch to it, so the fix was
  ported over rather than left for a third round to rediscover.

**Verified, not assumed:** full test suite run after every change (244
passing, up from 197 at the end of Round 7 — 47 new tests, all exercising
real behavior: actual concurrency timing, actual capacity-limit
enforcement, actual CLI argument parsing via `--help`, actual registry
persistence across separate `ProjectRegistry` instances); `python -m
autobot.cli --help` actually run to confirm the new flags parse, not just
that the file imports; `import autobot.web.app` actually run, not just
syntax-checked.

**Deliberately still deferred:**
- No dashboard endpoints for the orchestrator yet (`/api/orchestrator/...`
  mirroring Round 7's `/api/kaggle/jobs`). The CLI surface needed to exist
  and be tested first — a dashboard endpoint for concurrent multi-project
  dispatch is a reasonable Round 9 addition, not one to rush in alongside
  the CLI wiring itself.
- No automatic/unattended dispatch loop (nothing calls `dispatch_on_all()`
  on a timer). Every dispatch in this round is triggered by an explicit
  command — the same posture Kaggle's watchdog took before its own
  explicit unattended-autonomy conversation happened. If Dalton wants
  scheduled/autonomous multi-project dispatch later, that's its own
  explicit decision, same pattern as before.

## Sequencing

Ordered by what unlocks the most, and by what's cheapest to verify is
actually working (not just present).

1. ~~Confirm skill distillation is a closed loop.~~ **Done** — it wasn't
   (nothing ever called `save_skill`); now closed and tested. Still worth
   confirming on a real double-run that step 2 is visibly cheaper.
2. ~~Add the generic `computer_call` action.~~ **Done and tested.**
3. ~~Wire native UI extraction into OBSERVE.~~ **Done** — but only
   statically verified; needs a real run against an actual native app
   (Notepad is the cheapest first test, then Artemis).
   For apps with no accessibility tree at all (some scientific software
   genuinely has none), fall back to local OCR (Tesseract or EasyOCR) over
   a screenshot to locate text/buttons — vision-only, more expensive, but
   only needed as a last resort. **Not yet built.**
4. **API-level integration with other AI tools before UI automation of
   them.** If Autobot is going to drive Claude Code, do it through its
   CLI/SDK, not by taking screenshots of a chat window — an order of
   magnitude cheaper in tokens and far more reliable. Reserve
   vision-heavy UI automation for software that genuinely has no API
   (most GUI scientific tools). **Next up.**
5. **Higher-level OS adapters** on top of `Computer`: a `WindowAdapter`
   (`list_windows()`, `focus_window("Notepad")`, `close_app()`) and a
   `FileAdapter` for safe filesystem traversal — these make native-app
   skills shorter and cheaper to express than raw mouse/keyboard
   sequences. `computer/window.py` already covers part of this.
6. **Only after 2–5 are real:** revisit end-to-end "build an app, talk to
   Claude Code, check in from your phone" workflows. That's a composition
   of the capabilities above, not a new one — building it first, before
   the pieces underneath are solid, is exactly how the project got into
   the state this file exists to prevent.

### Round 3 — making failure legible and runs cheap
- **`autobot --doctor`** (`autobot/diagnostics.py`). Every serious bug in
  this project failed at a different layer but surfaced identically: "it
  doesn't work." The doctor checks each layer independently — Python
  version, required and optional packages, `.env`, which LLM key is set
  (never its value), approval mode, Chrome executable, whether anything is
  listening on the CDP port, writable dirs, and how many skills have been
  learned — and prints an actionable fix per failure. No LLM calls, no
  browser, pure stdlib, so it works even when the agent is badly broken.
- **Vision cost control** (`AUTOBOT_VISION_MODE=always|auto|never`).
  A screenshot is roughly 1-2k tokens versus a few hundred for the DOM
  text, and it was being sent on EVERY step. `auto` (the new default)
  spends it only where it pays: the first step, when the DOM is too sparse
  to act on (canvas apps, SPAs mid-render), and after a failed action —
  where the text state has demonstrably proven insufficient. In `never`
  mode the screenshot isn't even captured.
- **Self-correction ladder for clicks.** A failed click used to return a
  bare failure, and the model would typically re-issue the identical
  click. It now escalates through genuinely different mechanisms —
  CDP click, then scroll-into-view + retry, then `click_via_js()` which
  bypasses pointer hit-testing entirely (the fix for overlay/banner
  interception) — and if all fail, reports every method tried plus an
  explicit "do NOT retry this same click" with likely causes.

### Round 4 — paranoid static audit (no live LLM/browser access this pass)

No network access to openrouter.ai was available during this pass, so
nothing here was verified by a live agent run — everything below is either
(a) proven by direct execution of the affected code in isolation (the vault
key derivation, the anti_sleep dispatch fix, the offline test suites), or
(b) proven by tracing real import/call graphs with Grep across the whole
tree, never by reading a docstring and assuming it's accurate. Where a claim
below is a call-graph trace rather than an execution, it says so.

**Security fixes (all confirmed by direct execution, not just reading):**
- **`computer/vault.py` — every Windows install shared the same encryption
  key.** `_derive_key()` had a Linux branch and a macOS branch but no
  Windows branch, and read the POSIX-only `USER` env var (Windows uses
  `USERNAME`). Result: on Windows — the only platform this project actually
  ships to — `machine_id` was always empty and every install derived the
  identical hardcoded seed, so a copied `vault.json` could be decrypted on
  any other Windows machine with no access to the original required. Fixed
  with a real Windows machine ID (registry `MachineGuid`) plus, more
  importantly, a random 32-byte salt generated once and persisted
  per-install — the salt is what actually provides security regardless of
  whether any future platform's machine-ID lookup breaks again. **Breaking
  change**: any vault entries stored before this fix will not decrypt with
  the new key. Given the vault feature has no confirmed live use yet, this
  was judged worth it over preserving a predictable key.
- **`computer/clipboard.py` — PowerShell command injection.** The Windows
  fallback built `Set-Clipboard -Value '{text}'` by string interpolation.
  Any text containing a single quote — trivially reachable via
  `clipboard.copy()` on scraped web content — broke out of the quoted
  literal and executed arbitrary PowerShell. Fixed by passing text through
  an environment variable (`$env:X` is a data reference, never re-parsed as
  code) instead of ever interpolating it into the command string.
- **`web/app.py` `GET /api/run/{run_id}` — path traversal.** `run_id` from
  the URL was joined directly into a filesystem path with no containment
  check. Fixed: both paths are resolved and the result must stay under
  `runs_root`.
- **`web/app.py` `POST /api/agent/run` — TOCTOU race on module globals.**
  The "not already running" check and the writes that followed it were not
  atomic; two near-simultaneous requests could both pass the check, and the
  second silently overwrote `_agent_runner`, orphaning the first run with no
  way to cancel or query it. Fixed with a lock around the whole
  check-and-set.
- **`web/app.py` `POST /api/chat` — SUSPECTED still a stub** (found by a
  sub-agent, not yet independently re-verified line-by-line): always
  returns the same canned reply regardless of `req.message`, never touching
  the LLM, `TaskClassifier`, or `MissionAgent`. Same category as the
  `/api/human_input` bug fixed earlier — worth fixing before anything is
  built assuming this endpoint is real.

**Correctness fixes:**
- **`computer/computer.py` `get_tool_catalog()` derived each tool's
  dispatchable name from `tool.__class__.__name__.lower()` instead of its
  real attribute name.** This happened to match for most tools (`Mouse` ->
  `mouse`) but not `anti_sleep` (an `AntiSleepManager` instance), so the
  catalog advertised `computer.antisleepmanager.start()` — a name
  `dispatch.py`'s `getattr(computer, ...)` can never resolve. The entire
  anti-sleep feature (the background mouse-mover that keeps a long run's
  machine awake) was unreachable from any LLM-emitted call. Fixed generally
  — catalog names now come from the real attribute, so this class of bug is
  structurally impossible for any future tool, not just patched for this
  one. Verified end-to-end via the dispatcher, not just read.
- **`dom/native_extraction.py`** had two bare `except:` clauses during UIA
  tree traversal, which also swallow `KeyboardInterrupt`/`SystemExit` —
  meaning Ctrl+C during a slow native-window extraction would silently do
  nothing. Narrowed to `except Exception:`.
- **`run_grok_research_benchmark.py`** had a `\d` in a non-raw string
  (`SyntaxWarning`, harmless today but silences a real signal for the next
  actual bug of this kind).

**Major finding — a second orphaned-code layer, confirmed by import-graph
tracing (Grep for real `from ... import` statements, not just filename
mentions):**

The `MissionAgent`/`Orchestrator`/`ApprovalGuard`/`save_skill` orphaning
found in earlier rounds was not the whole picture. Confirmed **zero external
importers** for:
- **The entire `learning/` package** — `rl_controller.py`, `policy_memory.py`,
  `reward_computer.py`, `lesson_extractor.py`, `experience_store.py`. An
  apparently complete RL training pipeline (git history: "add RL pipeline,
  multi-agent orchestration, and adaptive waiting") that nothing in
  `agent/loop.py`, `agent/runner.py`, or anywhere else ever calls.
- **`agent/scheduler.py`'s `TaskScheduler`** — 486 lines, a real
  multi-task concurrent scheduler with priority queueing, a concurrency
  limit, and its own `AgentRunner` integration. It correctly imports and
  uses `agent/resource_manager.py`'s `ScreenLock` for time-slicing screen
  access between concurrent tasks — internally coherent — but nothing in
  `web/app.py` or `cli.py` ever imports `scheduler.py` itself, so none of
  this runs.
- **`agent/resource_manager.py`'s `ScreenLock`** — used only by the orphaned
  scheduler above; transitively unreachable.
- **`agent/orchestrator.py`'s `Orchestrator` class** (distinct from its
  `TaskClassifier`, which — see Round 3 — genuinely is wired into
  `AgentRunner.run()` now). `Orchestrator` itself, with its task
  decomposition and parallel sub-agent execution, is still never
  instantiated anywhere.
- **`agent/message_bus.py`** — used only by the orphaned `Orchestrator`;
  transitively unreachable.
- **`agent/evaluator.py`'s `EvaluationAgent`**, **`agent/planner.py`'s
  `ComplexityEstimator`**, **`agent/diagnostician.py`'s
  `TerminalStderrDiagnostician`**, **`agent/whatsapp_listener.py`'s
  `WhatsAppListener`** — each fully self-contained, each with zero external
  importers.

Not fixed this pass, and deliberately so: wiring any of these in is a real
design decision (should the RL pipeline actually train on live run data?
should the scheduler replace or sit alongside `AgentRunner`?) that
shouldn't be made unilaterally with zero live-run verification available.
Recorded here so it's a decision made on purpose next time, not
rediscovered by surprise a third time.

### Round 5 — CDP retirement, tool-catalog correctness, harness-side perception routing

By this round the architecture had already moved past everything Rounds
1-4 describe: `AgentLoop`, `MissionAgent`, `dom/extraction.py`,
`dom/page_snapshot.py`, and `browser/launcher.py` were gone from the live
`autobot` package namespace entirely (not even present as symlinks in the
live tree — only under `_archive/`), replaced by `CoreLoop`
(`agent/core_loop.py`), UIAutomation-first native-app perception
(`computer/window.py`), and the Chrome extension's DOM bridge
(`browser/extension_bridge.py`) for web page content. That transition
itself predates this round and isn't re-litigated here — this round is
about what was still live and wrong *within* that newer architecture.

**The one CDP path still actually wired in, found and cut:**
- `computer/computer.py` still imported and instantiated the old CDP
  `computer/browser.py`'s `Browser` class (`.url()`, `.click_element()`,
  `.fill()`, built on `dom/page_snapshot.py`'s websocket client to a
  `--remote-debugging-port` Chrome this codebase no longer launches), and
  listed `"browser"` in `_get_all_tools()`'s name list — meaning it still
  showed up in the LLM's tool catalog every step, indistinguishable from a
  tool that actually works. This is the direct root cause of the failing
  live-run transcript from earlier this cycle: the model called
  `computer.browser.url()`, got a silent blank result (no debug-port
  Chrome was listening), and had no signal that the tool itself, not its
  own reasoning, was broken. Fixed by removing the import, the
  instantiation, and the catalog name — `computer/browser.py` and its CDP
  dependency chain are now unreachable from any live entry point, not just
  archived alongside code that already was.
- New standing rule for this class of bug, written into
  `DESIGN_PHILOSOPHY.md`: a submodule attached in `Computer.__init__` is a
  promise the tool catalog makes to the model. Retiring a perception or
  actuation path means removing its attachment the same day, not
  "archiving the code and leaving the wiring in place for now."

**A second, separate bug behind the same failing transcript, found by
reading `window.py` end to end rather than assuming the CDP fix alone
explained the symptom:**
- `Window.focus()` used `auto.WindowControl(searchDepth=1,
  Name=title_query).Exists(0)`, which does an **exact** match on the
  window's `Name` property — despite the method's own docstring always
  saying "containing." A real Chrome window title looks like `"Traffic
  Flow Bench Pipeline | Kaggle — Google Chrome"`, never exactly `"Chrome"`,
  so `focus("Chrome")` never matched, `Exists(0)` was always `False`, and
  every caller's "if Chrome is already open, focus it" branch was
  unreachable — `navigate()` fell straight through to "launch a new
  Chrome" on every single call, even with Chrome already open on the right
  page. Fixed by enumerating top-level windows (the same source
  `list_all()` already uses) and matching case-insensitively as a
  substring, matching what the docstring — and every caller — always
  assumed it did.

**Item 5 of this round's plan — harness-side perception-source routing,**
implemented in `CoreLoop._observe()` (`agent/core_loop.py`): the harness
now checks the active window's title each step and, when it looks like
Chrome, appends an explicit hint that the UIAutomation tree above won't
show page content and `browser_text`/`browser_list` is what's actually
needed — instead of leaving "which perception source applies right now"
as an inference the model has to get right under time and token pressure,
with two overlapping tools in its catalog and no signal for which one
fits the current window. Deliberately a hint, not an eager fetch: it costs
one string comparison per step, not a network round trip to the extension
bridge, consistent with the existing `AUTOBOT_VISION_MODE=auto` pattern of
only paying for a more expensive perception path once the cheaper one has
proven insufficient.

**Docs:** `DESIGN_PHILOSOPHY.md` and `AUTOBOT_MISSION.md` were both fully
rewritten — the previous versions described the CDP/Playwright-centric
architecture as current, which after this round made them actively
contradictory with the code rather than just outdated. `README.md`'s top
"4 core pillars" section (`perception/`/`actuation/`/`governance/`
packages that only exist under `_archive/`) was corrected for the same
reason. `USE_CASES.md` and `DEPLOYMENT.md` were reviewed and left as-is —
both already self-caveat as aspirational/not-yet-built in README, and
weren't judged actively misleading enough to justify the edit this pass.

**Known limitation of this round, worth stating plainly:** the physical
`_archive/` folder (and its many symlinked CDP-era files — `agent/loop.py`,
`agent/mission_agent.py`, `dom/extraction.py`, `dom/page_snapshot.py`,
`browser/launcher.py`, the whole `learning/` package, etc.) was **not**
deleted from disk. The session doing this round's edits had no shell
access to the machine running the live code — only the ability to
overwrite file contents, not delete files or symlinks. Severing the import
wiring (done) makes this code unreachable from any live entry point;
physically removing it from disk is still a manual step if the user wants
it gone rather than archived.

**Also not verified by a live run:** the `window.py` and `computer.py`
fixes above were written and syntax-checked but not exercised against a
real Windows/UIAutomation session — the environment making these edits
had no way to run Python with `uiautomation` installed. Same verification
gap the "Verification standard" section below already calls out generally;
flagging it here specifically because both fixes target the exact failure
this round's investigation started from, and "compiles" isn't the bar this
document uses for "done."

### Round 6 — Kaggle unattended autonomy, a second AI backend, and the multi-project orchestrator's read-only half

This round followed a real screen-sharing session on the user's own
machine (live computer-use screenshot + web research against Antigravity's
official docs), not just code reading — three assumptions from earlier
planning turned out to be wrong once actually looked at, and the design
below reflects the corrected picture, not the original guess:

- **The "Claude Code panel" in VS Code is Claude Code's own official VS
  Code extension**, not a separate integration surface Autobot needs to
  build UI automation for. Confirmed by looking at the actual running VS
  Code window. This matters because it means the existing headless-CLI
  bridge pattern (`claude -p ... --output-format json`, already built and
  tested) is the correct integration point — there is no second "the panel
  itself" thing to also automate.
- **Antigravity has a real, documented headless CLI (`agy`)**, not just a
  GUI. Confirmed against antigravity.google's own CLI docs, not assumed
  from the product's GUI-first marketing. `agy -p "prompt" --output-format
  json` mirrors Claude Code's own headless mode closely enough (JSON
  envelope, session/conversation resume, a permission-scoping flag) that
  the existing bridge pattern extends to it directly rather than needing a
  new integration shape invented from scratch.
- **The live computer-use tools are genuinely click/read-tier restricted
  for IDEs, terminals, and browsers on this account** (view + left-click
  only for IDE/terminal windows, view-only for browsers, with an explicit
  instruction not to work around this via AppleScript/System
  Events/shell). This isn't a bug to route around — it's the concrete,
  live confirmation of `DESIGN_PHILOSOPHY.md`'s standing "CLI/API before
  UI automation" principle: on this exact class of target (other AI
  coding tools), UI automation is not just more expensive than an API,
  it's actively *unavailable* at the tier needed to type into them. The
  orchestrator feature below was designed around that constraint from the
  start, not discovered to need a redesign after hitting it.

**Kaggle: unattended autonomy as its own axis, not a fourth approval mode
(`agent/approval.py`).** The user's explicit ask — keep working on Kaggle
notebooks while I'm away from the computer — is a different question from
"how much do I trust the agent's judgment" (`mode`: strict/balanced/
trusted), which already existed. Conflating them would have meant either
weakening `strict` mode's meaning for everyone, or leaving unattended runs
stuck re-prompting into a terminal nobody's watching. Added a second,
independent `unattended` flag (env var `AUTOBOT_UNATTENDED`, mirroring how
`mode` already reads `AUTOBOT_APPROVAL_MODE`): CAUTION/DANGER auto-proceed
while unattended regardless of `mode` — including `strict` — since a
paused prompt nobody can answer isn't more careful, it's just a stall.
IRREVERSIBLE (a real competition `submit()`, above all) is the one
exception the "Safety principle" section below already commits to keeping
non-bypassable: it still blocks even while unattended, but the block is
immediate — no `wait_for_approval` call, no timeout clock ticking on a
decision nobody's there to make — logged and desktop-notified for the user
to review when they're back, rather than silently either approving itself
or hanging. `AUTOBOT_AUTO_APPROVE=1` (pre-existing, deliberately blunter)
is untouched — a separate override, not folded into this. 13 new
behavioral tests (`tests/test_unattended_approval.py`) drive this through
the real `ApprovalGuard.gate()` and the real `_ActionStub` `CoreLoop` uses,
not a hand-rolled stand-in, so a mismatch between the two can't hide.

Also fixed in the same file this round: `kaggle_tool.py`'s `push_kernel()`
docstring claimed CAUTION tier; the actual risk-classifier pattern
(`_SAFE_COMPUTER_CALL_RE`) has always made it SAFE, matching the user's
explicit "notebook iteration should have zero friction, submission always
stops for a human" line. The code was right; the comment was stale.
Corrected so it stops being a trap for the next person reading it.

**Antigravity: second headless-CLI backend, same bridge shape as Claude
Code.** `autobot/integrations/antigravity_bridge.py` (subprocess wrapper
around `agy -p`, no `shell=True`, same `{"ok","data","error"}` return
contract as `claude_code_bridge.py`) plus `autobot/computer/
antigravity_tool.py` (the thin `computer.antigravity.run()` wrapper for
the LLM-facing `computer_call` path). Wired into the real tool catalog —
`computer/computer.py`'s `_get_all_tools()` name list, the single source
of truth both the LLM's catalog and `dispatch.py`'s `getattr` resolution
read from (the exact site of a real bug fixed in Round 4: a catalog name
that doesn't match a real attribute is silently unreachable, not a loud
error) — and confirmed not just by unit tests but by instantiating a real
`Computer()` and calling the real `dispatch.dispatch_computer_call()`
with an actual LLM-shaped call string end to end. `--dangerously-skip-
permissions` gets its own DANGER-tier pattern in `approval.py` (a
different flag name from Claude Code's `acceptEdits`/`bypassPermissions`,
so it needed its own regex rather than accidentally relying on the other
one matching by coincidence); the default (`skip_permissions=False`)
falls through to `computer_call`'s generic CAUTION default, matching how
Claude Code's own "plan" mode default is treated. 21 + 6 new tests
(`tests/test_antigravity_bridge.py`, `tests/test_antigravity_tool.py`)
plus 5 more for the new DANGER pattern (`tests/
test_approval_new_patterns.py::TestAntigravitySkipPermissionsIsDanger`).

**Multi-project orchestrator, built read-only-first on purpose.** The
user's ask — track ~7 simultaneous AI-assisted projects (Claude Code in
VS Code, two Antigravity projects, more), understand each one's intent,
check in on progress, and eventually prompt them on his behalf — splits
into two capabilities with very different risk profiles, and only the
lower-risk one shipped this round:

- `autobot/knowledge/project_registry.py` — one JSON file per tracked
  project (mirrors `skill_distiller.py`'s existing storage convention:
  same directory-under-`knowledge/`, dataclass with `to_dict()`/
  `from_dict()`, filesystem-safe slug), storing the project's working
  directory, which backend drives it, and — critically — the user's own
  stated intent kept **verbatim**, never summarized or rewritten by
  Autobot at registration time, plus a running log of later intent notes
  and check-in history. 23 tests, real `tmp_path` fixtures rather than a
  mocked filesystem, since this module mostly *is* a filesystem wrapper.
- `autobot/agent/orchestrator_checkin.py` — asks a tracked project's AI
  backend for a status update (what's done, what's in progress, what's
  blocked, what needs a decision), explicitly instructing it *not* to
  start new work in response, and records the summary plus the resumed
  session/conversation id back to the registry so the next check-in
  continues the same conversation instead of starting cold every time.
  Calls `claude_code_bridge.run_headless()` / `antigravity_bridge.
  run_headless()` directly rather than going through the `computer.
  claude_code`/`computer.antigravity` LLM-facing wrappers, because this is
  orchestration Python that needs the full bridge result (specifically the
  session id), not an LLM tool call that needs a collapsed string — using
  the layer this code actually belongs to, not routing around the tool
  layer as a shortcut. `check_in_on_all()` treats one project's failure
  (backend not installed, timed out) as independent of the others — the
  whole point of a batch check-in is a full picture even when one project
  is temporarily unreachable. 24 new tests (`tests/
  test_orchestrator_checkin.py`), covering successful check-ins, registry
  updates, session resumption, a project-not-found miss, an unrecognized
  stored backend (simulating a hand-edited or older-schema registry file)
  failing clearly instead of crashing three frames down, an exception
  mid-dispatch converting to an error result instead of propagating, and
  one failing project not stopping a batch check-in on the rest.

**Deliberately not built this round: sending an actual follow-up
instruction to another AI on the user's behalf** ("go implement X now"
while he's not watching). This is a materially bigger decision than
reading a status report — it can cause real file writes and real
token/compute spend in someone else's project, unattended. The Kaggle
unattended-autonomy question above got an explicit, considered answer from
the user (the `AskUserQuestion` exchange this round, before any code was
written); the "can Autobot prompt my other AI tools for me while I'm away"
question hasn't, so it stays out of `orchestrator_checkin.py` — whose own
module docstring records this reasoning at length — until that same kind
of explicit decision exists for it. This is the same category of judgment
call the "Safety principle" section below already asks future work to
make deliberately rather than by default; recorded here so it's a decision
revisited on purpose, not skipped past because the read-only half shipped
and looked done.

**Verification for this round specifically:** every new module above has
real behavioral tests (97 new tests this round: 13 unattended-approval +
23 antigravity-bridge + 6 antigravity-tool + 5 antigravity-DANGER-pattern
+ 26 project-registry + 24 orchestrator-checkin — run against the actual
code, not mocks of the modules under test, per the "Verification
standard" below; 5 of those 97 came from the adversarial pass below, which
found real bugs the original tests didn't). The Antigravity catalog wiring specifically
was also confirmed by direct instantiation and a real dispatcher call, the
same standard Round 5's CDP-removal fix was held to, precisely because
this round started from an explicit user complaint that earlier work
(the Chrome extension drag-fix) had shipped "carelessly done" despite
multiple passes — syntax-checking and unit-testing alone were judged not
sufficient evidence of "actually wired" this time.

**Not yet built, tracked here so it isn't quietly dropped:** the Kaggle
side's actual end-user workflow beyond the approval-policy groundwork —
this round made unattended Kaggle iteration *possible* by removing the
policy obstacle, but didn't add new Kaggle-specific automation on top of
the existing `kaggle_tool.py`/`demo_kaggle_run.py`. Also not built: any
UI for the user to register/browse/forget tracked projects (the registry
above only has a Python API so far — a CLI or dashboard surface for it is
the natural next step once the read-only check-in loop has real usage to
learn from).

**Adversarial review pass (separate from the "write it, test it" work
above — a second, skeptical reader instructed specifically not to trust
that the author's own tests were sufficient) found two real bugs, both
fixed, both with regression tests, bringing this round's test count to
164:**
- **`project_registry.py`'s `_safe_name()` let two different project
  names collide onto the same storage file.** Two names differing only in
  separator characters (`"Foo Bar"` vs `"Foo-Bar"`) slugged to the
  identical filename, so registering the second silently overwrote the
  first — including carrying over a session id from the wrong backend
  (a Claude Code session id ending up passed to `agy --conversation`) and
  making the first project vanish from `list_all()` entirely. Fixed by
  appending a short hash of the case-folded name to the slug, so distinct
  names can never collide while re-registering the same name (any casing)
  still resolves to the same project, preserving the existing
  keep-the-session-on-re-register behavior.
- **`antigravity_bridge.py` misreported a missing/renamed project
  directory as "agy CLI not found on PATH."** `subprocess.run(cwd=...)`
  raises the *same* `FileNotFoundError` the code already caught for "the
  `agy` binary itself vanished from PATH" — so a project whose folder got
  moved or deleted since registration (a real scenario for the
  orchestrator, which passes `cwd=project.working_dir` straight from the
  registry) got told to go reinstall a CLI that was never the problem,
  masking the actual fix (correct the registry's stored path). Fixed with
  an explicit directory check before ever reaching `subprocess.run`, with
  its own clear error message.

Also flagged, deliberately not fixed as part of this pass: `project_registry.py`'s
`_save()` has no file locking (bare `write_text()`, no temp-file+rename) —
consistent with `skill_distiller.py`'s existing convention this module
was built to mirror, and not triggered by today's only caller
(`check_in_on_all()`'s genuinely sequential for-loop), but worth a real
decision before this registry gets a concurrent caller. And
`claude_code_bridge.py` has the identical cwd-misdiagnosis bug just fixed
in `antigravity_bridge.py` — out of scope for this round's diff, but the
two files are meant to be kept in sync line-by-line, so it should be
ported over in a future pass.

### Round 7 — async job watchdog, Kaggle SDK compat hardening, and real-competition-derived lessons

Unlike every prior round, this one had a genuine empirical corpus behind
it before any code was written: Dalton ran three real competitions
through Antigravity against his actual Kaggle account specifically to
find out where an LLM-driven agent breaks against real infrastructure —
`biohub_cell_tracking` (a strict Code Competition, 3D U-Net cell
tracking), `ieee_traffic_flow` (a physics-constrained state-reconstruction
benchmark), and `s6e9_ev_prediction` (a synthetic tabular Playground
competition, pushed to Top 5% / Rank #131 of 2,683). Each has a
`THINKING_AND_DECISIONS.md` in `competitions/` with the real experiment
log, and the cross-competition synthesis is
`AUTOBOT_AUTONOMOUS_EXECUTION_CHALLENGES.md` in the project root. These
aren't hypothetical — the leaderboard CSVs in `competitions/*/leaderboard/`
are real downloads with real competitor names and real timestamps, the
`submission.csv` files are real (7.8MB+), and the submission IDs and
kernel slugs in the THINKING_AND_DECISIONS.md files are bound to
Dalton's real Kaggle username. This round's job was translating that
corpus into actual code changes, not just reading it.

**The central finding, and the one this round is mostly about:** at
23:34 during the S6E9 run, a TabPFN kernel was pushed, a status check
returned RUNNING, and the agent told the user it was running and moved on
to the next task. 19 seconds later the kernel crashed
(`TabPFNLicenseError`). Nothing caught it — the agent doesn't run as a
daemon; once its turn ends it's fully dormant until the user speaks again
— and the failure sat undetected until the user manually asked "check if
it's really still running." The user's own root-cause diagnosis (recorded
in `s6e9_ev_prediction/THINKING_AND_DECISIONS.md` section 2.C) is exactly
right and is worth stating plainly: an LLM agent is a turn-based reactive
process (input → reason → tool calls → message → **halt**), not a
continuous loop, Kaggle doesn't push webhooks on failure, and roughly 80%
of cloud job failures happen in the first 60 seconds — precisely the
window an agent is most likely to have already stopped watching. No
amount of "try to remember to check back" prompting fixes a structural
gap; it needed actual scaffolding.

**What got built, in `autobot/computer/kaggle_watchdog.py` (new module,
pure state + one bounded blocking check, no thread/clock of its own):**
- `KaggleJobLedger` — a JSON-backed, on-disk record of every kernel job
  Autobot has dispatched (status, dispatch time, liveness-verified flag,
  error log, capped history). The point of putting this on disk rather
  than in conversation memory or process memory is exactly to survive the
  thing that broke in the TabPFN incident: a dead conversation, or a
  process that already exited.
- `verify_liveness()` — the "60-Second Liveness Verification Rule" from
  the lessons doc, as code: checks kernel status at t+30s and t+60s
  (configurable) before ever reporting a push as successfully launched.
  Wired into `kaggle_tool.py`'s `push_kernel()` as the default behavior
  (`verify_liveness=True`) — it now blocks for up to 60s and returns
  either "Liveness verified..." or "LIVENESS CHECK FAILED...", so an agent
  calling `push_kernel` literally cannot repeat the TabPFN mistake: the
  return value already contains the checked-at-t+60s status, not just the
  push acknowledgment. Skipped automatically when kernel-metadata.json has
  no `id` field to track (keeps the original, pre-Round-7 tests passing
  unchanged) or when explicitly disabled for a fire-and-forget dispatch.
- `poll_pending()` — a cheap, non-blocking sweep across every job the
  ledger knows about, for something *else* to call on its own schedule.
  That something else is `autobot/web/app.py`'s new
  `_kaggle_watchdog_loop()` background asyncio task, started from
  `lifespan()` and polling every 30s (`AUTOBOT_KAGGLE_WATCHDOG_INTERVAL`)
  — the FastAPI server is the one part of Autobot that's actually
  long-lived (a CLI `autobot "<task>"` run is one process that exits when
  the task finishes; it structurally cannot be the thing watching a
  multi-minute job). Status changes get pushed through the server's
  existing `_log()`/`_broadcast()` mechanism, so they show up in the
  dashboard's live log for free. A new `GET /api/kaggle/jobs` endpoint
  dumps the ledger on demand. For when the server isn't running at all, a
  new `autobot --jobs` CLI command does the same poll-and-print
  synchronously, once, and exits.

This is a deliberately modest version of the "decoupled supervisor
daemon" the lessons doc calls for — not a separate always-running OS
service, just the persistent process Autobot already has (the dashboard
server) doing something useful with its lifetime. A true standalone
daemon (survives even when the dashboard isn't running) is a bigger,
separate piece of scope and is listed below as explicitly deferred, not
silently dropped.

**Kaggle SDK drift, found and fixed (the proximate bug that started this
round, before the architecture work above):** the installed `kaggle==2.2.4`
(kagglesdk-backed) package had drifted from what `kaggle_tool.py` was
written against, confirmed live against Dalton's real, authenticated
account:
- `list_competitions()` — `competitions_list()` now returns an
  `ApiListCompetitionsResponse` wrapper, not a plain list. Fixed with a
  defensive `getattr(response, "competitions", response)` that works
  against both the old and new shapes. Also newly documented: `ref` is now
  a full URL, not a bare slug.
- `get_leaderboard()` — its old method, `competition_view_leaderboard()`,
  no longer exists on the installed `KaggleApi` at all (`AttributeError`).
  Rather than wire in Python's fuzzy suggestion
  (`competition_leaderboard_cli`) without verifying its return shape, this
  was rewritten to shell out to `kaggle competitions leaderboard
  download` — which is *independently proven*, not guessed: the real CSVs
  in `competitions/*/leaderboard/` on Dalton's machine are that exact
  command's output, with the exact header this method now parses
  (`Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,
  TeamMemberUserNames`). Reclassified SAFE-tier (read-only).
- **New:** `submit_code_competition(competition, kernel, version,
  file_path, message)` — strict Code Competitions (like Biohub Cell
  Tracking) reject a bare CSV upload via `submit()` outright (`400:
  Submission not allowed: This competition only accepts Submissions from
  Notebooks`) and require the kernel-version-bound form. The exact CLI
  command this shells out to
  (`kaggle competitions submit -c ... -k ... -v ... -f ... -m ...`) is
  the one proven working in
  `competitions/biohub_cell_tracking/THINKING_AND_DECISIONS.md` section
  4.4, against Dalton's real account. IRREVERSIBLE-tier, same as
  `submit()` — added to `approval.py`'s pattern list, not just
  `kaggle_tool.py`.
- **New:** `list_top_kernels(competition)` — the "Phase 0 SOTA Discovery"
  step from the lessons doc (`kaggle kernels list --competition ...
  --sort-by scoreDescending`, read-only, SAFE-tier). The IEEE Traffic Flow
  post-mortem in the lessons doc is the concrete argument for this
  existing at all: an agent built a baseline from a stale Sep-8 community
  notebook, unaware the organizers had published an official reference
  repo on Sep-10 that had already fixed the exact bug capping its score.
  `system_prompt.md` now tells the agent to call this before writing a
  baseline from scratch for a new competition, not after the first
  disappointing submission.
- The Python-API-based methods (`pull_kernel`/`push_kernel`/
  `kernel_status`/`kernel_output`/`list_competitions`/`download_data`/
  `submit`) were left on the Python API — no evidence any of them are
  broken, and the project's now-twice-confirmed lesson is to fix what's
  actually shown broken, not preemptively rewrite everything shelling out
  "just in case."

**Other real lessons captured but deliberately NOT turned into a hard
runtime gate this round** (recorded here so they're a decision made on
purpose, not lost): the lessons doc's "Mandatory Phase 0 SOTA Gate"
proposes a hard `MissingSOTAGateError` that blocks training code from
running at all until `list_top_kernels`-equivalent discovery has
happened. This round added the tool and the system-prompt instruction to
use it, but not the hard enforcement — Autobot has no reliable way from
inside `kaggle_tool.py` to know "has the agent actually looked at the
results" versus "did it just call the method and ignore the output," so a
hard gate here would either be trivially satisfiable (call it, discard
it) or need a much larger piece of state-tracking than this round's scope.
Worth revisiting if the softer version (prompt instruction) proves
insufficient in practice.

Similarly not built: the "OpenClaw dormant specialist lanes" multi-
subagent architecture the lessons doc sketches (isolated per-competition
conversation threads, a compute-resource semaphore for GPU/CPU slot
allocation across concurrent Kaggle kernels, event-driven wake instead of
polling). The watchdog above solves the specific, demonstrated failure
(silent job death going undetected); the full multi-agent orchestration
picture is a materially bigger architectural undertaking that deserves
its own round with its own explicit scoping conversation, not a rider on
this one.

**Verification for this round:** 33 new/changed tests (12 kaggle_watchdog
ledger/liveness/poll tests, 15 kaggle_tool tests covering the SDK fixes
and the two new CLI-backed methods, 4 approval-pattern tests for the new
SAFE/IRREVERSIBLE classifications, 2 CLI `--jobs` tests) — full suite run
or a rewritten regex might slip. `autobot/web/app.py` was actually
imported (not just syntax-checked) to confirm the new background task and
`/api/kaggle/jobs` route wire up without error, matching the verification
standard below. The two real SDK-compat bugs (list_competitions'
wrapper-object return, get_leaderboard's removed method) were confirmed
against Dalton's real, live, authenticated Kaggle account before being
fixed — not guessed at from documentation.

## Verification standard

Everything above marked "done and tested" has behavioral tests that
actually execute the code, not just a successful `compileall`. That
distinction matters here specifically: every major bug found in this
project so far — the dead DOM import, the missing `computer_call` field,
the uncalled `save_skill`, the hardcoded `/api/human_input` — was in code
that compiled perfectly and had simply never been run.

**Still unverified by a live run:** anything requiring a real browser, a
real LLM key, or a real native app. Those need a machine with the deps
installed and Chrome open — see the setup instructions in README.md.

## Self-correction (a real gap worth naming, not just "reasoning better")

A useful, concrete pattern for the loop: when a click doesn't produce the
expected state change, the next attempt shouldn't just retry the same
click harder — it should try a *different interaction category* (e.g.
right-click instead of left-click, scroll first, or check whether a modal
intercepted the click). This is worth encoding as an explicit fallback
ladder in the loop rather than leaving it to the LLM to reinvent each time
it happens, since that reinvention is exactly the kind of per-step
reasoning the skill-replay system (see Sequencing #1) is supposed to make
unnecessary on the second occurrence.

## Safety principle (do not relax this without a real conversation about it)

`trusted` mode means "don't ask me about clicks and shell commands." It
does **not** mean "do whatever, including things I can't undo." The
IRREVERSIBLE tier — deletion, money, credentials, sending/publishing
under the user's identity — stays hard-gated in every mode. This matters
more, not less, as more capability gets added (native app control, full
computer access): a wrong action across a bigger surface is a bigger
mistake, not a smaller one.

## Finding (2026-09-24): `verify_liveness`'s t+30s check has a false-positive-error mode under real multi-agent contention, and it silently corrupts capacity accounting afterward

Discovered live, not hypothesized, while running Gemma 4 Developer Agent
and IEEE AI Emulation competition work concurrently with another agent
already using this account (`autobot-gemma4-exp1-baseline-zeroshot` and
`autobot-emulation-exp3-dual-family-ensemble`, both pushed today) — this
is exactly the "several Kaggle-competition projects at once" scenario
`push_kernel`'s capacity enforcement was built for, so it's a realistic
environment to have caught this in, not a contrived edge case.

**What happened, twice, independently:** `push_kernel()` reported
`LIVENESS CHECK FAILED at t+[30]s: status=error` immediately after a
successful push. In both cases, a manual `kaggle kernels status <ref>`
moments later showed `RUNNING` — the kernel was actually fine. Kaggle's
own status API appears to report a transient `ERROR`-like state during
the first ~30s worker-assignment window (plausibly more likely to surface
under real scheduling contention, i.e. exactly when several kernels are
being dispatched around the same time across agents/projects), and
`verify_liveness` (`autobot/computer/kaggle_watchdog.py`) treats the
*first* `error` reading as final truth — it returns immediately
(`survived_init: False`) rather than using its second scheduled check
(`grace_checks=(30, 60)` — the 60s check never runs once 30s reads
`error`) to confirm.

**Why this is worse than a wrong log line:** `_TERMINAL_STATUSES` includes
`"error"`, and `KaggleJobLedger.pending()` (which
`active_hardware_count()` — the actual GPU/CPU capacity check
`push_kernel` gates on — is built from) excludes anything terminal. So a
false "error" doesn't just mislead a human reading the message once; it
permanently drops that job from `poll_pending()`'s future re-checks *and*
from capacity accounting, even though the job is really still occupying a
real Kaggle GPU/CPU slot. Two consecutive false positives during this
session briefly made the ledger think 0-1 GPU slots were in use when the
true number was 2/2 — the exact oversubscription scenario capacity
enforcement exists to prevent, self-inflicted by the liveness check
meant to protect it. (Worked around by hand this session via
`ledger.update(kernel, "running")` after manually confirming real status
— not a fix, just an unblock.)

**Suggested fix, not yet implemented:** don't let a single `error` reading
short-circuit `verify_liveness` before all `grace_checks` are exhausted —
either require two consecutive `error` reads (one full grace-check
interval apart) before treating it as real, or explicitly re-check once
more immediately before returning `survived_init: False`. Separately,
`_TERMINAL_STATUSES`/`pending()` conflating "confirmed dead" with "last
reading happened to be error" is the deeper issue — capacity accounting
arguably should trust a `kernels_status()` poll taken *at capacity-check
time*, not a cached liveness verdict from whenever the job was first
dispatched, for exactly this reason.

**Independent corroboration, same session (a parallel agent working
`umud_muscle_architecture`):** hit the identical t+30s false-error flicker
on both of its own pushes, and separately caught the ledger stale in the
*other* direction too — a job recorded `running`/`gpu` in the ledger that
was actually `COMPLETE` live. That confirms the deeper point above
concretely: `check_capacity()` (`kaggle_watchdog.py`) never makes a live
API call, it only ever trusts whatever `poll_pending()` last wrote — and
nothing in the current `push_kernel` path calls `poll_pending()` before
checking capacity. Two agents pushing concurrently this session happened
to have their staleness errors cancel out (real count matched what a
stale ledger implied, by luck), which is exactly the "worked by
coincidence" failure mode worth fixing before relying on this for real
multi-agent concurrency: `check_capacity()` should probably force a fresh
`kernels_status()` on every non-terminal job it's about to count, not
trust whatever the ledger says was true whenever it was last written.
