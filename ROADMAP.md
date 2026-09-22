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
real behavioral tests (92 new tests this round: 13 unattended-approval +
21 antigravity-bridge + 6 antigravity-tool + 5 antigravity-DANGER-pattern
+ 23 project-registry + 24 orchestrator-checkin — run against the actual
code, not mocks of the modules under test, per the "Verification
standard" below). The Antigravity catalog wiring specifically
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
