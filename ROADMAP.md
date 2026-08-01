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
- **Domain helpers.** `agent/domain/bio_synthesizer.py`,
  `dicom_synthesizer.py`, `materials_synthesizer.py`, `overleaf_helper.py`
  — stubs for exactly the genomics/DICOM/materials workflows that
  motivated this project. Need to be checked against what they actually
  do today versus what they're named for.
- **`system_prompt_full.md`** (600+ lines, far richer than the active
  `system_prompt.md`) is not loaded by `SystemPromptBuilder`. Worth
  deciding deliberately: merge the good parts in, or delete it so it
  stops looking authoritative.

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

## Verification standard

Everything above marked "done and tested" has behavioral tests that
actually execute the code, not just a successful `compileall`. That
distinction matters here specifically: every major bug found in this
project so far — the dead DOM import, the missing `computer_call` field,
the uncalled `save_skill`, the hardcoded `/api/human_input` — was in code
that compiled perfectly and had simply never been run.

**Still unverified by a live run:** anything requiring a real browser, a
real LLM key, or a real native app. Those need a machine with the deps
installed and Chrome open — see the run instructions in DOCUMENTATION.md.

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
