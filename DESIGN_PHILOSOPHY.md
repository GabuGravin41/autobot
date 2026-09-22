# Autobot Design Philosophy & Execution Guidelines

This document describes the architecture Autobot **actually runs today**
(CoreLoop, UIAutomation-first, Chrome-extension DOM bridge — Sep 2026) and
the rules that keep it from drifting back toward brittle, blind, or
over-engineered automation. The previous version of this file described a
CDP-first, screenshot-before-and-after-every-click architecture that this
project has since moved off of; that version is preserved under
`_archive/` for history, not as a second source of truth. If code and this
document ever disagree, that's a bug in one of them — fix the one that's
wrong rather than reading around it.

---

## The Prime Directive: No Blind Actions

The system must know where it is and what it's looking at before it acts.
That principle hasn't changed — what changed is what "knowing" costs and
how it's obtained.

**Forbidden:**
- Clicking coordinates that weren't just read from a live index (DOM
  index, UIAutomation `[N]` index, or a screenshot taken this step).
- Assuming a menu, dropdown, or page finished loading without checking.
- Reusing an index or coordinate from a previous step — layouts shift.
- Acting on a window without confirming it's actually the one you think
  it is (read the title; don't assume focus survived the last action).

**Required:**
1. **Observe before deciding.** Every step starts with a fresh read of the
   active window's state — see "Two Perception Sources" below for which
   one actually shows what.
2. **Act through the index that was just read**, not a remembered one.
3. **Verify after acting.** CoreLoop already re-observes after every
   action and compares against the previous screen; three unchanged
   observations in a row surfaces a stuck warning to the model rather than
   letting it silently repeat a dead action forever (see
   `agent/core_loop.py`'s stuck-detection in `_step()`).
4. **Screenshot is the fallback, not the default.** A screenshot costs
   roughly 1-2k tokens versus a few hundred for an index-based read, and it
   still has to be interpreted rather than acted on directly. Reach for it
   when the indexed state is genuinely insufficient — a canvas app, a
   custom WebGL UI, a QR code — not as the first move on every step.

---

## Two Perception Sources, Not One — and Not the Model's Job to Guess

Autobot looks at the computer through two different mechanisms, because
one perception source genuinely cannot see everything:

- **UIAutomation** (`computer/window.py`) — the primary source for every
  window. Native desktop apps (Notepad, Excel, DICOM viewers, Artemis) are
  fully described this way: `computer.window.extract_ui()` returns an
  indexed element tree, and `computer.window.click(N)` /
  `computer.window.type(N, text)` act on those same indices.
- **The Chrome extension's DOM bridge** (`browser/extension_bridge.py`,
  the `browser_text` / `browser_list` / `browser_click` / `browser_type` /
  `browser_paste` actions) — the only source that reliably sees actual web
  page *content*. UIAutomation's view of Chrome is limited to the
  toolbar/tabs; it does not consistently expose the page DOM. The bridge
  works through the already-installed, already-logged-in Autobot Chrome
  extension polling for commands — no debug port, no isolated profile, no
  CDP.

CDP was tried as a unifying layer for this (Playwright's
`connect_over_cdp()` against a `--remote-debugging-port` Chrome) and
retired: it required launching Chrome in a way that fought Windows'
SingletonLock, needed its own profile disconnected from whatever Chrome
the user actually had open, and broke in exactly the ways Claude Code and
other tools that *don't* depend on CDP did not. `computer/browser.py`'s
CDP client is kept under `_archive/` for reference; nothing in the live
package imports it, and `computer/computer.py` deliberately does not
attach it as a submodule — an unreachable tool that still shows up in the
model's catalog is worse than no tool at all, because the model has no way
to tell it apart from one that works.

Given two perception sources, the harness — not the model — should decide
which one is in play whenever that's mechanically determinable.
`CoreLoop._observe()` checks the active window's title every step and, when
it's Chrome, appends an explicit hint that the UIAutomation tree above
won't show page content and `browser_text`/`browser_list` is what's
actually needed. This is a hint, not an eager fetch — it doesn't call the
bridge itself, so it costs one string comparison, not a network round trip,
on every step. The general rule: **whenever the harness can determine which
tool applies from state it already has, tell the model directly instead of
leaving it as an inference the model has to get right under time and token
pressure.** A cheaper model benefits from this more than an expensive one
does, and cheap-model-friendliness is the actual design target here, not
"technically possible with a strong enough model."

---

## Rule: A Tool in the Catalog Must Actually Work

`Computer.get_tool_catalog()` auto-generates the LLM-facing tool list by
introspecting every submodule attached in `Computer.__init__`. That's
powerful — a new tool needs no separate catalog-writing step — but it also
means an attached submodule is a *promise*: the model will use it, because
it looks exactly as real as everything else in the list. Whenever an
architecture changes, the tool catalog changes in the same commit, not
after. Concretely: retiring a perception or actuation path means removing
its submodule from `Computer.__init__`'s attachment (and
`_get_all_tools()`'s name list) that same day — not archiving the code and
leaving the wiring in place "for now." "For now" is how the CDP-era
`computer.browser.url()` sat in the catalog silently returning blank
results for weeks after the code that made it work stopped running.

---

## Rule: Other AI Tools Get Driven Through Their CLI/API, Never Through Screenshots of Their Chat Window

When Autobot needs to operate another AI coding tool — Claude Code,
Antigravity, whatever comes next — the integration point is that tool's
own headless/scripting mode, not UI automation of its chat panel. Both
existing integrations (`autobot/integrations/claude_code_bridge.py`,
`antigravity_bridge.py`) are subprocess wrappers around a documented CLI
(`claude -p ... --output-format json`, `agy -p ... --output-format
json`) that return structured JSON directly — no reading pixels out of a
chat window, no guessing whether a reply has finished streaming, no
`shell=True` anywhere in either file (every caller-supplied value goes
into an argv list, never through a shell). This isn't a preference for
elegance; it's an order of magnitude cheaper in tokens and doesn't break
every time a UI redesign moves a button.

This got a real, live confirmation in Sep 2026, not just a design
argument: this account's computer-use tools are click/read-tier
restricted for exactly the target class this rule is about — IDE and
terminal windows get view-and-left-click only, browsers get view-only,
with an explicit instruction not to route around that restriction via
AppleScript, System Events, or shell commands. On this exact target,
UI automation isn't just the more expensive option, it's structurally
unavailable at the tier a real integration would need. Confirm a target
has a real CLI/API before reaching for computer-use as a fallback — and
if it doesn't, that's a reason to look harder for one, not a reason to
start scripting clicks.

---

## Rule: Never Manipulate Locked Chrome State

Do not attempt to copy SQLite cookie databases (`Network/Cookies`) or
leveldb files behind Windows DPAPI encryption while Chrome is running —
this corrupts files and crashes the browser. Autobot rides the user's
*existing*, already-open, already-logged-in Chrome (via the extension) or
a UIAutomation-focused instance of it — never a duplicated or exfiltrated
session.

---

## Rule: Self-Correction Escalates, It Doesn't Repeat

When an action doesn't produce the expected effect, the next attempt
should try a *different interaction category*, not the identical action
harder. CoreLoop's stuck-detection (three unchanged observations in a row)
is the current backstop for this; the model is explicitly told to try a
fundamentally different approach, use `screenshot` for visual context, or
call `human_input` rather than repeat. A useful next step here (not yet
built) is encoding specific fallback ladders — e.g. a failed click retries
via scroll-into-view before trying `screenshot`-guided coordinates — as
explicit escalation rather than leaving the whole ladder to the model to
reinvent on every occurrence.

---

## Rule: Successful Runs Should Get Cheaper, Not Just Complete

A run that finishes doesn't just return a result — `SkillDistiller` records
the proven action sequence as a learned skill (`autobot/knowledge/skills/`)
and re-injects it as context the next time a similar goal comes in, so the
second run of a repeated task doesn't re-derive the whole plan from
scratch. This is the actual mechanism for making a cheap model perform
like an expensive one over time on the tasks a given user actually
repeats: not a bigger model, a shorter path.

---

*This document is a working guide, not a museum piece — when the
architecture changes again, this file changes with it, in the same
change.*
