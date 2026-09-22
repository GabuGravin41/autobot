# Autobot: Mission & Architecture

> *"The computer was designed as a bicycle for the human mind. Autobot turns it into a co-pilot — co-owned by human and AI, operating on equal terms across the desktop."*

This document describes the architecture Autobot actually runs (Sep 2026).
An earlier version of this document described a CDP/Playwright-centric
design that has since been retired — see `DESIGN_PHILOSOPHY.md` for why,
and `ROADMAP.md` for the ground-truth log of what's wired up versus
aspirational. That older version is kept under `_archive/` for history.

---

## 1. The Mission

AI assistants are traditionally trapped inside chat windows, unable to
operate a computer the way a human does. Autobot is built to close that
gap — not by being the most capable possible agent on the most expensive
possible model, but by making a **cheap, fast model reliable enough at
computer control that it doesn't need to be an expensive one.** That's the
actual target: the same shift Cursor made for code editing — not "a model
smart enough to do anything," but a harness disciplined enough that a
modest model, given clean state and an unambiguous action set, performs
like a much larger one would have to without it.

Two commitments follow from that:
- **Equal capability, not equal cost.** Anything a human can see, click,
  type, or run on their machine, Autobot can perceive and act on — through
  the cheapest reliable mechanism for that specific surface, not
  uniformly through the most general (and most expensive/fragile) one.
- **Human-controlled autonomy.** A permission model the user sets — from
  "ask me about everything" to "full autonomy" — with a hard floor
  (deletion, money, credentials, publishing under the user's identity)
  that no mode, including full autonomy, is allowed to skip.

---

## 2. How It Actually Perceives and Acts

Not one universal mechanism — the cheapest reliable one per surface:

- **Native desktop apps** (Notepad, Excel, DICOM viewers, Artemis, VESTA,
  DAWs) — **UIAutomation** (`computer/window.py`). An indexed element tree,
  no vision model needed for the common case.
- **Web page content** — the already-installed **Chrome extension's DOM
  bridge** (`browser/extension_bridge.py`), acting on the user's real,
  already-logged-in Chrome. Not CDP: no debug-port launch, no isolated
  profile, no fighting Windows' SingletonLock over a Chrome the user
  already had open. `browser_text` / `browser_list` / `browser_click` /
  `browser_type` / `browser_paste` are the four actions this gives the
  agent loop.
- **Other AI tools with a real API or CLI** (Claude Code, and anything
  like it) — the actual API/CLI, never screenshots of a chat window. An
  order of magnitude cheaper in tokens and it doesn't depend on a UI that
  can redesign itself under you. `computer.claude_code.run(...)`,
  `computer.kaggle.*` are the working examples of this today.
- **Genuinely vision-only surfaces** (canvas, WebGL, QR codes) —
  screenshot, as the deliberate last resort, not the default: a screenshot
  costs roughly 1-2k tokens against a few hundred for an indexed read, and
  a cheap model gets less reliable at translating pixels into precise
  coordinates the more that mode is leaned on. `AUTOBOT_VISION_MODE=auto`
  spends it only where the cheaper source has already proven insufficient.
- **The rest of the machine** (shell, files, clipboard) — direct OS calls
  through `computer/*`, no indirection needed.

One rule threads all of this: **the harness decides which perception
source applies whenever that's mechanically determinable, and tells the
model plainly, instead of leaving two overlapping options for the model to
guess between under time and token pressure.** `CoreLoop._observe()`
checking the active window's title and appending a perception hint when
it's Chrome is the current concrete instance of this rule — expect more
of these as they're found, not fewer.

---

## 3. The Governance Model: The Permission Dial

* 🛡️ **Strict** — Nothing risky runs without a live "Allow" click.
* ⚡ **Balanced (default)** — Safe reads/navigation auto-proceed; anything
  risky pauses for approval.
* 🚀 **Trusted** — Clicks and shell commands don't stop you. This does
  **not** extend to the floor below.

**The floor, in every mode without exception:** deletion, financial
transactions, credential entry, sending or publishing under the user's
identity. "Trusted" means "don't ask me about clicks" — it never means
"do whatever, including things I can't undo." This matters more, not
less, as capability grows: a wrong action across a bigger surface is a
bigger mistake, not a smaller one.

---

## 4. What Makes This Get Cheaper Over Time, Not Just Wider

Two mechanisms exist specifically so repeated work doesn't re-pay full
reasoning cost every run:
- **Skill distillation** (`autobot/knowledge/skill_distiller.py`) — a
  successful run's proven action path gets saved and re-injected as
  context the next time a similar goal comes in, so the second run of a
  repeated task is a shorter path, not a fresh derivation.
- **API/CLI-first tool integration** (Kaggle, Claude Code) — reaching a
  tool through its real interface instead of automating its UI is not
  just more reliable, it's a standing cost reduction: no vision calls, no
  DOM traversal, no retry ladder for a page that redesigned itself.

The corollary, learned the expensive way: **capability that gets built and
never wired into a real, callable path doesn't just fail to help — it
becomes exactly the kind of clutter that makes the next round of work
slower and the model's own tool catalog less trustworthy.** See
`ROADMAP.md`'s verification standard: something only counts as "done" once
a real run, or a real automated test, exercises it — not when the file
compiles.

---

## 5. Strategic Vision

Autobot doesn't require a locked-down cloud container or a third-party
SaaS subscription — it runs against the computer the user already has,
under permissions the user already controls. The bet is that reliability
comes from disciplined scaffolding around a modest model, not from
routing every task to the largest one available.
