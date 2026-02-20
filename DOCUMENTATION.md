# Autobot — Project Documentation

Autobot is a **local desktop automation controller** that runs repetitive workflows using your existing Chrome profile and desktop. It can drive web apps (WhatsApp Web, Google Docs, Overleaf, Grok, etc.) and system actions (clipboard, hotkeys, VS Code) with optional AI planning.

---

## Table of contents

1. [Overview](#1-overview)
2. [Installation & run](#2-installation--run)
3. [Architecture](#3-architecture)
4. [Configuration](#4-configuration)
5. [Run folder layout](#5-run-folder-layout)
6. [Workflows & presets](#6-workflows--presets)
7. [Quick commands (planner)](#7-quick-commands-planner)
8. [Adapters reference](#8-adapters-reference)
9. [Tool-call stress test](#9-tool-call-stress-test)
10. [Autonomous mode & AI planner](#10-autonomous-mode--ai-planner)
11. [Notes & troubleshooting](#11-notes--troubleshooting)

---

## 1. Overview

- **Purpose:** Run multi-step workflows (open sites, type, send messages, copy/paste, compile, download) without writing one-off scripts.
- **Browser modes:**
  - **`human_profile`** — Uses your real Chrome (via `webbrowser` / Chrome executable). Typing and keys use `pyautogui`. Best when DevTools automation is blocked or you need real cookies/sessions.
  - **`devtools`** — Playwright-controlled Chrome with a dedicated profile. Full DOM/selector control.
  - **`auto`** — Tries devtools; falls back to human_profile if Chrome blocks automation (e.g. “non-default data directory”).
- **Execution:** Workflows are **plans** made of **steps**. Each step is an **action** (e.g. `open_url`, `adapter_call`, `clipboard_set`, `screenshot`) with optional condition, retries, and `continue_on_error`.
- **Runs:** Every run writes a **run folder** (human-readable name) with `history.json`, `artifacts.json`, `screenshots/`, `console.log`, and `about.txt`.

---

## 2. Installation & run

### Requirements

- Python 3.10+
- Chrome (for Playwright and/or human profile)

### Install

```bash
pip install -r requirements.txt
playwright install chrome
```

### Run the UI

```bash
python -m autobot
# or
python -m autobot.main
```

### Run the stress test from CLI

```bash
python -m autobot.run_stress "phone|docs_url|download_path|message"
# or set topic via env:
set AUTOBOT_STRESS_TOPIC=+1234|https://...|C:/path/to/pdf|Message
python -m autobot.run_stress
```

### Organize old run files into folders

```bash
python -m autobot.organize_runs       # dry run
python -m autobot.organize_runs --do  # migrate runs/*.json into named folders
```

---

## 3. Architecture

### High-level flow

```
UI / CLI / Autonomous loop
    → Planner (text → WorkflowPlan)
    → Engine.run_plan(plan)
        → For each step: Engine._execute_action(action, args)
            → BrowserController (goto, press, screenshot)
            → AdapterManager (adapter_call → BaseAdapter.execute)
            → Built-in actions (clipboard, desktop_*, wait, screenshot, etc.)
    → Run folder written (history, artifacts, screenshots, about.txt)
```

### Main modules

| Module | Role |
|--------|------|
| **`engine.py`** | `AutomationEngine`: runs `WorkflowPlan` steps, rate limiting, run directory creation, `_write_run_history` and artifacts. Defines all built-in actions (`open_url`, `adapter_call`, `screenshot`, `clipboard_set`, etc.). |
| **`browser_agent.py`** | `BrowserController`: starts Playwright (devtools) or uses system browser (human_profile). `goto`, `search`, `press`, `screenshot`, `fill`/`click` (devtools only). |
| **`focus_manager.py`** | `FocusManager`: brings target window to front by title keywords (e.g. “chrome”, “whatsapp”) using `pyautogui`; blocks typing when Autobot/Cursor is focused. |
| **`adapters/manager.py`** | `AdapterManager`: registry of adapters, policy (strict/balanced/trusted), sensitive-action prepare/confirm. |
| **`adapters/base.py`** | `BaseAdapter`: session health, selectors, human_nav, `run_human_nav()`, `_ensure_human_target_focus()`, `fill_any`/`click_any`. |
| **`workflows.py`** | Defines `WorkflowPlan`s: `tool_call_stress_workflow`, `website_builder_workflow`, `research_paper_workflow`, `console_fix_assist_workflow`, etc. |
| **`planner.py`** | `build_plan_from_text()`: maps natural-language-style commands to `WorkflowPlan` (e.g. “search …”, “run tool stress …”, “adapter …”). |
| **`ui.py`** | Tkinter UI: workflow presets, topic input, adapter panel, AI Planner Chat, browser mode dropdown, Run / Stop / Run Benchmarks. |
| **`autonomy.py`** | Autonomous loop: diagnose → plan (LLM or fallback) → execute → repeat; uses engine and optional LLM. |
| **`llm_brain.py`** | LLM integration for plan generation (Gemini / OpenAI-compat); produces `PlanDraft` with steps. |

### Key data types

- **`TaskStep`** — `action`, `args`, `save_as`, `description`, `condition`, `retries`, `continue_on_error`.
- **`WorkflowPlan`** — `name`, `description`, `steps: list[TaskStep]`.
- **`ExecutionResult`** — `success`, `completed_steps`, `total_steps`, `state` (includes `run_dir`, `last_run_history_path`, etc.).

---

## 4. Configuration

### Environment variables

| Variable | Purpose |
|----------|---------|
| `AUTOBOT_BROWSER_MODE` | `auto` \| `human_profile` \| `devtools`. Default: `auto`. |
| `AUTOBOT_OPEN_NEW_TAB` | `1` or `0`. Default: `1`. In human_profile, open each URL in a new tab (leave current tab open). |
| **Load waits** | In human_profile, seconds to wait after opening slow sites. Set to `0` to skip. `AUTOBOT_WHATSAPP_LOAD_WAIT` (default 8), `AUTOBOT_WHATSAPP_CHAT_LOAD_WAIT` (5), `AUTOBOT_OVERLEAF_LOAD_WAIT` (5), `AUTOBOT_GROK_LOAD_WAIT` (4), `AUTOBOT_GOOGLE_DOCS_LOAD_WAIT` (4). |
| `AUTOBOT_CHROME_USER_DATA_DIR` | Directory for Playwright Chrome profile (devtools). |
| `AUTOBOT_CHROME_PROFILE_DIR` | Profile name (default `Default`). |
| `AUTOBOT_CHROME_EXECUTABLE` | Path to Chrome binary (human_profile / fallback). |
| `AUTOBOT_CHROME_SOURCE_USER_DATA_DIR` | Source Chrome user data for bootstrap. |
| `AUTOBOT_CHROME_LAUNCH_TIMEOUT_MS` | Launch timeout (default 15000). |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Enables Gemini for AI planner. |
| `AUTOBOT_LLM_PROVIDER` | `gemini` \| `openai_compat`. |
| `OPENAI_API_KEY` or `XAI_API_KEY` | For `openai_compat` (e.g. Grok). |
| `AUTOBOT_OPENAI_BASE_URL` | API base URL (default `https://api.x.ai/v1`). |
| `AUTOBOT_LLM_MODEL` | Model name (default `gemini-1.5-flash`). |
| `AUTOBOT_STRESS_TOPIC` | Default topic for `run_stress` CLI: `phone\|docs_url\|download_path\|message`. |

Secrets (`.env`, `*.key`, `secrets.json`) are in `.gitignore`.

---

## 5. Run folder layout

Each run creates a **folder** under `runs/` with a **human-readable name**:  
`plan_YYYY-MM-DD_HH-MM-SS` (e.g. `tool_call_stress_2026-02-20_10-56-21`).

| Item | Description |
|------|-------------|
| **`about.txt`** | One-line summary: plan name, started/finished, success, steps. |
| **`history.json`** | Full step log, state snapshot, adapter telemetry. |
| **`artifacts.json`** | For human review: `whatsapp_message_sent`, `pdf_downloaded`, `doc_text_preview`, `latex_text_preview`, `run_dir`, `history_path`. |
| **`screenshots/`** | Step screenshots (e.g. `01_whatsapp_sent.png`, `02_google_doc_typed.png`, …). |
| **`console.log`** | Console output when run from CLI. |

Legacy single-file runs (`runs/*.json`) can be migrated into this layout with:

```bash
python -m autobot.organize_runs --do
```

---

## 6. Workflows & presets

Defined in **`workflows.py`** and exposed as UI presets:

| Preset | Description |
|--------|-------------|
| **tool_call_stress** | WhatsApp → Google Docs → Grok (LaTeX) → Overleaf (compile, download) → cleanup; optional phone/docs_url/download_path/message. |
| **website_builder** | Open VS Code, CASA AI, Grok, Google search for layout ideas. |
| **research_paper** | Open Grok, DeepSeek, Overleaf, Google search for references. |
| **console_fix_assist** | Open local URL, capture console errors, copy to clipboard. |

Workflow parameters (e.g. topic) are passed from the UI “Topic” field or from the planner text.

---

## 7. Quick commands (planner)

The planner turns natural-language-style **text** into a **WorkflowPlan**. Used by the UI and autonomous mode.

| Command | Effect |
|--------|--------|
| `search <query>` | Google search. |
| `open <url\|target>` | Open URL or named target (e.g. overleaf, grok, vscode). |
| `run benchmarks` | Run internal benchmark suite. |
| `run tool stress <phone>\|<docs_url>\|<path>\|<message>` | Build tool_call_stress workflow. |
| `run <command>` | Run OS command. |
| `browser mode` | Show browser mode status. |
| `list adapters` | List adapter action libraries. |
| `adapter telemetry` | Show adapter telemetry. |
| `adapter policy <strict\|balanced\|trusted>` | Set adapter policy. |
| `adapter prepare <name> <action> [json]` | Prepare sensitive action; returns token. |
| `adapter confirm <token>` | Execute prepared sensitive action. |
| `adapter <name> <action> [json]` | Call adapter action (with policy/confirm as needed). |
| Other | Passed to AI planner if configured, else error. |

---

## 8. Adapters reference

Adapters are **per-app** layers that expose actions (open, type, send, etc.) and use either **selectors** (devtools) or **human_nav** (keyboard/shortcuts) in human_profile mode.

| Adapter | Description |
|---------|-------------|
| **whatsapp_web** | Open home, open chat by phone/name, type message, send. Params: `phone`, `chat`, `text`; optional `use_search` for open by phone. |
| **instagram_web** | Open home, human_nav flows. |
| **overleaf_web** | Open dashboard, replace editor text, compile, download PDF. |
| **google_docs_web** | Open new doc, open by URL, type text, copy all. |
| **grok_web** | Open home, ask from clipboard, copy visible response. |
| **vscode_desktop** | Open VS Code. |

- **Selectors:** `autobot/adapters/selectors/<adapter>.json` (e.g. `message_input`, `login_google_button`).
- **Human nav:** `autobot/adapters/human_nav/<adapter>.json` (sequences of hotkey, type, press, sleep).
- **Sensitive actions** (e.g. send message, download PDF) may require **confirmation** in the UI or `confirmed: true` in params; in **strict** policy they require prepare → token → confirm.

---

## 9. Tool-call stress test

End-to-end chain to verify tool-calling and capture evidence (screenshots, artifacts).

1. Set browser mode to **human_profile** (UI: dropdown + Apply).
2. **UI:** Preset **tool_call_stress**, Topic: `phone|docs_url|download_path|message`.  
   **CLI:** `python -m autobot.run_stress "phone|docs_url|download_path|message"` or `AUTOBOT_STRESS_TOPIC=... python -m autobot.run_stress`.
3. Run. When finished, open the **Run folder** from the log; check `about.txt`, `screenshots/`, `artifacts.json`.

With **empty** phone or download path, WhatsApp and file-download steps are skipped; the rest of the chain still runs and screenshots are captured.

---

## 10. Autonomous mode & AI planner

- **Autonomous mode (UI):** Enter a goal; the system runs a loop: diagnose → plan → execute (with step limit and loop limit). Plans come from the **LLM** (if API keys set) or from the **planner** (quick commands).
- **AI Planner Chat:** Enter a prompt → generate plan (LLM) → preview steps → execute. Uses `llm_brain.generate_plan_draft()` and then engine.
- **LLM config:** Set `GOOGLE_API_KEY`/`GEMINI_API_KEY` for Gemini, or `OPENAI_API_KEY`/`XAI_API_KEY` and `AUTOBOT_LLM_PROVIDER=openai_compat` for Grok/OpenAI-compat.

---

## 11. Notes & troubleshooting

- **Chrome profile:** If Playwright fails with “non-default data directory” or profile locked, use a dedicated automation profile dir or close all Chrome windows. In many cases **human_profile** avoids this.
- **Human profile:** Keep the **Chrome window focused** (or click it) before running so new tabs and typing go to the right place. Typing is blocked when Autobot/Cursor is focused.
- **WhatsApp Web:** Phone is normalized to digits. If opening by link is unreliable, use **`use_search: true`** in `open_chat` params so the adapter opens chat via search (Ctrl+Alt+/ → type number → Enter).
- **Login:** Adapters can try “Continue with Google” when detected; otherwise rely on saved sessions or manual sign-in.
- **Runs:** All run output is under `runs/` (ignored by git). Use `about.txt` and folder names to find a specific run.

---

*This document reflects the Autobot codebase and README as of the last update.*
