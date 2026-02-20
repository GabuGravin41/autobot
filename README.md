# Autobot

Autobot is a local desktop automation controller designed to execute your repetitive workflows through your existing laptop setup.

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

## Install

```bash
pip install -r requirements.txt
playwright install chrome
```

## Run

```bash
python -m autobot.main
```

## Environment variables

- `AUTOBOT_CHROME_USER_DATA_DIR` (optional)
- `AUTOBOT_CHROME_PROFILE_DIR` (optional, default: `Default`)
- `AUTOBOT_CHROME_EXECUTABLE` (optional)
- `AUTOBOT_CHROME_SOURCE_USER_DATA_DIR` (optional, default: local Chrome user-data root for bootstrap)
- `AUTOBOT_CHROME_SOURCE_PROFILE_DIR` (optional, default: same as `AUTOBOT_CHROME_PROFILE_DIR`)
- `AUTOBOT_CHROME_LAUNCH_TIMEOUT_MS` (optional, default: `15000`)
- `AUTOBOT_BROWSER_MODE` (optional: `auto`, `human_profile`, `devtools`; default: `auto`)
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` (optional, enables LLM planner in autonomous mode)
- `AUTOBOT_LLM_PROVIDER` (optional: `gemini` or `openai_compat`)
- `OPENAI_API_KEY` or `XAI_API_KEY` (for `openai_compat`, including Grok-compatible endpoints)
- `AUTOBOT_OPENAI_BASE_URL` (optional, default: `https://api.x.ai/v1`)
- `AUTOBOT_LLM_MODEL` (optional, default: `gemini-1.5-flash`)

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

## Notes

- If launch fails on default Chrome profile restrictions, let Autobot use a dedicated automation profile directory.
- First launch may require one-time sign-in in the automation profile if cookies cannot be copied.
- In `human_profile` mode, URL/search and keyboard flows use your real Chrome session; some DOM-selector actions are limited.
- Human-mode safety guard blocks typing if Autobot/Cursor window appears focused, to prevent runaway self-trigger loops.
- AI Planner Chat panel allows prompt -> plan preview -> execute workflow (uses configured provider or safe fallback).
- End-to-end tool-calling stress workflow is available as preset `tool_call_stress`.
- Desktop actions are intentionally explicit and coordinate-based to keep behavior predictable.
- This is a foundation for larger autonomous loops; extend workflows in `autobot/workflows.py`.
- Autonomous mode blocks obvious bulk-messaging intents by default. Use consent-based, explicit tasks only.
- Login behavior:
  - Adapters include a `attempt_google_continue_login` action to click "Continue with Google" when present.
  - If not present, rely on profile-saved password autofill or ask user to intervene.
- WhatsApp Web (human mode): The adapter opens WhatsApp home first, waits for load, then opens the chat by phone. Phone numbers are normalized to digits only. If opening by direct link is unreliable, use search instead by passing `use_search: true` in the adapter params for `open_chat` (e.g. in a workflow or UI adapter call). Keep the Chrome window focused (or click it) before running so the new tab is visible.
