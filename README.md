# Autobot

Autobot is a local desktop automation controller designed to execute your repetitive workflows through your existing laptop setup.

## Current capabilities

- Uses your real Chrome profile (cookies/session) via persistent Playwright context.
- Includes autonomous multi-loop mode:
  - diagnose -> plan -> execute -> retest
  - loop limits, per-loop step caps, and cancellation
  - conditional step execution, retries, continue-on-error semantics
- Runs one-shot commands:
  - `search <query>`
  - `open <url|target>`
  - `run <os command>`
  - `open path <local path>`
  - `switch window`
  - `type <text>`
  - `wait <seconds>`
  - `list adapters`
  - `adapter <name> <action> <json_params>`
- Runs preset workflows:
  - `website_builder`
  - `research_paper`
  - `console_fix_assist`
- Includes stateful app adapters:
  - `whatsapp_web`
  - `instagram_web`
  - `overleaf_web`
  - `vscode_desktop`
- Adapter actions are explicit and per-site, with confirmation gates for sensitive operations such as message send and PDF download.
- UI includes an adapter panel with action docs and a required checkbox for sensitive actions.
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
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` (optional, enables LLM planner in autonomous mode)
- `AUTOBOT_LLM_PROVIDER` (optional: `gemini` or `openai_compat`)
- `OPENAI_API_KEY` or `XAI_API_KEY` (for `openai_compat`, including Grok-compatible endpoints)
- `AUTOBOT_OPENAI_BASE_URL` (optional, default: `https://api.x.ai/v1`)
- `AUTOBOT_LLM_MODEL` (optional, default: `gemini-1.5-flash`)

## Notes

- Close existing Chrome windows before starting Autobot if you hit profile-lock errors.
- Desktop actions are intentionally explicit and coordinate-based to keep behavior predictable.
- This is a foundation for larger autonomous loops; extend workflows in `autobot/workflows.py`.
- Autonomous mode blocks obvious bulk-messaging intents by default. Use consent-based, explicit tasks only.
- Login behavior:
  - Adapters include a `attempt_google_continue_login` action to click "Continue with Google" when present.
  - If not present, rely on profile-saved password autofill or ask user to intervene.
