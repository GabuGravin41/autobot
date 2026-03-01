# Wait Times and Human-Speed Behavior

**Principle: the system must work. We do not optimize for speed.** Sites and LLMs need time to load and respond. The engine waits after adapter actions so pages and AI responses are ready before the next step.

## How it works

- After each **adapter_call** (e.g. `grok_web.open_home`, `grok_web.ask_latex_from_clipboard`), the engine looks up a **load wait** in seconds.
- Waits are defined in:
  1. **`autobot/load_waits.json`** (optional) — override per adapter and action.
  2. **`engine.DEFAULT_LOAD_WAITS`** — built-in defaults if the file is missing or doesn’t list that action.

## Suggested waits (seconds)

| Adapter / action | Suggested | Notes |
|------------------|-----------|--------|
| **grok_web** open_home | 8 | Page and session load |
| **grok_web** ask_latex_from_clipboard | 90–120 | Grok can take 1–2 min to answer |
| **grok_web** copy_visible_response | 5 | After response is visible |
| **chatgpt_web** open_home | 10 | ChatGPT front end load |
| **chatgpt_web** send_message | 90–120 | ChatGPT often takes 1–2 min to respond |
| **overleaf_web** open_dashboard / open_project | 6–8 | Project list and editor load |
| **overleaf_web** compile_project | 15+ | Compilation can be slow |
| **whatsapp_web** open_home | 20 | QR / session load |
| **leetcode_web** submit | 15 | Submission and result |
| **kaggle_web** run_notebook | 60–120 | Kernel run time |
| **kaggle_web** submit_to_competition | 10 | Submit click and confirmation |

## Editing waits

- **Without code changes:** edit `autobot/load_waits.json`. Use integer seconds per action. Restart the backend so it reloads the file.
- **With code changes:** adjust `DEFAULT_LOAD_WAITS` in `autobot/engine.py`.

If a flow fails because the next step runs too early (e.g. “element not found” or empty response), **increase** the wait for the previous action. Prefer being patient over being fast.
