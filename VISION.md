# Autobot Vision & Priorities

This document captures product direction: **functionality over speed**, a professional UI, and an AI that plans with tool calls and state feedback.

---

## 1. Functionality over speed

- **Priority:** Can it do the job? Speed and parallelism can be optimized later.
- The system must **control the computer like a human**: open sites, click buttons, type, copy/paste, navigate Overleaf/Kaggle/LeetCode/Artemis, and run full pipelines.
- If it can’t do exactly what the user wants in the way they want, it’s not useful. Reliability and correctness come first.

---

## 2. User interface (frontend)

- **Professional look:** The UI should feel polished and “more professional,” not just acceptable.
- **Sharp containers:** Prefer **sharp corners** (e.g. border-radius ~1–4px). Too much roundness is not desired; cards and modals should look crisp.
- **Color:** Avoid purely monochromatic themes. Support **double-ended / beam-style** accents: two bright, distinct colors (e.g. gradient or primary + secondary) so the interface feels alive and clear, not flat single-color.
- **Improvements:** Ongoing work on layout, hierarchy, and visual effects to make the interface more appealing and trustworthy.

---

## 3. AI & LLM routing

- **Default route:** **OpenRouter** with **DeepSeek V3.1 (free)** model: `deepseek/deepseek-chat-v3.1:free`.
- **Role of the AI:**
  - Understand **natural language** and complex, multi-step tasks.
  - Know what **can and can’t** be controlled (browser, desktop, files, etc.).
  - **Produce a plan that would actually work** using the available tools.
- **Tool calls:** Expose clear tools (open URL, click, type, navigate, etc.) so the AI doesn’t “burn tokens” guessing—it uses tool calls with the right parameters (e.g. URL, selector, text).
- **State feedback:** Pass back **state** to the AI: UI state, screen state, computer state (e.g. “page loaded,” “console errors,” “clipboard content,” “last run result”). The AI should use this feedback to decide next steps (RAG-like: retrieve state, then generate next action).
- **Iteration:** Support back-and-forth: run step → get feedback → adjust plan → run again (e.g. fix code from errors, retry until success).

---

## 4. Target use cases (what “works” should mean)

The system should eventually be able to automate or assist end-to-end in these domains:

- **Gene annotation pipeline:** Load FASTA files → open Artemis → read values → spreadsheets → BLAST → record results, handle error margins, save to text files.
- **Brain segmentation pipeline:** Automate the full pipeline appropriate to the user’s tools and data.
- **Machine learning / Kaggle:**
  - Go to Kaggle, run training, manage notebooks.
  - Join a competition: take problem statement → iterate with AI (e.g. Grok/ChatGPT/DeepSeek) → refine code → run → see performance/errors → change prompt or code → repeat. Not about “100 submissions in 10 minutes,” but about **correctness**: back-and-forth until the solution works.
- **Coding contests (e.g. IEEE Extreme):** Paste code to AI → get code back → test → paste errors/results back → repeat until code is correct.
- **LeetCode-style workflow:**
  - Go to LeetCode, pick N problems (e.g. first 20).
  - For each problem: send to AI → get code → submit → read performance (e.g. pass/fail, accuracy) → iterate with AI until a target (e.g. ≥85% acceptance) → move to next problem.
  - Goal: complete all N problems with at least the desired performance level (e.g. 85%+ on each).

These are success criteria for “the system works”: it can automate or heavily assist such pipelines, with the AI using tools and state feedback to iterate until the task is done.

---

## 5. Technical implications

- **Engine:** Prefer **reliability and correctness** over raw speed (waits, retries, state checks, human-paced pacing where appropriate).
- **Adapters & human_nav:** Robust navigation (e.g. Overleaf, Grok, Kaggle, LeetCode) so the system “knows how to navigate” and doesn’t get stuck on wrong pages or elements.
- **State and feedback:** Structured state (clipboard, run results, console errors, screenshots, last action outcome) passed into the planner so the AI can decide what to do next.
- **LLM config:** Default to OpenRouter + DeepSeek V3.1 (free); document how to switch provider/model for power users.
- **Wait times / human speed:** Sites and LLMs (Grok, ChatGPT) need time to load and respond (often 60–120s). Per-action waits: `autobot/load_waits.json` and `engine.DEFAULT_LOAD_WAITS`. See **WAIT_TIMES.md**. We do not optimize for speed.
- **WebSocket and logs:** Logs stream over WebSocket; frontend falls back to polling **GET /api/logs** if the connection fails. One bad client does not affect others.
- **Plan evolves with state:** When an action plan fails in autonomous mode, the brain is re-asked with failure context (`last_error`) and can suggest fallbacks (wait longer, different action, human help). The plan adapts instead of stopping.

---

*This vision prioritizes: **Does it work?** over **Is it fast?** and aligns UI, AI routing, and use cases toward a fully useful automation platform.*

---

## Deployment and packaging

The goal is a **frontend on the internet** (e.g. Vercel — no download, usable from phone or any browser) and a **downloadable agent** on the user’s computer that controls it. The two communicate so you can **see** (and eventually interact with) the screen from the web app. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the deployment vision, packaging options, and browser↔laptop connectivity.

---

## How to run it today

- **Vision doc:** You’re reading it. Root file: `VISION.md`.
- **Code iteration:** Use preset **code_iteration** (Topic = test command, e.g. `pytest`) or quick task **run code iteration** / **code iteration pytest**. Then use **Autonomous mode** with Goal “Make tests pass” and Diagnostics command `pytest` so the AI gets state and suggests next steps each loop.
- **State feedback:** The AI now receives richer state (last command output, saved keys like `latex_text`, `doc_text`, run dir, loop index) so it can decide what to do next and iterate until the goal is met.
