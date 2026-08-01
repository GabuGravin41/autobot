# 🛡️ Autobot Core Design Philosophy & Execution Guidelines

This document outlines the **foundational design philosophy, execution rules, and browser control guidelines** for Autobot. It serves as an authoritative guide for developers and AI agents working on this codebase to ensure we never fall back into brittle, over-engineered, or blind automation loops.

---

## 🚨 ABSOLUTE LAW: "NO BLIND ACTIONS" — The Prime Directive

> **The system is FORBIDDEN from making any blind decision. It MUST know where it is. Every action MUST be preceded by a grounded, evidence-based observation from at least one of: (a) a live screenshot, (b) DOM/accessibility tree inspection via CDP, or (c) a verified UI state read. Not once, not even a single click, shall be executed without knowing with certainty what element is being targeted.**

### What "blind" means in practice (FORBIDDEN behaviours):
- Hardcoding pixel coordinates `(x, y)` without first reading them from a screenshot or DOM query for the current page load.
- Assuming a dropdown is open without taking a screenshot to confirm it.
- Clicking a sequence of UI elements without verifying each intermediate state.
- Inferring page layout from a previous session — layouts can shift (banners, modals, scrolled state).
- Taking any OS-level input action (mouse click, key press, clipboard paste) on a window without first confirming foreground window identity.

### What grounded decisions look like (REQUIRED behaviours):
1. **Locate before clicking**: Take a screenshot (or query DOM). Read the element's bounding box / pixel position from that live observation. Only then click at the confirmed center.
2. **Verify after clicking**: Take another screenshot immediately after. Confirm the expected state change appeared (dropdown opened, modal appeared, page changed). If not, STOP and diagnose — do not proceed.
3. **CDP-first for DOM**: When Chrome is open, use Chrome DevTools Protocol (CDP) to query the DOM and get element bounding boxes, text content, and visibility. This is more accurate than pixel estimation.
4. **Screenshot-second for visuals**: For elements not in the accessibility tree (custom dropdowns, canvas UI, overlays), fall back to screenshot-based pixel analysis.
5. **Never chain steps blindly**: Each step is a separate verify → act → verify cycle. A failure at any step must be caught and handled before the next step begins.

---

## 🎯 1. Our Aim & Goal

Autobot is a **Sovereign Autonomous Digital Agent**. Its goal is to execute complex, multi-turn digital tasks (e.g., multi-turn academic research on Grok, LaTeX synthesis, Overleaf compilation, Kaggle automation) with **human-level precision and adaptability**.

### Core Pillars:
1. **Human-Parallelism**: Autobot acts on the computer precisely the way a human user does.
2. **Visual Truth**: Autobot sees what the human sees using multimodal visual screenshots on every single step.
3. **Pragmatic Synergy**: Autobot seamlessly combines DOM accessibility tools, visual perception, and OS desktop controls (mouse/keyboard).
4. **Local Sovereignty**: Operates using the user's authentic logged-in accounts and desktop environment without requiring raw session exfiltration or brittle database mirroring.

---

## ⚖️ 2. Core Design Philosophy

### Rule #1: "Look and Act Like a Human" (No Over-Engineered Hacks)
* **Never manipulate locked user databases**: Do NOT attempt to copy SQLite cookie databases (`Network/Cookies`) or leveldb files behind Windows DPAPI encryption while Chrome is active. This corrupts files and causes crashes.
* **Never guess mysterious profile names**: Do NOT guess directory names (`Profile 1` vs `Profile 2`). Look at the screen! If Chrome opens a profile picker ("Who's using Chrome?"), take a screenshot, visually locate the user's profile tile (e.g. Dalton's brown paper + pencil avatar with $e^{i\pi}+1=0$), scroll down if needed, and click it with the mouse.
* **Ride Along on Existing Sessions**: If Chrome is already open on the desktop with logged-in accounts (Grok, Overleaf), interact with that window directly rather than spawning duplicate browser instances.

### Rule #2: "Zero Blind Actions" (Focus & Verify First)
* **Verify Window Focus**: Before issuing any keyboard (`Ctrl+T`, `Ctrl+V`, `Enter`) or mouse action, Autobot MUST locate the exact window handle (`hwnd`), restore it (`SW_MAXIMIZE`), send an `Alt` keypress to unlock Windows foreground restrictions, and verify focus via window title.
* **CDP DOM Query First**: Before clicking any element in a web page, query the DOM via CDP (`document.querySelectorAll`, `getBoundingClientRect`) to get the exact bounding box of the target element. Use this to compute the precise click target. Never use hardcoded coordinates without DOM verification.
* **Screenshot Before + After Every Click**: Take a screenshot BEFORE every click to confirm starting state. Take a screenshot AFTER every click to confirm the expected state change. If the state did not change as expected, HALT and diagnose.
* **Dropdown Verification**: When a dropdown is expected to open, take a screenshot and confirm the dropdown is visible (look for its text items) before clicking any item within it.
* **Fail Fast & Report Anomaly**: If an unexpected dialog or error banner appears (e.g., "Restore pages?", cookie banners), pause immediately, report the visual state, handle it cleanly, screenshot to confirm it's gone, then proceed.

---

## 🛠️ 3. Tool Synergy Matrix

Autobot does not rely on a single input mechanism. It categorizes and combines three primary tool sets based on the situation:

| Tool Layer | Best Used For | Example Actions |
| :--- | :--- | :--- |
| **DOM & Accessibility Tree** | Structural text extraction, form field detection, fast page inspection | `browser_snapshot`, element query, text scraping |
| **Multimodal Vision** | Visual perception, identifying un-indexed UI popups, verifying page load states | `screenshot`, visual target localization |
| **OS Desktop Controls** | Interacting with native desktop Chrome UI, profile pickers, global shortcuts, clipboard paste | `pyautogui.click()`, `mouse.scroll()`, `pyperclip.copy()`, `win32gui.SetForegroundWindow()` |

---

## 📁 4. Chrome Profile Mapping Reference

For this machine, the Chrome profile mappings are:

| Profile Directory | Account Email | Visual Avatar / Identifier | Key Logged-In Services |
| :--- | :--- | :--- | :--- |
| **`Default`** (PRIMARY) | `daltonomondi588@gmail.com` | "Person 1" — Brown paper & pencil with $e^{i\pi}+1=0$ | Overleaf, Grok, Google AI Studio, OpenAI, Google Services |
| **`Profile 1`** | `daltonomondi04@gmail.com` | "Person 2" — Geometric polar bear | Grok, X.ai OAuth |
| **`Profile 2`** | `6153.2023@students.ku.ac.ke` | KU Crest / Logo | University Services |

*Default Command for Primary Account Launch*:
```cmd
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --profile-directory="Default" "https://grok.com"
```

---

## 🚀 5. Blueprint for Multi-Step Tasks (Grok + Overleaf Benchmark)

When executing complex benchmarks:
1. **Phase 1 (Clean Launch & Verification)**: Launch Chrome with `--profile-directory="Default"`, focus & maximize window, send `Escape` to dismiss any "Restore pages" popup, and capture screenshot to confirm logged-in state.
2. **Phase 2 (Structured Multi-Turn Research)**: Execute turn-by-turn prompts (survey $\rightarrow$ mathematical derivation $\rightarrow$ device specs $\rightarrow$ LaTeX synthesis). Capture screenshot after every turn.
3. **Phase 3 (Overleaf Project Creation & Compilation)**: Open Overleaf tab, create blank project, paste synthesized LaTeX via clipboard (`pyperclip` + `Ctrl+A` + `Ctrl+V`), trigger compilation (`Ctrl+Enter`), and visually verify PDF output.

---

*Autobot Architecture Team — Sovereignty through Visual & Pragmatic Automation.*
