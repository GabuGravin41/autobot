# AutoBot: Mission, Architecture & Autonomous Computing Manifesto

> *"The computer was designed as a bicycle for the human mind. AutoBot transforms the computer into an autonomous co-pilot—co-owned by human and AI, operating on equal terms across the entire digital desktop."*

---

## 1. Executive Summary & Mission

### The Mission
**AutoBot** is built to create a true **Autonomous OS Co-Pilot**. It is designed to solve a fundamental limitation of modern AI: AI assistants are traditionally trapped inside isolated web chatboxes or restricted sandbox environments, unable to operate your physical computer the way you do.

AutoBot establishes a **co-ownership paradigm** between human and AI:
- **Equal Capability**: Anything a human can see, click, type, execute, or manage on a laptop, AutoBot can perceive, decide, and actuate.
- **Human-Controlled Autonomy**: You set the permission dial (*Observer* → *Supervised* → *Full Autonomy*), retaining 100% authority over safety and irreversible actions.

---

## 2. What We Are Building: The Autonomous Computer Engine

Rather than building an isolated browser extension or a simple CLI wrapper, AutoBot is a unified **OS Co-Pilot Engine** built on a closed-loop architecture inspired by autonomous vehicles:

```
                  ┌──────────────────────────────────────────┐
                  │          Human Permission Dial           │
                  │   (Observer | Supervised | Full Auto)    │
                  └────────────────────┬─────────────────────┘
                                       │
┌─────────────────────────┐            ▼            ┌─────────────────────────┐
│    PERCEPTION PILLAR    ├────────────────────────►│    ACTUATION PILLAR     │
│                         │   PERCEIVE → PLAN → ACT │                         │
│ • Desktop Screen Vision │                         │ • Native OS Input       │
│ • Browser CDP DOM Tree  │    ┌──────────────┐     │ • Browser CDP Engine    │
│ • Active OS Windows     │    │  Agent Loop  │     │ • Window Focus Manager  │
│ • Terminal / CLI Output │    └──────────────┘     │ • Shell / CLI Engine    │
└─────────────────────────┘                         └─────────────────────────┘
```

---

## 3. Core Functionalities

1. **Multi-Modal Screen & Vision Perception**: Captures live desktop screenshots to navigate image-only UIs (Canvas, WebGL, QR codes, native desktop applications) where text-only DOM trees are blind.
2. **CDP & DOM Tree Extraction**: Directly queries Chrome DevTools Protocol (CDP) for structured ARIA accessibility trees and element indices.
3. **Native Desktop Input Control**: Drives mouse cursor movement, coordinate clicks, text typing, and global hotkeys via native OS drivers (`PyAutoGUI` / Win32).
4. **Subprocess & Terminal Execution**: Runs shell commands, code snippets, file system operations, and background services directly on the host machine.
5. **Governance & Safety Gating**: Evaluates actions before execution to prevent accidental file deletion, unintended financial transactions, or unauthorized external messaging.

---

## 4. Laptop Actuators & Classes of Movement

To operate a laptop gracefully, AutoBot categorizes every physical and virtual input mechanism into **5 Classes of Movement / Actuators**:

### Class I: Native OS Input Actuators
* **Cursor Movement & Clicking**: Absolute and relative screen coordinate mouse positioning, left/right/double clicking, and click-and-drag operations.
* **Keyboard Typing & Single Keypresses**: Simulated physical keypresses (`Enter`, `Tab`, `Escape`, `Backspace`, arrow keys) and text string typing.
* **Global Hotkey Sequences**: System-level key combinations (`Alt + Tab`, `Win + D`, `Ctrl + Shift + Esc`, `Alt + Space`).

### Class II: Window & Desktop Management Levers
* **Window Focus & Restoration**: Bringing hidden or background application windows (VS Code, Excel, Chrome, Terminal) to the foreground using native OS APIs (`SetForegroundWindow`, `ShowWindow`).
* **Window Geometry & Workspaces**: Minimizing, maximizing, snapping, and resizing active application windows.
* **Multi-Monitor & DPI Scaling**: Adjusting coordinate mapping across varying screen resolutions and display scaling factors.

### Class III: Browser Protocol Actuators (CDP / Playwright)
* **Indexed DOM Element Interaction**: Targeting interactive web elements by unified index (`click_element(index)`, `fill_element(index, text)`).
* **Tab & Context Management**: Enumerating open browser tabs, switching active tabs, opening new tabs, and managing browser contexts.
* **URL Navigation & History**: Direct page navigation (`goto`), page reloads, and back/forward navigation.

### Class IV: System CLI & Process Actuators
* **Subprocess Execution**: Spawning and monitoring shell scripts, Python programs, system diagnostic tools, and background daemons.
* **File System Manipulation**: Creating, reading, editing, moving, and deleting local files and workspace directories.
* **Environment & Process Controls**: Inspecting environment variables, process lists, CPU/memory usage, and process termination (`taskkill`).

### Class V: Clipboard & Inter-Process Communication (IPC)
* **OS Clipboard Buffer**: Reading from and writing to the system copy/paste buffer (`Ctrl + C`, `Ctrl + V`).
* **Local REST & WebSockets IPC**: Exposing local FastAPI endpoints and WebSockets for real-time frontend dashboard telemetry.

---

## 5. Architectural Bottlenecks & Strategic Solutions

| Bottleneck | Root Cause | AutoBot Solution |
| :--- | :--- | :--- |
| **Profile & Single-Instance Lock Stalls** | Background Chrome instances hold file handles on `SingletonLock`. | **Graceful CDP Launcher**: Detects CDP availability, clears stale locks cleanly, and falls back to isolated automation profiles when necessary. |
| **DOM vs. Vision Disconnect** | Web apps with canvas, QR codes, or custom WebGL renderers produce empty DOM trees. | **Hybrid Perception**: Combines screen screenshots with DOM trees so the AI always *sees* visual elements. |
| **Context Window Token Bloat** | Large single-page apps (SPAs) generate thousands of DOM nodes. | **Smart DOM Compression**: Filters out non-interactive layout containers and strips hidden elements before sending to LLM. |
| **Asynchronous UI Latency** | Web Workers, SPA state updates, or network delays cause premature action retries. | **Reflection & Verification Steps**: Incorporates deliberate wait/verify checks between action cycles rather than infinite refresh loops. |

---

## 6. The Governance Model: The Permission Dial

To ensure complete safety without sacrificing capability, AutoBot operates under a 3-tier **Permission Dial**:

* 🛡️ **Level 0 (Observer / Advisor)**: Read-only perception. AutoBot analyzes the screen/logs and suggests actions, but cannot execute OS mutations without manual user approval.
* ⚡ **Level 1 (Supervised Co-Pilot - Default)**: Auto-approves safe read and navigation actions. Pauses and prompts the user before executing potentially **irreversible actions** (deleting files, sending external messages, financial transactions).
* 🚀 **Level 2 (Full Autonomy)**: Executes multi-step OS and browser missions independently with real-time status reporting.

---

## 7. Strategic Vision

AutoBot proves that autonomous computing does not require locked-down cloud containers or third-party SaaS subscriptions. By combining native OS control, browser protocol access, multi-modal vision, and user-controlled governance, AutoBot turns your existing laptop into an **Autonomous AI Workstation**.
