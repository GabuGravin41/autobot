# Autobot UI: Design Requirements & Aesthetic Guide

To make Autobot feel like a premium, state-of-the-art AI assistant, the UI should prioritize **clarity**, **real-time feedback**, and a **"Jarvis-like" sophisticated aesthetic**.

## 1. Core UI Components

### A. The "Brain" (Chat Interface)
- **Central Focus**: A wide, conversational area where the user gives commands.
- **Thinking State**: DeepSeek R1's reasoning process (the `<think>` tags) could be hidden behind a "View Reasoning" accordion to keep the main chat clean but transparent.
- **Rich Output**: If the AI extracts info (e.g., from an email), it should be rendered in a clean "Card" format, not just raw text.

### B. "State Awareness" Panel (The Pulse)
- **Current Action**: A prominent "Live Status" badge (e.g., *Status: Scanning Gmail*, *Status: Moving Mouse*).
- **Visual Feedback**: A small, live preview/thumbnail of the most recent `browser_snapshot` or screenshot so the user can see what the AI is seeing.
- **Time/Wait Bar**: A subtle progress bar when the AI is in a `wait` step, showing how much longer it's "pausing" for human-like realism.

### C. Workflow & Adapter Management
- **Tool Catalog**: A searchable sidebar of available adapters (WhatsApp, Overleaf, etc.) with toggles for permissions (e.g., "Allow Sensitive Actions").
- **Audit Log**: A technical terminal-style log (collapsible) for power users to see raw JSON inputs/outputs.

## 2. Recommended Aesthetic & Colors

### Color Palette (Obsidian & Electric)
- **Background**: `#0a0a0c` (Very deep charcoal/near-black).
- **Panels/Cards**: `#16161a` with 0.6 opacity (Glassmorphism effect).
- **Accents**: 
  - **Primary**: `#7c3aed` (Electric Violet) for buttons and active states.
  - **Secondary**: `#06b6d4` (Cyan) for status indicators and success logs.
  - **Thinking**: `#f59e0b` (Amber) for reasoning/paused states.
- **Borders**: `#27272a` (Subtle dark gray).

### Typography
- **Primary**: **Inter** or **Outfit** (Clean, modern, sans-serif).
- **Monospace**: **JetBrains Mono** for the logs and reasoning blocks.

## 3. Micro-Animations
- **Pulse Effect**: A slow glow on the "System Idle/Running" indicator.
- **Slide-in Logs**: New logs should slide in smoothly rather than just appearing.
- **Glass Hover**: Cards should slightly brighten or lift when hovered.
