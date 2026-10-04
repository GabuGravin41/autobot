# AUTOBOT EMPIRICAL COMPETITION RESEARCH RULES & CONTINUOUS EXECUTION PROTOCOL

> [!IMPORTANT]
> **PERMANENT CORE DIRECTIVE — NEVER FORGET OR BYPASS**:
> This project operates on strict **empirical engineering and continuous experimentation**.
> Never invent solutions purely in the prompt without experimental backing. Never claim unverified leaderboard breakthroughs. Every hypothesis must be proven through runnable probe scripts, offline evaluations, and active telemetry across all 4 tracked competitions.

---

## 1. The Four Competitions Under Active Supervision

The agent must continuously monitor and advance all 4 competitions without dropping any:

| Competition | Slug / Identifier | Goal & Metric | Current Standing |
| :--- | :--- | :--- | :--- |
| **1. Enveda CASMI 2026** ($50k) | `enveda-CASMI26-molecule-id-mass-spectra` | Molecule ID from Mass Spectra (Top-1 / Top-k) | **Rank #60** / 2,461 teams (`0.417` LB) |
| **2. RSNA Knee Detection** ($77k) | `rsna-knee-abnormality-detection` | 12 Abnormalities Macro ROC-AUC | **Rank #428** / 5,160 teams (`0.943` LB) |
| **3. ARC Prize 2026** ($850k) | `arc-prize-2026-arc-agi-3` | ARC-AGI-3 Few-shot Visual Fluid Intelligence | **Rank #797** / 3,697 teams (`4.04%` LB) |
| **4. Gemma 4 Developer Agent** ($65k) | `gemma-4-developer-agent` | SWE-bench Python issue resolution | **Rank #842** / 1,715 teams (`0.08` LB) |

---

## 2. Immutable Operational Rules

### Rule 1: No Unverified Submissions
- Never submit a candidate to Kaggle without prior verification:
  - In code/agent competitions (Gemma 4): verify the YAML and schema gates locally and on a test kernel.
  - In tabular/CV competitions (RSNA, Enveda): verify with out-of-fold CV or validation probes first.
  - In reasoning benchmarks (ARC Prize): verify on the 25 public puzzles before spending daily submission quotas.

### Rule 2: Continuous Research While Submissions Score
- When a submission is scoring (which takes 2–12 hours), the agent **must never sit idle**.
- Scoring time is active work time:
  1. Audit errors on previous runs (e.g., break down RSNA ROC-AUC per view: Sagittal vs. Coronal vs. Axial).
  2. Run data exploration scripts on Kaggle compute to uncover feature distributions and edge-case failures.
  3. Replicate techniques from past top-performing Kaggle solutions in focused probe scripts.
  4. Write the next iteration's probe scripts so improvements are ready when the score arrives.

### Rule 3: Live Ground-Truth Telemetry Over Assumptions
- Never report stale ranks or assumed leaderboard positions.
- Always poll live data via `kaggle competitions list` and `kaggle competitions submissions`.
- Track submission errors explicitly (e.g., notebook timeouts vs. memory vs. schema errors).

### Rule 4: Multi-Agent Parallel Investigation
- Use subagents (`invoke_subagent`) to parallelize tasks:
  - Subagent A: Data analysis & error audits.
  - Subagent B: Probe scripts & feature testing.
  - Root Agent: Coordinating execution, scheduling, and quota tracking.

### Rule 5: Keep Laptop Awake
- The Win32 Keep-Awake daemon (`keep_awake.ps1`) must run continuously with `0x80000003` to prevent system sleep, idle drops, and token expiration.
