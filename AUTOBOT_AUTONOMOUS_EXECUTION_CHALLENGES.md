# Autobot Autonomous Execution Challenges & Engineering Lessons
> **Context**: Real-time simulation of unattended, self-supervised multi-competition orchestration across Kaggle Cloud.

---

## 1. Asynchronous Lifecycle Management & The Non-Blocking Watchdog Law
### The Challenge
Kaggle kernels and submission scoring pipelines operate on radically different time horizons:
- Fast tabular training: 2 to 5 minutes.
- Heavy CPU state reconstruction (IEEE Traffic Flow): 12 to 15 minutes.
- Deep 3D U-Net video inference (Biohub Cell Tracking): 90 to 120+ minutes.
- Hidden test set scoring queues: 1 to 4 hours depending on Kaggle backend load.

### Autobot Failure Mode If Unassisted
If an LLM agent enters a synchronous polling loop, it:
1. Saturates the conversation context with repetitive polling messages.
2. Rapidly exhausts LLM token budgets and API rate limits.
3. Blocks other concurrent competitions from making progress.

### The Engineered Solution
1. **The 60-Second Liveness Verification Rule**:
   - Probe status at t+30s and t+60s.
   - If status == ERROR, immediately retrieve logs, patch the defect, and re-dispatch.
   - If status == RUNNING, the job has successfully passed initialization.
2. **Event-Driven Sleep State**:
   - Once liveness is verified, the agent commits state to disk and transitions to dormant sleep.
   - The orchestrator only wakes upon an explicit event (e.g. timer expiry, task finish, or user input).

---

## 2. Operating System & Platform Substrate Friction
### The Challenge
Developing locally on Windows while deploying remotely to Linux containers on Kaggle Cloud introduces subtle substrate anomalies:

1. **PowerShell UTF-8 Byte Order Mark (BOM)**:
   - PowerShell Out-File silently prepends a 3-byte BOM (0xEF, 0xBB, 0xBF).
   - When kaggle kernels push executes, Python standard json.load() crashes with Unexpected UTF-8 BOM.
   - Autobot Rule: Never use standard PowerShell redirection for JSON config. Always emit raw UTF-8 using explicit Python or .NET IO.

2. **Windows Terminal Character Mapping (charmap UnicodeEncodeError)**:
   - Remote Kaggle kernel stdout often contains UTF-8 status glyphs.
   - When the Kaggle CLI pipes remote stdout to the Windows command prompt, Python defaults to cp1252, causing immediate crash.
   - Autobot Rule: Standardize on ASCII stdout in all remote training scripts, or enforce PYTHONIOENCODING=utf-8 across all shell execution environments.

---

## 3. Public Leaderboard Probing vs. Private Generalization (The SOTA Illusion)
### The Challenge
In Kaggle competitions with synthetic data (Playground Series S6E9), community notebooks frequently reverse-engineer the public test split:
- They discover specific probability bands that artificially boost the public score (e.g. to 0.94656).
- These swaps are mathematically fitted only to the specific errors of one specific anchor prediction file.
- When an agent naively blends these swapped predictions with a fresh, well-regularized multi-seed model, the public score drops, creating the illusion that the ensemble is inferior.

### The Engineered Solution: Dual-Track Submission Portfolio
Autobot must strictly partition candidate submissions into two distinct tracks:
- **Track 1 (Public Calibration Anchor)**: Replicates the verified public leaderboard ceiling to secure high rank visibility (e.g., Exp 17, 0.94656, Top 4.88%).
- **Track 2 (Pure Generalization Ensemble)**: Uses pure Out-of-Fold (OOF) cross-validation with zero-tie ordinal ranking, unpolluted by public split probing, protecting against catastrophic private leaderboard shakeup.

---

## 4. Code Competition Submission Protocols
### The Challenge
Traditional competitions allow direct CSV uploads via API. However, sensitive competitions (Biohub) enforce strict Code Competition rules:
- Manual CSV upload fails with 400: Submission not allowed.
- Internet must be explicitly disabled (enable_internet: false).
- All wheels, weights, and dependencies must be mounted offline.
- Submission commands require kernel binding: kaggle competitions submit -c comp -k kernel -v version -f file.
- Hidden test evaluations run on private backend worker pools that do not expose real-time stdout logs.

### The Engineered Solution
Autobot must detect the competition submission protocol upfront:
- If code competition: bundle all dependencies into an offline dataset, push version with enable_internet: false, verify local proxy inference on validation sets, and trigger the notebook submission.

---

## 5. Dynamic Schema & Data Drift Auto-Patching
### The Challenge
Public pipelines often contain hardcoded assumptions that break on official evaluation sets:
- In IEEE Traffic Flow, the public baseline asserted 6,985,307 rows, whereas the official submission_key.csv contained 6,980,503 rows, causing a crash at the very final line of execution.
- External dataset authors may rename or mirror datasets under different usernames.

### The Engineered Solution
An autonomous agent must never assume public constants are immutable. It must inspect submission_key.csv dynamically at runtime, determine ground truth dimensions, and auto-patch assertions before execution.


---

## 6. The LLM Cognitive Failure Mode: Priority Decay & The Reinventing the Wheel Bias
### The Empirical Diagnosis
During this session, despite explicit user instructions to prioritize not reinventing the wheel and always start from the best available public solution, the agent suffered from a classic LLM cognitive vulnerability:
1. Generative Momentum over Research Grounding: When presented with a task, an LLM defaults to generative auto-completion--modifying existing code, inventing heuristic patches, and running local tweaks--rather than systematically verifying whether a vastly superior official repository or newer public codebase exists.
2. The Stale Anchor Trap: In IEEE Traffic Flow, the agent adopted a community notebook from September 8 (0.53880), unaware that on September 10, the organizers patched the benchmark, published an official repository (jacky850/trafficflowbench-public), and exposed that reading speeds from links.csv instead of fd_parameters.csv broke the fundamental diagram triangle, awarding S_physics = 0.0!
   - Mathematical proof: S_total = 0.35 * 0.90 + 0.30 * 0.39 + 0.15 * 0.0 + 0.20 * 0.53 = 0.53880. The pathetic score was a direct mathematical penalty for violating the fundamental diagram physics, which the official repo had already solved!

### The Autobot Architectural Remedy: The Mandatory Phase 0 SOTA Gate
Autobot cannot rely on LLM memory, prompt instructions, or goodwill to prioritize public code. The architecture must enforce this as a deterministic, unskippable software state machine:

Phase 0: Mandatory SOTA & Official Source Discovery (HARD LOCK)
- Query Kaggle API for top kernels (scoreDescending, voteCount)
- Query GitHub API for official benchmark repositories & organizer papers
- Download & parse AST of top reference solutions
- Check benchmark update changelogs & data patch announcements

If an LLM agent attempts to scaffold or train a custom baseline without completing Gate 0, the Autobot runtime throws a MissingSOTAGateError and aborts execution.

---

## 7. Empirical Validation: Phase 0 Enforcement Yields +0.12035 Score Leap
### The Empirical Result
When the agent stopped attempting ad-hoc prompt-driven patching and strictly executed the Phase 0 official source architecture (`jacky850/trafficflowbench-public`):
1. Triangle consistency was restored: speeds and densities conformed to empirical $v_f$ and $k_{crit}$ from `fd_parameters.csv`.
2. $S_{physics}$ was unlocked from $0.0$ back to valid physics regime ($>0.80$).
3. The Structural Kalman Filter (historical weekday x time profile + Kalman RTS deviation smoother) drastically reduced state RMSE across 6.98M cells.
4. **Leaderboard Score**: Soared from **`0.53880` $\to$ `0.65915`** (+0.12035 net gain, +22.3% relative improvement).
5. **Leaderboard Rank**: Climbed immediately from Rank #112 to **Rank #104**.

### Key Takeaway for Autobot Runtime
Ad-hoc LLM reasoning cannot out-compete published domain benchmarks. When Autobot enforces strict Phase 0 SOTA Discovery, it guarantees that every experiment begins at the mathematical boundary of existing knowledge rather than rediscovering known failure modes.

