# Google - The Gemma 4 Developer Agent Competition
## Accelerate Research in Autonomous Coding Agents · $65,000 USD Prize Pool
### Autobot Engineering Strategy, Architectural Analysis & SOTA Roadmap

---

## 1. Executive Summary & Competition Overview
- **Competition**: Google - The Gemma 4 Developer Agent Competition
- **Goal**: Build an autonomous software engineering agent using Google's **Gemma 4** (`gemma-4-31b-it-qat-w4a16-ct`) that resolves real-world Python GitHub repository issues (SWE-bench style) inside sandboxed Docker environments.
- **Metric**: **Resolution Rate [0.0 to 1.0]** (percentage of held-out software tasks completely resolved, verified by hermetic `pytest` in an isolated verification container).
- **Current Leaderboard Landscape**:
  - Public Baselines: `0.10` – `0.12` (10%–12% resolution rate).
  - Rank #1 Worldwide: `0.13` (13% resolution rate; Chandan Ranjan / 暗黑AGI / Alperen ÖZ / Daniel Olajide).
  - **Autobot Target**: **`0.17` (17% resolution rate)** — catapulting to **#1 in the World**.
- **Deadline**: December 2, 2026.
- **Compute Envelope**:
  - Evaluation Environment: **4 × NVIDIA L4 GPUs** (96 GB total GDDR6 VRAM).
  - Inference Server: **vLLM 0.19+** (`tensor_parallel_size = 4`, `max_model_len = 32,768`, INT4 W4A16 weight footprint ~16–18 GB).
  - Kaggle Execution Time Limit: **12 Hours**.

---

## 2. Technical Harness Architecture (`swegemma` + `adk-submission`)

### 2.1. The Declarative-Only Security Model
- Competitors **do not** submit Python code entrypoints (`agent.py`).
- Submissions must be packaged as **`submission.zip`** at `/kaggle/working/submission.zip` (< 3 GiB total uncompressed size).
- The harness compiles `agent.yaml`, `prompts/`, `sub_agents/`, `configs/`, and optional `adapters/` into Google ADK `BaseAgent` trees without dynamic `importlib` execution.
- Single Base Model Constraint: All agents in the tree must declare `gemma-4-31b-it-qat-w4a16-ct`.

### 2.2. Two-Container Lifecycle
1. **Container A (Agent Sandbox)**:
   - Cloned at `base_commit` (zero future git history).
   - Network disabled (`network_mode="none"`).
   - Agent interacts via 9 built-in tools (`run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch`, `search_similar_code`, `get_code_neighbors`, `get_code_subgraph`).
   - On completion or timeout, the harness extracts `agent_patch` via `git add -N . && git diff --binary HEAD`.
2. **Container B (Verification Sandbox)**:
   - Fresh container at `base_commit`.
   - Applies `agent_patch` using a 4-pass resilient patch engine.
   - **Anti-Tampering Reset**: Discards any changes made to test files (`test_*.py`, `*_test.py`, `tests/`) or config files (`conftest.py`, `pytest.ini`, `setup.cfg`).
   - Applies the hidden `task.test_patch`.
   - Runs `pytest`. If all tests pass (exit code 0), `resolved = True`.

---

## 3. Why Existing Baselines Plateau at 0.10–0.12 (Root Cause Analysis)

1. **Disabled / Starved Gemma 4 Thinking**:
   - Public baselines either set `include_thoughts: false` (turning off Gemma 4's native reasoning engine entirely) or set `thinking_budget: 2048`.
   - Complex repository bugs require multi-hop dependency tracing; without thinking, the model jumps to incorrect one-line assumptions.
2. **Premature Budget Cutoffs (`eval_config.yaml`)**:
   - Sample submissions configured `max_time_minutes: 4` and `max_tool_calls: 24`.
   - In SWE-bench, 24 tool calls is barely enough to locate the issue, leaving zero budget for reproduction and testing.
3. **Monolithic Prompt Pollution**:
   - Running exploration, file viewing, git diffs, and testing in a single agent conversation fills the context window with thousands of lines of output, degrading the model's instruction following.
4. **Test File Editing Trap**:
   - Naive agents edit test assertions to make tests pass locally, but Container B wipes test edits before verification, resulting in instant 0.0 scores.
5. **Lack of Targeted Pre-Submit Verification**:
   - Baselines make an edit and immediately call `submit_patch()` without ever executing a targeted `pytest path/to/test.py::test_func` in Container A.

---

## 4. The Autobot SOTA Architecture (Path to 0.17)

### Component 1: Calibrated Gemma 4 Thinking Engine
- `include_thoughts: true`
- `thinking_budget: 4096`
- Omit `thinking_level` (avoids OpenAI LiteLLM parameter mismatch).
- `temperature: 0.15`, `top_p: 0.92`, `seed: 42`.

### Component 2: Expanded Operational Budget (`eval_config.yaml`)
- `timeout_seconds: 180` (allows targeted test suites to run without timing out).
- `max_tool_calls: 48` (doubles baseline budget for deep inspection and verification).
- `max_time_minutes: 8.0` (well within the 12-hour total quota across tasks).
- `max_turns: 60`.

### Component 3: Elite Prompt Engineering (`prompts/system.md`)
- **Strict Anti-Tampering Directive**: Never touch `tests/`, `conftest.py`, or `pytest.ini`.
- **Targeted Code Intelligence**: Use `search_similar_code` strictly with exact function/class symbol names, not natural language sentences.
- **3-Tier Contextual Editing**: Require 3–5 lines of surrounding context to prevent `FileEditError` (multiple occurrence ambiguity).
- **Mandatory In-Sandbox Verification**: Run `pytest tests/path/to/test.py -k <test_name>` or `python3 -c "..."` to prove the fix works before calling `submit_patch()`.
- **Clean Diff Hygiene**: Run `git status --short` and `git diff --check` to ensure no accidental artifacts or syntax errors.

---

## 5. Experiment 6: Dual-Agent SOTA (Active Cloud Evaluation)
- **Kernel**: `daltongabrielomondi/autobot-gemma4-exp6-swe-sota`
- **Submission ID**: `56561900`
- **Status**: `SubmissionStatus.PENDING`
- **Architecture**: `swe_coder` (Root `LlmAgent`) + `code_analyzer` (`AgentTool` with `skip_summarization: true`).
- **Thinking Engine**: Enabled (`include_thoughts: true`, `thinking_budget: 4096`, `temperature: 0.15`).
- **Budgets**: `timeout_seconds: 180`, `max_tool_calls: 48`, `max_time_minutes: 8.0`, `max_turns: 60`.
- **Target**: Breakthrough over public 0.12–0.13 ceiling.

---

## 6. Experiment 7: 4-Stage Sequential Isolated Pipeline (Opus Architecture)
- **Kernel**: `daltongabrielomondi/autobot-gemma4-exp7-sequential-pipeline`
- **Status**: Kernel pushed & completed successfully on Kaggle (`submission.zip` compiled and ready for submission).
- **Architecture**:
  - `SequentialAgent` orchestrating 4 specialized stages:
    1. **`localizer`**: Scans symbols, call chains, and reads code. Capped at 12 tool calls. Emits structured `bug_card` verbatim to `output_key: bug_card`.
    2. **`reproducer`** (`include_contents: none`): Zero context bloat. Verifies bug failure on unpatched code via targeted test or inline heredoc in `/tmp`. Emits `repro_card` to `output_key: repro_card`.
    3. **`patcher`** (`include_contents: none`): Applies surgical `edit_file` with unique 3–4 line anchors (or whole function replacement for <40 lines). Includes guarded Python in-place fallback (`assert s.count(old) == 1`). Verifies syntax with `py_compile` and reruns repro command to confirm it passes. Emits `patch_note`.
    4. **`finalizer`** (`include_contents: none`): Quality assurance officer. Removes scratch files, enforces test anti-tampering (reverts any changes to `tests/`, `conftest.py`, `pytest.ini`), runs `git diff --check`, and calls `submit_patch()` once. Always.
- **Key Breakthrough**:
  - Eliminates context bloat: Each stage receives its own clean 32k window.
  - Eliminates `FileEditError`: Strict unique anchoring + whole function replacement + guarded fallback.
  - Eliminates "no patch submitted": `finalizer` guarantees `submit_patch()` is called on 100% of tasks.

