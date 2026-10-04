# ARC Prize 2026: ARC-AGI-3 Challenge
## Abstraction and Reasoning Corpus · $850,000 USD Prize Pool
### Autonomous Engineering Strategy, Benchmarks, & Handover Plan

---

## 1. Competition Overview
- **Competition**: ARC Prize 2026 - ARC-AGI-3.
- **Goal**: Create an AI capable of fluid intelligence that can learn new concepts on-the-fly from few-shot input/output demonstration grids without memorizing predefined task rules.
- **Format**: Featured Code Competition (`enable_internet: false` during scoring evaluation).
- **Scale**: 3,377 competing teams worldwide.
- **Current Baseline Rank**: Team Rank #1,001 (Score: 1.54% solved).
- **Leaderboard Top 10% Target**:
  - Current Top 10% cutoff is `3.65%` tasks solved (Rank #337 / 3,377).
  - Current Top 10 Rank is `9.96%` (rellik13); Rank #1 is `27.29%` (Tufa Labs).

---

## 2. Core Constraints & Submission Protocol
1. **Offline Evaluation**: The submission pipeline must run completely self-contained without internet access. All weights (Qwen-2.5/3.8, tokenizer checkpoints, LoRA adapters) must be mounted as offline Kaggle datasets.
2. **Submission Format**: `submission.parquet` containing predicted grids for test problems.
3. **Execution Time Quota**: Generous GPU compute allocated by competition hosts on dedicated Nvidia RTX Pro 6000 Ada (48GB VRAM), capped at 9–12 hours.
4. **Submission Protocol**: Code competition submission that evaluates against the hidden live Kaggle Arcade gateway (`http://gateway:8001/`).

---

## 3. Autopsy of Exp 2 (1.54% LB Score)
1. **Outdated Weights & Quantization**:
   - Exp 2 used `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` which lacked native NEXTN Multi-Token Prediction (MTP) speculative decoding and ModelOpt FP4 quantization.
   - Inference throughput was sluggish (~10-15 tokens/sec vs 60+ tokens/sec on FlashNext MTP), causing timeouts on complex ARC environments.
2. **Critical Agent Loop Defects (Identified by Scott Le Grand & Keith Tyser)**:
   - **Context Window Flooding (Duplicate Images)**: Historical messages preserved raw high-res base64 grid images, consuming up to 26% of the context budget and evicting previous tool outputs and reasoning traces.
   - **World Model Dropping**: The labeled-block parser used strict prefix matching, failing when the model output "World model (revised):" or "World model update:".
   - **Memory Wipeout on Retry**: `game_over` events wiped the entire world model state instead of retaining learned dynamics for level retries.
   - **Infinite Action Loops**: Identical code snippets re-executed indefinitely with zero actions executed.
   - **HUD Thrashing**: The agent lacked action-budget HUD bar masking, misinterpreting countdown bar cell changes as puzzle progress and thrashing no-ops.
   - **ACTION7 Failure**: ACTION7 (undo) was advertised in the action space but unmapped in the execution loop.

---

## 4. Experiment 3: RadixArk Qwen3.8 FlashNext NVFP4 MTP + AgentFix SOTA Reasoner
- **Kernel Slug**: `daltongabrielomondi/autobot-arc3-exp3-flashnext-agentfix-sota`
- **Model**: `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1` (135GB RadixArk ModelOpt NVFP4 checkpoint).
- **Runtime Environment**:
  - `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`
  - `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`
  - Machine: `NvidiaRtxPro6000` (48GB Ada GPU).
- **Core Engineering Enhancements**:
  1. **FlashNext MTP Speculative Decoding**:
     - Native 3-token NEXTN MTP speculative decoding (`TAAF_VLLM_MTP_TOKENS: 3`) providing ~3-4x throughput acceleration.
     - 16 vLLM sequences with CUDA graphs enabled (`cg32`).
  2. **AgentFix Suite (Runtime Monkey-Patches)**:
     - `AGENTFIX_IMAGES = 1`: Strips historical redundant grid images, retaining only recent text descriptions + current image, bounding token overhead to a flat 120 tokens.
     - `AGENTFIX_MEMORY = 1`: Tolerant labeled-block parser prevents world model dropouts; level retries preserve learned physics and transition rules.
     - `AGENTFIX_TIMING = 1`: Strict action execution time accounting.
  3. **Deep Puzzle Budget**:
     - Expanded `bm.solver.analyzer_timeout` from 900s to 1200s (20 minutes per puzzle domain), allowing comprehensive exploration on stubborn environments.
  4. **Watchdog Failure Recovery**:
     - Background vLLM watchdog polling `/v1/models` every 15s to automatically revive the server upon memory leaks or hardware stalls.
- **Benchmark Target**: $\ge 3.65\%$ (Surpassing Top 10% Cutoff) towards Keith Tyser's benchmark of **`8.62%` (Rank #12)**.
- **Official Score Result**: **`4.04%`** (Rank #711 worldwide).

---

## 5. Experiment 4: Pure Keith Tyser SOTA + Dynamic Per-Game Budget Fill
- **Kernel Slug**: `daltongabrielomondi/autobot-arc3-exp4-keith-budgetfill-sota`
- **Submission ID**: **`56815799`**
- **Date Submitted**: `2026-10-04 06:06:23 UTC`
- **Status**: `SubmissionStatus.PENDING`
- **Model**: `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`
- **Machine**: `NvidiaRtxPro6000` (Blackwell 96GB GPU).
- **Core Engineering Restorations & Enhancements**:
  1. **Full 32K Context Restored**: Removed the harmful 16K context limit (`LOCAL_ANALYZER_CONTEXT_WINDOW=16384`) from Exp 3, restoring the entire reasoning history and tool execution traces.
  2. **Zero AgentFix Regressions**: Eliminated Scott Le Grand's 2,707 lines of experimental monkey-patches which caused documented action refusals and a 29% activity drop. Restored Keith's pure verified 8-sequence solver.
  3. **Dynamic Per-Game Budget Fill**:
     - Formulated: `allocated_per_game = max(7920.0, (32400 - elapsed - 1500) / waves)`
     - Dynamically expands per-game solving budgets from 2.2 hours up to 7+ hours across available games within the 9-hour runtime ceiling.
  4. **Target Benchmark**: **`8.62% - 11.00%`** (Top 15 on ARC-AGI-3 Leaderboard).

