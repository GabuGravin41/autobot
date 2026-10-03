# Google — The Gemma 4 Developer Agent Competition

- Kaggle ref: `gemma-4-developer-agent` (Featured, $65,000; separate paper track `gemma-4-developer-agent-paper`, $35,000)
- Entry deadline: 2026-11-25. Final submission: 2026-12-02. Started ~2026-09-22.
- **Only 2 submissions allowed for the whole competition** (per user). Treat every submit as scarce — validate exhaustively offline first.
- We entered on day 1. As of 2026-09-24, every visible leaderboard row is 0.00 — nobody has a working submission yet. First non-zero score is a real, achievable win condition independent of eventual leaderboard rank.

## What this competition actually is

Not a tabular/CSV competition. You submit `submission.zip`: a declarative **Google ADK agent config** (`agent.yaml` + optional `prompts/`, `configs/`, `sub_agents/`, `skills/`, LoRA `adapters/`) — no executable Python entrypoint. Google's own infrastructure compiles this YAML into a live ADK agent and runs it against held-out SWE-bench-style tasks: given a GitHub issue/problem statement and a repo snapshot, the agent must produce a **unified git diff patch** that makes a hidden `test_patch` pass (fail-to-pass) without breaking existing tests (pass-to-pass). Score = fraction of tasks resolved.

Public training data (`tasks.jsonl`, 129 tasks across fastapi/fastapi, Textualize/rich, psf/requests, encode/httpx): each task ships `problem_statement`, `hints_text`, `base_commit`, reference `patch` + `test_patch`, a repo snapshot tarball (`snapshots/`), an AST call/dependency graph (`graphs/`), and semantic code embeddings (`embeddings/`) that back two agent tools (`search_similar_code`, `get_code_neighbors`, `get_code_subgraph`). 22.4GB total — lives entirely on Kaggle's side (competition data mount / attached dataset), never downloaded to this machine.

### The single-base-model rule (hard constraint)
Every `model:` field across the whole submission (root agent, every sub-agent, every AgentTool) must resolve to **exactly one** declared base model, and it must be one of the pre-registered Gemma 4 aliases. `validate_single_declared_model` rejects anything else — **MedGemma and Qwen are not valid entries**, this is not a "bring your own base model" competition. See `THINKING_AND_DECISIONS.md` §2 for which Gemma 4 sizes actually exist as downloadable Kaggle Model instances (not all registry-listed aliases do).

### The harness is NOT locally runnable
`HARNESS_README.md` documents a `swegemma eval` CLI and describes `swegemma`/`adk-submission`/`adk-eval-core` as the evaluation stack, but **none of those packages ship anywhere** — not in the competition dataset (checked all 782 files), not on PyPI, not on GitHub (searched). A competitor asked the organizers about this in the official forum (topic 742882, unanswered as of this writing) — I hit the identical wall independently. Practical consequence: nobody, including us, can run the real Phase-1-agent / Phase-2-pytest-verification loop the graders use. We built our own approximate local harness (see exp1) purely to sanity-check that our agent's *mechanics* (tool-calling, patch extraction, edit_file matching) are sound before spending one of 2 submissions — it is not a faithful reproduction of the grader and its pass rate is not the real score.

### Compute
- Server-side grading runs on 4x NVIDIA L4 (96GB VRAM total), vLLM, `max_model_len=32768`, `max_loras=8`, `max_lora_rank=128`.
- Our own dev/training work runs entirely on **Kaggle's GPU kernels** (T4x2 / P100) — not this machine (no local GPU work, per user instruction; also this disk is at 100% capacity, 1.6GB free).
- AutoBot's existing Kaggle watchdog/ledger (`~/.autobot/kaggle_jobs.json`, `autobot/computer/kaggle_watchdog.py`) enforces the real account-wide 2-GPU/4-CPU slot cap — shared with whatever else is running (another agent is concurrently working `ieee_traffic_flow`, `biohub_cell_tracking`, `arc_prize_2026`, `ieee_ai_emulation` in this same repo). Always check `autobot --jobs` / the ledger before pushing a GPU kernel here.

## Base model decision
`gemma-4-e4b-it` (Elastic 4B, instruction-tuned). Reasoning in `THINKING_AND_DECISIONS.md` §2 — short version: `gemma-4-9b-it` (our first pick) turns out not to exist as a downloadable Kaggle Model instance despite being a registered eval alias; `e4b-it` does, is explicitly pre-registered, small enough for QLoRA on a single Kaggle T4/P100, and is available with a ready-made INT4 (`w4a16-ct`) variant for cheap inference iteration.

## Submission gating
Real `kaggle competitions submit` is IRREVERSIBLE-tier in AutoBot's approval system (`autobot/agent/approval.py`) — it always requires the user's live approval, unattended mode just logs-and-skips rather than bypassing it. We prepare and locally-validate `submission.zip` candidates; the user decides when to spend one of the 2 submissions.
