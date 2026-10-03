# Peer review and deck plan: "Autonomous AI Systems for Long-Horizon Engineering"

I checked the paper against the repo, and several headline numbers don't match your own records. As written, the paper would likely be rejected on credibility before anyone gets to the ideas. The underlying story (long-horizon agents fail quietly, and the fixes are systems engineering) is good and worth publishing. The fix is to make it smaller and fully supported by evidence.

## 0. Blocking issues: the paper's numbers vs. the repo

| Claim in paper | What the repo records | Source |
|---|---|---|
| Traffic **0.79155, Rank #88** | Best score is Exp 5 at **0.69045**; latest recorded rank is **#104 / 133** | `competitions/ieee_traffic_flow/THINKING_AND_DECISIONS.md:144,160,182` |
| Biohub **0.953, Top 8%** | 0.953 at **Rank #460 / 3,906**, which is about the **top 11.8%** | `biohub_cell_tracking/THINKING_AND_DECISIONS.md:36` |
| UMUD **0.45976, #45** | **0.43918 at #45** (normalized MAE, lower is better) | `umud_muscle_architecture/exp5_.../build_submission.py:7` |
| Soil **59.99 EMD, #144** | No source found anywhere except the paper and the figure/deck scripts | `papers/.../generate_figures.py:93` |
| SWE-bench Exp 8 "calibrated" | **No resolve rate recorded.** The leader is at 0.13 | `gemma_4_developer_agent/THINKING_AND_DECISIONS.md:13` |
| Earth: Exp 6 crash → Exp 7 clip at 0.05, "autonomous" | Repo history: the clip fix landed in **exp2 (1e-3 floor)**. In exp2, **46% of height predictions hit the clip floor** and the leaderboard score got *worse* (0.274 → 0.330). There is a logged **"User directive (2026-09-24): approved"**. Latest rank is #76 / 94 | `ieee_ai_emulation/THINKING_AND_DECISIONS.md:92-94,162,193,209` |
| Supervisor does "Autopsy → Patch Code → Re-push" | The supervisor **downloads logs and sets a flag** (`state["emulation_exp7_error"] = True`). It never patches or re-pushes. The pollers are hardcoded per experiment (`poll_ai_emulation_exp7`, `poll_gemma4_quota_and_dispatch`) | `autobot_autonomous_supervisor.py:53-120` |
| (missing) | **S6E9: Rank #131 / 2,683, top 4.88%.** This is your strongest verified result and the paper leaves it out | `s6e9_ev_prediction/THINKING_AND_DECISIONS.md:306` |

The same unverified numbers are hardcoded in `generate_figures.py` and `generate_deck.py`, so the deck would repeat them.

---

## 1. Methodological and theoretical review

### 1.1 The "Dual-Loop Reactive Engine"

**What works:** separating fast, deterministic liveness checks from slow, LLM-driven strategic review is a sound design, and it is the paper's real contribution.

**Problems:**
1. **It isn't new, and the paper cites nothing.** This is a supervision tree (Erlang/OTP) or a liveness probe plus a reconciler (Kubernetes), with an LLM acting as the reconciler. Present it that way and cite that lineage. Also cite the agent literature: SWE-agent, OpenHands, AIDE, MLE-bench, RE-Bench, METR's time-horizon work, Reflexion and Voyager. A paper with no citations is a desk reject at most venues.
2. **The diagram overstates what the code does.** The inner loop is a deterministic poller and does not heal anything. Healing, meaning writing the patch, happens in the outer LLM session and sometimes needs your approval. Redraw the diagram so every node is labelled **deterministic**, **LLM** or **human**. Reviewers will read the code.
3. **The inner loop isn't general.** It is a per-experiment script, not a "concurrent cloud and local job poller." Either generalize it (a declarative job registry) or describe it honestly as bespoke.
4. **The outer loop depends on a proprietary tool (Antigravity), and its cost isn't counted.** That undercuts both the "open architecture" framing and the cost section.
5. **"Metacognitive" is anthropomorphic.** "Scheduled LLM audit" says the same thing more precisely.
6. **There is no ablation.** The central claim is that the outer loop prevents stalls, but no stall-hours are reported with and without it. You need metrics:
   - mean time to detect (MTTD) and mean time to repair (MTTR)
   - stall-hours avoided
   - share of wall-clock time spent productively
   - human interventions per 12 hours
7. **Keep-awake overclaim.** `SetThreadExecutionState` prevents idle sleep only while the calling process is alive. It does not stop network-adapter power management, Windows Update reboots or lid-close policy. Change "guarantees" to "prevents idle sleep." `ES_DISPLAY_REQUIRED` is also unnecessary for headless work.

### 1.2 SWE-bench budget arithmetic

The arithmetic itself is correct:
- Old budget: 120 × (8.0 + 1.2) = 1,104 min = 18.4 h
- New budget: 120 × (3.5 + 0.75) = 510 min = 8.5 h

Five problems with the reasoning around it:
1. **The overhead changes from 1.2 to 0.75 minutes with no explanation.** Either measure it or keep 1.2. At 1.2 the bound is 564 min (9.4 h), which is still safe and more honest.
2. **The "hard ceiling" depends on how the limit is enforced.** If `max_time_minutes` is only checked between turns, an in-flight turn (22 s of generation plus up to a 60 s tool timeout) can overrun each task by about 1.4 min:

   120 × (3.5 + 1.4 + 0.75) = 678 min = **11.3 h**

   That still passes, but the safety margin shrinks from 3.5 h to 0.7 h. Say how the limit is enforced.
3. **Fixed costs are left out.** vLLM cold start, loading a 31B model across 4 × L4 GPUs, and Docker image pulls are one-time costs. Add a fixed term: T = T₀ + N·(b + o).
4. **N ≈ 120 is an assumption.** Show sensitivity. At 4.25 min per task, the 12 h limit breaks at N ≈ 169. Plot the maximum safe N for each configuration.
5. **`max_turns: 25` does nothing.** At 22 s per turn, 3.5 min allows about 9.5 turns, so the time cap always binds first. More importantly, **cutting the budget almost certainly lowers the resolve rate**, and you report no resolve rate. The claim that halving `thinking_budget` doubles throughput "while preserving deep reasoning" is unsupported. It needs an A/B comparison of resolve rates on a public dev split. The throughput gain is 2.18×.

Reframe this section as a **constrained optimization problem**: maximize expected resolve rate subject to the 12 h limit. That is a real contribution; "we made the numbers smaller" is not.

### 1.3 The Earth-system boundary failure

1. **The mechanism doesn't hold up as stated.** Predicting in log space guarantees positive values after exponentiation. Reaching 0.000 means one of three things: `expm1` of a negative number, rounding at export, or a delta model working in linear space. State which one it was.
2. **The deeper lesson is failing fast, not clipping.** An invariant checked only at export costs 2.1 hours of compute. Check invariants **after every autoregressive hop** and abort, or fall back, as soon as one is violated.
3. **Clipping hides the underlying problem.** Your own logs show 46% of predictions hitting the clip floor and the score getting worse. Clipping guarantees the file passes validation, not that the model is good. Always report the clip-floor hit rate. This finding is worth including: it's rare, honest, and publishable.
4. **The autonomy claim needs accounting.** State exactly what the supervisor did (detected the error and fetched logs), what the LLM did (diagnosed and patched), and what the human did (approved). Report the times between these steps.
5. **Report the outcome.** "4-Var Delta Residual" is not a metric. Report the leaderboard score and rank.

### 1.4 Economic analysis

The arithmetic checks out: $0.29 of electricity, and $2.69–$7.09 in total. The analysis itself doesn't:

1. **The ratios don't match the paper's own inputs.**

   | Comparison | Paper claims | Recomputed from the paper's figures |
   |---|---|---|
   | Devin | 50–80× | $240–480 vs $2.69–7.09 = **34×–178×** |
   | Human team | 800–1,000× | $4,320 vs $2.69–7.09 = **609×–1,606×** |
   | AutoML per day | $600 | $50k–120k/yr ÷ 365 = **$137–329**; there is no path to $600 |
   | Abstract | "100×–1,000×" | Inconsistent with the paper's own Section 5 |

   The dollar-sign bars in the ASCII chart aren't to scale either.
2. **It compares the cost of inputs, not outcomes.** This is the fatal flaw. The claim is that $5 of Autobot does the same job as $4,320 of grandmasters, but the verified ranks are bottom-quartile traffic, the top 12% on Biohub and #76/94 on Earth. **"Competitive parity with human grandmaster teams" must be removed.** Use **cost per unit of outcome** instead (dollars per leaderboard percentile gained, dollars per resolved task).
3. **Costs that are left out:**
   - the market value of the sponsored 4 × L4 GPUs (roughly $3–4/hr on-demand at a cloud provider, so about $25–35 per 8.5 h run; check current pricing)
   - the tokens and subscription for the orchestrating LLM (Antigravity and its underlying model)
   - amortized hardware
   - **human supervision time**
   - development cost amortized over runs

   Present two figures: **marginal cost** and **fully-loaded cost**.
4. **The 2.4M-token count needs telemetry.** Show a breakdown by model.
5. **Devin pricing must be cited with a date.** Its pricing has changed several times.
6. **Tone.** "Proving" and "astounding" should go, and so should "free-tier exploitation," which reads badly with respect to platform terms of service. Use "free-tier utilization."

### 1.5 The sharpest reviewer objections, and how to preempt them

| # | Objection | Preemption |
|---|---|---|
| 1 | "The reported scores don't match the public leaderboards." | Fix every number. Add a table of leaderboard snapshots with date, rank / field size and kernel ID. |
| 2 | "How autonomous was this really?" | Add an **Autonomy Accounting** table: every human action during the 12 h run, with a timestamp and a classification (approval, fix or strategy). |
| 3 | "There is no ablation of the dual loop." | Run 12 h with the outer loop disabled, or replay logs, and report stall-hours, MTTD and MTTR. |
| 4 | "N = 1 run, and these are anecdotes." | Present the case studies explicitly as *failure-mode case studies*, not benchmarks. Add a Threats to Validity section. |
| 5 | "Cost-parity claims without outcome parity." | Use cost per outcome, include fully-loaded cost, and drop the "grandmaster" claim. |
| 6 | "SOTA" is misused. | Remove "SOTA" everywhere except where you actually hold #1. |
| 7 | "No related work." | Add Section 2: Related Work (agent harnesses, MLE-bench/RE-Bench, supervision theory). |
| 8 | "It isn't reproducible." | The links are `file:///C:/Users/...` paths. Publish the repo and reference a commit hash, with redacted logs. |
| 9 | "Section 6 is off-topic and overclaims." | Move it to an appendix or cut it. "Solves the first two WHO delays" is wrong: in the Three Delays model, delays 1 and 2 are the decision to seek care and reaching a facility, and data interoperability doesn't address either. |
| 10 | "The authorship is anonymous or synthetic." | Name the human authors. Disclose that AI assisted with the writing and with the experiments. |

---

## 2. Ten-slide keynote deck

The deck is built on the stronger, honest thesis: **"Long-horizon agents fail quietly; here's the systems engineering that catches it."** Values marked ⟨verify⟩ must be filled from the corrected tables.

**Slide 1: "Long-Horizon Autonomy Fails Quietly"**
- *Subtitle:* Architecture, failure modes and economics of a 12-hour autonomous engineering agent.
- *Visual:* a 12-hour horizontal timeline. Green is productive time; red gaps are stalls, labelled with their causes.
- *Bullets:*
  - 6 domains running in parallel, one 12 h unattended run
  - 4 recurring failure classes
  - Marginal cost about $3–7 per run
- *Speaker notes:* "Everyone benchmarks what agents can do in two minutes. We ran one for twelve hours across six problems at once. The model rarely failed on intelligence. It failed on arithmetic, on silence, and on physics. This talk covers those failures and the architecture that catches them."

**Slide 2: "From Seconds to Hours"**
- *Subtitle:* New failure classes appear as the horizon grows.
- *Visual:* a log-scale horizon axis (seconds for chat, minutes for agent chains, hours for Autobot). Beneath each band are the failure modes that first appear at that scale.
- *Bullets:*
  - Short horizon: the model's quality dominates
  - Long horizon: operations dominate (budgets, liveness, drift)
  - Human-in-the-loop hides these failures; autonomy exposes them
- *Speaker notes:* "When a human re-prompts every two minutes, the human is the watchdog. Remove the human and every silent failure becomes hours of lost compute."

**Slide 3: "Four Ways Agents Die Overnight"**
- *Subtitle:* A failure taxonomy from real incidents.
- *Visual:* a 2×2 grid (budget compounding, silent stall, boundary drift, latency floor). Each cell shows an incident count and hours lost ⟨from logs⟩.
- *Bullets:* one incident per cell, each with the time it cost.
- *Speaker notes:* "None of these is a hallucination. Every one of them would pass a single-turn benchmark."

**Slide 4: "Two Loops: Deterministic Liveness, LLM Strategy"**
- *Subtitle:* A supervision tree with an LLM as the reconciler.
- *Visual:* two concentric loops. The inner loop (2 min) is grey and deterministic. The outer loop (30 min) is blue and LLM-driven. The human approval gate is orange. Every edge is labelled with who acts.
- *Bullets:*
  - Inner loop: poll, validate, fetch the traceback (no LLM, fails cheaply)
  - Outer loop: audit, diagnose, patch, reschedule, handle quota windows
  - Measured MTTD / MTTR: ⟨X⟩ / ⟨Y⟩ min
- *Speaker notes:* "The cheap loop only has to notice. The expensive loop only has to think. Keeping them apart is what stops a silent stall from lasting all night."

**Slide 5: "The 18.4-Hour Mistake"**
- *Subtitle:* Budgets compound across the evaluation split.
- *Visual:* three horizontal bars against a red 12 h line: Exp 6/7 at 18.4 h, Exp 8 nominal at 8.5 h, Exp 8 worst-case overrun at 11.3 h. An inset plots the maximum safe N against the per-task budget.
- *Bullets:*
  - 120 × (8.0 + 1.2) min = 18.4 h. Killed twice.
  - T = T₀ + N·(b + o) must be checked *before dispatch*
  - Budget is a constrained optimization: resolve rate vs. wall clock
- *Speaker notes:* "Eight minutes per task sounds reasonable until you multiply it by 120. We now compute the worst-case bound before any dispatch and refuse to submit above 80% of the limit."

**Slide 6: "Failing at Hour 2.1 Instead of Minute 1"**
- *Subtitle:* Check invariants at every hop, not at export.
- *Visual:* eight autoregressive hops with the height trajectory drifting to zero at an arid site. The original assertion sits at the end; the proposed per-hop checks sit at each hop.
- *Bullets:*
  - The final-step assertion wasted 2.1 h of compute
  - The clip fix passed validation, but 46% of predictions hit the floor and the score got worse
  - Lesson: a clip is a guardrail, not a model fix. Always report the clip hit rate.
- *Speaker notes:* "This is the most honest slide in the deck. The self-healing loop made the file valid. It didn't make the model right. Autonomy needs quality signals, not just validity signals."

**Slide 7: "Results Across Six Domains"**
- *Subtitle:* Verified leaderboard positions, with the level of autonomy for each.
- *Visual:* a dot plot of rank percentile per domain, with each dot coloured by autonomy level (fully autonomous, human-approved, human-directed).
- *Bullets:*
  - Best: S6E9 top 4.88% (#131 / 2,683)
  - Biohub #460 / 3,906 (top 12%)
  - Other domains ⟨verified values⟩
- *Speaker notes:* "These aren't grandmaster results, and we don't claim they are. The claim is that one agent held six campaigns together, unattended, at mid-field quality."

**Slide 8: "Three Engineering Laws"**
- *Subtitle:* What we would build into any long-horizon agent.
- *Visual:* three columns: Paranoid Liveness, Budget Proofs, Physical Invariants.
- *Bullets:*
  - Silent failures outnumber loud ones ⟨ratio from logs⟩
  - Deep sub-agent trees add a latency floor: 4 agents × 4 turns × 25 s = 6.7 min, which is above a 3.5 min budget
  - Enforce constraints by projection, then measure how often the projection is active
- *Speaker notes:* "Each law comes from a specific incident on the earlier slides. None of them needs a better model."

**Slide 9: "What Autonomy Actually Costs"**
- *Subtitle:* Marginal cost, fully-loaded cost, and cost per outcome.
- *Visual:* two stacked bars (marginal about $3–7; fully loaded including GPU market value, the orchestrator LLM and human supervision). A second panel shows cost per percentile point gained.
- *Bullets:*
  - Marginal: $2.69–7.09 per 12 h
  - Fully loaded: ⟨$X⟩
  - Human supervision: ⟨Y⟩ min per 12 h
- *Speaker notes:* "The cheapest number is real, but it's the marginal cost. Here is the honest total, and here is what it buys per unit of result."

**Slide 10: "Limitations and Roadmap"**
- *Subtitle:* From bespoke supervisor to general runtime.
- *Visual:* a roadmap in three swimlanes: Measure, Generalize, Prove.
- *Bullets:*
  - Measure: ablate the outer loop; log MTTD, MTTR and interventions
  - Generalize: declarative job registry; per-hop invariant library
  - Prove: a pre-dispatch budget verifier, and a public repo with logs
- *Speaker notes:* "Every claim on this slide is something a skeptic can check. That's what the next version of the paper needs."

---

## 3. Concrete revisions

### Do now
1. **Reconcile every number** with `competitions/*/THINKING_AND_DECISIONS.md`. Report rank as "#r / field size (top p%)" with a snapshot date. Remove "SOTA" everywhere.
2. **Rewrite the final sentence of the abstract**, for example: *"…we report the marginal and fully-loaded costs of a 12-hour unattended campaign ($2.69–$7.09 marginal) and argue that, for mid-field competitive results, autonomous orchestration shifts cost from engineer hours to supervision minutes."*
3. **Redraw the architecture diagram** so each node is labelled deterministic, LLM or human. Delete the disconnected `InnerLoopExec` node.
4. **Add Section 2: Related Work.**
5. **Add an Autonomy Accounting table** covering the 12 h run, built from `AUTOBOT_12H_RUN.md` plus approvals.
6. **SWE-bench section:**
   - fixed-cost term T₀
   - justification for the overhead value
   - how the time limit is enforced, with the overrun bound
   - a max-safe-N plot
   - Exp 8 resolve rate, or state plainly that it is pending
7. **Earth section:**
   - correct the log-space mechanism
   - fix the exp numbering to match the repo
   - report the clip-floor hit rate and the leaderboard outcome
   - propose per-hop checks
8. **Economics:** recompute the ratios, fix the AutoML figure, add fully-loaded cost and cost per outcome, date-stamp the pricing citations, and replace the ASCII bars with a to-scale log chart.
9. **Move Section 6 to an appendix**, and delete the WHO-delays claim.
10. **Add Threats to Validity:** a single run, a single operator, leaderboard overfitting, and platform-specific limits.
11. **Reproducibility:** publish the repo and reference a commit hash. Replace the `file:///` links. The SEQUOY path is outside this repo.
12. **Regenerate the figures and deck** from `generate_figures.py` and `generate_deck.py` after the fixes, since both hardcode the stale numbers.

### Roadmap (turns the critique into contributions)
- **Outer-loop ablation study.** This is the single highest-value experiment for the paper.
- **Pre-dispatch budget verifier:** compute the worst-case bound from `eval_config.yaml` and refuse to dispatch above 0.8 × the limit.
- **Generalize the supervisor** from `poll_<exp>` functions into a declarative job registry.
- **A per-hop invariant library** that reports clip-activation telemetry.
- **Cost telemetry per model call,** so the economics section is measured rather than estimated.

Nothing in the repo has been edited. The fix-up plan is in the plan file. The next step would be steps 1, 3 and 12: correct the numbers in the paper and in both generator scripts, then regenerate the figures and deck.