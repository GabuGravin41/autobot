# CHI-Bench (IEEE Big Data Cup 2027 Challenge 07)
### Status: BLOCKED on 3 items only you can unblock — read this first

- Kaggle ref: `chi-bench`. Entry deadline 2026-11-07. Final submission 2026-11-15. Prize pool $3,000 + Reliability Award.
- Marked by user as the highest-priority competition of this batch ("a fresh competition by IEEE that I really need to win").

## What it actually is
Not a model-training competition. You build an **agent + orchestration harness** that drives long-horizon healthcare-administration workflows (prior authorization, utilization management, care management) inside `χ-World`, a simulator exposed via MCP/REST tools, grounded in a 1,279-document policy handbook. Scored by a pinned LLM judge (`claude-opus-4-7`, 3 votes/rubric) plus deterministic workflow checks. You can use *any* model/harness/hardware — this is not restricted to Kaggle notebooks by the competition itself.

## Checked 2026-09-24 — 3 hard blockers found
1. **`ANTHROPIC_API_KEY` is not set anywhere in this environment.** Required for the pinned workspace judge and simulator counterparts the harness calls during local dev/eval. I have no way to provision this myself — it needs an Anthropic API account/billing on your side.
2. **No Docker anywhere available to me.** This machine has no `docker` binary at all, and Kaggle kernels don't support nested Docker (no privileged containers) — so I can't run `cb docker build` / `cb data verify` either locally (blocked by your "Kaggle resources only" instruction) or inside a Kaggle notebook (technically unsupported). CHI-Bench's own harness is built around `uv` + Docker on a real host or cloud VM, not a Kaggle-kernel-native workflow like our other competitions.
3. **The Managed-Care Operations Handbook is gated on HuggingFace** (`actava/managed-care-operations-handbook`) and requires you to personally request and receive approval on huggingface.co before anyone can download it — this is a manual human step tied to your HF identity, not something an agent can do on your behalf.

## What I did without those
- Confirmed the public (non-gated) parts: `chi-bench` GitHub repo, public task dataset (`actava/chi-bench`, Apache-2.0), quickstart docs — these don't require the handbook or API key to *read*, just to *run*.
- Did not install Docker locally (would violate your explicit "no local resources" instruction) and did not attempt to fabricate/borrow credentials.

## To unblock (needs you specifically)
1. Request handbook access at the HF link in the competition page, using your HF account.
2. Provide an `ANTHROPIC_API_KEY` (e.g. as an environment variable, or tell me where to source one).
3. Decide where this actually runs: since Kaggle notebooks can't host Docker, this likely needs either (a) explicit one-off permission to install Docker Desktop on this machine, or (b) a separate cloud VM (GCP/AWS/a Docker-capable box) — worth deciding deliberately rather than me picking a cloud provider and spinning up billable infrastructure on your behalf without asking.

Once unblocked, next step is `git clone https://github.com/actava-ai/chi-bench` and work through the quickstart's smoke test on one task before scaling up.
