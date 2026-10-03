# Ideas for Autobot: Robust Long-Horizon Orchestration on Weak Models

> Grounded entirely in real evidence: the 9 competition `THINKING_AND_DECISIONS.md`
> post-mortems in `competitions/*/` (run largely via Antigravity on Gemini 3.8
> Flash — itself a mid-tier model, not a frontier one), plus
> `AUTOBOT_AUTONOMOUS_EXECUTION_CHALLENGES.md`, `DESIGN_PHILOSOPHY.md`, and
> `AUTOBOT_MISSION.md`. Every recommendation below cites the specific
> competition/section it comes from — this is not a generic "best practices
> for agents" list. Nothing here has been implemented yet; this is the
> ideas-and-suggestions pass requested before any code changes.

## 0. The framing this confirms, not just informs

`AUTOBOT_MISSION.md` already states the target precisely: *"not the most
capable possible agent on the most expensive possible model... a harness
disciplined enough that a modest model, given clean state and an unambiguous
action set, performs like a much larger one would have to without it."*

The 9 competitions are a real, if informal, existence proof of this thesis
under load — a mid-tier model (Gemini 3.8 Flash) running unattended,
multi-hour, multi-competition orchestration, and *mostly succeeding*
precisely where the harness carried weight the model couldn't (the
liveness-verification rule, the capacity semaphore, the SOTA gate). It also
shows exactly where the harness *didn't yet* carry enough weight — those
gaps are what moving to DeepSeek V4.1 / Nemotron-class models will expose
harder, because those models will very likely be weaker than Gemini 3.8
Flash at the specific things that already broke here: tool-call format
discipline, self-correction after a nudge, and multi-task attention.
Everything below is either (a) a confirmed gap the docs already surfaced,
with a concrete fix, or (b) an extension of a pattern the docs show already
works, pushed further because a weaker model needs more of it.

---

## 1. Fix first — confirmed by repeated, independent real-world evidence

These aren't single-incident guesses. Each was hit independently, by
different agents, on different competitions, days apart — that repetition is
itself the signal that they're real architectural gaps, not one-off flukes.

### 1a. The liveness grace window is measurably too short, and treats one bad read as fatal

`push_kernel()`'s `grace_checks=(30, 60)` default reported
`LIVENESS CHECK FAILED ... status=error` on runs that were, moments later,
confirmed genuinely `RUNNING` or already `COMPLETE` — independently in:
- `soil_grain_size_photos` (exp1 *and* exp2, both real CPU runs of
  255s/269s) — explicitly diagnosed: *"the current `grace_checks=(30, 60)`
  default window is too short to distinguish 'genuinely crashed' from
  'still in the queued/init state that Kaggle's status API transiently
  reports as ERROR before settling into RUNNING.'"*
- `gemma_4_developer_agent` (exp4/LoRA SFT push, and reconfirmed on exp3 and
  a `biohub-exp2` ledger entry in the same session)
- `ieee_ai_emulation` (exp3 push)
- `umud_muscle_architecture` (run 1) — traced to the same
  "running→error→running flicker" pattern visible elsewhere in this
  account's own job history.
- The original TabPFN incident that motivated the rule in the first place
  (`s6e9_ev_prediction`) was the opposite failure (missed a *real* crash),
  which is why the current implementation is conservative in exactly the
  wrong direction for these cases.

**Suggestion**: don't treat a single `error` read as fatal. Two concrete,
cheap options, not mutually exclusive:
- Require a *second* consecutive non-RUNNING read before declaring failure
  (i.e. extend to a 3rd check, or re-poll once more before returning),
  since every false-positive case here resolved to RUNNING/COMPLETE within
  the existing window.
- Distinguish "transient status-API flicker" from "real crash" by whether
  an error log actually exists — a real crash always has traceback
  content; the transient flicker cases had no error log to show. This is
  cheap (one more API call only on the ERROR branch) and directly
  falsifiable against the pattern already observed.
- Consider a longer default window specifically for jobs expected to run
  more than ~1 minute (most of these are CPU jobs in the few-minutes
  range, not the near-instant scripts the original 30/60s window was
  probably tuned against).

### 1b. Capacity checks trust a possibly-stale ledger, with a real TOCTOU race now that dispatch is concurrent

`kaggle_watchdog.py`'s `check_capacity()` reads the local ledger's *cached*
status, never a live API call. Confirmed stale in both directions
independently:
- `umud_muscle_architecture`, entry 1: one job the ledger listed as
  `running` (excluded from the active count when it shouldn't have been)
  and one listed as `error` (excluded when it was actually `RUNNING`) —
  *"two independent staleness errors happened to cancel out here... but
  that's luck, not a property of the mechanism."*
- `gemma_4_developer_agent`, §6/§9: ledger entries corrected by hand
  multiple times from direct `kaggle kernels status` checks before trusting
  them for a capacity decision.
- `ieee_ai_emulation`, exp5: explicitly did a full live `kaggle kernels
  list --mine` sweep before pushing specifically *because* the on-disk
  ledger was known-stale.

This was tolerable when one human-directed agent checked capacity
occasionally. It's a real correctness risk now that `orchestrator_dispatch.py`
can fire off genuinely concurrent dispatches, and — as these docs make
clear — **multiple independent Antigravity/Claude Code sessions are already
sharing the same real Kaggle account and the same `~/.autobot/kaggle_jobs.json`
file**, not just Autobot's own internal semaphore. Two processes can both
read "1/2 GPU slots used," both decide there's room, and both push,
oversubscribing the account's real 2-GPU cap. Atomic *writes* (already
fixed, Round 8) don't prevent this — the race is between the *read* and
the *write* in two different processes.

**Suggestion**:
- Add a TTL-based live-status refresh: before a capacity decision, refresh
  any ledger entry older than N seconds via one real `kaggle kernels
  status` call (cheap — only for entries near the cap, not a full sweep
  every time).
- Add real file locking (e.g. `filelock`, or an OS advisory lock) around
  the check-then-register sequence in `push_kernel()`, so "check capacity"
  and "register the new job" become one atomic critical section shared
  across processes — closing the TOCTOU gap that atomic writes alone don't
  close.

### 1c. `kernel_output()` / CLI log download crashes on Windows when the kernel log has non-cp1252 characters

`soil_grain_size_photos`, §5: both the CLI (`kaggle kernels output`) and
this repo's own `Kaggle.kernel_output()` crash with `UnicodeEncodeError:
'charmap' codec can't encode character '▉'` on Windows when a kernel's log
contains a Unicode progress-bar character (here: a `tqdm` download bar from
inside the kernel). The crash is on the **log file write**, not the
download — `submission.csv` still saves fine, only the `.log` silently
truncates to 0 bytes. Confirmed fix: `PYTHONUTF8=1` in the environment.

This is the same *category* of bug already fixed for kernel stdout printing
(`system_prompt.md`'s BOM/cp1252 warnings), but it's a different code path
(file write during log download) that evidently wasn't covered by the
existing fix.

**Suggestion**: set `PYTHONUTF8=1` (or open the log file with explicit
`encoding="utf-8", errors="replace"`) inside `kernel_output()`'s own
subprocess/file-write path, not left as something every caller has to
remember — exactly the same fix shape already applied elsewhere in this
codebase for the identical class of bug.

### 1d. Windows short-path (8.3) form breaks file reads

`soil_grain_size_photos`, §5: a path containing the short Windows form
(`C:\...\USER1~1\...`, which `kaggle kernels pull` and this session's own
Bash `/tmp` mapping both printed) causes "File does not exist" even though
the file is there — resolved by re-issuing the long form. Single
occurrence so far, but cheap and mechanical to guard against.

**Suggestion**: normalize short-name path components to their long form
before handing a path to any tool that isn't a plain shell (a few lines
via `ctypes.windll.kernel32.GetLongPathNameW` or equivalent).

---

## 2. Tool-call robustness for weaker models — the highest-leverage section for this specific ask

`gemma_4_developer_agent` §7 (exp1-3) is the single most directly relevant
document to "running on DeepSeek V4.1 / Nemotron" of everything in this
repo, because it's the one place a *smaller* model (`gemma-4-e4b-it`, an
"Elastic 4B", ~8B real params) was driven through a hand-rolled tool-call
protocol and its failure modes were logged in detail. What happened:

> "every tool call had a malformed args shape: `{"tool": "read_file",
> "filepath": "..."}` (flat, not nested under `"args"`)... across 4 repeated
> attempts the model never converged on the one shape that would've worked."

> "model ignored the fenced-JSON convention entirely and used bare
> pseudo-Python instead — `call: read_file(...)` — repeated verbatim every
> single turn with zero adaptation even after 'no valid tool_call block'
> nudges."

Two things stand out for weak-model orchestration specifically: the model
didn't converge on the right shape even after repeated failures, and it
didn't respond to a natural-language correction at all. Both are
exactly the failure modes to expect *more*, not less, of from smaller
open models than from Gemini 3.8 Flash.

**Suggestions, roughly in order of leverage:**

1. **Prefer native/constrained decoding over a prompted convention,
   wherever the serving stack allows it.** If DeepSeek V4.1 / Nemotron are
   served through vLLM/SGLang/TGI (or any stack with guided/grammar-constrained
   decoding — outlines, xgrammar, lm-format-enforcer), use JSON-schema-constrained
   generation or the model's own native function-calling template for
   Autobot's tool calls, instead of a hand-rolled "please emit fenced JSON"
   prompt. This is the single fix that would have prevented *both* exp1-3
   failure modes at the source: the decoder — not the model's compliance —
   guarantees schema-valid output. The gemma4 doc itself flags this as a
   real, un-taken option: `AutoProcessor`'s own sample code implied a native
   `tool_call_parser='gemma4'` convention existed, and the hand-rolled
   JSON-fence prompt was "fighting the model's natural output shape" the
   whole time.
2. **Where constrained decoding isn't available, make the parser
   permissive rather than the model precise.** Accept multiple observed
   argument shapes (nested dict, flat top-level kwargs, positional list)
   via `inspect.signature`-based coercion, and recognize multiple call
   syntaxes (fenced JSON, and a `name(args)` pseudo-Python fallback via
   `ast.literal_eval`) — this is literally what the gemma4 harness had to
   retrofit after real failures, and it's a general, reusable pattern
   worth building into Autobot's own tool-dispatch layer once, rather than
   rediscovering per-project.
3. **Never let one malformed call crash the whole run.** `read_file`
   crashed a task outright (`IsADirectoryError`) on an unvalidated path.
   Every `Computer.*` tool handler should validate its own inputs and
   return a structured error string, never raise — mirroring
   `DESIGN_PHILOSOPHY.md`'s "self-correction escalates" rule, but applied
   to the tool-dispatch layer, not just perception.
4. **Don't trust a natural-language nudge to fix a weak model's format
   drift.** The repeated-verbatim-zero-adaptation finding is the strongest
   evidence in this whole corpus that "tell the model it did it wrong" is
   not a reliable correction mechanism for a smaller model. Where #1/#2
   above aren't fully sufficient, prefer deterministic retry/coercion logic
   in the harness over another round of prompted correction.

---

## 3. "Looks done" is not "is done" — make the check automatic, not remembered

Two incidents are the same failure class, in different competitions, and
both are exactly the kind of thing a weaker model is less likely to
remember to check for itself:

- `gemma_4_developer_agent` §9, v4: kernel reported `COMPLETE`, produced a
  plausible `adapter_model.safetensors`, but an uncaught OOM during
  `.backward()` meant **zero optimizer steps actually ran** — the "trained"
  adapter was still at initialization. Nothing about the terminal status
  said so; only reading the raw log revealed it. Explicitly flagged as
  *"the one worth remembering longest... always check
  `n_real_optimizer_steps_completed`/`training_effectively_empty`... before
  trusting an adapter."*
- `ieee_traffic_flow` §11 (exp7): Task 4 silently defaulted to all-zeros
  because a sanity check on a nonexistent file path caused every corridor
  to be skipped — `.isna().any()` and shape checks both passed cleanly on
  an all-zero column. Already partially captured in
  `AUTOBOT_AUTONOMOUS_EXECUTION_CHALLENGES.md` §6 (active value assertions,
  not shape checks) — the gemma4 incident shows the same class of bug can
  hide behind a *process-level* completion status too, not just a
  data-level check.

**Suggestion**: build a small, reusable library of generic post-run
invariant checks into the harness itself — non-zero-value-ratio per output
column, row/id count against a freshly-verified manifest (not a possibly
stale sample file — see the UMUD finding below), and a
"did-real-work-happen" proof for training jobs specifically (steps
completed > 0, loss actually changed) — and run them automatically after
every dispatch that produces an artifact, rather than depending on the
model to think to write the assertion. This converts "a disciplined agent
adds this check" into "the harness always adds this check," which matters
more, not less, for a weaker model.

Related, smaller finding worth folding into the same helper:
`umud_muscle_architecture` §5 found `sample_submission.csv` can itself be a
truncated 2-row *format example*, not the real 309-row manifest — taking it
as authoritative silently truncated a submission. Fix already applied
locally (compare against the real on-disk test-file glob count); worth
generalizing into a shared "verify your row manifest against reality, not
just a sample file" helper other competition scripts can call, since the
doc explicitly flags this as *"a repo-wide risk, not specific to this
competition."*

---

## 4. CV/OOF validation has its own blind spot — a "domain coverage gate" to sit next to the SOTA gate

Two independent competitions burned real effort (one burned real
submissions) on a validation scheme that looked strong on paper but never
actually covered the axis the real test set varies on:

- `soil_grain_size_photos` §11: OOF EMD looked great (46.94); real LB score
  came in 72% worse (80.96) than *every individual unblended model's* OOF
  number. Root cause, found only by building a dedicated diagnostic kernel:
  **100% of training images were shot on Android phones, 100% of test
  images on iPhones — zero camera overlap.** No CV scheme built only from
  the training pool could ever have caught this, because there was no
  iPhone data to hold out.
- `ieee_ai_emulation` §9: OOF scaled MSE 9-18x more optimistic than the
  real LB score. Root cause: `GroupKFold`-by-site only tests generalization
  to new *sites* under the *same historical climate* — it never tests
  generalization to the future SSP126/585 climate scenarios the real test
  set requires, which is the actual point of the competition.

Both write-ups explicitly name this as the same category of gap. That's
two-for-two on "a good-looking OOF number was trusted past the point its
own validation scheme could actually vouch for" — exactly the kind of
mistake a model (weak or strong, but especially a weak one that can't
reason as carefully about what a CV scheme does and doesn't cover) is
likely to keep repeating without a structural check.

**Suggestion**: formalize a **Domain Coverage Gate** the same way
`AUTOBOT_AUTONOMOUS_EXECUTION_CHALLENGES.md` §6 formalized the Phase 0 SOTA
Gate — a deterministic, unskippable check, not a prompted reminder. Before
any OOF number is allowed to be treated as an LB proxy: enumerate every
categorical/metadata axis available (camera model, site id, scenario flag,
time period, etc.), compare train vs. test coverage on each, and loudly
flag any axis with zero or near-zero overlap. This wouldn't have *fixed*
either underlying problem (there's genuinely no iPhone data to train on),
but it would have surfaced the risk *before* spending a submission or a
long compute run on a model whose OOF number was structurally incapable of
predicting real performance — which is exactly what happened in both
cases.

---

## 5. Sharpen the SOTA gate itself: distinguish a real pipeline from a memorized answer sheet

`umud_muscle_architecture` §2 found the actual highest-scoring public
kernel on a small (309-image), fixed, long-lived public test set was not a
reproducible method at all — it was a frozen, hand-tuned, leaderboard-probed
answer table: a gzip+base64-encoded prediction blob, SHA-256-gated, that
the notebook's own text describes as embedding *"only a scored prediction
anchor... not competition images or labels."* A second, even higher-scoring
kernel was a **literal hardcoded CSV pasted into the notebook**, titled as
if it were a segmentation model. Both were correctly rejected by the human
task brief's own judgment (*"copying a hardcoded/leaked answer table would
produce a great local number with zero methodological content... isn't in
the spirit of the exercise"*) — but that judgment call was made by a human
reading the notebook carefully, not by the SOTA-gate mechanism itself.

This matters specifically for weaker models: "find the top-scoring public
kernel and build on it" is exactly the kind of instruction a weak model is
likely to satisfy with shallow pattern-matching (find highest score, copy
it) rather than the careful methodological read that caught this here.

**Suggestion**: extend the Phase 0 SOTA Gate with a cheap, deterministic
"does this actually run inference on the mounted data" heuristic before a
public kernel is adopted as a baseline — flag (not necessarily block, but
surface for explicit review) kernels with: no code path that reads the
actual test images/files at all; an unusually dense cluster of
many-decimal-place hardcoded constants; large embedded base64/binary blobs;
or explicit self-declared "anchor"/"frozen"/checksum-gated language. None
of these need to be perfectly reliable — a deterministic flag that forces
a second look beats nothing, especially for a model that can't reliably
make this judgment call unprompted.

---

## 6. Long-horizon multi-agent scaffolding — validate, then push further

### 6a. Already correct, now confirmed by real combat, not just design

`s6e9_ev_prediction` §2B/§5 independently arrived at — and this repo's
`orchestrator_dispatch.py` + `kaggle_watchdog.py`'s capacity semaphore now
implement — the same OpenClaw-inspired pattern: stateless turns over a
stateful on-disk workspace, strict per-lane isolation, and a global
concurrency throttle rather than unbounded parallel dispatch. §4I of the
same doc reports this actually working under real concurrent dispatch
(two subagents, one CPU one GPU competition, "zero interference"). Worth
treating this as validated architecture, not just a good idea — and worth
noting in `ROADMAP.md` that the real-world validation predates the formal
build.

### 6b. "Cognitive Thrashing" is the strongest argument yet for stricter lane isolation

`s6e9_ev_prediction` §2B names the exact failure mode to design against:
*"when an unassisted frontier model... attempts multi-tasking across 3
complex competitions, it suffers Cognitive Thrashing: Attention Saturation
[context fills with mismatched feature names/schemas], Optimistic Launch
Bias [treating an async call as mission-complete], Reinvent-the-Wheel
Syndrome."* This was observed on Gemini 3.8 Flash. A weaker model should be
expected to thrash *sooner*, not later.

**Suggestion**: make single-active-dispatch-per-project an explicit,
enforced rule, not just a convention: `orchestrator_dispatch.py` /
`ProjectRegistry` could refuse a new dispatch to a project whose last
dispatch hasn't been explicitly resolved (a `status: open_loop` flag,
cleared only by an explicit "this thread is closed" signal), mirroring the
already-named-but-not-yet-enforced *"Linear Step-Completion Invariant: never
transition to a new experiment until the active task is either verified
complete or actively patched and re-verified"* (`s6e9_ev_prediction` §2B).
Currently this is a rule the model is asked to follow; making the registry
itself refuse the violation removes it from the set of things a weak model
has to remember.

### 6c. A cheap, already-demonstrated pattern worth formalizing: read-only research subagents

`s6e9_ev_prediction` §5E: *"delegating historical solution mining to
Read-Only Research Scout Subagents that run locally at zero Kaggle compute
cost"* before any real compute is spent. Worth making this a first-class
dispatch mode (e.g. a `permission_mode`/tool-allowlist preset in
`orchestrator_dispatch.py` that structurally cannot call
`push_kernel`/`submit`), so a weaker model doing "just research" can't
accidentally spend real budget mid-thought — removing a failure mode by
construction rather than by instruction.

### 6d. Submission Economics as a structural gate, not just a written policy

`s6e9_ev_prediction` §4H's Submission Economics Law (high-allowance =
active-probe mode; low-allowance = Triple-Gate Approval: OOF beats current
best, format/distribution integrity passes, rank-correlation with the best
known-good anchor holds) is currently a human-written playbook rule.
**Suggestion**: encode the Triple-Gate check as an actual pre-submit
function in `kaggle_tool.py` (warn or require explicit override rather than
silently allowing) for any competition under, say, 5 submissions/day —
converting a policy a disciplined model follows into a check the harness
enforces regardless of model quality.

---

## 7. A pattern worth productizing as-is: the CHI-Bench blockers report

`competitions/chi_bench/THINKING_AND_DECISIONS.md` is a genuinely excellent
model of graceful failure for *any* long-horizon task, independent of model
strength: it states exactly which blockers exist (missing API key, no
Docker, a gated dataset needing the user's own HF approval), does
everything possible short of those blockers, and stops cleanly with a
specific, actionable ask — rather than guessing, faking a workaround, or
silently producing a degraded result.

**Suggestion**: this exact shape (`Status: BLOCKED on N items only you can
unblock`, followed by what was verified without them, then a numbered
unblock list) is worth turning into an actual helper — e.g.
`autobot.agent.blockers.report_blocked(items: list[Blocker])` — so a weaker
model has a template to fall into rather than needing to invent
well-structured blocker prose under pressure each time. Given the gemma4
and tool-calling findings above, giving a weak model a rigid structure to
fill in is generally safer than trusting it to freeform something
equally good.

One more note, not a code suggestion: CHI-Bench is *itself* a real,
externally-scored long-horizon agentic-orchestration benchmark (an agent +
harness driving multi-step healthcare workflows through MCP/REST tools,
graded by a pinned LLM judge plus deterministic checks) — once its
blockers clear, it's a plausible real eval target for Autobot's own
weak-model orchestration reliability, separate from anything Kaggle-specific.
Worth keeping in mind as a dogfooding opportunity, not something to chase
now.

---

## Summary table

| # | Finding | Evidence (competitions) | Confidence |
|---|---|---|---|
| 1a | Liveness grace window too short / single bad read treated as fatal | soil, gemma4, ieee_ai_emulation, umud, s6e9 (5x independent) | Very high |
| 1b | Ledger staleness + TOCTOU race under concurrent dispatch | umud, gemma4, ieee_ai_emulation | High |
| 1c | `kernel_output()` Windows unicode crash | soil | High (exact repro + fix known) |
| 1d | Windows short-path (8.3) resolution | soil | Medium (single occurrence) |
| 2 | Tool-call format brittleness on smaller models | gemma4 | Very high, directly on-target |
| 3 | Terminal status ≠ correctness; needs automatic invariant checks | gemma4, ieee_traffic_flow | High |
| 4 | OOF/CV blind to real test-distribution shift | soil, ieee_ai_emulation (2x independent) | High |
| 5 | SOTA gate can't yet detect memorized/leaked answer kernels | umud | Medium-high |
| 6 | Multi-agent lane isolation / cognitive thrashing | s6e9 | Already partly built; extend enforcement |
| 7 | Structured blocker-reporting as a reusable pattern | chi_bench | Low effort, clear win |

Nothing above has been implemented. Happy to build any subset of this next
— §1 (the four confirmed infra fixes) and §2 (tool-call robustness) are the
highest-leverage, lowest-risk starting points given the stated goal of
running long-horizon orchestration on DeepSeek V4.1 / Nemotron-class models.
