# Autobot — Test Plan (Kaggle + Claude Code autonomy push)

Two layers: fix-level tests that must pass before anything ships, and a
graduated live-system test plan that increases stakes one step at a time.
Do not skip ahead — each step exists because an earlier failure mode was
identified, and each is cheap to run.

## Layer 1 — Code-level tests (before touching the device again)

1. **`list_competitions()` regression test** — mock a response object with
   a `.competitions` attribute (the new kagglesdk shape) and confirm the
   method unwraps it correctly and preserves all six fields
   (ref/title/description/deadline/category/reward). Also test the
   defensive fallback path: a plain list input (old SDK shape) should still
   work via `getattr(response, "competitions", response)`.
2. **`get_leaderboard()` regression test** — blocked until the real
   replacement method and its return shape are confirmed (see
   ANTIGRAVITY_BRIEFING.md's diagnostic). Once confirmed, mock that exact
   shape and test the top-20 truncation and field mapping.
3. **`ref`-as-URL follow-through** — check whether `download_data()`,
   `submit()`, and `get_leaderboard()` assume `competition` is a bare slug
   (e.g. `'titanic'`) rather than a full URL
   (`'https://www.kaggle.com/competitions/titanic'`). If any of them build
   a URL or path from the slug internally, a `list_competitions()`-derived
   `ref` fed straight into them could break silently. Write a test that
   calls each of these three methods with a full-URL-style ref and confirms
   either (a) it works, or (b) `kaggle_tool.py` normalizes the ref to a bare
   slug before passing it down — pick one deliberately, don't leave it
   ambiguous.
4. **Full existing suite** — run all of `tests/` (164+ tests as of Round 6)
   and confirm nothing regressed. This is a hard gate before any device
   push.
5. **`autobot --doctor` self-check** — re-run `run_all()` in a clean
   environment and confirm it still exits 0/nonzero correctly and every
   FAIL includes an actionable fix (this is already covered by
   `test_diagnostics.py`, re-run it after any diagnostics.py edits).

## Layer 2 — Live device smoke tests (cheapest first)

Run these in order on Dalton's real machine. Stop and fix forward if any
step fails — do not proceed to a more expensive/risky step on a shaky
foundation.

6. **`autobot --doctor`** — confirm every check that matters for this test
   run is OK: an LLM key configured, Kaggle credentials found
   (`kaggle.json` present), Claude Code CLI found (`claude` on PATH),
   Antigravity CLI found (`agy` on PATH, now also relevant since Antigravity
   itself is joining), Chrome present, unattended mode correctly reported
   as off by default.
7. **Trivial non-Kaggle smoke test** — `autobot "open Notepad and type
   hello autobot"`. Confirms the core perceive/decide/act loop, the LLM
   client, and UI Automation control all work end-to-end before any
   external API is involved.
8. **Kaggle read-only smoke test** — a task that only calls
   `computer.kaggle.list_competitions` / `pull_kernel` / `kernel_status` /
   `kernel_output` (all SAFE-tier, no approval friction) against a real,
   low-stakes community competition kernel. Confirms real Kaggle API
   round-trips work through `kaggle_tool.py` post-fix, not just the mocked
   unit tests.
9. **Claude Code round-trip smoke test** — `computer.claude_code.run(...)`
   against the kernel pulled in step 8, `permission_mode="plan"` (read-only,
   the default) first. Confirms the headless CLI bridge works and returns
   something Autobot can act on.

## Layer 3 — First real supervised competition run

10. **One low-stakes community competition, fully supervised** — the
    complete sequence from the system prompt's documented workflow:
    `pull_kernel` → `claude_code.run` (now with `permission_mode="acceptEdits"`,
    which is DANGER-tier and will pause for approval — confirm that pause
    actually happens) → `push_kernel` → `kernel_status` polling →
    `kernel_output` → hand results back to Claude Code for next-step
    strategy. Approval mode: default (balanced), NOT unattended — Dalton
    approves each gated step live. Do not call `submit()` yet.
11. **First real `submit()` call** — same competition, still supervised.
    Confirm it requires live approval in every mode as designed (this
    should be provable from `approval.py`'s IRREVERSIBLE classification,
    but confirm it behaves that way in practice, not just in the test
    suite).
12. **Score/leaderboard read-back** — once a submission has a score, use
    the now-fixed `get_leaderboard()` to confirm Autobot can read its own
    standing and feed it back to Claude Code for a next-iteration decision.

## Layer 4 — Widening scope (only after Layer 3 succeeds cleanly)

13. **Second community competition, still supervised** — confirms step 10
    generalizes rather than having been tuned to one specific competition's
    quirks.
14. **First frontier competition, supervised** — higher-stakes data
    volumes, longer kernel runtimes, likely GPU quota considerations. Watch
    specifically for timeout/polling behavior on longer-running kernels.
15. **Antigravity as a second executor** — since Antigravity now has native
    device access and is being looped in, verify Autobot's existing
    `antigravity_bridge.py` CLI integration against a real task the same way
    Claude Code was verified in steps 9-10, so both coding-agent backends
    are proven, not just one.
16. **First unattended run** — only after 13-15 are clean. Set
    `AUTOBOT_UNATTENDED=1` on a single low-stakes community competition,
    with the Kaggle-specific unattended-autonomy policy from Round 6 doing
    the gating (re-read `approval.py`'s unattended rules immediately before
    this test, since this is the first time they'll run against a real
    external system rather than the test suite's mocks). Watch it end to
    end rather than walking away, even though the point is that it
    theoretically doesn't need supervision — first run of a new autonomy
    tier always gets watched.
17. **First unattended frontier-competition run** — the actual target
    scenario from Dalton's original ask. Only after every step above is
    clean.

## Notes

- Steps 1-5 are blocking: don't run any Layer 2+ step against a
  `kaggle_tool.py` that hasn't had its confirmed bugs fixed and tested.
- Steps 6-9 are cheap and fast — no reason to skip them even if confident,
  since they're what would have caught the current SDK-drift bug months
  earlier had they existed.
- Regenerate the Kaggle API token before Layer 2 if that hasn't happened
  yet (the live token was exposed in chat earlier this session).
