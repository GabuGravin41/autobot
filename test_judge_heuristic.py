"""Behavioral test for JudgeAgent's fast heuristic pre-check.

Root cause this guards against: the heuristic ran BEFORE any LLM call and
matched loose substrings ("error:", "timed out", "unable to", "could not
complete") ANYWHERE in the result text — including inside the agent's own
free-form description of what it actually accomplished. A run that
researched and wrote a paper discussing, say, measurement error or a
switching threshold would get silently marked FAILED before the LLM judge
ever saw it — directly relevant to the Overleaf/Grok benchmark, whose
whole point is producing exactly that kind of scientific prose.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


from autobot.agent.judge import JudgeAgent  # noqa: E402

judge = JudgeAgent(llm_client=None, model="fake")  # heuristic path never touches llm_client


# ── THE regression: successful, technical/scientific content ───────────────
scientific_success_cases = [
    "Successfully compiled the paper. The abstract discusses the measurement "
    "error of the switching threshold, which was reduced to within tolerance.",
    "Task complete. The device timed out at 10ns as expected per the design "
    "spec, demonstrating the intended switching behavior.",
    "Done. The model was unable to converge below 1e-6 loss by design — this "
    "matches the theoretical noise floor described in the paper.",
    "Compiled the LaTeX document. It reports that we could not complete the "
    "measurement below the quantum limit, which is the expected physical result.",
    "Successfully completed. Error: 0.02 (2%) matches the target precision "
    "specified in the device requirements section.",
]
for i, text in enumerate(scientific_success_cases):
    result = judge._fast_heuristic_check("write a paper about X", text)
    check(f"scientific success case {i+1} not misclassified by heuristic",
          result is None or result.success is True,
          f"heuristic said success={getattr(result, 'success', None)}: {text[:60]}...")


# ── Genuine failures must still be caught precisely ─────────────────────────
def test_real_error_wrapper():
    text = "Error: Connection refused\n\nTraceback:\nRuntimeError: ..."
    r = judge._fast_heuristic_check("goal", text)
    check("real 'Error:' wrapper still caught", r is not None and r.success is False)

def test_real_traceback():
    text = "Something happened.\nTraceback (most recent call last):\n  File..."
    r = judge._fast_heuristic_check("goal", text)
    check("real traceback still caught", r is not None and r.success is False)

def test_max_steps_hit():
    text = "Agent ran 25 steps without calling 'done'.\nLast steps:\n..."
    r = judge._fast_heuristic_check("goal", text)
    check("real max-steps summary still caught", r is not None and r.success is False)

def test_llm_unavailable():
    text = "The LLM failed 3 times in a row, so the agent cannot make progress."
    r = judge._fast_heuristic_check("goal", text)
    check("real LLMUnavailableError text still caught", r is not None and r.success is False)

def test_empty_result():
    r = judge._fast_heuristic_check("goal", "")
    check("truly empty result still caught", r is not None and r.success is False)

def test_short_real_success_not_penalized():
    # Previously: anything under 20 chars without the literal word "done"
    # failed automatically. "Sent the email." is a complete, real success.
    r = judge._fast_heuristic_check("send an email", "Sent the email.")
    check("short-but-real success is NOT auto-failed (goes to LLM or matches success phrase)",
          r is None or r.success is True, str(r))

def test_prior_judge_success_shortcut():
    text = "did the thing\n\n[Judge Verification: SUCCESS] looks right"
    r = judge._fast_heuristic_check("goal", text)
    check("prior judge success shortcut still works", r is not None and r.success is True)

def test_explicit_success_phrase():
    r = judge._fast_heuristic_check("goal", "Task complete, everything worked.")
    check("explicit success phrasing still caught", r is not None and r.success is True)


test_real_error_wrapper()
test_real_traceback()
test_max_steps_hit()
test_llm_unavailable()
test_empty_result()
test_short_real_success_not_penalized()
test_prior_judge_success_shortcut()
test_explicit_success_phrase()

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
