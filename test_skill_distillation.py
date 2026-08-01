"""Behavioral test for SkillDistiller's write path (roadmap #1).

Loads skill_distiller.py directly so it runs without the autobot package's
heavier deps. Verifies the loop actually closes: distill -> save -> find ->
prompt context, plus the Windows-illegal-filename regression.
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parent / "autobot" / "knowledge" / "skill_distiller.py"
spec = importlib.util.spec_from_file_location("skill_distiller", SRC)
m = importlib.util.module_from_spec(spec)
sys.modules["skill_distiller"] = m
spec.loader.exec_module(m)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  — {detail}" if detail and not cond else ""))


def result(action_name, success, error=None):
    return SimpleNamespace(action_name=action_name, success=success, error=error)

def step(next_goal, results, url="https://overleaf.com"):
    return SimpleNamespace(
        agent_output=SimpleNamespace(next_goal=next_goal),
        action_results=results,
        url_after=url,
    )


tmp = Path(tempfile.mkdtemp())
try:
    d = m.SkillDistiller(skills_dir=tmp)
    GOAL = "Open Overleaf and compile the perovskite LaTeX paper"

    history = [
        step("Navigate to overleaf", [result("navigate", True)]),
        step("Click New Project", [result("click", False, "element 5 not found"),
                                   result("click", True)]),
        step("Paste LaTeX and compile", [result("computer_call", True),
                                         result("press_key", True)]),
    ]

    # ---- distill writes a skill ----
    skill = d.distill_from_run(goal=GOAL, history=history, result="compiled OK")
    check("distill returns a skill", skill is not None)
    check("skill file written", len(list(tmp.glob("*.json"))) == 1,
          f"files={list(tmp.glob('*.json'))}")

    # ---- only successful steps become proven steps ----
    check("proven steps recorded", len(skill.proven_steps) == 3, str(skill.proven_steps))
    check("failed action became a lesson",
          any("not found" in l for l in skill.lessons_learned), str(skill.lessons_learned))

    # ---- keywords are content words, not stopwords ----
    check("keywords exclude stopwords",
          "the" not in skill.keywords and "and" not in skill.keywords, str(skill.keywords))
    check("keywords include content words",
          "overleaf" in skill.keywords, str(skill.keywords))

    # ---- THE CLOSED LOOP: a later run with the same goal finds it ----
    found = d.find_matching_skill(GOAL)
    check("saved skill is findable", found is not None and found.name == skill.name)

    ctx = d.get_skill_prompt_context(GOAL)
    check("prompt context is non-empty", bool(ctx) and "PREVIOUSLY LEARNED SKILL" in ctx)
    check("prompt context lists proven steps", "Navigate to overleaf" in ctx)

    # ---- unrelated goal does NOT match ----
    check("unrelated goal does not match",
          d.find_matching_skill("check the weather in Nairobi") is None)


    # ---- second success bumps count, keeps shorter path ----
    shorter = [step("Do it in one go", [result("navigate", True)])]
    again = d.distill_from_run(goal=GOAL, history=shorter, result="ok")
    check("success_count incremented", again.success_count == 2, str(again.success_count))
    check("shorter proven path preferred", len(again.proven_steps) == 1, str(again.proven_steps))
    check("still only one file", len(list(tmp.glob("*.json"))) == 1)

    # ---- REGRESSION: Windows-illegal filename characters ----
    nasty = "Research: perovskites? <part 1> | v2 *draft*"
    s2 = d.distill_from_run(goal=nasty, history=history, result="ok")
    check("illegal filename chars handled", s2 is not None and (tmp / f"{d._safe_name(nasty)}.json").exists())

    # ---- REGRESSION: one generic word in common must not trigger a match ----
    # find_matching_skill used to return on the FIRST keyword hit, so a skill
    # with keywords like ["list","windows"] fired on "list the files...",
    # injecting irrelevant proven-steps and stale lessons into an unrelated
    # task. Misleading guidance is worse than none.
    generic = d.distill_from_run(
        goal="list the open windows",
        history=[step("List windows", [result("computer_call", True)])],
        result="ok")
    check("generic skill saved for the test", generic is not None)
    check("single shared word does not match",
          d.find_matching_skill("list the files in my project directory") is None,
          str(getattr(d.find_matching_skill("list the files in my project directory"), "name", None)))
    check("genuinely matching goal still matches",
          d.find_matching_skill("list the open windows on screen") is not None)

    # ---- no successful steps -> nothing saved ----
    before = len(list(tmp.glob("*.json")))
    none_ok = d.distill_from_run(
        goal="a doomed goal", history=[step("try", [result("click", False, "nope")])], result="failed")
    check("all-failed run saves nothing",
          none_ok is None and len(list(tmp.glob("*.json"))) == before)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
