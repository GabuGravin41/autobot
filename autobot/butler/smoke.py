"""
`autobot butler smoke` — prove the whole loop works on THIS machine.

What it does (about 2-5 minutes; uses a little of your Claude or
Antigravity usage for one small coding turn):

  1. Tool check: claude / agy / code / latexmk / pdflatex / kaggle / git.
  2. LaTeX: compiles a two-line document with the same check the butler uses.
  3. Kaggle: authenticates and reads your job ledger (no pushes, no submits).
  4. The real thing: creates a throwaway git project with a deliberately
     broken function and a failing unit test, queues a coding task for the
     worker, and runs the butler until the task finishes. It passes only if
     Autobot's OWN check (the unit test) passes afterwards.

It uses its own database under ~/.autobot/smoke/, so a running butler
never picks these tasks up. The report is written to
~/.autobot/smoke_report.md — paste it back to Claude if anything fails.
"""
from __future__ import annotations

import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

BUGGY = '''def add(a, b):
    """Return the sum of a and b."""
    return a - b


def mean(values):
    """Arithmetic mean of a non-empty list."""
    return sum(values) / (len(values) + 1)
'''

TESTS = '''import unittest

from calc import add, mean


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mean(self):
        self.assertEqual(mean([2, 4, 6]), 4)


if __name__ == "__main__":
    unittest.main()
'''

TEX = r"""\documentclass{article}
\begin{document}
Autobot smoke test: $e^{i\pi} + 1 = 0$.
\end{document}
"""


def run_smoke(worker: str | None = None, keep: bool = False) -> int:
    from autobot.butler.checks import run_check
    from autobot.butler.cli import doctor_checks
    from autobot.butler.daemon import ButlerDaemon
    from autobot.butler.store import ButlerStore
    from autobot.butler.workers import pick_worker
    from autobot.integrations.cli_exec import run_cli
    from autobot.paths import autobot_home

    home = autobot_home()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = home / "smoke" / stamp
    root.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Autobot smoke test {stamp}", "",
                        f"- Python {sys.version.split()[0]} on {platform.platform()}",
                        f"- Autobot home: {home}", ""]
    results: dict[str, bool] = {}

    def section(title: str) -> None:
        print(f"\n== {title}")
        lines.append(f"## {title}")

    def note(ok: bool | None, text: str) -> None:
        tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
        print(f"[{tag}] {text}")
        lines.append(f"- **{tag}** {text}")

    # 1. tools
    section("1. Tools")
    for name, ok, detail in doctor_checks():
        note(ok if ok else None if "optional" in name or "VS Code" in name else False, f"{name}: {detail}")

    # 2. LaTeX
    section("2. LaTeX compile")
    tex_dir = root / "latex"
    tex_dir.mkdir()
    (tex_dir / "main.tex").write_text(TEX, encoding="utf-8")
    r = run_check({"type": "latex_compiles", "main": "main.tex", "timeout": 300}, tex_dir)
    results["latex"] = r.passed
    note(r.passed, f"latex_compiles: {r.detail[:300]}")

    # 3. Kaggle
    section("3. Kaggle credentials and ledger")
    try:
        from autobot.computer.kaggle_watchdog import KaggleJobLedger, make_kaggle_api
        make_kaggle_api()
        ledger = KaggleJobLedger()
        results["kaggle"] = True
        note(True, f"authenticated; ledger at {ledger.path} has {len(ledger.all())} job(s), "
                   f"{len(ledger.pending())} not finished")
    except Exception as e:
        results["kaggle"] = False
        note(False, f"Kaggle not usable: {e}")

    # 4. End-to-end coding task
    section("4. End-to-end: the butler fixes a bug with a real worker")
    chosen = pick_worker(worker)
    if chosen is None:
        note(False, "no worker CLI (claude or agy) on PATH — cannot run the end-to-end test")
        results["e2e"] = False
    else:
        proj = root / "project"
        proj.mkdir()
        (proj / "calc.py").write_text(BUGGY, encoding="utf-8")
        (proj / "test_calc.py").write_text(TESTS, encoding="utf-8")
        if shutil.which("git"):
            run_cli(["git", "init", "-q"], cwd=str(proj), timeout=60)
            run_cli(["git", "-c", "user.email=autobot@local", "-c", "user.name=Autobot", "add", "."], cwd=str(proj), timeout=60)
            run_cli(["git", "-c", "user.email=autobot@local", "-c", "user.name=Autobot", "commit", "-qm", "broken"],
                    cwd=str(proj), timeout=60)
        py = Path(sys.executable).name
        check_cmd = f'"{sys.executable}" -m unittest -q' if " " in sys.executable else f"{sys.executable} -m unittest -q"
        pre = run_check({"type": "command", "run": check_cmd}, proj)
        note(not pre.passed, f"before: the unit tests fail, as intended ({py})")

        store = ButlerStore(root / "smoke.db")
        task = store.add_task(
            "coding", "Smoke test: fix calc.py",
            "The unit tests in test_calc.py fail because calc.py has bugs. Fix calc.py so the tests pass. "
            "Do not change the tests.",
            project_dir=str(proj), worker=chosen,
            criteria=[{"type": "command", "run": check_cmd, "timeout": 300}],
            config={"max_attempts": 2, "worker_timeout_min": 15, "branch": False},
        )
        note(None, f"queued task #{task.id} for {chosen}; running the butler (inline, no manager model)…")
        t0 = time.time()
        try:
            daemon = ButlerDaemon(store=store, llm=None, log=lambda m: None)
            for _ in range(20):
                daemon.run_until_idle(max_ticks=50)
                t = store.get_task(task.id)
                if t.status in ("done", "failed", "needs_you", "paused", "cancelled"):
                    break
                if t.status == "waiting":
                    time.sleep(min(60, max(1, t.next_wake_at - time.time())))
            t = store.get_task(task.id)
            ok = t.status == "done"
            results["e2e"] = ok
            note(ok, f"task finished with status '{t.status}' after {time.time() - t0:.0f}s")
            post = run_check({"type": "command", "run": check_cmd}, proj)
            note(post.passed, "after: Autobot's own run of the unit tests " + ("passes" if post.passed else "still fails"))
            results["e2e"] = results["e2e"] and post.passed
            lines += ["", "### Task history", "```"]
            for e in store.events(task.id, limit=100):
                lines.append(f"{time.strftime('%H:%M:%S', time.localtime(e.ts))} {e.kind:<14} {e.message[:1500]}")
            lines.append("```")
            for ap in store.approvals("pending", task_id=task.id):
                lines.append(f"- Inbox item left: [{ap.kind}] {ap.summary}")
        except Exception:
            results["e2e"] = False
            note(False, "the butler crashed:\n```\n" + traceback.format_exc()[-3000:] + "\n```")

    section("Summary")
    for k, v in results.items():
        note(v, k)
    overall = all(results.get(k) for k in ("e2e",))
    note(overall, "Autobot can run a real task end-to-end on this machine" if overall
         else "End-to-end run did not succeed — see details above")
    report = home / "smoke_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nFull report: {report}")
    if not keep and overall:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if overall else 1
