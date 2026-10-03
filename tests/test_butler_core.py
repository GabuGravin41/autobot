"""Butler store, checks and daemon mechanics (no real workers)."""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from autobot.butler import checks
from autobot.butler.daemon import AlreadyRunning, ButlerDaemon
from autobot.butler.store import DONE, NEEDS_YOU, QUEUED, WAITING, WORKING, ButlerStore

PY = sys.executable


@pytest.fixture
def store(tmp_path):
    return ButlerStore(tmp_path / "b.db")


# ── store ────────────────────────────────────────────────────────────────────

class TestStore:
    def test_add_and_get_keeps_intent_verbatim(self, store):
        intent = "Fix the  flaky test.\nKeep the API stable!"
        t = store.add_task("coding", "", intent, project_dir="/x")
        assert store.get_task(t.id).intent == intent
        assert t.title == "Fix the  flaky test.\nKeep the API stable!"[:80].strip()

    def test_empty_intent_rejected(self, store):
        with pytest.raises(ValueError):
            store.add_task("coding", "t", "   ")

    def test_claim_due_is_exclusive_across_threads(self, store):
        for i in range(20):
            store.add_task("coding", f"t{i}", "x")
        claimed: list[int] = []
        lock = threading.Lock()

        def worker():
            for _ in range(10):
                got = store.claim_due(limit=1)
                with lock:
                    claimed.extend(t.id for t in got)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert sorted(claimed) == sorted(set(claimed))   # nobody claimed a task twice
        assert len(claimed) == 20

    def test_claim_respects_wake_time_and_priority(self, store):
        later = store.add_task("coding", "later", "x", start_at=time.time() + 3600)
        low = store.add_task("coding", "low", "x", priority=9)
        high = store.add_task("coding", "high", "x", priority=1)
        got = store.claim_due(limit=5)
        assert [t.id for t in got] == [high.id, low.id]
        assert store.get_task(later.id).status == QUEUED

    def test_recover_orphans(self, store):
        t = store.add_task("coding", "t", "x")
        store.update_task(t.id, status=WORKING)
        assert store.recover_orphans() == [t.id]
        assert store.get_task(t.id).status == QUEUED
        assert any(e.kind == "recovered" for e in store.events(t.id))

    def test_decision_only_once(self, store):
        a = store.add_approval(None, "question", "q?")
        store.decide(a.id, True, response="yes")
        with pytest.raises(ValueError):
            store.decide(a.id, False)
        assert store.get_approval(a.id).response == "yes"
        assert [x.id for x in store.unhandled_decisions()] == [a.id]
        store.mark_handled(a.id)
        assert store.unhandled_decisions() == []

    def test_merge_state(self, store):
        t = store.add_task("coding", "t", "x")
        store.merge_state(t.id, phase="work", session="s1")
        store.merge_state(t.id, feedback="f")
        assert store.get_task(t.id).state == {"phase": "work", "session": "s1", "feedback": "f"}

    def test_cooldowns(self, store):
        store.set_cooldown("claude_code", time.time() + 60, "limit")
        assert store.cooldown_until("claude_code") > time.time()
        assert store.cooldown_until("antigravity") == 0


# ── checks ───────────────────────────────────────────────────────────────────

class TestChecks:
    def test_command_pass_and_fail(self, tmp_path):
        ok = checks.run_check({"type": "command", "run": f'"{PY}" -c "print(42)"', "contains": "42"}, tmp_path)
        bad = checks.run_check({"type": "command", "run": f'"{PY}" -c "import sys; sys.exit(2)"'}, tmp_path)
        assert ok.passed and not bad.passed and "exit 2" in bad.detail

    def test_missing_program_fails_not_crashes(self, tmp_path):
        r = checks.run_check({"type": "command", "run": "no-such-program-xyz --v"}, tmp_path)
        assert not r.passed and "not found" in r.detail

    def test_file_exists(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        assert checks.run_check({"type": "file_exists", "path": "a.txt"}, tmp_path).passed
        assert not checks.run_check({"type": "file_exists", "path": "b.txt"}, tmp_path).passed

    def test_path_cannot_escape_project(self, tmp_path):
        r = checks.run_check({"type": "file_exists", "path": "../../etc/passwd"}, tmp_path)
        assert not r.passed and "escapes" in r.detail

    def test_csv_rows_semicolon_and_zero_column(self, tmp_path):
        # The UMUD sample file was ';'-delimited; the ODME column collapsed to all zeros.
        (tmp_path / "s.csv").write_text("id;flow\n1;0\n2;0\n3;0\n", encoding="utf-8-sig")
        r = checks.run_check({"type": "csv_rows", "path": "s.csv", "equals": 3}, tmp_path)
        assert r.passed
        r = checks.run_check({"type": "csv_rows", "path": "s.csv", "nonzero_column": "flow"}, tmp_path)
        assert not r.passed and "all zeros" in r.detail
        r = checks.run_check({"type": "csv_rows", "path": "s.csv", "min": 309}, tmp_path)
        assert not r.passed

    def test_contains_and_min_count(self, tmp_path):
        (tmp_path / "m.md").write_text("# Eigenvalues\nRemarks\nsee https://a.org and http://b.org")
        assert checks.run_check({"type": "contains", "path": "m.md", "patterns": ["eigenvalues", "remarks"]}, tmp_path).passed
        r = checks.run_check({"type": "contains", "path": "m.md", "patterns": ["Jordan form"]}, tmp_path)
        assert not r.passed and "Jordan form" in r.detail
        assert checks.run_check({"type": "min_count", "path": "m.md", "pattern": r"https?://", "min": 2}, tmp_path).passed
        assert not checks.run_check({"type": "min_count", "path": "m.md", "pattern": r"https?://", "min": 3}, tmp_path).passed

    @pytest.mark.skipif(not (shutil.which("latexmk") or shutil.which("pdflatex")), reason="no TeX")
    def test_latex_compiles_and_reports_errors(self, tmp_path):
        (tmp_path / "main.tex").write_text(r"\documentclass{article}\begin{document}Hi $x^2$\end{document}")
        assert checks.run_check({"type": "latex_compiles", "main": "main.tex"}, tmp_path).passed
        (tmp_path / "bad.tex").write_text(r"\documentclass{article}\begin{document}\undefinedmacro\end{document}")
        r = checks.run_check({"type": "latex_compiles", "main": "bad.tex"}, tmp_path)
        assert not r.passed and "undefined" in r.detail.lower()

    @pytest.mark.parametrize("check,ok", [
        ({"type": "command", "run": "pytest -q"}, True),
        ({"type": "command", "run": "python -m pytest tests"}, True),
        ({"type": "command", "run": "npm test"}, True),
        ({"type": "command", "run": "npm install evil"}, False),
        ({"type": "command", "run": "rm -rf build"}, False),
        ({"type": "command", "run": "python -c \"import os; os.remove('x')\""}, False),
        ({"type": "command", "run": "python -m http.server"}, False),
        ({"type": "command", "run": "pytest -q && git push"}, False),
        ({"type": "latex_compiles", "main": "main.tex"}, True),
        ({"type": "unknown"}, False),
    ])
    def test_validate_model_proposed(self, check, ok):
        assert (checks.validate(check, proposed=True) is None) == ok

    def test_user_checks_may_use_any_program_but_no_shell_ops(self):
        assert checks.validate({"type": "command", "run": "my_script.bat --fast"}) is None
        assert checks.validate({"type": "command", "run": "a | b"}) is not None


# ── daemon ───────────────────────────────────────────────────────────────────

class _Playbook:
    """Minimal playbook double: marks tasks done, can be told to crash or sleep."""

    def __init__(self, store, crash_ids=(), sleep=0.0):
        self.store, self.crash_ids, self.sleep = store, set(crash_ids), sleep
        self.running = 0
        self.max_running = 0
        self._lock = threading.Lock()

    def step(self, task):
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            time.sleep(self.sleep)
            if task.id in self.crash_ids:
                raise RuntimeError("boom")
            self.store.update_task(task.id, status=DONE)
        finally:
            with self._lock:
                self.running -= 1

    def on_decision(self, a):
        self.store.mark_handled(a.id)


class TestDaemon:
    def test_run_until_idle_and_crash_isolation(self, store):
        a = store.add_task("coding", "a", "x")
        b = store.add_task("coding", "b", "x")
        d = ButlerDaemon(store=store, playbook=_Playbook(store, crash_ids={a.id}), llm=None)
        d.run_until_idle()
        assert store.get_task(b.id).status == DONE
        ta = store.get_task(a.id)
        assert ta.status == QUEUED and ta.next_wake_at > time.time() + 60   # backed off, not dead
        assert any("boom" in e.message for e in store.events(a.id))

    def test_step_that_forgets_to_set_status_is_not_stranded(self, store):
        class Lazy(_Playbook):
            def step(self, task):
                pass
        t = store.add_task("coding", "t", "x")
        ButlerDaemon(store=store, playbook=Lazy(store), llm=None).run_until_idle()
        assert store.get_task(t.id).status == QUEUED

    def test_concurrency_is_bounded(self, store):
        from concurrent.futures import ThreadPoolExecutor
        for i in range(6):
            store.add_task("coding", f"t{i}", "x")
        pb = _Playbook(store, sleep=0.3)
        d = ButlerDaemon(store=store, playbook=pb, llm=None, max_concurrent=2)
        d._executor = ThreadPoolExecutor(max_workers=2)
        t0 = time.time()
        while time.time() - t0 < 10 and any(t.status != DONE for t in store.list_tasks(None)):
            d.tick()
            time.sleep(0.05)
        d._executor.shutdown(wait=True)
        assert all(t.status == DONE for t in store.list_tasks(None))
        assert pb.max_running == 2

    def test_single_instance(self, store):
        d1 = ButlerDaemon(store=store, playbook=_Playbook(store), llm=None)
        d2 = ButlerDaemon(store=store, playbook=_Playbook(store), llm=None)
        d1.acquire_single_instance()
        try:
            with pytest.raises(AlreadyRunning):
                d2.acquire_single_instance()
        finally:
            d1._lock.release()

    def test_stop_file_ends_run_forever(self, store):
        from autobot.paths import autobot_home
        d = ButlerDaemon(store=store, playbook=_Playbook(store), llm=None, tick_seconds=0.05)
        th = threading.Thread(target=d.run_forever)
        th.start()
        time.sleep(0.3)
        (autobot_home() / "butler.stop").write_text("x")
        th.join(5)
        assert not th.is_alive()
        assert store.get_meta("daemon")["pid"] is None
