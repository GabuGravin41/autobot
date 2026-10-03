"""`autobot butler ...` commands against an isolated AUTOBOT_HOME."""
from __future__ import annotations

import json

from autobot.butler.cli import main
from autobot.butler.store import ButlerStore


def test_add_list_show_inbox_answer(tmp_path, capsys):
    proj = tmp_path / "p"
    proj.mkdir()
    assert main(["add", "coding", "Fix the parser bug", "--dir", str(proj), "--check", "pytest -q",
                 "--worker", "claude_code", "--priority", "2"]) == 0
    out = capsys.readouterr().out
    assert "Task #1 queued" in out and "isn't running" in out

    store = ButlerStore()
    t = store.get_task(1)
    assert t.criteria == [{"type": "command", "run": "pytest -q"}] and t.criteria_source == "user"
    assert t.priority == 2 and t.project_dir == str(proj.resolve())

    main(["list"])
    assert "Fix the parser bug" in capsys.readouterr().out

    q = store.add_approval(1, "question", "Which branch?")
    main(["inbox"])
    assert "Which branch?" in capsys.readouterr().out
    assert main(["answer", str(q.id), "main"]) == 0
    assert store.get_approval(q.id).response == "main"
    assert main(["answer", str(q.id), "again"]) == 1           # can't decide twice

    main(["show", "1"])
    out = capsys.readouterr().out
    assert "pytest -q" in out and "History:" in out


def test_add_rejects_shell_operators(tmp_path, capsys):
    assert main(["add", "coding", "x", "--dir", str(tmp_path), "--check", "pytest && git push"]) == 2


def test_learning_options_and_json_check(tmp_path, capsys):
    spec = tmp_path / "spec.md"
    spec.write_text("Audience: KMO round 2")
    main(["add", "learning", "Inequalities module", "--dir", str(tmp_path), "--outline", "AM-GM; Cauchy-Schwarz",
          "--spec-file", str(spec), "--check", json.dumps({"type": "file_exists", "path": "main.pdf"})])
    t = ButlerStore().get_task(1)
    assert t.config["outline"] == ["AM-GM", "Cauchy-Schwarz"] and "KMO" in t.config["spec"]
    assert t.criteria == [{"type": "file_exists", "path": "main.pdf"}]


def test_pause_resume_cancel_and_status(tmp_path, capsys):
    main(["add", "research", "Who funds olympiad programs in East Africa?"])
    store = ButlerStore()
    store.merge_state(1, phase="awaiting_user", resume_phase="work")
    store.add_approval(1, "question", "q")
    main(["pause", "1"])
    assert store.get_task(1).status == "paused"
    main(["resume", "1"])
    t = store.get_task(1)
    assert t.status == "queued" and t.state["phase"] == "work"
    assert store.approvals("pending", task_id=1) == []
    main(["cancel", "1"])
    assert store.get_task(1).status == "cancelled"
    main(["status"])
    assert "NOT running" in capsys.readouterr().out
