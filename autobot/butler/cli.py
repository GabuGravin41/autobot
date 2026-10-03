"""
`autobot butler ...` — give the butler work, check on it, answer it.

  autobot butler run                         start the daemon (leave this window open)
  autobot butler stop                        ask a running daemon to stop
  autobot butler status                      is it running, what's queued, what needs you
  autobot butler add LANE "what you want" [options]
  autobot butler list [--all]
  autobot butler show ID
  autobot butler inbox                       approvals and questions waiting for you
  autobot butler approve ID [--note TEXT]
  autobot butler reject ID [--note TEXT]
  autobot butler answer ID "your answer"
  autobot butler pause|resume|cancel ID
  autobot butler doctor                      check claude / agy / code / latexmk / kaggle
  autobot butler gmail-auth                  connect Gmail once (see GMAIL_SETUP.md)
  autobot butler smoke [--worker claude_code|antigravity]
                                             real end-to-end test on this machine

Lanes: coding, document, learning, research, kaggle, email.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from autobot.butler.lanes import LANES
from autobot.butler.store import (
    ACTIVE_STATUSES,
    CANCELLED,
    PAUSED,
    QUEUED,
    ButlerStore,
)


def _fmt_age(ts: float | None) -> str:
    if not ts:
        return "never"
    d = time.time() - ts
    if d < 90:
        return f"{int(d)}s ago"
    if d < 5400:
        return f"{int(d / 60)}m ago"
    return f"{d / 3600:.1f}h ago"


def _parse_check(text: str) -> dict:
    """--check accepts JSON ({"type": ...}) or a plain command ("pytest -q")."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    return {"type": "command", "run": text}


def cmd_add(store: ButlerStore, a: argparse.Namespace) -> int:
    from autobot.butler import checks as checks_mod
    criteria = [_parse_check(c) for c in (a.check or [])]
    for c in criteria:
        why = checks_mod.validate(c, proposed=False)
        if why:
            print(f"Invalid check {c}: {why}")
            return 2
    config: dict = json.loads(a.config_json) if a.config_json else {}
    if a.max_attempts:
        config["max_attempts"] = a.max_attempts
    if a.read_only:
        config["writes"] = False
    if a.main_tex:
        config["main_tex"] = a.main_tex
    if a.format:
        config["format"] = a.format
    if a.outline:
        config["outline"] = [x.strip() for x in a.outline.split(";") if x.strip()]
    if a.style:
        config["style"] = a.style
    if a.spec_file:
        config["spec"] = Path(a.spec_file).read_text(encoding="utf-8")
    if a.competition:
        config["competition"] = a.competition
    if a.rounds:
        config["max_rounds"] = a.rounds
    if a.no_branch:
        config["branch"] = False
    if a.timeout_min:
        config["worker_timeout_min"] = a.timeout_min
    if a.mode:
        config["mode"] = a.mode
    if a.every:
        config["every_minutes"] = a.every
    if a.query:
        config["query"] = a.query
    if a.open_vscode:
        config["open_vscode"] = True
    project = str(Path(a.dir).expanduser().resolve()) if a.dir else None
    task = store.add_task(
        lane=a.lane, title=a.title or a.intent.strip().splitlines()[0][:80], intent=a.intent,
        project_dir=project, worker=a.worker, criteria=criteria, config=config, priority=a.priority,
    )
    print(f"Task #{task.id} queued in lane '{task.lane}'" + (f" for {project}" if project else "") + ".")
    if not criteria:
        lane = LANES[a.lane]
        if lane.default_checks(config):
            print("  Checks: lane defaults (" + "; ".join(checks_mod.describe(c) for c in lane.default_checks(config)) + ")")
        elif lane.propose_criteria:
            print("  No checks given: the worker will propose some first; unsafe ones are dropped.")
        elif not lane.rounds:
            print("  No checks: you'll be asked to review the result.")
    daemon = store.get_meta("daemon") or {}
    if not daemon.get("pid"):
        print("  The butler isn't running yet. Start it with:  autobot butler run")
    return 0


def cmd_list(store: ButlerStore, a: argparse.Namespace) -> int:
    tasks = store.list_tasks(None if a.all else ACTIVE_STATUSES)
    if not tasks:
        print("No tasks." if a.all else "No active tasks. (autobot butler list --all to include finished ones)")
        return 0
    for t in tasks:
        phase = t.state.get("phase", "start")
        wake = ""
        if t.status == "waiting" and t.next_wake_at > time.time():
            wake = f" (wakes in {int((t.next_wake_at - time.time()) / 60) + 1}m)"
        print(f"#{t.id:<4} {t.status:<10} {t.lane:<9} {phase:<13} {t.worker or '-':<12} {t.title[:60]}{wake}")
    return 0


def cmd_show(store: ButlerStore, a: argparse.Namespace) -> int:
    t = store.get_task(a.id)
    if not t:
        print(f"No task #{a.id}")
        return 1
    print(f"#{t.id} [{t.status}] {t.title}\nLane: {t.lane}   Worker: {t.worker or 'auto'}   Phase: {t.state.get('phase', 'start')}"
          f"   Attempts: {t.attempts}\nFolder: {t.state.get('cwd') or t.project_dir or '-'}\n\nIntent:\n{t.intent}\n")
    if t.criteria:
        from autobot.butler.checks import describe
        print(f"Checks ({t.criteria_source}):")
        for c in t.criteria:
            print(f"  - {describe(c)}")
        print()
    if t.result:
        print(f"Result:\n{t.result}\n")
    print("History:")
    for e in store.events(t.id, limit=a.limit):
        msg = e.message.replace("\n", "\n      ")
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(e.ts))} {e.kind:<14} {msg[:1500]}")
    return 0


def cmd_inbox(store: ButlerStore, a: argparse.Namespace) -> int:
    items = store.approvals("pending")
    if not items:
        print("Inbox is empty.")
        return 0
    for ap in items:
        task = f"task #{ap.task_id}" if ap.task_id else "general"
        print(f"[{ap.id}] {ap.kind.upper():<14} ({task}, {_fmt_age(ap.created_at)})\n    {ap.summary}")
        detail = ap.payload.get("summary") or ap.payload.get("result")
        if detail:
            text = str(detail)
            if len(text) > 2500:
                text = text[:2500] + f"\n… (full text: autobot butler show {ap.task_id})"
            print("    " + text.replace("\n", "\n    "))
        verbs = {"question": "answer ID \"...\" | reject ID", "review": "approve ID | reject ID --note \"what to fix\"",
                 "kaggle_submit": "approve ID | reject ID", "done": "approve ID (dismiss)",
                 "send_email": "approve ID (sends it) | reject ID (keeps it as a Gmail draft)",
                 "digest": "approve ID (dismiss)"}
        print(f"    -> autobot butler {verbs.get(ap.kind, 'approve ID | reject ID')}\n")
    return 0


def cmd_decide(store: ButlerStore, a: argparse.Namespace, approved: bool, response: str | None) -> int:
    try:
        ap = store.decide(a.id, approved, via="cli", response=response)
    except (KeyError, ValueError) as e:
        print(e)
        return 1
    print(f"{'Approved' if approved else 'Rejected'} #{ap.id} ({ap.kind}).")
    if ap.kind == "kaggle_submit" and approved:
        print("The butler will submit on its next tick (within a few seconds if it's running).")
    return 0


def cmd_state(store: ButlerStore, a: argparse.Namespace, status: str) -> int:
    t = store.get_task(a.id)
    if not t:
        print(f"No task #{a.id}")
        return 1
    if status == CANCELLED:
        store.expire_pending_for_task(t.id)
    if status == QUEUED and t.state.get("phase") == "awaiting_user":
        # Resuming a task that was waiting on you: continue where it would have.
        store.expire_pending_for_task(t.id)
        store.merge_state(t.id, phase=t.state.get("resume_phase", "work"))
    store.update_task(t.id, status=status, next_wake_at=time.time())
    store.log(t.id, "user", f"You set this task to {status}.")
    print(f"Task #{t.id} is now {status}.")
    return 0


def cmd_status(store: ButlerStore, a: argparse.Namespace) -> int:
    daemon = store.get_meta("daemon") or {}
    hb = store.get_meta("heartbeat")
    alive = bool(daemon.get("pid")) and hb and time.time() - float(hb) < 120
    print(f"Daemon: {'running (pid %s, heartbeat %s)' % (daemon.get('pid'), _fmt_age(hb)) if alive else 'NOT running'}")
    if daemon.get("llm") is not None or alive:
        print(f"Manager model: {daemon.get('llm') or 'none (playbooks only)'}")
    counts: dict[str, int] = {}
    for t in store.list_tasks(None, limit=10000):
        counts[t.status] = counts.get(t.status, 0) + 1
    print("Tasks: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"))
    pending = store.approvals("pending")
    print(f"Inbox: {len(pending)} item(s) waiting for you" + ("  ->  autobot butler inbox" if pending else ""))
    for r in store.resources():
        if r["cooldown_until"] > time.time():
            print(f"Cooling down: {r['name']} until {time.strftime('%H:%M', time.localtime(r['cooldown_until']))} ({r['reason']})")
    print(f"State folder: {store.path.parent}")
    return 0


def cmd_stop(store: ButlerStore, a: argparse.Namespace) -> int:
    from autobot.paths import autobot_home
    (autobot_home() / "butler.stop").write_text(str(time.time()))
    print("Stop requested; the daemon will finish its current checks and exit within a few seconds.")
    return 0


def cmd_run(store: ButlerStore, a: argparse.Namespace) -> int:
    from autobot.butler.daemon import AlreadyRunning, ButlerDaemon
    d = ButlerDaemon(store=store, max_concurrent=a.max_concurrent, tick_seconds=a.tick)
    try:
        d.run_forever()
    except AlreadyRunning as e:
        print(e)
        return 1
    return 0


def doctor_checks() -> list[tuple[str, bool, str]]:
    """(name, ok, detail) for everything the butler depends on. No quota is spent."""
    from autobot.integrations.cli_exec import run_cli
    out: list[tuple[str, bool, str]] = []

    def version(cmd: list[str]) -> tuple[bool, str]:
        exe = shutil.which(cmd[0])
        if not exe:
            return False, "not found on PATH"
        r = run_cli(cmd, timeout=60)
        text = (r.stdout or r.stderr).strip().splitlines()
        return r.ok, f"{exe} — {text[0][:100] if text else ''}"

    for name, cmd, why in [
        ("claude (Claude Code)", ["claude", "--version"], "coding/research worker"),
        ("agy (Antigravity)", ["agy", "--version"], "coding/research worker"),
        ("code (VS Code)", ["code", "--version"], "opens results for your review"),
        ("latexmk", ["latexmk", "-v"], "document/learning checks"),
        ("pdflatex", ["pdflatex", "--version"], "document/learning checks"),
        ("kaggle", ["kaggle", "--version"], "kaggle lane"),
        ("git", ["git", "--version"], "coding lane branches/diffs"),
    ]:
        ok, detail = version(cmd)
        out.append((f"{name} [{why}]", ok, detail))

    try:
        from autobot.llm import get_manager_llm
        llm = get_manager_llm()
        out.append(("manager model (optional)", True,
                    llm.describe() if llm else "none configured — escalations go to your inbox"))
    except Exception as e:
        out.append(("manager model (optional)", False, str(e)))

    from autobot.paths import autobot_home
    home = autobot_home()
    in_onedrive = "onedrive" in str(home).lower()
    out.append(("state folder", not in_onedrive, f"{home}" + (" — inside OneDrive; set AUTOBOT_HOME elsewhere" if in_onedrive else "")))
    return out


def cmd_doctor(store: ButlerStore, a: argparse.Namespace) -> int:
    bad = 0
    for name, ok, detail in doctor_checks():
        print(f"[{'OK ' if ok else 'MISSING'}] {name}: {detail}")
        bad += 0 if ok else 1
    workers = [w for w in ("claude", "agy") if shutil.which(w)]
    print()
    if not workers:
        print("No worker CLI found — the butler can't do any work until claude or agy is on PATH.")
        return 1
    print("Ready. Run `autobot butler smoke` once to prove the whole loop works on this machine "
          "(it spends a few minutes of your Claude/Antigravity usage).")
    return 0


def cmd_smoke(store: ButlerStore, a: argparse.Namespace) -> int:
    from autobot.butler.smoke import run_smoke
    return run_smoke(worker=a.worker, keep=a.keep)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autobot butler", description="Autobot's always-on task manager.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Start the daemon in this window")
    r.add_argument("--max-concurrent", type=int, default=None)
    r.add_argument("--tick", type=float, default=5.0)
    sub.add_parser("stop", help="Ask a running daemon to stop")
    sub.add_parser("status", help="Daemon health, task counts, inbox size")

    ad = sub.add_parser("add", help="Queue a task")
    ad.add_argument("lane", choices=sorted(LANES))
    ad.add_argument("intent", help="What you want, in your own words (stored verbatim)")
    ad.add_argument("--title")
    ad.add_argument("--dir", help="Project folder the worker works in")
    ad.add_argument("--worker", choices=["auto", "claude_code", "antigravity", "vscode"], default="auto")
    ad.add_argument("--check", action="append",
                    help='Acceptance check: a command ("pytest -q") or JSON ({"type":"file_exists","path":"x"}). Repeatable.')
    ad.add_argument("--priority", type=int, default=5, help="1 = most urgent, 9 = least (default 5)")
    ad.add_argument("--max-attempts", type=int)
    ad.add_argument("--timeout-min", type=float, help="Max minutes per worker turn (default 30)")
    ad.add_argument("--read-only", action="store_true", help="Worker may not change files")
    ad.add_argument("--no-branch", action="store_true", help="Coding: don't create an autobot/task-N git branch")
    ad.add_argument("--main-tex", help="Document/learning: main .tex file (default main.tex)")
    ad.add_argument("--format", choices=["tex", "md"], help="Learning/document output format (default tex)")
    ad.add_argument("--outline", help="Learning: outline items separated by ';' (each must become a section)")
    ad.add_argument("--spec-file", help="Learning/document: a file with the full specification")
    ad.add_argument("--style", help="Style rules for the writing")
    ad.add_argument("--competition", help="Kaggle: competition slug")
    ad.add_argument("--rounds", type=int, help="Kaggle: max rounds (default 4)")
    ad.add_argument("--mode", choices=["triage", "compose"], help="Email: triage new mail (default) or compose new emails")
    ad.add_argument("--every", type=float, help="Repeat every N minutes (e.g. email triage --every 120)")
    ad.add_argument("--query", help="Email triage: Gmail search query (default: newer_than:2d, no promotions/social)")
    ad.add_argument("--open-vscode", action="store_true", help="Coding: open the folder in VS Code when done, for your review")
    ad.add_argument("--config-json", help="Advanced: raw JSON config")

    ls = sub.add_parser("list", help="List tasks")
    ls.add_argument("--all", action="store_true")
    sh = sub.add_parser("show", help="Task details and history")
    sh.add_argument("id", type=int)
    sh.add_argument("--limit", type=int, default=30)
    sub.add_parser("inbox", help="Things waiting for you")
    for name in ("approve", "reject"):
        x = sub.add_parser(name)
        x.add_argument("id", type=int)
        x.add_argument("--note", default=None)
    an = sub.add_parser("answer")
    an.add_argument("id", type=int)
    an.add_argument("text")
    for name in ("pause", "resume", "cancel"):
        x = sub.add_parser(name)
        x.add_argument("id", type=int)
    sub.add_parser("doctor", help="Check the tools the butler relies on")
    ga = sub.add_parser("gmail-auth", help="Connect Gmail (one-time browser sign-in; see GMAIL_SETUP.md)")
    ga.add_argument("--no-browser", action="store_true")
    sm = sub.add_parser("smoke", help="Real end-to-end test on this machine")
    sm.add_argument("--worker", choices=["claude_code", "antigravity"], default=None)
    sm.add_argument("--keep", action="store_true", help="Keep the temporary test project")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    store = ButlerStore()
    if a.cmd == "add":
        return cmd_add(store, a)
    if a.cmd == "list":
        return cmd_list(store, a)
    if a.cmd == "show":
        return cmd_show(store, a)
    if a.cmd == "inbox":
        return cmd_inbox(store, a)
    if a.cmd == "approve":
        return cmd_decide(store, a, True, a.note)
    if a.cmd == "reject":
        return cmd_decide(store, a, False, a.note)
    if a.cmd == "answer":
        return cmd_decide(store, a, True, a.text)
    if a.cmd == "pause":
        return cmd_state(store, a, PAUSED)
    if a.cmd == "resume":
        return cmd_state(store, a, QUEUED)
    if a.cmd == "cancel":
        return cmd_state(store, a, CANCELLED)
    if a.cmd == "status":
        return cmd_status(store, a)
    if a.cmd == "stop":
        return cmd_stop(store, a)
    if a.cmd == "run":
        return cmd_run(store, a)
    if a.cmd == "doctor":
        return cmd_doctor(store, a)
    if a.cmd == "smoke":
        return cmd_smoke(store, a)
    if a.cmd == "gmail-auth":
        from autobot.integrations import gmail
        try:
            who = gmail.authorize(open_browser=not a.no_browser)
        except gmail.GmailNotConfigured as e:
            print(e)
            return 1
        print(f"Gmail connected for {who}. Autobot can read mail and create drafts; it sends only what you approve.")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
