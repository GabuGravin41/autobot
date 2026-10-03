"""
The butler: Autobot's always-on task manager.

    store.py     durable tasks / events / inbox (SQLite in ~/.autobot)
    lanes.py     kinds of work and the standing guidance each hands to a worker
    workers.py   Claude Code / Antigravity as the workers that do the real work
    checks.py    deterministic acceptance checks — how "done" is decided
    playbook.py  the per-task state machine
    daemon.py    the loop that owns time
    cli.py       `autobot butler ...`
"""
