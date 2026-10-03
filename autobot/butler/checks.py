"""
Deterministic acceptance checks — how the butler decides "done" without
trusting anyone's say-so, the model's included.

The two most expensive failures in the Kaggle logs looked like success from
the outside: a training run that reported COMPLETE after zero optimizer
steps, and a submission whose ODME column silently collapsed to all zeros.
A weak model is even more inclined to accept "it finished" as "it worked".
So completion is decided by these checks, run by the daemon itself.

A check is a small dict. Supported types:

  {"type": "command", "run": "pytest -q", "expect_exit": 0, "timeout": 900,
   "contains": "passed"}                     # optional substring of stdout+stderr
  {"type": "file_exists", "path": "out/report.pdf", "min_bytes": 1}
  {"type": "csv_rows", "path": "submission.csv", "min": 1, "max": null,
   "equals": null, "nonzero_column": null}   # rows excluding header
  {"type": "latex_compiles", "main": "main.tex"}
  {"type": "contains", "path": "notes.md", "patterns": ["Eigenvalues", "Remarks"],
   "min_count": null}                         # every pattern must appear (case-insensitive)
  {"type": "min_count", "path": "report.md", "pattern": "https?://", "min": 5}  # regex occurrences

Paths are relative to the task's project folder and may not escape it.
Commands run WITHOUT a shell (argv split, PATHEXT honored on Windows).
Commands proposed by a model (not typed by the user) must start with an
allowlisted program — see SAFE_PROPOSED_PROGRAMS.
"""
from __future__ import annotations

import csv
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autobot.integrations.cli_exec import resolve_exe, run_cli

# Programs a model may propose as an acceptance check. Anything else must be
# written by the user. (Test runners, compilers, linters — things that check,
# not things that change the world.)
SAFE_PROPOSED_PROGRAMS = {
    "pytest", "py.test", "python", "python3", "py", "npm", "pnpm", "yarn", "npx", "node",
    "cargo", "go", "dotnet", "mvn", "gradle", "make", "ruff", "flake8", "mypy", "pyright",
    "tsc", "eslint", "latexmk", "pdflatex", "xelatex", "lualatex", "chktex",
}
_SAFE_SUBCOMMANDS = {
    # program: allowed first arguments (None = any)
    "npm": {"test", "run", "exec"}, "pnpm": {"test", "run"}, "yarn": {"test", "run"},
    "cargo": {"test", "check", "build", "clippy"}, "go": {"test", "vet", "build"},
    "dotnet": {"test", "build"}, "mvn": {"test", "verify"}, "gradle": {"test", "check"},
    "make": {"test", "check"},
}
_PYTHON_SAFE_MODULES = {"pytest", "unittest", "compileall", "py_compile", "mypy", "ruff", "flake8", "json.tool"}

VALID_TYPES = {"command", "file_exists", "csv_rows", "latex_compiles", "contains", "min_count"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def describe(check: dict) -> str:
    t = check.get("type")
    if t == "command":
        return f"`{check.get('run')}` exits {check.get('expect_exit', 0)}"
    if t == "file_exists":
        return f"{check.get('path')} exists"
    if t == "csv_rows":
        return f"{check.get('path')} has the expected rows"
    if t == "latex_compiles":
        return f"{check.get('main', 'main.tex')} compiles to PDF"
    if t == "contains":
        return f"{check.get('path')} covers {len(check.get('patterns') or [])} required items"
    if t == "min_count":
        return f"{check.get('path')} has >= {check.get('min')} matches of {check.get('pattern')!r}"
    return str(check)


def _split(command: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(command)
    # posix=False keeps Windows backslashes intact but leaves quotes on tokens.
    out = []
    for tok in shlex.split(command, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        out.append(tok)
    return out


def validate(check: Any, proposed: bool = False) -> str | None:
    """None if the check is well-formed (and, for model-proposed checks, safe);
    otherwise a reason string."""
    if not isinstance(check, dict):
        return "not an object"
    t = check.get("type")
    if t not in VALID_TYPES:
        return f"unknown check type {t!r}"
    if t == "command":
        run = check.get("run")
        if not isinstance(run, str) or not run.strip():
            return "command check needs 'run'"
        if any(op in run for op in ("&&", "||", "|", ";", ">", "<", "`", "$(")):
            return "shell operators are not supported (commands run without a shell)"
        if proposed:
            argv = _split(run)
            prog = Path(argv[0]).name.lower()
            for ext in (".exe", ".cmd", ".bat"):
                prog = prog.removesuffix(ext)
            if prog not in SAFE_PROPOSED_PROGRAMS:
                return f"'{prog}' is not an allowlisted check program for model-proposed checks"
            if prog in _SAFE_SUBCOMMANDS and (len(argv) < 2 or argv[1] not in _SAFE_SUBCOMMANDS[prog]):
                return f"'{prog} {' '.join(argv[1:2])}' is not an allowed check subcommand"
            if prog in ("python", "python3", "py"):
                if "-m" in argv:
                    mod = argv[argv.index("-m") + 1] if argv.index("-m") + 1 < len(argv) else ""
                    if mod not in _PYTHON_SAFE_MODULES:
                        return f"python -m {mod} is not an allowed check module"
                elif "-c" in argv:
                    return "python -c is not allowed in model-proposed checks"
    elif t in ("file_exists", "csv_rows", "contains", "min_count"):
        if not isinstance(check.get("path"), str) or not check["path"].strip():
            return f"{t} check needs 'path'"
        if t == "contains" and not (isinstance(check.get("patterns"), list) and check["patterns"]):
            return "contains check needs a non-empty 'patterns' list"
        if t == "min_count" and not isinstance(check.get("pattern"), str):
            return "min_count check needs 'pattern'"
    return None


def _safe_path(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    root_r = root.resolve()
    if p != root_r and root_r not in p.parents:
        raise ValueError(f"path {rel!r} escapes the project folder")
    return p


def _tail(text: str, n: int = 1200) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "…" + text[-n:]


def run_check(check: dict, cwd: str | Path) -> CheckResult:
    root = Path(cwd)
    name = describe(check)
    try:
        t = check.get("type")
        if t == "command":
            argv = _split(check["run"])
            res = run_cli(argv, cwd=str(root), timeout=float(check.get("timeout", 900)))
            if res.not_found:
                return CheckResult(name, False, f"program not found: {argv[0]}")
            if res.timed_out:
                return CheckResult(name, False, f"timed out after {check.get('timeout', 900)}s")
            output = (res.stdout or "") + "\n" + (res.stderr or "")
            expected = int(check.get("expect_exit", 0))
            if res.returncode != expected:
                return CheckResult(name, False, f"exit {res.returncode} (expected {expected}):\n{_tail(output)}")
            needle = check.get("contains")
            if needle and needle not in output:
                return CheckResult(name, False, f"output did not contain {needle!r}:\n{_tail(output)}")
            return CheckResult(name, True, _tail(output, 300))

        if t == "file_exists":
            p = _safe_path(root, check["path"])
            if not p.exists():
                return CheckResult(name, False, "missing")
            size = p.stat().st_size if p.is_file() else 1
            if size < int(check.get("min_bytes", 1)):
                return CheckResult(name, False, f"only {size} bytes")
            return CheckResult(name, True, f"{size} bytes")

        if t == "csv_rows":
            p = _safe_path(root, check["path"])
            if not p.is_file():
                return CheckResult(name, False, "missing")
            with p.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
                sample = fh.read(4096)
                fh.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.reader(fh, dialect)
                header = next(reader, None)
                col = check.get("nonzero_column")
                col_idx = header.index(col) if (col and header and col in header) else None
                if col and col_idx is None:
                    return CheckResult(name, False, f"column {col!r} not in header {header}")
                rows = nonzero = 0
                for row in reader:
                    if not row:
                        continue
                    rows += 1
                    if col_idx is not None and col_idx < len(row):
                        try:
                            if float(row[col_idx] or 0) != 0:
                                nonzero += 1
                        except ValueError:
                            nonzero += 1
            if check.get("equals") is not None and rows != int(check["equals"]):
                return CheckResult(name, False, f"{rows} rows, expected exactly {check['equals']}")
            if check.get("min") is not None and rows < int(check["min"]):
                return CheckResult(name, False, f"{rows} rows, expected at least {check['min']}")
            if check.get("max") is not None and rows > int(check["max"]):
                return CheckResult(name, False, f"{rows} rows, expected at most {check['max']}")
            if col_idx is not None and nonzero == 0:
                return CheckResult(name, False, f"column {col!r} is all zeros across {rows} rows")
            return CheckResult(name, True, f"{rows} rows" + (f", {nonzero} non-zero {col!r}" if col_idx is not None else ""))

        if t == "latex_compiles":
            return _latex(root, check)

        if t in ("contains", "min_count"):
            p = _safe_path(root, check["path"])
            if not p.is_file():
                return CheckResult(name, False, "missing")
            text = p.read_text(encoding="utf-8", errors="replace")
            if t == "contains":
                missing = [s for s in check["patterns"] if str(s).lower() not in text.lower()]
                if missing:
                    return CheckResult(name, False, f"missing {len(missing)}: " + "; ".join(map(str, missing[:10])))
                return CheckResult(name, True, f"all {len(check['patterns'])} present")
            n = len(re.findall(check["pattern"], text, flags=re.IGNORECASE))
            ok = n >= int(check.get("min", 1))
            return CheckResult(name, ok, f"{n} matches")

        return CheckResult(name, False, f"unknown check type {t!r}")
    except Exception as e:  # a broken check must fail loudly, never pass
        return CheckResult(name, False, f"check error: {type(e).__name__}: {e}")


def _latex(root: Path, check: dict) -> CheckResult:
    main = check.get("main", "main.tex")
    name = describe(check)
    src = _safe_path(root, main)
    if not src.is_file():
        return CheckResult(name, False, f"{main} missing")
    pdf = src.with_suffix(".pdf")
    started = time.time()
    if resolve_exe("latexmk"):
        engine_flag = {"xelatex": "-xelatex", "lualatex": "-lualatex"}.get(check.get("engine", ""), "-pdf")
        # -g forces a real rebuild: without it, latexmk says "All targets are
        # up-to-date" when the worker already compiled, doesn't touch the PDF,
        # and a fresh-PDF check fails for no reason (seen in the first live run).
        argv = ["latexmk", engine_flag, "-g", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", src.name]
        res = run_cli(argv, cwd=str(src.parent), timeout=float(check.get("timeout", 600)))
    else:
        engine = check.get("engine") or "pdflatex"
        argv = [engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", src.name]
        res = run_cli(argv, cwd=str(src.parent), timeout=float(check.get("timeout", 600)))
        if res.ok:  # second pass for references
            res = run_cli(argv, cwd=str(src.parent), timeout=float(check.get("timeout", 600)))
    if res.not_found:
        return CheckResult(name, False, "no LaTeX engine found on PATH (latexmk / pdflatex)")
    log = src.with_suffix(".log")
    errors = ""
    if log.is_file():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        errs = [l for l in lines if l.startswith("!") or ":error:" in l.lower() or re.match(r".*:\d+: ", l)]
        errors = "\n".join(errs[:15])
    if not res.ok or not pdf.is_file() or pdf.stat().st_mtime < started - 1:
        return CheckResult(name, False, errors or _tail(res.stdout + res.stderr))
    return CheckResult(name, True, f"{pdf.name} built ({pdf.stat().st_size} bytes)")


def run_checks(checks: list[dict], cwd: str | Path) -> list[CheckResult]:
    return [run_check(c, cwd) for c in checks]


def summarize(results: list[CheckResult]) -> str:
    if not results:
        return "(no checks)"
    return "\n".join(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}" + ("" if r.passed else f"\n    {r.detail[:800]}")
                     for r in results)
