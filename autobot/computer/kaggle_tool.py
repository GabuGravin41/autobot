"""
Kaggle Tool — Wrapper for the official Kaggle API + the real `kaggle` CLI.

Kernel methods (pull_kernel/push_kernel/kernel_status/kernel_output) close
a real gap: this class already had list_competitions, download_data,
submit, and get_leaderboard, but nothing to read or write a kernel's
actual notebook/script source — which is exactly what an Autobot-to-
Claude-Code workflow needs ("read the competition code, hand it to Claude
Code, push back what it writes"). They call the same
kaggle.api.kaggle_api_extended.KaggleApi methods the official `kaggle` CLI
itself calls (kernels_pull/kernels_push/kernels_status/kernels_output) —
same official API, used as a library instead of shelling out to the CLI,
so results come back as real Python values/exceptions instead of text to
parse.

Two of this class's methods deliberately shell out to the real `kaggle`
CLI binary via subprocess instead of calling a Python API method:
get_leaderboard() and submit_code_competition(). This is not the
project's default choice — pull_kernel/push_kernel/kernel_status/
kernel_output/list_competitions all still use the Python API — but it's
the deliberate, evidence-based one for these two specifically:

- The Python API's leaderboard method (`competition_view_leaderboard`) was
  found to no longer exist at all in kaggle==2.2.4 (AttributeError, see
  ROADMAP.md), and its suggested replacement's return shape was unverified.
- `daltongabrielomondi`'s real competition runs (competitions/*/
  THINKING_AND_DECISIONS.md) independently arrived at the same answer:
  the leaderboard CSVs actually sitting in competitions/*/leaderboard/ on
  his machine are byte-for-byte the output of
  `kaggle competitions leaderboard download`, and code-competition
  submissions (Biohub) were only reachable through the documented
  `kaggle competitions submit -c comp -k kernel -v version -f file` CLI
  form, not a plain `competition_submit()` call — Code Competitions reject
  a bare CSV upload outright (`400: Submission not allowed`).

Given the Python API has already drifted once, and a live human (via
Antigravity) already proved out the CLI form against a real account, that
proven command is more trustworthy than guessing at whichever Python
method name the fuzzy AttributeError suggestion pointed at. If the
`kaggle` CLI's own interface ever drifts, at least the failure will be a
loud, obvious "command not found" / non-zero exit rather than a silent
wrong-shape return value.

Deliberately NOT changed here: submit() (the plain CSV path) stays a
single, explicit, IRREVERSIBLE-tier action, and submit_code_competition()
is IRREVERSIBLE too (see autobot/agent/approval.py) — a real competition
submission is never something a kernel-iteration loop can slide into
unnoticed, whichever code path it goes through.
"""
import csv
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from autobot.computer.kaggle_watchdog import (
    DEFAULT_LEDGER_PATH,
    KaggleJobLedger,
    check_capacity as _check_capacity,
    make_kaggle_api,
    refresh_for_capacity,
    verify_liveness as _verify_liveness,
)
from autobot.util.filelock import FileLock

logger = logging.getLogger(__name__)


def _utf8_env() -> dict:
    """Environment for a child `kaggle` process that can't crash on non-cp1252 text.

    `kaggle kernels output` writes the kernel's log to disk with Python's
    default encoding — cp1252 on Windows — and dies on the first tqdm
    progress-bar glyph ('▉'), leaving a 0-byte .log (soil_grain_size_photos
    §5). PYTHONUTF8 must be set when the interpreter STARTS, so it has to
    go on the child process, not on this one.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_kaggle_cli(args: List[str], timeout: int = 120) -> str:
    """
    Run the real `kaggle` CLI binary and return its stdout.

    errors="replace" matches the rest of the project's Windows-console
    convention (see cli.py's _make_stdout_unicode_safe docstring) — a
    remote kernel's stdout piped back through this can contain glyphs a
    cp1252 environment can't decode; we'd rather get '?' than crash.
    """
    try:
        result = subprocess.run(
            ["kaggle", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_utf8_env(),
        )
    except FileNotFoundError:
        raise RuntimeError(
            "kaggle CLI not found on PATH. Run: pip install kaggle"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"kaggle {' '.join(args)} timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


class Kaggle:
    """
    Kaggle API wrapper for autonomous competition participation.
    Requires kaggle-api package and ~/.kaggle/kaggle.json credentials.
    """

    def __init__(self, ledger_path: str | Path | None = None):
        self._api = None
        self._ledger_path = ledger_path
        self._ledger: KaggleJobLedger | None = None

    def _get_api(self):
        if self._api is None:
            try:
                self._api = make_kaggle_api()
            except RuntimeError as e:
                logger.error(f"Kaggle API authentication failed: {e}")
                raise
        return self._api

    def _get_ledger(self) -> KaggleJobLedger:
        if self._ledger is None:
            self._ledger = KaggleJobLedger(self._ledger_path)
        return self._ledger

    def list_competitions(self, search: str = None) -> List[Dict[str, Any]]:
        """List active competitions.

        kaggle>=2.x (kagglesdk-backed) wraps the result in an
        ApiListCompetitionsResponse instead of returning a plain list — the
        real list is at `.competitions`. Confirmed live against kaggle==2.2.4
        on 2026-09-20: field names on each competition object are unchanged
        (ref/title/description/deadline/category/reward), except `ref` is now
        a full URL (e.g. 'https://www.kaggle.com/competitions/titanic')
        instead of the old short slug (e.g. 'titanic'). Callers that feed a
        list_competitions()-derived ref into download_data/submit/
        get_leaderboard should not assume it is a bare slug.
        """
        api = self._get_api()
        response = api.competitions_list(search=search)
        comps = getattr(response, "competitions", response)
        return [
            {
                "ref": c.ref,
                "title": c.title,
                "description": c.description,
                "deadline": str(c.deadline),
                "category": c.category,
                "reward": c.reward,
            }
            for c in comps
        ]

    def list_top_kernels(self, competition: str, sort_by: str = "scoreDescending", limit: int = 10) -> List[Dict[str, Any]]:
        """
        List the top public kernels/notebooks for a competition — the
        "Phase 0 SOTA Discovery" step the lessons in
        AUTOBOT_AUTONOMOUS_EXECUTION_CHALLENGES.md call for: check what
        already works before scaffolding a baseline from scratch (that
        doc's IEEE Traffic Flow post-mortem shows the cost of skipping
        this — a stale Sep-8 community notebook scored 0.53880 when the
        organizers' own Sep-10 official repo, unused, would have avoided
        the exact bug that capped it).

        Read-only — never touches the user's own kernels or submissions.
        Uses the CLI (not the Python API) since this is a listing/search
        operation whose exact Python method name is not something this
        codebase has independently verified against the installed SDK
        version; the CLI's tabular output format is stable and documented.
        """
        if not competition:
            raise ValueError("list_top_kernels: competition ref required")
        output = _run_kaggle_cli([
            "kernels", "list",
            "--competition", competition,
            "--sort-by", sort_by,
            "--page-size", str(limit),
        ])
        lines = [l for l in output.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            return []
        # CLI table format: header, '---' separator, then fixed-width rows.
        header_line = lines[0]
        col_starts = [i for i, ch in enumerate(header_line) if i == 0 or (header_line[i - 1] == " " and ch != " ")]
        col_names = [header_line[a:b].strip() for a, b in zip(col_starts, col_starts[1:] + [None])]
        rows = []
        for line in lines[2:2 + limit]:
            values = [line[a:b].strip() for a, b in zip(col_starts, col_starts[1:] + [None])]
            rows.append(dict(zip(col_names, values)))
        return rows

    def download_data(self, competition: str, path: str = "./data"):
        """Download competition data files."""
        api = self._get_api()
        os.makedirs(path, exist_ok=True)
        api.competition_download_files(competition, path=path, quiet=False)
        logger.info(f"Downloaded data for {competition} to {path}")
        return f"Files downloaded to {path}"

    def submit(self, competition: str, file_path: str, message: str) -> str:
        """
        Submit a file directly to a competition (plain CSV upload path).

        Only valid for ordinary (non-Code-Competition) competitions — a
        strict Code Competition (e.g. one requiring GPU inference inside a
        notebook, like Biohub Cell Tracking) rejects this with
        `400: Submission not allowed: This competition only accepts
        Submissions from Notebooks` and needs submit_code_competition()
        instead. IRREVERSIBLE-tier: always requires live human approval,
        see autobot/agent/approval.py.
        """
        api = self._get_api()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Submission file not found: {file_path}")

        api.competition_submit(file_path, message, competition)
        logger.info(f"Submitted {file_path} to {competition}: {message}")
        return f"Successfully submitted to {competition}"

    def submit_code_competition(self, competition: str, kernel: str, version: int, file_path: str, message: str) -> str:
        """
        Submit a Code Competition's output by binding it to a specific
        kernel version — the path submit() cannot take.

        Empirically proven command (competitions/biohub_cell_tracking/
        THINKING_AND_DECISIONS.md section 4.4, run against Dalton's real
        account 2026-09): `kaggle competitions submit -c <competition>
        -k <kernel> -v <version> -f <file> -m <message>`. Shells out to
        the real CLI rather than guessing a Python API equivalent — see
        this module's docstring for why.

        `file_path` must exist locally (kernel_output() downloads it) even
        though the CLI call itself only submits the already-computed
        remote kernel version — this is a real submission integrity check,
        not decoration: it stops Autobot from binding an approval to a
        kernel version whose output was never actually verified locally.

        IRREVERSIBLE-tier, same as submit() — always requires live human
        approval (see autobot/agent/approval.py's _IRREVERSIBLE_PATTERNS).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Submission file not found: {file_path}")
        if not competition or not kernel:
            raise ValueError("submit_code_competition: competition and kernel required")
        output = _run_kaggle_cli([
            "competitions", "submit",
            "-c", competition,
            "-k", kernel,
            "-v", str(version),
            "-f", file_path,
            "-m", message,
        ], timeout=60)
        logger.info(f"Submitted (code competition) {file_path} to {competition} via kernel {kernel} v{version}")
        return output.strip() or f"Successfully submitted to {competition} via {kernel} v{version}"

    def get_leaderboard(self, competition: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get the current public leaderboard for a competition.

        Rewritten Sep 2026: the Python API's `competition_view_leaderboard`
        method no longer exists in kaggle==2.2.4 (raises AttributeError —
        see ROADMAP.md's Round 7 entry). Rather than wire in the fuzzy
        AttributeError suggestion (`competition_leaderboard_cli`) without
        verifying its return shape, this shells out to
        `kaggle competitions leaderboard download`, which is proven —
        the real CSVs it produces sit in competitions/*/leaderboard/ in
        this repo from Dalton's actual competition runs, with the exact
        header this method parses: Rank,TeamId,TeamName,
        LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames.

        Read-only — downloads a public leaderboard snapshot, does not
        touch the user's own submissions or account state.
        """
        if not competition:
            raise ValueError("get_leaderboard: competition ref required")
        with tempfile.TemporaryDirectory() as tmp:
            _run_kaggle_cli(["competitions", "leaderboard", "download", "-c", competition, "-p", tmp])
            csv_files = list(Path(tmp).glob("*.csv"))
            if not csv_files:
                raise RuntimeError(
                    f"kaggle competitions leaderboard download produced no CSV for {competition}"
                )
            with open(csv_files[0], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        return [
            {
                "rank": int(row["Rank"]) if row.get("Rank", "").isdigit() else row.get("Rank"),
                "teamName": row.get("TeamName"),
                "score": float(row["Score"]) if row.get("Score") else None,
                "submissionCount": int(row["SubmissionCount"]) if row.get("SubmissionCount", "").isdigit() else row.get("SubmissionCount"),
                "lastSubmissionDate": row.get("LastSubmissionDate"),
            }
            for row in rows[:top_n]
        ]

    def pull_kernel(self, kernel: str, path: str = "./kernel") -> str:
        """Download a kernel's current source + kernel-metadata.json into `path`. Read-only."""
        if not kernel:
            raise ValueError("pull_kernel: kernel slug required (e.g. 'username/kernel-name')")
        api = self._get_api()
        os.makedirs(path, exist_ok=True)
        api.kernels_pull(kernel, path, metadata=True)
        logger.info(f"Pulled kernel {kernel} to {path}")
        return f"Kernel {kernel} pulled to {path}"

    def kernel_status(self, kernel: str) -> str:
        """Poll a kernel's current run status (queued/running/complete/error). Read-only."""
        if not kernel:
            raise ValueError("kernel_status: kernel slug required")
        api = self._get_api()
        status = api.kernels_status(kernel)
        return str(status)

    def kernel_output(self, kernel: str, path: str = "./kernel_output") -> str:
        """Download a completed kernel run's output files (and its log) into `path`. Read-only."""
        if not kernel:
            raise ValueError("kernel_output: kernel slug required")
        os.makedirs(path, exist_ok=True)
        # Via the CLI in a UTF-8 child process (see _utf8_env): the in-process
        # API call writes the log with this interpreter's default encoding and
        # crashes on Windows as soon as the log contains a progress-bar glyph.
        try:
            _run_kaggle_cli(["kernels", "output", kernel, "-p", path, "-o"], timeout=600)
        except RuntimeError as e:
            if "not found on PATH" not in str(e):
                raise
            api = self._get_api()
            api.kernels_output(kernel, path)
        logger.info(f"Downloaded output of kernel {kernel} to {path}")
        return f"Output of kernel {kernel} downloaded to {path}"

    def push_kernel(
        self,
        path: str,
        verify_liveness: bool = True,
        grace_checks: tuple = (30, 60),
        sleep_fn=None,
        enforce_capacity: bool = True,
        competition: str | None = None,
    ) -> str:
        """
        Push local changes at `path` and trigger a new run of the kernel
        described by its kernel-metadata.json.

        NOT read-only — this changes a live Kaggle kernel and spends real
        run/GPU quota. It is, however, explicitly SAFE-tier in CoreLoop's
        computer_call risk classifier (see approval.py's
        _SAFE_COMPUTER_CALL_RE / tests/test_approval_new_patterns.py's
        TestKaggleKernelOpsAreSafe) — the user drew this line deliberately:
        iterating on a notebook (push/pull/status/output) should be
        frictionless in every approval mode including strict, while a real
        competition submit() always stops for a live human decision.

        Capacity enforcement (Round 8, Sep 2026): before ever calling the
        real API, checks kernel-metadata.json's `enable_gpu` field and
        refuses to push (raises RuntimeError, no API call made) if doing so
        would exceed Kaggle's account-wide hardware slot limit
        (KAGGLE_GPU_SLOT_LIMIT=2, KAGGLE_CPU_SLOT_LIMIT=4 — see
        kaggle_watchdog.py's module-level constants for why those specific
        numbers). This is what makes running several Kaggle-competition
        projects at once (see orchestrator_dispatch.py) safe rather than a
        way to silently oversubscribe a quota that's shared across the
        whole account, not per-project. Disable with
        `enforce_capacity=False` only if you're certain of what you're
        doing (e.g. a competition whose real limit differs from the
        standard account defaults).

        Liveness verification (Sep 2026): by default, once the push
        succeeds this blocks for up to `grace_checks[-1]` seconds (60s by
        default) to confirm the kernel actually survived initialization
        before returning — see autobot/computer/kaggle_watchdog.py's
        module docstring for the real, documented incident this closes
        (a kernel silently crashed 19 seconds after reporting RUNNING, and
        the agent had already moved on and told the user it was fine).

        Both capacity enforcement and liveness verification are skipped
        automatically if kernel-metadata.json has no `id` field to track —
        there's nothing to register in the ledger without one, so there's
        nothing to meaningfully enforce or verify either.
        """
        if not path:
            raise ValueError("push_kernel: path required")
        meta_path = os.path.join(path, "kernel-metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No kernel-metadata.json in {path} — call pull_kernel first, "
                f"or run 'kaggle kernels init -p {path}' to create one."
            )

        try:
            metadata = json.loads(Path(meta_path).read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            metadata = {}
        kernel_id = metadata.get("id")
        hardware = ("gpu" if metadata.get("enable_gpu") in (True, "true", "True") else "cpu") if kernel_id else None

        ledger = self._get_ledger() if kernel_id else None
        api = self._get_api()

        # Check-capacity -> push -> register is one critical section across
        # every process on this machine (several agents share one Kaggle
        # account and one ledger file). Without the lock, two processes can
        # both see a free slot and both push. The ledger is re-read inside
        # the lock and suspect entries are refreshed from the live API first.
        lock = FileLock(Path(ledger.path).with_suffix(".lock"), timeout=120) if kernel_id else None
        if lock:
            lock.acquire()
        try:
            if kernel_id:
                ledger.load()
                if enforce_capacity:
                    try:
                        refresh_for_capacity(api, ledger)
                    except Exception as e:  # never let a refresh failure block a push
                        logger.warning(f"push_kernel: live capacity refresh failed: {e}")
                    has_capacity, capacity_msg = _check_capacity(ledger, hardware)
                    if not has_capacity:
                        raise RuntimeError(
                            f"push_kernel: refusing to dispatch {kernel_id} — {capacity_msg}. "
                            f"Kaggle's account-wide {hardware.upper()} slot limit would be "
                            f"exceeded. Check 'autobot --jobs' for what's active, wait for one "
                            f"to finish, or pass enforce_capacity=False if you're certain this "
                            f"account's real limit differs."
                        )
                    logger.debug(f"push_kernel: capacity check for {kernel_id} ({hardware}): {capacity_msg}")

            result = api.kernels_push(path)
            logger.info(f"Pushed kernel from {path}: {result}")
            summary = str(result)

            if not kernel_id:
                logger.debug(f"push_kernel: no 'id' in {meta_path}, skipping tracking/verification")
                return summary

            ledger.register(kernel_id, competition=competition, path=path, hardware=hardware)
        finally:
            if lock:
                lock.release()

        if not verify_liveness:
            return summary

        kwargs = {"grace_checks": grace_checks}
        if sleep_fn is not None:
            kwargs["sleep_fn"] = sleep_fn
        liveness = _verify_liveness(api, ledger, kernel_id, **kwargs)

        if liveness["survived_init"]:
            return f"{summary}\nLiveness verified at t+{liveness['checked_at']}s: status={liveness['status']}"
        return (
            f"{summary}\nLIVENESS CHECK FAILED at t+{liveness['checked_at']}s: "
            f"status={liveness['status']}. error_log={liveness.get('error_log')}"
        )
