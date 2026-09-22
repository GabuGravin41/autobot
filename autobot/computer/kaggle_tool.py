"""
Kaggle Tool — Wrapper for the official Kaggle API.

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

Deliberately NOT extended here: nothing changes about submit() — a real
competition submission stays a single, explicit, IRREVERSIBLE-tier action
(see autobot/agent/approval.py's _IRREVERSIBLE_PATTERNS), not something a
kernel-iteration loop can slide into unnoticed.
"""
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Kaggle:
    """
    Kaggle API wrapper for autonomous competition participation.
    Requires kaggle-api package and ~/.kaggle/kaggle.json credentials.
    """

    def __init__(self):
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self._api = KaggleApi()
                self._api.authenticate()
            except Exception as e:
                logger.error(f"Kaggle API authentication failed: {e}")
                raise RuntimeError(f"Kaggle API not configured: {e}")
        return self._api

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

    def download_data(self, competition: str, path: str = "./data"):
        """Download competition data files."""
        api = self._get_api()
        os.makedirs(path, exist_ok=True)
        api.competition_download_files(competition, path=path, quiet=False)
        logger.info(f"Downloaded data for {competition} to {path}")
        return f"Files downloaded to {path}"

    def submit(self, competition: str, file_path: str, message: str) -> str:
        """Submit a file to a competition."""
        api = self._get_api()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Submission file not found: {file_path}")
        
        api.competition_submit(file_path, message, competition)
        logger.info(f"Submitted {file_path} to {competition}: {message}")
        return f"Successfully submitted to {competition}"

    def get_leaderboard(self, competition: str) -> List[Dict[str, Any]]:
        """Get current leaderboard for a competition."""
        api = self._get_api()
        lb = api.competition_view_leaderboard(competition)
        return [
            {
                "teamName": item.teamName,
                "rank": item.rank,
                "score": item.score,
            }
            for item in lb[:20] # Top 20
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
        """Download a completed kernel run's output files into `path`. Read-only."""
        if not kernel:
            raise ValueError("kernel_output: kernel slug required")
        api = self._get_api()
        os.makedirs(path, exist_ok=True)
        api.kernels_output(kernel, path)
        logger.info(f"Downloaded output of kernel {kernel} to {path}")
        return f"Output of kernel {kernel} downloaded to {path}"

    def push_kernel(self, path: str) -> str:
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
        competition submit() always stops for a live human decision. This
        docstring previously claimed CAUTION tier, which was stale relative
        to that decision — corrected Sep 2026 so the comment matches the
        code instead of the other way around.
        """
        if not path:
            raise ValueError("push_kernel: path required")
        meta_path = os.path.join(path, "kernel-metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No kernel-metadata.json in {path} — call pull_kernel first, "
                f"or run 'kaggle kernels init -p {path}' to create one."
            )
        api = self._get_api()
        result = api.kernels_push(path)
        logger.info(f"Pushed kernel from {path}: {result}")
        return str(result)
