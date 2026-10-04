"""
Kaggle Submission Sentinel & Autonomous Quota Watcher
------------------------------------------------------
Continuously monitors Kaggle competition submission quotas, calculates
the countdown to UTC midnight (00:00:00 UTC) quota resets, detects slot
availability, and automatically triggers queued model submissions with
live scoring watchers and persistent audit logging.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SubmissionSentinel")

# Target competitions configuration
COMPETITIONS_CONFIG = {
    "rsna-knee-abnormality-detection": {
        "name": "RSNA Knee Abnormality Detection ($77k)",
        "daily_limit": 5,
        "primary_file": "submission.csv",
    },
    "arc-prize-2026-arc-agi-3": {
        "name": "ARC Prize 2026: ARC-AGI-3 ($850k)",
        "daily_limit": 1,
        "primary_file": "submission.parquet",
    },
    "enveda-casmi-2026": {
        "name": "Enveda CASMI 2026 ($50k)",
        "daily_limit": 5,
        "primary_file": "submission.csv",
    },
}


class SubmissionSentinel:
    def __init__(self, state_file: str = "sentinel_state.json"):
        self.state_file = Path(state_file)
        self.api = None
        self._init_api()
        self.staged_queue: List[Dict[str, Any]] = []
        self.load_state()

    def _init_api(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Successfully authenticated with Kaggle API.")
        except Exception as e:
            logger.error(f"Failed to authenticate with Kaggle API: {e}")
            self.api = None

    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.staged_queue = data.get("staged_queue", [])
                logger.info(f"Loaded {len(self.staged_queue)} staged tasks from {self.state_file}")
            except Exception as e:
                logger.warning(f"Could not load state from {self.state_file}: {e}")
        else:
            self.staged_queue = []

    def save_state(self):
        try:
            data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "staged_queue": self.staged_queue,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save state to {self.state_file}: {e}")

    def stage_submission(
        self,
        competition: str,
        kernel_slug: str,
        file_path: str,
        message: str,
        version: Optional[int] = None,
        auto_submit_on_reset: bool = True,
    ):
        """Add a candidate model submission to the staged execution queue."""
        task = {
            "id": f"{competition}_{int(time.time())}",
            "competition": competition,
            "kernel_slug": kernel_slug,
            "file_path": file_path,
            "version": version,
            "message": message,
            "auto_submit_on_reset": auto_submit_on_reset,
            "status": "QUEUED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submitted_at": None,
            "submission_ref": None,
            "public_score": None,
        }
        self.staged_queue.append(task)
        self.save_state()
        logger.info(f"Staged submission for {competition}: '{message}' ({kernel_slug})")

    @staticmethod
    def get_time_to_utc_midnight() -> timedelta:
        """Calculate exact time remaining until 00:00:00 UTC daily quota reset."""
        now = datetime.now(timezone.utc)
        tomorrow_utc = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return tomorrow_utc - now

    def get_competition_status(self, comp_id: str) -> Dict[str, Any]:
        """Query submission history for a competition and compute quota status."""
        if not self.api:
            return {"error": "API not authenticated"}

        config = COMPETITIONS_CONFIG.get(comp_id, {"daily_limit": 5, "name": comp_id})
        try:
            subs = self.api.competition_submissions(comp_id)
        except Exception as e:
            return {"competition": comp_id, "error": str(e)}

        now_utc = datetime.now(timezone.utc)
        today_date = now_utc.date()

        # Count submissions made today (UTC)
        today_subs = []
        pending_subs = []
        for s in subs:
            # Parse submission date
            date_str = str(getattr(s, "date", ""))
            try:
                # Format: 2026-10-04T14:59:14.193Z or 2026-10-04 14:59:14
                clean_date = date_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_date)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt.date() == today_date:
                    today_subs.append(s)
            except Exception:
                pass

            status_str = str(getattr(s, "status", ""))
            if "PENDING" in status_str:
                pending_subs.append(s)

        used_today = len(today_subs)
        daily_limit = config.get("daily_limit", 5)
        remaining = max(0, daily_limit - used_today)

        latest_sub = subs[0] if subs else None
        latest_info = None
        if latest_sub:
            latest_info = {
                "ref": getattr(latest_sub, "ref", None),
                "date": str(getattr(latest_sub, "date", "")),
                "description": getattr(latest_sub, "description", ""),
                "status": str(getattr(latest_sub, "status", "")),
                "public_score": getattr(latest_sub, "publicScore", None),
            }

        return {
            "competition": comp_id,
            "name": config.get("name", comp_id),
            "daily_limit": daily_limit,
            "used_today": used_today,
            "remaining_today": remaining,
            "pending_count": len(pending_subs),
            "can_submit_now": (remaining > 0 and len(pending_subs) == 0),
            "latest_submission": latest_info,
        }

    def print_dashboard(self):
        """Render a formatted status dashboard to terminal."""
        now_utc = datetime.now(timezone.utc)
        time_to_reset = self.get_time_to_utc_midnight()
        hours, remainder = divmod(int(time_to_reset.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n" + "=" * 80)
        print(f"  AUTOBOT KAGGLE SUBMISSION SENTINEL · UTC TIME: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  COUNTDOWN TO 00:00:00 UTC QUOTA RESET: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print("=" * 80)

        print(f"{'Competition':<36} | {'Limit':<5} | {'Used':<4} | {'Left':<4} | {'Pending':<7} | {'Latest LB'}")
        print("-" * 80)
        for comp_id in COMPETITIONS_CONFIG:
            info = self.get_competition_status(comp_id)
            if "error" in info:
                print(f"{comp_id:<36} | ERROR: {info['error'][:35]}")
                continue
            limit = info["daily_limit"]
            used = info["used_today"]
            left = info["remaining_today"]
            pending = info["pending_count"]
            latest = info.get("latest_submission") or {}
            score = latest.get("public_score") or "N/A"
            status = latest.get("status", "").replace("SubmissionStatus.", "")
            lb_str = f"{score} ({status})" if score != "N/A" else status
            print(f"{comp_id:<36} | {limit:<5} | {used:<4} | {left:<4} | {pending:<7} | {lb_str}")

        print("-" * 80)
        print(f"  STAGED SUBMISSION QUEUE: {len(self.staged_queue)} task(s)")
        for idx, task in enumerate(self.staged_queue):
            print(f"    [{idx + 1}] [{task['status']}] {task['competition']} -> {task['kernel_slug']}")
            print(f"        Message: {task['message']}")
            if task.get("public_score"):
                print(f"        Result Score: {task['public_score']}")
        print("=" * 80 + "\n")

    def execute_submission(self, task: Dict[str, Any]) -> bool:
        """Execute a submission command for a staged task."""
        comp = task["competition"]
        kernel = task["kernel_slug"]
        file_path = task["file_path"]
        msg = task["message"]
        version = task.get("version")

        logger.info(f"Triggering submission for {comp} using kernel {kernel}...")

        # Build kaggle command
        cmd = ["kaggle", "competitions", "submit", "-c", comp]
        if kernel:
            cmd.extend(["-k", kernel])
        if version is not None:
            cmd.extend(["-v", str(version)])
        cmd.extend(["-f", file_path, "-m", f'"{msg}"'])

        cmd_str = " ".join(cmd)
        logger.info(f"Executing: {cmd_str}")

        import subprocess

        try:
            res = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (res.stdout + "\n" + res.stderr).strip()
            logger.info(f"Submission output:\n{output}")

            if "Successfully submitted" in output or res.returncode == 0:
                task["status"] = "SUBMITTED"
                task["submitted_at"] = datetime.now(timezone.utc).isoformat()
                self.save_state()
                return True
            else:
                logger.warning(f"Submission failed or returned non-zero code: {output}")
                return False
        except Exception as e:
            logger.error(f"Error executing submission: {e}")
            return False

    def watch_submission_scoring(self, comp_id: str, max_wait_sec: int = 7200, poll_sec: int = 30):
        """Watch the newest submission until scoring finishes and print the public score."""
        logger.info(f"Monitoring scoring progress for competition {comp_id}...")
        start_time = time.time()
        while time.time() - start_time < max_wait_sec:
            info = self.get_competition_status(comp_id)
            latest = info.get("latest_submission")
            if latest:
                status = latest.get("status", "")
                score = latest.get("public_score")
                ref = latest.get("ref")
                logger.info(f"Submission #{ref}: Status = {status}, Score = {score}")
                if "COMPLETE" in status or "SUCCESS" in status:
                    logger.info(f"SCORING COMPLETE! Public Score: {score}")
                    return score
                elif "ERROR" in status:
                    logger.error(f"SCORING FAILED: Submission #{ref} encountered an error.")
                    return None
            time.sleep(poll_sec)
        logger.warning("Scoring watcher timed out.")
        return None

    def run_sentinel_loop(self, poll_interval: int = 60, auto_submit: bool = True):
        """Main sentinel loop that monitors quotas and triggers staged tasks when live."""
        logger.info(f"Starting Sentinel Loop (poll interval: {poll_interval}s, auto_submit: {auto_submit}).")
        while True:
            try:
                self.print_dashboard()

                # Check if we have queued tasks ready to submit
                if auto_submit:
                    for task in self.staged_queue:
                        if task["status"] == "QUEUED":
                            comp_id = task["competition"]
                            comp_status = self.get_competition_status(comp_id)
                            if comp_status.get("can_submit_now"):
                                logger.info(f"QUOTA AVAILABLE for {comp_id}! Initiating submission...")
                                success = self.execute_submission(task)
                                if success:
                                    logger.info("Submission dispatched. Waiting 30s before scoring watch...")
                                    time.sleep(30)
                                    score = self.watch_submission_scoring(comp_id, max_wait_sec=5400, poll_sec=45)
                                    task["public_score"] = score
                                    task["status"] = "COMPLETE" if score else "EVALUATED"
                                    self.save_state()
                            else:
                                logger.info(
                                    f"Task for {comp_id} queued but cannot submit now. "
                                    f"(Remaining: {comp_status.get('remaining_today')}, "
                                    f"Pending: {comp_status.get('pending_count')})"
                                )

                time_to_reset = self.get_time_to_utc_midnight()
                # If within 2 minutes of reset, sleep in smaller increments
                sleep_time = min(poll_interval, max(10, int(time_to_reset.total_seconds())))
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Sentinel loop terminated by user.")
                break
            except Exception as e:
                logger.error(f"Error in sentinel loop: {e}", exc_info=True)
                time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Submission Sentinel")
    parser.add_argument("--once", action="store_true", help="Print dashboard once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--no-auto-submit", action="store_true", help="Monitor only, do not auto submit")
    args = parser.parse_args()

    sentinel = SubmissionSentinel()
    if args.once:
        sentinel.print_dashboard()
    else:
        sentinel.run_sentinel_loop(
            poll_interval=args.interval,
            auto_submit=not args.no_auto_submit,
        )


if __name__ == "__main__":
    main()
