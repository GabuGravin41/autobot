"""
Generator script to build competitions/submission_sentinel/notebook.ipynb
with full AST syntax verification and IPython notebook structure.
"""

import json
import ast
from pathlib import Path


def create_sentinel_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Cell 0: Header Markdown
    cell0_md = """# Kaggle Submission Sentinel & Autonomous Quota Watcher
### Real-Time Quota Tracking · 00:00:00 UTC Countdown · Autonomous Triggering & Scoring Monitor

This notebook is an autonomous sentinel that monitors Kaggle competition submission quotas, sits and waits for the daily submission reset (at 00:00:00 UTC) or for pending evaluations to clear, and immediately triggers queued submissions when slots open.

---
### Monitored Arenas
1. **RSNA Knee Abnormality Detection ($77k)**: 5 submissions/day (Target: Exp 10 Production ViT-VLM Hybrid)
2. **ARC Prize 2026: ARC-AGI-3 ($850k)**: 1 submission/day (Target: Exp 5 Balanced AgentFix SOTA)
"""
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": cell0_md.splitlines(keepends=True)})

    # Cell 1: Environment & Kaggle API Verification
    cell1_code = '''import os
import sys
import time
import json
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from IPython.display import display, HTML, clear_output

# Authenticate Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("[SUCCESS] Kaggle API authenticated successfully.")
except Exception as e:
    print(f"[WARNING] Kaggle API authentication notice: {e}")
    api = None
'''
    ast.parse(cell1_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_code.splitlines(keepends=True)})

    # Cell 2: Staged Queue Configuration
    cell2_code = '''# Competition Quota & Target Staging Configuration
COMPETITIONS_CONFIG = {
    "rsna-knee-abnormality-detection": {
        "name": "RSNA Knee Abnormality Detection",
        "daily_limit": 5,
        "primary_file": "submission.csv",
    },
    "arc-prize-2026-arc-agi-3": {
        "name": "ARC Prize 2026: ARC-AGI-3",
        "daily_limit": 1,
        "primary_file": "submission.parquet",
    },
}

# Define candidate submissions to stage
# When quota is live or resets at 00:00:00 UTC, the sentinel will automatically submit these.
STAGED_SUBMISSIONS = [
    {
        "id": "rsna_knee_exp10",
        "competition": "rsna-knee-abnormality-detection",
        "kernel_slug": "daltongabrielomondi/autobot-rsna-knee-exp10-vit-vlm-hybrid-sota",
        "version": None,
        "file_path": "submission.csv",
        "message": "Autobot RSNA Knee Exp 10: Production 20-Tail DINOv2 + Medical VLM Arbiter Consensus SOTA",
        "status": "QUEUED",
    },
    {
        "id": "arc_prize_exp5",
        "competition": "arc-prize-2026-arc-agi-3",
        "kernel_slug": "daltongabrielomondi/autobot-arc3-exp5-balanced-agentfix-sota",
        "version": None,
        "file_path": "submission.parquet",
        "message": "Autobot ARC3 Exp 5: Balanced AgentFix SOTA (NVFP4 MTP3, 16 seq, image dedup, memory retention)",
        "status": "QUEUED",
    },
]

print(f"Staged {len(STAGED_SUBMISSIONS)} candidate submission(s) ready for dispatch.")
'''
    ast.parse(cell2_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell2_code.splitlines(keepends=True)})

    # Cell 3: Sentinel Core Engine Implementation
    cell3_code = '''class KaggleSentinelEngine:
    """Core sentinel logic for monitoring quotas, calculating UTC resets, and executing submissions."""

    def __init__(self, api_client, config: Dict[str, Any], staged_queue: List[Dict[str, Any]]):
        self.api = api_client
        self.config = config
        self.queue = staged_queue

    @staticmethod
    def get_time_to_utc_midnight() -> timedelta:
        now = datetime.now(timezone.utc)
        tomorrow_utc = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return tomorrow_utc - now

    def query_competition_status(self, comp_id: str) -> Dict[str, Any]:
        if not self.api:
            return {"error": "Kaggle API client not authenticated"}

        cfg = self.config.get(comp_id, {"daily_limit": 5, "name": comp_id})
        try:
            subs = self.api.competition_submissions(comp_id)
        except Exception as e:
            return {"competition": comp_id, "error": str(e)}

        now_utc = datetime.now(timezone.utc)
        today_date = now_utc.date()

        today_subs = []
        pending_subs = []
        for s in subs:
            date_str = str(getattr(s, "date", ""))
            try:
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
        daily_limit = cfg.get("daily_limit", 5)
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
            "name": cfg.get("name", comp_id),
            "daily_limit": daily_limit,
            "used_today": used_today,
            "remaining_today": remaining,
            "pending_count": len(pending_subs),
            "can_submit_now": (remaining > 0 and len(pending_subs) == 0),
            "latest_submission": latest_info,
        }

    def dispatch_submission(self, task: Dict[str, Any]) -> bool:
        comp = task["competition"]
        kernel = task["kernel_slug"]
        msg = task["message"]
        file_path = task.get("file_path", "submission.csv")
        version = task.get("version")

        cmd = ["kaggle", "competitions", "submit", "-c", comp]
        if kernel:
            cmd.extend(["-k", kernel])
        if version is not None:
            cmd.extend(["-v", str(version)])
        cmd.extend(["-f", file_path, "-m", f'"{msg}"'])

        cmd_str = " ".join(cmd)
        print(f"[DISPATCH] Executing: {cmd_str}")

        try:
            res = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (res.stdout + "\\n" + res.stderr).strip()
            print(f"[DISPATCH OUTPUT] {output}")
            if "Successfully submitted" in output or res.returncode == 0:
                task["status"] = "SUBMITTED"
                task["submitted_at"] = datetime.now(timezone.utc).isoformat()
                return True
            return False
        except Exception as e:
            print(f"[DISPATCH ERROR] {e}")
            return False

    def watch_scoring(self, comp_id: str, max_wait_sec: int = 5400, poll_sec: int = 30) -> Optional[str]:
        print(f"[SCORING WATCHER] Tracking live evaluation for {comp_id}...")
        start_time = time.time()
        while time.time() - start_time < max_wait_sec:
            info = self.query_competition_status(comp_id)
            latest = info.get("latest_submission")
            if latest:
                status = latest.get("status", "")
                score = latest.get("public_score")
                ref = latest.get("ref")
                print(f"  [SUBMISSION #{ref}] Status: {status} | Score: {score}")
                if "COMPLETE" in status or "SUCCESS" in status:
                    print(f"  [SUCCESS] Scoring Complete! Public Score: {score}")
                    return score
                elif "ERROR" in status:
                    print(f"  [ERROR] Scoring failed for #{ref}")
                    return None
            time.sleep(poll_sec)
        print("[WARNING] Scoring watcher timed out.")
        return None
'''
    ast.parse(cell3_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell3_code.splitlines(keepends=True)})

    # Cell 4: Visual Dashboard Renderer
    cell4_code = '''def render_dashboard_html(engine: KaggleSentinelEngine):
    """Render rich HTML status dashboard."""
    now_utc = datetime.now(timezone.utc)
    time_to_reset = engine.get_time_to_utc_midnight()
    hours, remainder = divmod(int(time_to_reset.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    rows = []
    for comp_id in engine.config:
        info = engine.query_competition_status(comp_id)
        if "error" in info:
            rows.append(f"""
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px; font-weight: bold;">{comp_id}</td>
                <td colspan="5" style="color: #ff6b6b; padding: 10px;">{info['error']}</td>
            </tr>
            """)
            continue

        latest = info.get("latest_submission") or {}
        score = latest.get("public_score") or "—"
        status = latest.get("status", "").replace("SubmissionStatus.", "")
        status_color = "#51cf66" if "COMPLETE" in status else ("#fcc419" if "PENDING" in status else "#ff6b6b")
        avail_badge = '<span style="background: #2b8a3e; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">READY</span>' if info["can_submit_now"] else '<span style="background: #868e96; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">WAITING</span>'

        rows.append(f"""
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 10px; font-weight: 600;">{info['name']}</td>
            <td style="padding: 10px; text-align: center;">{info['daily_limit']}</td>
            <td style="padding: 10px; text-align: center;">{info['used_today']}</td>
            <td style="padding: 10px; text-align: center; font-weight: bold; color: {'#51cf66' if info['remaining_today'] > 0 else '#ff6b6b'};">{info['remaining_today']}</td>
            <td style="padding: 10px; text-align: center;">{avail_badge}</td>
            <td style="padding: 10px;"><span style="color: {status_color}; font-weight: bold;">{score}</span> <span style="font-size: 11px; opacity: 0.7;">({status})</span></td>
        </tr>
        """)

    queue_rows = []
    for idx, t in enumerate(engine.queue):
        status_color = "#51cf66" if t["status"] in ["COMPLETE", "SUBMITTED"] else "#339af0"
        queue_rows.append(f"""
        <tr style="border-bottom: 1px solid #222;">
            <td style="padding: 8px;">{idx + 1}</td>
            <td style="padding: 8px; font-weight: 500;">{t['competition']}</td>
            <td style="padding: 8px;"><code>{t['kernel_slug']}</code></td>
            <td style="padding: 8px;"><span style="color: {status_color}; font-weight: bold;">{t['status']}</span></td>
            <td style="padding: 8px; font-weight: bold; color: #51cf66;">{t.get('public_score') or '—'}</td>
        </tr>
        """)

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #1a1b1e; color: #e9ecef; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); max-width: 950px; margin: 10px auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #373a40; padding-bottom: 12px;">
            <div>
                <h2 style="margin: 0; color: #74c0fc; font-size: 20px;">AUTOBOT SUBMISSION SENTINEL</h2>
                <div style="font-size: 12px; color: #909296;">UTC Timestamp: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            </div>
            <div style="text-align: right; background: #25262b; padding: 8px 16px; border-radius: 8px; border: 1px solid #373a40;">
                <div style="font-size: 11px; color: #ced4da; text-transform: uppercase; letter-spacing: 0.5px;">Quota Reset Countdown</div>
                <div style="font-size: 22px; font-weight: bold; color: #ffd43b; font-family: monospace;">{hours:02d}h {minutes:02d}m {seconds:02d}s</div>
            </div>
        </div>

        <h3 style="margin-top: 18px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #adb5bd;">Competition Quota Status</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead>
                <tr style="border-bottom: 2px solid #373a40; text-align: left; color: #868e96; font-size: 11px; text-transform: uppercase;">
                    <th style="padding: 8px 10px;">Competition</th>
                    <th style="padding: 8px 10px; text-align: center;">Daily Limit</th>
                    <th style="padding: 8px 10px; text-align: center;">Used</th>
                    <th style="padding: 8px 10px; text-align: center;">Available</th>
                    <th style="padding: 8px 10px; text-align: center;">Status</th>
                    <th style="padding: 8px 10px;">Latest LB Score</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>

        <h3 style="margin-top: 20px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #adb5bd;">Staged Submission Pipeline</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <thead>
                <tr style="border-bottom: 2px solid #373a40; text-align: left; color: #868e96; font-size: 11px; text-transform: uppercase;">
                    <th style="padding: 8px;">#</th>
                    <th style="padding: 8px;">Competition</th>
                    <th style="padding: 8px;">Kernel</th>
                    <th style="padding: 8px;">Queue State</th>
                    <th style="padding: 8px;">Public Score</th>
                </tr>
            </thead>
            <tbody>
                {''.join(queue_rows)}
            </tbody>
        </table>
    </div>
    """
    clear_output(wait=True)
    display(HTML(html))
'''
    ast.parse(cell4_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell4_code.splitlines(keepends=True)})

    # Cell 5: Main Autonomous Sentinel Watch Loop
    cell5_code = '''# Initialize Engine
sentinel = KaggleSentinelEngine(
    api_client=api,
    config=COMPETITIONS_CONFIG,
    staged_queue=STAGED_SUBMISSIONS,
)

# Configuration:
# - POLL_INTERVAL_SECONDS: Frequency to refresh quota and countdown
# - AUTO_SUBMIT: Whether to trigger submission automatically when quota becomes live
POLL_INTERVAL_SECONDS = 60
AUTO_SUBMIT = True
MAX_CYCLES = 120  # Runs up to 2 hours in interactive session, or set to None for continuous loop

print("[START] Running Autonomous Submission Sentinel...")
cycle = 0
try:
    while MAX_CYCLES is None or cycle < MAX_CYCLES:
        cycle += 1
        # Render visual dashboard
        render_dashboard_html(sentinel)

        # Check for executable tasks
        if AUTO_SUBMIT:
            for task in sentinel.queue:
                if task["status"] == "QUEUED":
                    comp_id = task["competition"]
                    status = sentinel.query_competition_status(comp_id)
                    if status.get("can_submit_now"):
                        print(f"\\n>>> [SLOT DETECTED] Submitting queued task for {comp_id}!")
                        success = sentinel.dispatch_submission(task)
                        if success:
                            time.sleep(30)
                            score = sentinel.watch_scoring(comp_id, max_wait_sec=5400, poll_sec=45)
                            task["public_score"] = score
                            task["status"] = "COMPLETE" if score else "EVALUATED"
                            render_dashboard_html(sentinel)

        # Dynamic sleep interval
        time_to_reset = sentinel.get_time_to_utc_midnight()
        sleep_sec = min(POLL_INTERVAL_SECONDS, max(10, int(time_to_reset.total_seconds())))
        time.sleep(sleep_sec)

except KeyboardInterrupt:
    print("\\n[STOP] Sentinel monitoring stopped by user.")
'''
    ast.parse(cell5_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell5_code.splitlines(keepends=True)})

    # Cell 6: Summary & Export Cell
    cell6_code = '''# Save final session audit log
session_summary = {
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "tasks": sentinel.queue,
}
with open("sentinel_audit_log.json", "w", encoding="utf-8") as f:
    json.dump(session_summary, f, indent=2)

print("[COMPLETE] Session summary exported to sentinel_audit_log.json:")
print(json.dumps(session_summary, indent=2))
'''
    ast.parse(cell6_code)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell6_code.splitlines(keepends=True)})

    # Write target notebook
    target_nb = Path("competitions/submission_sentinel/notebook.ipynb")
    with open(target_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"[SUCCESS] Authored {target_nb} ({len(nb['cells'])} cells) with zero AST errors.")


if __name__ == "__main__":
    create_sentinel_notebook()
