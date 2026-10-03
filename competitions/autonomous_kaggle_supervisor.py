"""
Autobot Autonomous Kaggle Supervisor Daemon
Monitors running GPU kernels (Biohub Exp 9 and 9B), handles automatic submission,
fetches leaderboard scores, and dispatches queued experiments (RSNA Exp 4) upon slot clearance.
"""

import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = REPO_ROOT / "competitions" / "AUTONOMOUS_SUPERVISOR_LOG.md"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_cmd(cmd: list[str], timeout=60) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT))
        return res.returncode, (res.stdout + "\n" + res.stderr).strip()
    except Exception as e:
        return 1, str(e)

def check_kernel_status(slug: str) -> str:
    code, out = run_cmd(["kaggle", "kernels", "status", slug], timeout=30)
    for line in out.splitlines():
        if "KernelWorkerStatus." in line or "has status" in line:
            return line.strip()
    return out.strip()

def submit_kernel(slug: str, competition: str, version: int, desc: str) -> bool:
    log(f"SUBMITTING KERNEL: {slug} (v{version}) to {competition}...")
    cmd = [
        "kaggle", "competitions", "submit",
        competition,
        "-k", slug,
        "-v", str(version),
        "-f", "submission.csv",
        "-m", desc
    ]
    code, out = run_cmd(cmd, timeout=60)
    log(f"Submission command output:\n{out}")
    return code == 0

def check_latest_submission_score(competition: str) -> str:
    code, out = run_cmd(["kaggle", "competitions", "submissions", competition], timeout=45)
    lines = [l for l in out.splitlines() if "submission" in l or "Autobot" in l]
    if lines:
        return lines[0].strip()
    return "No recent submissions found"

def main():
    log("=== AUTOBOT AUTONOMOUS KAGGLE SUPERVISOR ACTIVE ===")
    
    biohub_comp = "biohub-cell-tracking-during-development"
    k_exp9 = "daltongabrielomondi/autobot-biohub-exp9-det955-sota"
    k_exp9b = "daltongabrielomondi/autobot-biohub-exp9b-div070-sota"
    
    exp9_submitted = False
    exp9b_submitted = False
    rsna_pushed = False
    rsna_submitted = False
    
    consecutive_checks = 0
    while consecutive_checks < 180: # Run for up to ~3 hours
        consecutive_checks += 1
        try:
            # Check Exp 9
            st9 = check_kernel_status(k_exp9)
            log(f"Status Exp 9 ({k_exp9}): {st9}")
            if ("COMPLETE" in st9 or "complete" in st9.lower()) and not exp9_submitted:
                log(">>> Biohub Exp 9 has COMPLETED! Initiating submission...")
                ok = submit_kernel(
                    slug=k_exp9,
                    competition=biohub_comp,
                    version=1,
                    desc="Autobot Exp 9: Calibrated Detection Recall SOTA (DET_THRESHOLD=0.955)"
                )
                if ok:
                    exp9_submitted = True
                    
            # Check Exp 9B
            st9b = check_kernel_status(k_exp9b)
            log(f"Status Exp 9B ({k_exp9b}): {st9b}")
            if ("COMPLETE" in st9b or "complete" in st9b.lower()) and not exp9b_submitted:
                log(">>> Biohub Exp 9B has COMPLETED! Initiating submission...")
                ok = submit_kernel(
                    slug=k_exp9b,
                    competition=biohub_comp,
                    version=1,
                    desc="Autobot Exp 9B: Optimal Division Weight SOTA (w_div=0.70)"
                )
                if ok:
                    exp9b_submitted = True
                    
            # If either Biohub experiment finished, automatically dispatch RSNA Exp 4 into the freed GPU slot
            if (exp9_submitted or exp9b_submitted) and not rsna_pushed:
                log(">>> GPU slot freed! Pushing RSNA Exp 4 (0.946 Ryokucha Quintuple Ensemble SOTA)...")
                rsna_dir = REPO_ROOT / "competitions" / "rsna_knee" / "exp4_ryokucha_0946_sota"
                code, out = run_cmd(["kaggle", "kernels", "push", "-p", str(rsna_dir)])
                log(f"RSNA Exp 4 push output:\n{out}")
                if code == 0:
                    rsna_pushed = True
                    
            # Monitor RSNA Exp 4 if pushed
            if rsna_pushed and not rsna_submitted:
                k_rsna = "daltongabrielomondi/autobot-rsna-knee-exp4-ryokucha-0946-sota"
                st_rsna = check_kernel_status(k_rsna)
                log(f"Status RSNA Exp 4 ({k_rsna}): {st_rsna}")
                if "COMPLETE" in st_rsna or "complete" in st_rsna.lower():
                    log(">>> RSNA Exp 4 has COMPLETED! Initiating submission...")
                    ok = submit_kernel(
                        slug=k_rsna,
                        competition="rsna-knee-abnormality-detection",
                        version=1,
                        desc="Autobot Exp 4: 0.946 Ryokucha Quintuple Ensemble SOTA"
                    )
                    if ok:
                        rsna_submitted = True
                    
            # Poll leaderboard scoring if any were submitted
            if exp9_submitted or exp9b_submitted:
                latest_sub = check_latest_submission_score(biohub_comp)
                log(f"Biohub Latest Submission Status: {latest_sub}")
                
            if rsna_submitted:
                latest_rsna = check_latest_submission_score("rsna-knee-abnormality-detection")
                log(f"RSNA Latest Submission Status: {latest_rsna}")
        except Exception as err:
            log(f"Supervisor loop encountered error: {err}")
            
        time.sleep(60)

if __name__ == "__main__":
    main()
