# Kaggle Submission Sentinel & Autonomous Quota Watcher
## Autonomous 00:00:00 UTC Daily Reset Monitor & Pipeline Dispatcher

### Overview
The **Kaggle Submission Sentinel** is an autonomous system engineered to monitor Kaggle competition submission quotas, compute the real-time countdown to daily quota resets (at 00:00:00 UTC), detect when slots become active, and automatically trigger staged model submissions with live scoring watchers.

---

### Key Capabilities
1. **Multi-Competition Quota Tracking**:
   - `rsna-knee-abnormality-detection` ($77k): 5 submissions/day.
   - `arc-prize-2026-arc-agi-3` ($850k): 1 submission/day.
   - `enveda-casmi-2026` ($50k): 5 submissions/day.
2. **UTC Midnight Countdown**:
   - Automatically parses current UTC timestamps and computes exact hours, minutes, and seconds remaining until daily reset.
3. **Autonomous Submission Dispatch**:
   - Maintains a staged task queue.
   - Executes `kaggle competitions submit -c <comp> -k <kernel> -f <file> -m <message>` when daily slots open or pending submissions finalize.
4. **Scoring Watcher**:
   - Once submitted, automatically polls the evaluation status until `SubmissionStatus.COMPLETE` or error, logging the official public score to `sentinel_audit_log.json`.

---

### File Structure
- `notebook.ipynb`: Interactive Jupyter Notebook featuring rich HTML status cards, dynamic meters, and execution cells.
- `build_notebook.py`: Generator script that rebuilds `notebook.ipynb` with AST syntax verification.
- `sentinel.py`: Standalone CLI runner and daemon for headless local/server operation.
- `kernel-metadata.json`: Kaggle cloud kernel specification.

---

### Usage

#### Option A: Interactive Jupyter Notebook
Open `competitions/submission_sentinel/notebook.ipynb` in VS Code / Jupyter Lab and execute the cells.

#### Option B: Terminal CLI / Daemon
```bash
# Print current quota dashboard once
python competitions/submission_sentinel/sentinel.py --once

# Run continuous autonomous sentinel daemon (polls every 60s)
python competitions/submission_sentinel/sentinel.py --interval 60
```
