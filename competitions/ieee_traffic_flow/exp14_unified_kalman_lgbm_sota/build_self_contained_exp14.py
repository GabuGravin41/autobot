import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
MODULES = [
    "config.py",
    "network_utils.py",
    "resume.py",
    "aux_features.py",
    "physics_corrections.py",
    "task1_state.py",
    "task2_queue.py",
    "task4_odme.py",
    "submission.py",
]

encoded_modules = {}
for m in MODULES:
    p = SRC_DIR / m
    content = p.read_bytes()
    encoded_modules[m] = base64.b64encode(content).decode('ascii')

main_template = f'''"""
Autobot Traffic Flow Bench Exp 14: Unified SOTA Pipeline (0 GPU consumed).
Self-contained script execution on Kaggle Cloud CPU.
"""
import os
import sys
import time
import base64
from pathlib import Path

# 0. Unpack embedded source modules
src_dir = Path("src")
src_dir.mkdir(parents=True, exist_ok=True)

EMBEDDED_MODULES = {encoded_modules}

for fname, b64_data in EMBEDDED_MODULES.items():
    target = src_dir / fname
    target.write_bytes(base64.b64decode(b64_data))
    
if str(src_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(src_dir.resolve()))
if "src" not in sys.path:
    sys.path.insert(0, "src")

print("Embedded modules unpacked successfully into src/.")

# 1. Pipeline Execution
from config import locate_release_root
from task1_state import run_task1
from task2_queue import run_task2
from task4_odme import run_task4
from submission import merge_and_validate_submission

print("=== AUTOBOT TRAFFIC FLOW BENCH EXP 14: UNIFIED SOTA ===")
t0 = time.time()

REL = locate_release_root()
print("Competition Release Root:", REL)
SPLITS = ['validation', 'private']
work_dir = Path("work")
work_dir.mkdir(parents=True, exist_ok=True)

# 1. Task 1 State Reconstruction & Task 3 Physics
state_out = work_dir / "state_submission.csv"
print("\\n=== Running Task 1 & 3 (State Reconstruction & Physics) ===")
t1_rows = run_task1(REL, SPLITS, state_out)
print(f"Task 1 complete: {{t1_rows:,}} rows written to {{state_out}}")

# 2. Task 2 Queue Forecasting
queue_out = work_dir / "queue_submission.csv"
print("\\n=== Running Task 2 (Calibrated Precision Queue Forecasting) ===")
t2_rows = run_task2(REL, SPLITS, queue_out)
print(f"Task 2 complete: {{t2_rows:,}} rows written to {{queue_out}}")

# 3. Task 4 ODME (Regularized NNLS)
odme_out = work_dir / "odme_submission.csv"
print("\\n=== Running Task 4 (Regularized NNLS ODME) ===")
t4_rows = run_task4(REL, SPLITS, odme_out)
print(f"Task 4 complete: {{t4_rows:,}} rows written to {{odme_out}}")

# 4. Final Submission Assembly & Triple-Gate Verification
print("\\n=== Assembling Final Submission Contract ===")
key_file = REL / "submission_key.csv"
final_out = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")

total_rows = merge_and_validate_submission(
    state_path=state_out,
    queue_path=queue_out,
    odme_path=odme_out,
    key_file=key_file,
    out_path=final_out,
)

print(f"Submission saved to {{final_out}}: {{total_rows:,}} rows. Total time: {{time.time() - t0:.1f}}s")
print("=== END AUTOBOT TRAFFIC EXP 14 SOTA ===")
'''

(HERE / "main.py").write_text(main_template, encoding="utf-8")
print(f"Generated self-contained main.py ({(HERE / 'main.py').stat().st_size / 1024:.1f} KB)")
