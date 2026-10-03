"""Build Autobot IEEE Traffic Flow Bench Exp 12 Notebook.

Assembles all modular components into a single clean notebook:
- config.py
- network_utils.py
- resume.py
- aux_features.py
- physics_corrections.py
- task1_state.py (Treiber-Helbing ASM)
- task2_queue.py (Dynamic Kinematic Shockwave & Saturation)
- task4_odme.py (Regularized NNLS)
- submission.py (Triple-gate validation)
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
DST_NB = HERE / "notebook.ipynb"

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

def make_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    d = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d

def main():
    cells = []
    
    # Title Markdown
    cells.append(make_cell("markdown", (
        "# Autobot Traffic Flow Bench - Exp 12: Kinematic Shockwaves & Adaptive Smoothing SOTA\n\n"
        "**Upgrades**:\n"
        "- **Task 2 (Queue)**: Dynamic Kinematic Shockwave & Pre-Breakdown Saturation Engine.\n"
        "- **Task 1 & 3 (State & Physics)**: Treiber-Helbing Adaptive Smoothing Method (ASM) with directional wave propagation.\n"
        "- **Task 4 (ODME)**: Regularized NNLS solver.\n"
        "- **Hardware Target**: 100% Kaggle Cloud 4-Core CPU (0 GPU quota consumed)."
    )))
    
    # Cell 1: Setup work dir
    cells.append(make_cell("code", (
        "import os, pathlib\n"
        "pathlib.Path('src').mkdir(parents=True, exist_ok=True)\n"
        "pathlib.Path('work').mkdir(parents=True, exist_ok=True)\n"
        "print('Working directories initialized.')"
    )))
    
    # Modular source cells
    for mod_name in MODULES:
        mod_path = SRC_DIR / mod_name
        if not mod_path.exists():
            raise FileNotFoundError(f"Missing required source module: {mod_path}")
        code = mod_path.read_text(encoding="utf-8")
        cells.append(make_cell("markdown", f"### Module: `src/{mod_name}`"))
        cells.append(make_cell("code", f"%%writefile src/{mod_name}\n{code}"))
        
    # Execution cell
    cells.append(make_cell("markdown", "### Pipeline Execution & Submission Assembly"))
    exec_code = (
        "import sys, time, shutil\n"
        "sys.path.insert(0, 'src')\n"
        "from pathlib import Path\n"
        "from config import locate_release_root\n"
        "from task1_state import run_task1\n"
        "from task2_queue import run_task2\n"
        "from task4_odme import run_task4\n"
        "from submission import merge_and_validate_submission\n\n"
        "t0 = time.time()\n"
        "REL = locate_release_root()\n"
        "print('Competition Release Root:', REL)\n"
        "SPLITS = ['validation', 'private']\n\n"
        "# 1. Task 1 State Reconstruction & Task 3 Physics\n"
        "print('=== Running Task 1 (Treiber-Helbing ASM) ===')\n"
        "run_task1(REL, SPLITS, Path('work/state_submission.csv'))\n\n"
        "# 2. Task 2 Queue Forecasting (Kinematic Shockwave)\n"
        "print('=== Running Task 2 (Kinematic Shockwave) ===')\n"
        "run_task2(REL, SPLITS, Path('work/queue_submission.csv'))\n\n"
        "# 3. Task 4 ODME (Regularized NNLS)\n"
        "print('=== Running Task 4 (Regularized NNLS) ===')\n"
        "run_task4(REL, SPLITS, Path('work/odme_submission.csv'))\n\n"
        "# 4. Merge, Verify, and Output Final Submission\n"
        "KEY = REL / 'submission_key.csv'\n"
        "merge_and_validate_submission(\n"
        "    Path('work/state_submission.csv'),\n"
        "    Path('work/queue_submission.csv'),\n"
        "    Path('work/odme_submission.csv'),\n"
        "    KEY,\n"
        "    Path('/kaggle/working/submission.csv')\n"
        ")\n\n"
        "# Clean up scratch work directories\n"
        "shutil.rmtree('work', ignore_errors=True)\n"
        "shutil.rmtree('src/__pycache__', ignore_errors=True)\n"
        "print(f'Pipeline finished successfully in {time.time()-t0:.1f}s.')\n"
    )
    cells.append(make_cell("code", exec_code))
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Successfully generated {DST_NB} ({len(cells)} cells, {DST_NB.stat().st_size:,} bytes).")

if __name__ == "__main__":
    main()
