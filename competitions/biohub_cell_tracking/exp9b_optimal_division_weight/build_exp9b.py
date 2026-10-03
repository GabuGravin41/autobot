"""
Build Biohub Exp 9B: Optimal ILP Division Weight SOTA (ILP_DIVISION_WEIGHT 0.78 -> 0.70).
Unlocks additional true biological mitotic forks globally in the integer programming solver.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "exp6_harmonic_multihop_sota" / "harmonic_multihop_sota.ipynb"
DST = HERE / "harmonic_div070_sota.ipynb"

def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    
    # 1. Update Cell 2: BIOHUB_ILP_DIVISION_WEIGHT
    cell2 = "".join(nb['cells'][2]['source'])
    assert 'os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.78"' in cell2
    cell2 = cell2.replace('os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.78"', 'os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.70"')
    nb['cells'][2]['source'] = cell2

    # 2. Update Title in Cell 0
    cell0 = "".join(nb['cells'][0]['source'])
    new_title = "# Biohub Cell Tracking: Exp9B Optimal Division Weight (ILP_DIVISION_WEIGHT 0.78 -> 0.70, Global Solver Forks)"
    cell0 = re.sub(r"# Biohub Cell Tracking:.*", new_title, cell0)
    nb['cells'][0]['source'] = cell0

    # 3. Clear execution outputs
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    # 4. Compile check on all code cells
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", "".join(c['source']))
            compile(body, f"cell{i}", "exec")

    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"SUCCESS: Wrote {DST} ({DST.stat().st_size} bytes, {len(nb['cells'])} cells)")
    print("Verification: All code cells compiled successfully, JSON valid.")

if __name__ == "__main__":
    main()
