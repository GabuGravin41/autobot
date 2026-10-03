"""
Build Biohub Exp 9: Calibrated Detection Recall SOTA (DET_THRESHOLD 0.965 -> 0.955).
Directly targets the 1.0-weight edge metric to break out of the 110-team 0.954 tie cluster.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "exp6_harmonic_multihop_sota" / "harmonic_multihop_sota.ipynb"
DST = HERE / "harmonic_det955_sota.ipynb"

def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    
    # 1. Update Cell 2: BIOHUB_DET_THRESHOLD and READMIT_MIN_SCORE
    cell2 = "".join(nb['cells'][2]['source'])
    assert 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"' in cell2
    cell2 = cell2.replace('os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"', 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.955"')
    cell2 = cell2.replace('os.environ["BIOHUB_READMIT_MIN_SCORE"] = "0.965"', 'os.environ["BIOHUB_READMIT_MIN_SCORE"] = "0.955"')
    nb['cells'][2]['source'] = cell2

    # 2. Update Cell 4: _EXPECTED_NUMERIC guard
    cell4 = "".join(nb['cells'][4]['source'])
    assert '"BIOHUB_DET_THRESHOLD": 0.965,' in cell4
    cell4 = cell4.replace('"BIOHUB_DET_THRESHOLD": 0.965,', '"BIOHUB_DET_THRESHOLD": 0.955,')
    nb['cells'][4]['source'] = cell4

    # 3. Update Title in Cell 0
    cell0 = "".join(nb['cells'][0]['source'])
    new_title = "# Biohub Cell Tracking: Exp9 Calibrated Detection Recall (DET_THRESHOLD 0.965 -> 0.955, Break 0.954 Tie Wall)"
    cell0 = re.sub(r"# Biohub Cell Tracking:.*", new_title, cell0)
    nb['cells'][0]['source'] = cell0

    # 4. Clear execution outputs
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    # 5. Compile check on all code cells
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", "".join(c['source']))
            compile(body, f"cell{i}", "exec")

    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"SUCCESS: Wrote {DST} ({DST.stat().st_size} bytes, {len(nb['cells'])} cells)")
    print("Verification: All code cells compiled successfully, JSON valid.")

if __name__ == "__main__":
    main()
