"""Build Biohub Exp 8: Grandmaster Iterative Flow + Global ILP Division SOTA

Fuses Aman Atar's 0.965+ grandmaster iterative flow pipeline:
  - 3D morphogenetic tissue velocity flow field motion relinking (seed K=16, radius=48um)
  - DeepCenter 3D U-Net Test-Time Augmentation (TTA=1)
  - Lower detection threshold (0.960) with sub-threshold peak readmission (0.965)
  - Gap-density adaptive linking (5.8 um base)
  - Short-track spurious noise filter (min len=6) with division component preservation
WITH our proven mathematical discovery:
  - BIOHUB_ILP_DIVISION_WEIGHT: 1.2 -> 0.78
    (unlocks the global integer linear program to optimize mitotic division forks directly)
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "research_amanatar_opt" / "optimized-biohub-max-score.ipynb"
DST = HERE / "main.ipynb"

OVERRIDES = {
    "BIOHUB_ILP_DIVISION_WEIGHT": ("1.2", "0.78")
}

REQUIRED = {
    "BIOHUB_ILP_DIVISION_WEIGHT": "0.78",
    "BIOHUB_ILP_APPEARANCE_WEIGHT": "0.0",
    "BIOHUB_DET_THRESHOLD": "0.960",
    "BIOHUB_GAP_CLOSE_UM": "5.8",
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": "6",
    "BIOHUB_DEEPCENTER_TTA": "1",
    "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed",
    "BIOHUB_MOTION_RELINK_FLOW_K": "16",
    "BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM": "48.0",
    "BIOHUB_READMIT_RADIUS_UM": "4",
    "BIOHUB_READMIT_MIN_SCORE": "0.965",
    "BIOHUB_GAPFILL_MAX_GAP": "3",
    "BIOHUB_GAPFILL_STEP_UM": "5.0",
    "BIOHUB_VALIDATOR_ENABLE": "0",
    "BIOHUB_ILP_TIMEOUT_S": "1200",
    "BIOHUB_REPAIR_DEADLINE_S": "27000",
}

ASSIGN = r'os\.environ\["{key}"\] = ["\']([^"\']*)["\']'

def main():
    print(f"Loading base notebook from {SRC}...")
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for key, (old, new) in OVERRIDES.items():
            pat = re.compile(ASSIGN.format(key=key))
            src, n = pat.subn(lambda m: m.group(0).replace(f'"{old}"', f'"{new}"'), src)
            if n > 0:
                print(f"Patched {key}: {old} -> {new} ({n} occurrence(s))")
        
        cell["source"] = src
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    for key, want in REQUIRED.items():
        found = re.findall(ASSIGN.format(key=key), code)
        assert found, f"{key}: no os.environ assignment found"
        assert all(v == want for v in found), f"{key}: expected {want}, found {found}"
        print(f"OK  {key} = {want}")
    
    assert "v1284_head" in code or "v1284-head" in code, "V1284 sub-voxel head reference missing!"
    print("OK  V1284 sub-voxel regression head verified present.")

    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    
    # Round-trip validity check
    json.loads(DST.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", c["source"])
            compile(body, f"cell{i}", "exec")
            
    print(f"Successfully generated {DST} ({DST.stat().st_size} bytes, {len(nb['cells'])} cells) - JSON valid, code cells compile cleanly!")

if __name__ == "__main__":
    main()
