"""Build Exp 7 notebook from verified Exp 6 (0.954 LB, Rank #295 / Top 7.49%) notebook.

Key Innovation vs Exp 6:
1. Retains proven BIOHUB_ILP_DIVISION_WEIGHT = "0.78" which scored 0.954.
2. Enables BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER = "1" to filter unphysical forks
   (sister > 8.0 um or parent > 10.5 um), dropping invalid secondary branches to continuous single tracks.
3. Updated title for Exp 7.

Usage: python build_exp7.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "exp6_harmonic_multihop_sota" / "harmonic_multihop_sota.ipynb"
DST = HERE / "harmonic_geometry_filter.ipynb"

REQUIRED = {
    "BIOHUB_ILP_DIVISION_WEIGHT": "0.78",
    "BIOHUB_ILP_APPEARANCE_WEIGHT": "0.0",
    "BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER": "1",
    "BIOHUB_SAFE_DIV_MAX_UM": "9.0",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "14.0",
    "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.6",
    "BIOHUB_SAFE_DIV_DIVERGE_UM": "2.25",
    "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.25",
    "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5",
    "BIOHUB_MOTION_RELINK_LEARNED_BONUS": "1.0",
    "BIOHUB_GAPFILL_MAX_GAP": "3",
    "BIOHUB_GAPFILL_STEP_UM": "5.0",
    "BIOHUB_VALIDATOR_ENABLE": "0",
    "BIOHUB_ILP_TIMEOUT_S": "1200",
    "BIOHUB_REPAIR_DEADLINE_S": "27000",
}

OLD_TITLE = ("# Biohub Cell Tracking: Exp6 Harmonic MultiHop "
             "(Exp4 0.953 base + ILP division fix, division_weight 1.2 -> 0.78)")
NEW_TITLE = ("# Biohub Cell Tracking: Exp7 Harmonic Geometry Filter "
             "(Exp6 0.954 base + BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER=1 for Silver Medal push)")

ASSIGN = r'os\.environ\["{key}"\] = ["\']([^"\']*)["\']'

def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    
    # Check if BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER assignment already exists
    code_text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    has_filter_env = 'os.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"]' in code_text

    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        
        # If this cell configures the environment and has BIOHUB_ILP_DIVISION_WEIGHT, insert geometry filter
        if 'os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.78"' in src and not has_filter_env:
            src = src.replace(
                'os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.78"',
                'os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.78"\nos.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"] = "1"'
            )
            has_filter_env = True
        elif 'os.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"] = "0"' in src:
            src = src.replace(
                'os.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"] = "0"',
                'os.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"] = "1"'
            )

        if cell["cell_type"] == "markdown":
            src = src.replace(OLD_TITLE, NEW_TITLE)
        cell["source"] = src
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for key, want in REQUIRED.items():
        found = re.findall(ASSIGN.format(key=key), code)
        assert found, f"{key}: no os.environ assignment found"
        assert all(v == want for v in found), f"{key}: expected {want}, found {found}"
        print(f"OK  {key} = {want}  ({len(found)} assignment(s))")
    assert "v1284_head" in code or "v1284-head" in code, "sub-voxel head reference missing"

    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.loads(DST.read_text(encoding="utf-8"))  # round-trip validity check
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", "".join(c["source"]))
            compile(body, f"cell{i}", "exec")
    print(f"Wrote {DST} ({DST.stat().st_size} bytes, {len(nb['cells'])} cells) - JSON valid, code cells compile")

if __name__ == "__main__":
    main()
