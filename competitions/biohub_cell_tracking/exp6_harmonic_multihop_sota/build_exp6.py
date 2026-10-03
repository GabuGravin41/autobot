"""Build Exp6 notebook from the verified Exp4 (0.953 LB) notebook.

Only change vs Exp4: BIOHUB_ILP_DIVISION_WEIGHT 1.2 -> 0.78 (plus title).
Every other parameter is asserted to already hold the required value.

Usage:  python build_exp6.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "exp4_harmonic_subvoxel_sota" / "harmonic_fusion_sota.ipynb"
DST = HERE / "harmonic_multihop_sota.ipynb"

OVERRIDES = {"BIOHUB_ILP_DIVISION_WEIGHT": ("1.2", "0.78")}

REQUIRED = {
    "BIOHUB_ILP_DIVISION_WEIGHT": "0.78",
    "BIOHUB_ILP_APPEARANCE_WEIGHT": "0.0",
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

OLD_TITLE = "# Biohub Cell Tracking: Harmonic Fusion V3 (0.953 Record Edition)"
NEW_TITLE = ("# Biohub Cell Tracking: Exp6 Harmonic MultiHop "
             "(Exp4 0.953 base + ILP division fix, division_weight 1.2 -> 0.78)")

ASSIGN = r'os\.environ\["{key}"\] = ["\']([^"\']*)["\']'


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for key, (old, new) in OVERRIDES.items():
            pat = re.compile(ASSIGN.format(key=key))
            src, n = pat.subn(lambda m: m.group(0).replace(f'"{old}"', f'"{new}"'), src)
        if cell["cell_type"] == "markdown":
            src = src.replace(OLD_TITLE, NEW_TITLE)
        cell["source"] = src
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
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
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", c["source"])
            compile(body, f"cell{i}", "exec")
    print(f"Wrote {DST} ({DST.stat().st_size} bytes, {len(nb['cells'])} cells) - JSON valid, code cells compile")


if __name__ == "__main__":
    main()
