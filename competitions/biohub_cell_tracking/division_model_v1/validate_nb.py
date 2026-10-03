import json
import re
from pathlib import Path

nb_path = Path(__file__).parent / "division_train_v1.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
print(f"Cells: {len(nb['cells'])}")
errors = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", src)
        try:
            compile(body, f"cell{i}", "exec")
        except SyntaxError as e:
            errors.append(f"Cell {i}: {e}")
            print(f"SYNTAX ERROR Cell {i}: {e}")
            # Print the offending lines
            lines = body.splitlines()
            lineno = e.lineno or 0
            for ln in range(max(0, lineno-3), min(len(lines), lineno+2)):
                print(f"  {ln+1:4d}: {lines[ln]}")
if not errors:
    print("All code cells: COMPILE OK")
print(f"Notebook size: {nb_path.stat().st_size:,} bytes")
