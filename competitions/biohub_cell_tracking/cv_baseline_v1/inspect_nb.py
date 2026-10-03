import json
from pathlib import Path

nb_path = Path(__file__).parent.parent / "exp9_calibrated_detection_recall" / "harmonic_det955_sota.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    snippet = src[:150].replace("\n", "\\n").strip()
    print(f"Cell {i:2d} [{c['cell_type']:8s}]: {len(src):6d} chars | {snippet!r}")
