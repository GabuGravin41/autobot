"""
Build and stage RSNA Knee Exp 4: 0.946 Ryokucha Quintuple Ensemble SOTA.
Replicates the verified 0.946 benchmark incorporating DINOv2 (20 tails),
A5 Attention Pooling, RadImageNet E10/E11/E13, and Raptor CoAtNet DepthZone SWA.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "research_ryokucha"
DST_DIR = HERE / "exp4_ryokucha_0946_sota"
DST_DIR.mkdir(exist_ok=True, parents=True)

SRC_NB = SRC_DIR / "rsna-knee-d4-blend-0946-ours10.ipynb"
DST_NB = DST_DIR / "rsna-knee-d4-blend-0946-ours10.ipynb"
SRC_META = SRC_DIR / "kernel-metadata.json"
DST_META = DST_DIR / "kernel-metadata.json"

def main():
    # 1. Clean and update metadata
    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = "daltongabrielomondi/autobot-rsna-knee-exp4-ryokucha-0946-sota"
    meta["title"] = "Autobot RSNA Knee Exp4 Ryokucha 0946 SOTA"
    meta["is_private"] = True
    # Filter out empty string datasets
    meta["dataset_sources"] = [d for d in meta.get("dataset_sources", []) if d.strip()]
    DST_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"OK: Wrote clean metadata to {DST_META} with {len(meta['dataset_sources'])} datasets.")

    # 2. Process notebook
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # 3. Compile verification
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", "".join(c["source"]))
            compile(body, f"cell{i}", "exec")

    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"SUCCESS: Wrote {DST_NB} ({DST_NB.stat().st_size} bytes, {len(nb['cells'])} cells). All cells compile.")

if __name__ == "__main__":
    main()
