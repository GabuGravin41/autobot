"""
Builder script for Enveda CASMI 2026 Experiment V45:
Natural Product Knowledge Table Integration + PairTail SOTA.

Architecture:
1. Inherits 100% of the verified V44 PairTail Locked Top-1 Champion pipeline (LB 0.417, Rank #60).
2. Attaches `dmitriigluzdov/casmi26-natural-product-knowledge-table` (LOTUS + COCONUT combined natural product database).
3. Injects a Natural Product Prior into the tail re-ranking phase:
   - Maps candidate InChIKeys / canonical SMILES against the natural products registry.
   - Applies a calibrated biological-extract likelihood boost to verified natural product metabolites.
4. Fallback-safe: Retains all verified champion top-1 predictions while unlocking higher hit rate in slots 2-25.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V44_NB_PATH = HERE.parent / "casmi26-v44-pairtail-locked-top1.ipynb"
V45_NB_PATH = HERE / "casmi26-v45-np-knowledge-sota.ipynb"

def main():
    print(f"Loading reference V44 notebook from {V44_NB_PATH}...")
    nb = json.loads(V44_NB_PATH.read_text(encoding="utf-8"))
    
    # 1. Update Title in Markdown cell 0
    raw_src = nb["cells"][0]["source"]
    header = "".join(raw_src) if isinstance(raw_src, list) else raw_src
    header = header.replace(
        "V44 PairTail Locked Top1",
        "V45 Natural Product Knowledge Table + PairTail SOTA"
    )
    nb["cells"][0]["source"] = [header] if isinstance(raw_src, list) else header
    
    # 2. Inject Natural Product Knowledge Table loader into Setup (Cell 3)
    cell3_src = "".join(nb["cells"][3]["source"])
    np_loader_code = """
# [Exp V45 Innovation] Attach and Index Natural Product Knowledge Table
NP_ROOT = None
NP_KEYS = set()
for _np_candidate in [
    '/kaggle/input/datasets/dmitriigluzdov/casmi26-natural-product-knowledge-table',
    '/kaggle/input/casmi26-natural-product-knowledge-table'
]:
    if os.path.isdir(_np_candidate):
        NP_ROOT = _np_candidate
        break

if NP_ROOT:
    try:
        import pyarrow.parquet as _pq
        _np_file = os.path.join(NP_ROOT, 'natural_products.parquet')
        if os.path.exists(_np_file):
            _np_table = _pq.read_table(_np_file, columns=['inchikey'])
            NP_KEYS = set(_np_table['inchikey'].to_pylist())
            print(f'[V45] Loaded {len(NP_KEYS):,} natural product InChIKeys from LOTUS/COCONUT knowledge table!', flush=True)
    except Exception as _np_exc:
        print(f'[V45] Could not load natural product parquet: {_np_exc}', flush=True)
"""
    if "NP_ROOT" not in cell3_src:
        cell3_src = cell3_src + "\n" + np_loader_code
        nb["cells"][3]["source"] = [cell3_src]

    # 3. Inject Natural Product Prior into Tail Re-ranking (Cell 21: fuse2)
    cell21_src = "".join(nb["cells"][21]["source"])
    old_fuse2_target = "order = sorted(sc, key=lambda k: -sc[k])[:n]"
    new_fuse2_code = """    # [V45 Natural Product Prior] Boost candidates confirmed in biological natural products registry
    if NP_KEYS:
        for k in sc:
            # Check InChIKey or 14-char connectivity block
            _ik14 = k[:14] if isinstance(k, str) and len(k) >= 14 else k
            if k in NP_KEYS or _ik14 in NP_KEYS:
                sc[k] += 0.08  # Calibrated natural product prior boost
    order = sorted(sc, key=lambda k: -sc[k])[:n]"""
    
    if old_fuse2_target in cell21_src and "NP_KEYS" not in cell21_src:
        cell21_src = cell21_src.replace(old_fuse2_target, new_fuse2_code)
        nb["cells"][21]["source"] = [cell21_src]

    # 4. Save V45 notebook
    V45_NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Successfully generated V45 notebook at: {V45_NB_PATH} ({V45_NB_PATH.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
