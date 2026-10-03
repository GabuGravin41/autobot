"""
Build Biohub CV Baseline Validator Notebook - V2 (Cache-Accelerated).

Key difference from v1:
  - Mounts daltongabrielomondi/biohub-cv-edge-cache-v1 dataset
  - Injects edge_cache pre-population code into Cell 7 (offline packages)
    → 128 6bba detection files are pre-loaded from the dataset BEFORE Cell 9 runs
    → Cell 9 GPU inference finds cached results and SKIPS detection entirely
    → Total runtime reduced from 9+ hours to ~2-3 hours (ILP+scoring only)

All other changes from v1 are preserved:
  1. BIOHUB_VALIDATOR_ENABLE = "1"
  2. BIOHUB_VALIDATOR_N_PER_TYPE = "999"
  3. val_stems overridden to ALL 6bba training videos in Cell 13
  4. BIOHUB_VALIDATOR_ENABLE removed from Cell 4 guard

Usage: python build_cv_baseline_v2.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "exp9_calibrated_detection_recall" / "harmonic_det955_sota.ipynb"
DST  = HERE / "cv_baseline_validator_v2.ipynb"

CV_FOLD_PREFIX = "6bba"

# ── Cell 2 overrides ─────────────────────────────────────────────────────────
ENV_OVERRIDES = {
    'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"':
        'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"',
    'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"':
        'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "999"',
}

# ── Cell 7 injection: pre-populate edge_cache from dataset ───────────────────
# This code runs right at the END of Cell 7 (offline package install),
# before Cell 9 (GPU detection).  When the detector checks the cache it finds
# all 6bba files already present → skips GPU inference for those videos.
EDGE_CACHE_INJECT = """
# ── CV BASELINE V2: Pre-populate edge_cache from pre-computed dataset ────────
import shutil, pathlib as _pl

_CACHE_SRC = _pl.Path("/kaggle/input/biohub-cv-edge-cache-v1/edge_cache")
_CACHE_DST = _pl.Path("/kaggle/working/edge_cache")
_CACHE_DST.mkdir(parents=True, exist_ok=True)

if _CACHE_SRC.exists():
    _npz_files = list(_CACHE_SRC.glob("*.npz"))
    _copied = 0
    for _f in _npz_files:
        _dst = _CACHE_DST / _f.name
        if not _dst.exists():
            shutil.copy2(_f, _dst)
            _copied += 1
    print(f"[CV v2] Edge cache pre-populated: {len(_npz_files)} files total, {_copied} newly copied")
    print(f"  Source: {_CACHE_SRC}")
    print(f"  Dest:   {_CACHE_DST}")
else:
    print(f"[CV v2] WARNING: Edge cache dataset NOT found at {_CACHE_SRC}")
    print("[CV v2] Will run full detection (slow, may timeout).")
# ── END edge_cache pre-population ────────────────────────────────────────────
"""

# ── Cell 13 injection: override val_stems to 6bba-only ──────────────────────
VAL_STEMS_OVERRIDE = f"""
    # ── CV VALIDATOR V2: Override to embryo-disjoint {CV_FOLD_PREFIX} fold ──
    _cv_all = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr"))
    _cv_fold = [s for s in _cv_all if s.startswith("{CV_FOLD_PREFIX}")]
    print(f"CV FOLD: replacing {{len(val_stems)}} default stems with {{len(_cv_fold)}} {CV_FOLD_PREFIX}-only stems")
    val_stems = _cv_fold
    for _s in val_stems:
        print(f"  {{_s}}  has_gt_div={{division_flags.get(_s, '?')}}")
    # ── END CV FOLD OVERRIDE ─────────────────────────────────────────────────
"""

NEW_TITLE = (
    "# Biohub CV Baseline Validator V2: Cache-Accelerated 6bba Fold\\n\\n"
    "**v2 key change**: 128 pre-computed 6bba edge_cache files are loaded from "
    "`daltongabrielomondi/biohub-cv-edge-cache-v1` before detection runs.\\n"
    "GPU inference is **skipped** for all cached videos → only ILP+scoring+PP sweep runs (~2-3h).\\n\\n"
    "**NO SUBMISSION** generated. Pure CV validation run.\\n\\n"
    "**Scores**: `edge_jaccard`, `division_jaccard`, `proxy_score` per video + aggregate."
)
OLD_TITLE_PATTERN = r"# Biohub Cell Tracking: Exp9.*"


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"Loaded: {SRC} ({len(nb['cells'])} cells)")

    # ── Cell 0: Update title ─────────────────────────────────────────────────
    cell0 = "".join(nb["cells"][0]["source"])
    cell0 = re.sub(OLD_TITLE_PATTERN, NEW_TITLE, cell0)
    nb["cells"][0]["source"] = cell0
    print("Cell 0: title updated")

    # ── Cell 2: Apply env overrides ──────────────────────────────────────────
    cell2 = "".join(nb["cells"][2]["source"])
    for old, new in ENV_OVERRIDES.items():
        if old not in cell2:
            key = old.split('"')[1]
            new_val = new.split('"')[-2]
            pat = re.compile(rf'os\.environ\["{re.escape(key)}"\] = "[^"]*"')
            cell2, n = pat.subn(f'os.environ["{key}"] = "{new_val}"', cell2)
            if n == 0:
                cell2 += f'\nos.environ["{key}"] = "{new_val}"'
                print(f"  Appended: {key} = {new_val}")
            else:
                print(f"  Fallback replaced: {key} = {new_val}")
        else:
            cell2 = cell2.replace(old, new)
            print(f"  OK: {old[:60]!r}")
    if 'os.environ["BIOHUB_VALIDATOR_ENABLE"]' not in cell2:
        cell2 = 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"\n' + cell2
    nb["cells"][2]["source"] = cell2
    print("Cell 2: env overrides applied")

    # ── Cell 4: Relax config guard ───────────────────────────────────────────
    cell4 = "".join(nb["cells"][4]["source"])
    cell4 = re.sub(
        r'["\']BIOHUB_VALIDATOR_ENABLE["\']\\s*:\\s*["\'][^"\']*["\'],?\\s*\\n?',
        "", cell4,
    )
    nb["cells"][4]["source"] = cell4
    print("Cell 4: VALIDATOR_ENABLE removed from guard")

    # ── Cell 7: Inject edge_cache pre-population ─────────────────────────────
    cell7 = "".join(nb["cells"][7]["source"])
    # Append to the END of cell 7 (after all pip install / package setup)
    cell7 = cell7.rstrip() + "\n" + EDGE_CACHE_INJECT
    nb["cells"][7]["source"] = cell7
    print("Cell 7: edge_cache pre-population injected")

    # ── Cell 13: Inject val_stems override ──────────────────────────────────
    cell13 = "".join(nb["cells"][13]["source"])
    marker = "print(val_stems)"
    if marker in cell13:
        cell13 = cell13.replace(marker, marker + "\n" + VAL_STEMS_OVERRIDE)
        print("Cell 13: val_stems 6bba-only override injected")
    else:
        fallback = 'print(f"VALIDATOR: selected'
        if fallback in cell13:
            idx = cell13.index(fallback)
            eol = cell13.index("\n", idx)
            cell13 = cell13[:eol+1] + VAL_STEMS_OVERRIDE + cell13[eol+1:]
            print("Cell 13: val_stems override injected (fallback)")
        else:
            print("WARNING: Could not inject val_stems override!")
    nb["cells"][13]["source"] = cell13

    # ── Clear outputs ─────────────────────────────────────────────────────────
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # ── Write ─────────────────────────────────────────────────────────────────
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.loads(DST.read_text(encoding="utf-8"))  # round-trip check
    print(f"\nJSON round-trip: OK")

    # ── Compile-check all code cells ─────────────────────────────────────────
    nb2 = json.loads(DST.read_text(encoding="utf-8"))
    for i, c in enumerate(nb2["cells"]):
        if c["cell_type"] == "code":
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", "".join(c["source"]))
            try:
                compile(body, f"cell{i}", "exec")
            except SyntaxError as e:
                print(f"SYNTAX ERROR cell {i}: {e}")
                raise

    print(f"\nSUCCESS: {DST}")
    print(f"  Size: {DST.stat().st_size:,} bytes  Cells: {len(nb2['cells'])}")
    print(f"  All code cells: COMPILE OK")
    print(f"\nKey advantage over v1:")
    print(f"  - 128 pre-cached 6bba detection files loaded from Kaggle dataset")
    print(f"  - GPU detection (~7h) is SKIPPED entirely")
    print(f"  - Only ILP+scoring+PP sweep runs (~2-3h)")


if __name__ == "__main__":
    main()
