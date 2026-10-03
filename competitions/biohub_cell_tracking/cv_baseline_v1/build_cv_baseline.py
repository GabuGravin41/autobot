"""
Build Biohub CV Baseline Validator Notebook.

Uses Exp9 (0.956 public) as base. Key changes:
  1. BIOHUB_VALIDATOR_ENABLE = "1"  (was "0" in all submission notebooks)
  2. BIOHUB_VALIDATOR_N_PER_TYPE = "999"  (effectively all videos per prefix)
  3. CV_FOLD_PREFIX = "6bba"  →  val_stems is overridden to ALL 6bba training videos
  4. No submission CSV written at end (CV-only run)
  5. Score breakdown printed: edge_jaccard, division_jaccard, proxy_score per video

The 6bba embryo (density p50 ~114) is the closest proxy to the private ea36 embryo
(density p50 ~60-100) in the training data. Using it as hold-out gives us the
best estimate of private performance without any public-LB leakage.

Usage: python build_cv_baseline.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "exp9_calibrated_detection_recall" / "harmonic_det955_sota.ipynb"
DST = HERE / "cv_baseline_validator.ipynb"

CV_FOLD_PREFIX = "6bba"   # hold out all videos starting with this prefix

# ── Cell 2 overrides ────────────────────────────────────────────────────────
ENV_OVERRIDES = {
    # Enable the validator (was "0" in all submission notebooks)
    'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"':
        'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"',
    # Set N per type very high so ALL videos of each prefix are selected
    # (we then filter to 6bba only via val_stems override below)
    'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"':
        'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "999"',
}

# ── Cell 13 injection: override val_stems to 6bba-only ──────────────────────
# Injected as INDENTED code inside the existing `if VALIDATOR_ENABLE and TRAIN_DIR.exists():` block.
# This avoids breaking the elif/else chain that follows.
VAL_STEMS_OVERRIDE = f"""
    # ── CV VALIDATOR: Override to embryo-disjoint {CV_FOLD_PREFIX} fold ──────
    _cv_all = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr"))
    _cv_fold = [s for s in _cv_all if s.startswith("{CV_FOLD_PREFIX}")]
    print(f"CV FOLD: replacing {{len(val_stems)}} default stems with {{len(_cv_fold)}} {CV_FOLD_PREFIX}-only stems")
    val_stems = _cv_fold
    for _s in val_stems:
        print(f"  {{_s}}  has_gt_div={{division_flags.get(_s, '?')}}")
    # ── END CV FOLD OVERRIDE ──────────────────────────────────────────────────
"""

# ── New header markdown for Cell 0 ──────────────────────────────────────────
OLD_TITLE_PATTERN = r"# Biohub Cell Tracking: Exp9.*"
NEW_TITLE = (
    "# Biohub CV Baseline Validator: 6bba Embryo-Disjoint Fold\n\n"
    "**Purpose**: Establish a reliable local CV score on the 6bba training fold.\n"
    "The 6bba embryo (density p50 ~114) is the closest proxy to the private ea36 embryo\n"
    "(density p50 ~60-100). This score is our ground truth to beat with the division model.\n\n"
    "**NO SUBMISSION** is generated. This is a pure validation run.\n\n"
    "**Scores reported**: `edge_jaccard`, `division_jaccard`, `proxy_score` per video + aggregate."
)


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"Loaded source: {SRC} ({len(nb['cells'])} cells)")

    # ── Cell 0: Update title ─────────────────────────────────────────────────
    cell0_src = "".join(nb["cells"][0]["source"])
    cell0_src = re.sub(OLD_TITLE_PATTERN, NEW_TITLE, cell0_src)
    nb["cells"][0]["source"] = cell0_src
    print("Cell 0: title updated")

    # ── Cell 2: Apply env overrides ──────────────────────────────────────────
    cell2_src = "".join(nb["cells"][2]["source"])
    for old, new in ENV_OVERRIDES.items():
        if old not in cell2_src:
            # Try without the specific value (might already be different)
            # Just set it explicitly if not found
            print(f"  WARNING: Could not find exact string: {old!r}")
            print(f"  Attempting fallback injection...")
            # Find the key and replace whatever value it has
            key = old.split('"')[1]
            new_val = new.split('"')[-2]
            pat = re.compile(rf'os\.environ\["{re.escape(key)}"\] = "[^"]*"')
            cell2_src, n = pat.subn(f'os.environ["{key}"] = "{new_val}"', cell2_src)
            if n == 0:
                # Key not present at all — append it
                cell2_src += f'\nos.environ["{key}"] = "{new_val}"'
                print(f"  Appended: {key} = {new_val}")
            else:
                print(f"  Fallback replaced {n} occurrence(s): {key} = {new_val}")
        else:
            cell2_src = cell2_src.replace(old, new)
            print(f"  OK: {old[:60]!r} -> {new[:60]!r}")

    # Also ensure BIOHUB_VALIDATOR_ENABLE is set to "1" if not already set
    if 'os.environ["BIOHUB_VALIDATOR_ENABLE"]' not in cell2_src:
        cell2_src = 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"\n' + cell2_src
        print("  Prepended BIOHUB_VALIDATOR_ENABLE = 1")
    nb["cells"][2]["source"] = cell2_src
    print("Cell 2: env overrides applied")

    # ── Cell 4: Relax configuration guard (remove VALIDATOR check) ───────────
    cell4_src = "".join(nb["cells"][4]["source"])
    # Remove BIOHUB_VALIDATOR_ENABLE from the strict guard dict so it doesn't
    # fail when we change it from "0" to "1"
    cell4_src = re.sub(
        r'["\']BIOHUB_VALIDATOR_ENABLE["\']\s*:\s*["\'][^"\']*["\'],?\s*\n?',
        "",
        cell4_src,
    )
    nb["cells"][4]["source"] = cell4_src
    print("Cell 4: removed VALIDATOR_ENABLE from config guard")

    # ── Cell 13: Inject val_stems override AFTER the existing selector block ─
    cell13_src = "".join(nb["cells"][13]["source"])

    # Find the insertion point: right after the print(val_stems) line
    # which is the end of the default selection logic
    insertion_marker = "print(val_stems)"
    if insertion_marker in cell13_src:
        cell13_src = cell13_src.replace(
            insertion_marker,
            insertion_marker + "\n" + VAL_STEMS_OVERRIDE,
        )
        print("Cell 13: val_stems override injected after default selector")
    else:
        # Fallback: inject after "VALIDATOR: selected" print
        fallback_marker = 'print(f"VALIDATOR: selected'
        if fallback_marker in cell13_src:
            idx = cell13_src.index(fallback_marker)
            end_of_line = cell13_src.index("\n", idx)
            cell13_src = cell13_src[:end_of_line+1] + VAL_STEMS_OVERRIDE + cell13_src[end_of_line+1:]
            print("Cell 13: val_stems override injected at fallback marker")
        else:
            print("WARNING: Could not find injection marker in Cell 13!")
    nb["cells"][13]["source"] = cell13_src

    # ── Clear all outputs ────────────────────────────────────────────────────
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # ── Write output ─────────────────────────────────────────────────────────
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ── Validate: JSON round-trip ────────────────────────────────────────────
    json.loads(DST.read_text(encoding="utf-8"))
    print(f"\nJSON round-trip: OK")

    # ── Validate: compile all code cells ────────────────────────────────────
    nb2 = json.loads(DST.read_text(encoding="utf-8"))
    for i, c in enumerate(nb2["cells"]):
        if c["cell_type"] == "code":
            body = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", "".join(c["source"]))
            try:
                compile(body, f"cell{i}", "exec")
            except SyntaxError as e:
                print(f"SYNTAX ERROR in cell {i}: {e}")
                raise

    print(f"\nSUCCESS: Wrote {DST}")
    print(f"  Size: {DST.stat().st_size:,} bytes")
    print(f"  Cells: {len(nb2['cells'])}")
    print(f"  All code cells: COMPILE OK")
    print(f"\nWhat this notebook does:")
    print(f"  1. Runs full detection+ILP pipeline on ALL {CV_FOLD_PREFIX} training videos")
    print(f"  2. Scores against GT GEFF files (edge_jaccard + division_jaccard)")
    print(f"  3. Reports per-video and aggregate proxy_score")
    print(f"  4. Runs post-process sweep (gap_close_um, motion_relink, etc.)")
    print(f"  5. Reports which pp config wins on the {CV_FOLD_PREFIX} fold")
    print(f"\nThis score is our ground truth to beat with the division model.")


if __name__ == "__main__":
    main()
