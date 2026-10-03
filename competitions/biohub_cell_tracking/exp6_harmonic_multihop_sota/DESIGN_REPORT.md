# Exp 6 — Harmonic MultiHop SOTA: Design Report

**Base:** Exp 4 `exp4_harmonic_subvoxel_sota/harmonic_fusion_sota.ipynb` (public LB **0.953**, rank #525/3935)
**Kernel:** `daltongabrielomondi/autobot-biohub-exp6-harmonic-multihop-sota`
**Hypothesis under test:** Letting the global ILP produce mitotic forks (currently impossible) improves division recall without the false-positive fragmentation that cost Exp 5 0.004.

## 1. The single change vs Exp 4

| Parameter | Exp 4 | Exp 6 |
|---|---|---|
| `BIOHUB_ILP_DIVISION_WEIGHT` | 1.2 | **0.78** |

This is a **single-variable ablation** against the 0.953 baseline, so any LB delta can be attributed to the ILP fork change alone. The markdown title was also updated.

## 2. Why 1.2 disables ILP divisions (checked against the Exp 4 code)

These values come from the Exp 4 notebook itself:

- The ILP solver is built with `edge_weight = ILP_EDGE_WEIGHT * EdgeAttr("edge_prob")`, where `ILP_EDGE_WEIGHT = -1.0`. So each selected edge costs `-p`.
- `BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"`.
- When the ILP adds a second outgoing edge (probability p2) to a parent, the objective changes by Δ = −p2 + w_div − w_app = w_div − p2.
- The fork is taken only if Δ < 0, i.e. **p2 > w_div**. With w_div = 1.2 and p2 ≤ 1.0, the ILP never forks. All Exp 4 divisions came from post-hoc `SAFE_DIV` insertion.
- Roger's notebook (`research_roger/biohub-the-proxy-is-inverted-the-ilp-can-t-fork.ipynb`) confirms this with scipy `milp`. At the exact tie p2 = w_div, HiGHS and SCIP return different answers. A value of 0.78 avoids any practical tie.

**Effect of 0.78:** any parent whose second-best outgoing edge has fused `edge_prob > 0.78` becomes a fork, as long as that daughter is not claimed by a better parent.

## 3. Parameters retained from Exp 4 (already present; unchanged)

| Group | Parameter | Value |
|---|---|---|
| Cytokinesis gates | `SAFE_DIV_MAX_UM` | 9.0 |
| | `SAFE_DIV_SISTER_MAX_UM` | 14.0 |
| | `SAFE_DIV_SISTER_SYMMETRY_TAU` | 0.6 |
| | `SAFE_DIV_DIVERGE_UM` | 2.25 |
| | `DEEPCENTER_SAFE_DIV_THRESHOLD` | 0.25 |
| | `SAFE_DIV_FRAME_FRAC_CAP` / `GLOBAL_FRAC_CAP` | 0.0076 / 0.00375 |
| Motion relink | `MOTION_RELINK_TIGHT_UM` / `LEARNED_BONUS` | 5.5 / 1.0 |
| Gap fill | `GAPFILL_MAX_GAP` / `GAPFILL_STEP_UM` | 3 / 5.0 |
| Gap close | `GAP_CLOSE_MAX_GAP` / `GAP_CLOSE_UM` | 2 / 5.0 (Exp 4 value, not changed) |
| Runtime | `VALIDATOR_ENABLE` | 0 |
| | `ILP_TIMEOUT_S` / `REPAIR_DEADLINE_S` | 1200 / 27000 |
| Sub-voxel head | `biohub-v1284-head-s075/v1284_head.pt` + `v1284_coordinate_refinement.py` | preserved |

## 4. Risks and caveats

1. **ILP forks skip the SAFE_DIV gates.** The 9.0 / 2.25 / 0.6 / 0.25 constraints only apply to post-hoc safe-division insertion. The filter that would check ILP-produced forks (`BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER`) defaults to `"0"` and Exp 4 never sets it. At 0.78, geometrically implausible forks can pass through unchecked, which is the same failure mode that took Exp 5 down to 0.949. If Exp 6 scores below 0.953, the next ablation should be Exp 6 plus `BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER=1`, or a stricter weight such as 0.85–0.90.
2. **No local evidence exists for 0.78.** We have no validation run (the validator is disabled), so there is no measured division count or F1 at 0.78. The value is a prior, not a calibration.
3. **"MultiHop" is only a name.** No multi-hop linking logic was added. Runtime should match Exp 4, since the ILP problem size is unchanged. A lower division weight could make the solve slower, but it stays capped by the 1200 s timeout.

## 5. Build status

- `kernel-metadata.json`: written as specified.
- `harmonic_multihop_sota.ipynb`: **currently a byte-for-byte copy of Exp 4 (division weight still 1.2).** In the authoring session, in-place editing (sed, Python, and the notebook editor on a 257 KB file) was blocked by permissions.
- `build_exp6.py`: a reproducible build script. It regenerates the notebook from Exp 4, applies the 1.2 → 0.78 override, checks every parameter in §3, clears outputs, round-trips the JSON, and compiles every code cell.

**Before dispatch, run:**

```bash
cd competitions/biohub_cell_tracking/exp6_harmonic_multihop_sota
python build_exp6.py        # must print OK for all 14 keys and "JSON valid"
kaggle kernels push -p .
```

Do **not** push without running the build step. The notebook as it stands would re-run Exp 4 and waste a submission.
