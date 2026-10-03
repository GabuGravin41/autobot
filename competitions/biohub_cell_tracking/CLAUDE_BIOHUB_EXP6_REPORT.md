Exp 6 is **not ready to dispatch yet**. The notebook in the Exp 6 folder is still an exact copy of Exp 4, with the division weight at 1.2. The session blocked every way I had to edit a 257 KB notebook: `sed -i`, running Python, and Read/Edit/NotebookEdit, which refuse files over 256 KB. Pushing it now would just re-run Exp 4 and waste a submission. I also haven't run anything, so the notebook's JSON hasn't been validated yet.

**In `competitions/biohub_cell_tracking/exp6_harmonic_multihop_sota/`:**
- **`kernel-metadata.json`** – written exactly as you specified.
- **`harmonic_multihop_sota.ipynb`** – the copy of Exp 4 described above.
- **`build_exp6.py`** – rebuilds the notebook from Exp 4 with the division weight changed from 1.2 to 0.78 and a new title. It then checks all 14 required settings, clears outputs, re-reads the JSON to confirm it's valid, and checks that every code cell compiles.
- **`DESIGN_REPORT.md`** – the design log.

**To finish, run:**
```bash
cd competitions/biohub_cell_tracking/exp6_harmonic_multihop_sota
python build_exp6.py      # expect 14 "OK" lines + "JSON valid"
kaggle kernels push -p .
```

**What I found in the Exp 4 code:**
- **The ILP fix is real.** Exp 4 sets edge cost to −1.0 × edge probability, appearance weight to 0.0 and division weight to 1.2. A fork only happens if the second daughter's edge probability is above the division weight, and probabilities never exceed 1.0, so the ILP never produced a division. Roger's notebook shows the same thing.
- **Everything else you listed was already in Exp 4.** That includes the division limits (9.0 / 14.0 / 0.6 / 2.25 / 0.25), relinking (5.5 / 1.0), validator off, and the 1200 s / 27000 s timeouts. Your "gap=3, step=5.0" matches the gap-fill settings (`GAPFILL_MAX_GAP` / `GAPFILL_STEP_UM`). The separate gap-close setting stays at Exp 4's 2 gaps / 5.0 µm. The v1284 sub-voxel head is still loaded. So Exp 6 changes exactly one thing compared with Exp 4.

**Risks:**
- **The division limits don't apply to ILP divisions.** They only filter the divisions added after the ILP. The setting that would check ILP divisions, `BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER`, is off by default and Exp 4 never turns it on. At 0.78, false divisions like the ones that cost Exp 5 could get through. If Exp 6 scores below 0.953, try that filter on, or a weight of 0.85–0.90, next.
- **0.78 hasn't been measured.** The validator is off, so this is an educated guess.
- **Nothing "multi-hop" was added.** It's only in the name. I didn't verify the 16-minute runtime either; it should be about the same as Exp 4.

Slack is still unauthorized. Connect it in your claude.ai connector settings if you need it.