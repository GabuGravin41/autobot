No code or config has been changed; the plan is saved at `C:\Users\User 1\.claude\plans\you-are-claude-opus-immutable-nebula.md`. The short version: two small experiments off Exp 6 can plausibly get us out of the tie, but 0.958 is probably out of reach with parameter changes alone.

**1. Why 110 teams tie at 0.954.** Nothing in the method is capping everyone. It's a copy cluster: the teams all run the same public weights and variants of the same public notebook, so they produce nearly the same cells and links. The leaderboard shows three decimals, so small differences all display as 0.954. Our Exp 6 is the public notebook that argued the ILP (the global solver that picks the links) could never create a division. The metric, per that notebook, is `adjusted_edge_jaccard + 0.1 · division_jaccard`, where the edge term is adjusted for how many cells you predict and only a fraction of cells are labelled. Link accuracy dominates the score; divisions count for at most 0.1.

**2. The three avenues, checked against the Exp 6 notebook:**
- **A (detection, ensembles, TTA):** mostly used up. The notebook already runs 8-view flip/rotation TTA on the main detector, a two-model ensemble, TTA on the features used for linking, and the sub-voxel head. The one untested lever of real size is the **detection threshold**: 0.965 in our runs, 0.955 in another public lineage. The same public notebook shows that removing cells looks good on our local check but hurts on the test set, so a lower threshold (more cells) is the direction the test set may reward.
- **B (ILP costs):** every link has cost `-p` (zero or negative), so the solver never refuses an allowed link because its probability is low. A division happens exactly when the second-best child's probability is above the division weight. The appearance, disappearance and offset settings interact with each other and with which short tracks survive. I couldn't check how the tracking library handles that, so don't restructure the costs this close to the deadline. The only knob with a measured gain is the division weight (1.2 → 0.78 gave about +0.001), so the safe move is one more step.
- **C (post-processing):** stop. Exp 7 lost 0.024 and Exp 5 lost 0.005, and the metric punishes false divisions directly.

**3. Private leaderboard risk:** the detection threshold is the most robust change, the division weight is medium, and cost restructuring or geometry filters are low. Always keep Exp 6 as one of your two final submissions, as a floor.

**4. Recommended experiments.** Build each from Exp 6 with a copy of `build_exp6.py`, changing one setting:
- **Exp 9 (primary):** `BIOHUB_DET_THRESHOLD` 0.965 → **0.955**. The notebook's built-in value check around cell 111 must also be set to 0.955, or the run aborts.
- **Exp 9B (run in parallel):** `BIOHUB_ILP_DIVISION_WEIGHT` 0.78 → **0.70**.
- **Exp 10:** combine whichever of the two scored at least 0.954. They affect different parts of the metric, so the gains should roughly add. If Exp 9 improves, also try 0.945 if you have a spare slot.
- **Final selection:** Exp 6 plus the best of Exp 9, 9B and 10.

In the Kaggle logs, check that the cell count rises a few percent in Exp 9, that the ILP still solves within its 1200 s limit, that the TTA and sub-voxel head log lines are present, and that Exp 9B shows more ILP divisions than Exp 6.

**Honest ceiling:** 0.958 needs +0.004, as much as the whole sub-voxel head gained, and no single parameter change credibly gives that. A realistic target is 0.955–0.956, which clears the tie and reaches the silver cutoff.