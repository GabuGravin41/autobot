# Biohub Cell Tracking: Post-Mortem & Lessons Learned

> Written: 2026-09-30. Final private score: 0.920. 3rd place private: 0.967. Gap: 0.047.

---

## The Core Failure

**We never understood the data before writing a single line of model code.**

The private test embryo (ea36) has roughly half the median cell density of the public test embryo (fdad), and about a quarter of the training density.

| Embryo | Split | Density p10 | Density p50 | Density p90 |
|--------|-------|-------------|-------------|-------------|
| 44b6 | Train | 134 | 355 | 627 |
| 6bba | Train | 54 | 114 | 445 |
| fdad | **Public test** | [40, 60) | [160, 260) | [260, 420) |
| ea36 | **Private test** | [40, 60) | **[60, 100)** | [160, 260) |

We tuned our entire pipeline against the public leaderboard (fdad), which was density-similar to training. We never once asked: *what does the private test set actually look like?*

---

## The Probing Technique We Missed

**This is legal and the top teams do it on day 1.**

Run a minimal diagnostic notebook on Kaggle that:
1. Counts detections per frame in the hidden test videos
2. Reports the embryo prefixes and video counts
3. Outputs density percentiles (p10, p50, p90) per prefix

Submit this as a "probing submission." You get a score back, but more importantly you learn the test set's structure. The 3rd place knew ea36's density range through exactly this technique.

> **Rule for future competitions:** Before any model work, submit a probing notebook to characterize the test set. Always.

---

## The Validator We Never Built

Our `BIOHUB_VALIDATOR_ENABLE` was set to `"0"` for the entire competition. We never had a working local validation score.

**Every experiment was validated only by the public leaderboard, which the 12th place explicitly warns is not predictive of private performance.**

The 12th place used 5-fold video-level CV on the 199 training videos. Their CV over-read their private by ~0.011. That's a small and calibrated gap. Our public-to-private gap was 0.036.

> **Rule for future competitions:** Build a local validator on day 1. Accept that it will over-read private slightly. Tune only against local CV, never against public LB.

The correct validation split for this problem:
- Embryo-disjoint (never split one embryo across train/val)
- Hold out one embryo prefix entirely
- Run the full pipeline (detect → link → post-process) on the held-out set
- Evaluate with the official `score.py` (it was available in `official_repo/`)

---

## The Metric Asymmetry We Underweighted

From the 12th place writeup:
- One missed **edge** costs: 7.47e-6
- One missed **division** costs: **5.5e-4 (as much as 74 edges)**

**A single division is worth 74 edges.** We spent weeks chasing edge Jaccard by tuning detection thresholds. We should have been building a division model.

The break-even precision for adding a division:
- Public: 32% (i.e., 1-in-3 added divisions need to be real to break even)
- Private: ~25% (sparser embryo, more false positives)

Our ILP division weight (0.78 or 0.70) was a heuristic. The 3rd place trained a dedicated division parent model + pre/post models + LightGBM calibration. Their division Jaccard was ~0.47; ours was likely <0.20.

---

## The Public/Private Sign Flip

The 12th place documented 17 parent-child pairs with **opposite signs on public vs private**. 15 of 17 were negative on public and positive on private.

Examples:
| Change | Public | Private |
|--------|--------|---------|
| Lateral TTA phase shift | -0.013 | +0.001 |
| Fusion weight 0.475 → 0.30 | -0.004 | +0.001 |
| Division model (12th place) | +0.001 | +0.003 |

**Changes that look bad on public often help on private.**

Our own experiments confirmed this: Risky C (aggressive division penalty, fewer divisions) scored *lower* on public (~0.93?) but *equal* on private (0.920) compared to our best public submission (0.956 public → 0.920 private).

> **Rule for future competitions:** When public and private distributions are known to differ, treat public LB deltas as noise. Prefer configurations that match the *private* distribution's characteristics.

---

## What the Top Teams Had That We Didn't

### 3rd Place (0.967 private)
1. ✅ Local validator (5-fold CV on 199 videos)
2. ✅ Probing the hidden test set for density
3. ✅ Trained dedicated division model (parent + pre + post + LightGBM calibration)
4. ✅ Trained dense 3D flow model (not velocity heuristic)
5. ✅ 3-architecture detector ensemble (EfficientNetV2-L + EfficientNet-B7 + SegResNet 3D)
6. ✅ Per-video affine motion model for coordinate refinement

### 12th Place (0.946 private) — **Started from EXACTLY the same public pack as us**
1. ✅ Started: pilkwang detector + prvsiyan chain = 0.897 private (comparable to our range)
2. ✅ Built consolidation stage (rebuilt divisions, re-solved ambiguous links)
3. ✅ Built mitosis specialist (+0.001 public / +0.003 private)
4. ✅ Built final repair stage (9 rules: admit missed divisions, merge duplicates, re-stitch)
5. ✅ Built learned recentring stage (+0.001 public / +0.003 private)
6. ✅ Their core insight: "Own your base models, or build the OOF world on day 1"

### Us (0.920 private)
- ❌ No local validator
- ❌ No probing of test set
- ❌ No trained division model (ILP heuristic weight 0.70-0.92)
- ❌ No trained flow model (velocity heuristic)
- ❌ Single detector architecture (pilkwang UNet)
- ❌ Coordinate refinement via frozen V1284 head (not per-video affine)

---

## What Actually Worked (Salvageable Insights)

1. **Global ILP division weight 0.78 cracked the 0.954 tie cluster** (Exp 6, +0.007 public from baseline). This was a genuine discovery that unlocked ILP divisions.
2. **BIOHUB_VALIDATOR_ENABLE = "0" prevented timeouts.** This was the right architectural decision.
3. **Risky C's high division penalty (0.92) was better on private** despite worse public score. This validated the sparse-embryo hypothesis retrospectively.
4. **Sub-voxel V1284 head** was a meaningful improvement (Exp 4 → 0.953 from 0.947).

---

## The Correct Approach for a Redo (Late Submission Strategy)

### Phase 1: Data Understanding (2 days)
1. Probe ea36: count detections per frame, get density percentiles
2. Visualize ea36 alongside 44b6 and 6bba — understand the appearance differences
3. Build 5-fold CV validator, verify it tracks true performance within 0.015

### Phase 2: Density-Aware Tuning (2 days)
1. Run 5-fold CV with current pipeline on training data
2. Identify: what detection threshold maximizes CV for a sparse embryo?
3. Identify: what gap-closing distance works for sparse frames?
4. Tune ALL knobs against CV, NEVER against public LB

### Phase 3: Division Model (5-10 days)
1. Train a division parent classifier (CNN on spatio-temporal crops)
2. Calibrate with LightGBM/XGBoost on held-out folds
3. Integrate into ILP as learned division costs (replace fixed weight)

### Phase 4: Flow Model (5-10 days)
1. Train a dense 3D displacement field predictor
2. Use as motion-compensated coordinates for gap closing
3. Use flow residuals as division signal

### Phase 5: Ensemble and Submit
1. 3+ detection models (different architectures, seeds)
2. Combine with trained division + flow costs
3. Final ILP with all learned costs
4. Target: ~0.940-0.950 private

---

## Architecture Ideas for Future Similar Competitions

### State Space Models (Mamba/SSM)
- **LiMTrack**: Mamba-based long-range temporal propagation for visual object tracking
- **Application**: Replace node-transformer edge predictor with a Mamba block
  - Each cell maintains a hidden state across all 100 frames (not just 2)
  - Linear complexity vs. quadratic for attention
  - Natural gap-filling: forward pass predicts, backward pass confirms
- **Concrete change**: In `predict_unet_transformer.py`, replace the attention edge scorer with a bidirectional Mamba layer

### Symplectic Transformers
- **Core idea**: Inject Hamiltonian mechanics into the motion model
- **Application**: Replace velocity heuristic with a symplectic integrator
  - Learns a conserved energy function for tissue motion
  - Prevents long-term drift (energy conservation = no accumulating error)
  - **Division signal for free**: divisions violate energy conservation → the non-Hamiltonian residual IS the division signal
- **Reference**: StretchTime (ICML 2026) introduces Symplectic Positional Embeddings (SyPE) — a drop-in RoPE replacement
- **Caveat**: Embryo tissue is not a closed system. Need dissipative/driven Hamiltonian (SymODEN with control)

### Priority Order for Implementation
1. Local validator (free, immediate)
2. Trained division model (10x leverage vs. detection tuning)
3. Dense flow model (cleans all downstream stages)
4. Mamba edge predictor (architectural improvement)
5. Symplectic flow (principled but complex — last)

---

## The One-Line Summary

> **We optimized for the wrong embryo, validated against the wrong distribution, and attacked the metric's lowest-leverage component (edge Jaccard) instead of its highest-leverage component (divisions).**

---

## Rules for the Next Competition

1. **Day 1: Probe the test set.** Understand its distribution before touching a model.
2. **Day 1: Build a local CV validator.** Never tune against public LB alone.
3. **Before any tuning: calculate the metric's component leverages.** What is 1 unit of each component worth?
4. **Prefer conservatism when public/private distributions differ.** Sparse test data rewards precision, not recall.
5. **Divisions, segments, and rare events dominate metrics.** Invest in those first.
6. **"More is better" is rarely true for Jaccard metrics.** Precision trades off against recall.
