I've written the strategy to `C:\Users\User 1\.claude\plans\you-are-claude-opus-soft-tiger.md`. I checked three things in the repo, and they change the picture.

**1. Why so many teams sit at 0.943**
- **The 0.943 score comes from shared public weights.** Our research folder has the 0.943 public notebook (`research_haideptry/...coatnet-d4-0943`), and most of the cluster is forks of it. Exp 3 uses the same four CoAtNet models it does, so it produces nearly the same predictions and the same score.
- **A public notebook already scores 0.946** (`research_ryokucha/rsna-knee-d4-blend-0946-ours10.ipynb`). It uses datasets we don't attach: raptor-knee maxspan/native384/native384dense, coatnet-d4-depthzone-swa3, global96, e11-diverse-heads-v20, v52 RadImageNet heads, and `pilkwang/rsna-knee-llm-labels`. Expect the cluster to move up to 0.946 soon.
- **Our CV is not an honest held-out score.** CV rose 0.006 while the leaderboard rose 0.004, with a steady gap of about 0.02. The public checkpoints were trained on data that overlaps our validation folds, so our per-target weight tuning has probably learned which model overfit most.
- **Two smaller limits:**
  - Training labels are extracted from free-text radiology reports (Spanish, Dutch, English), and most teams use the same LLM-extracted labels, so everyone shares the same label errors.
  - Mean pooling across 20–40 slices weakens findings that appear on only 1–3 slices.

**2A. Flip TTA: swapping medial and lateral is not the right rule**
- A mirrored right knee is just a normal left knee. In the original orientation, medial stays medial when you flip, so the labels don't change.
- Our pipelines already convert every right knee into a standard left-knee orientation before the model sees it. In that orientation, flipping shows the model something it has never seen: medial on the wrong side.
- **Rules:**
  - No horizontal flip for these models.
  - Use orientation-safe TTA instead: shift the slice window by ±1 slice, crop at 133/140/147 mm, vary the intensity window, and shift ±4 px in-plane.
  - A flip with the medial/lateral outputs swapped (and MCL discarded from the flipped pass) is worth trying only as a side experiment.
- **Laterality audit:** if the left/right detection fails on some studies, their medial and lateral are silently swapped. That affects 5 of the 12 targets (Med/Lat Meniscus, Med/Lat OA, MCL). We should measure how often that happens.

**2B. Pooling**
- The targets hurt most by mean pooling are Fracture, MCL, ACL and meniscal root/radial tears.
- Effusion, Synovitis, OA and Baker's are spread across many slices, so pooling matters less there.
- If there's time, add one new model with attention pooling for those 6 localized targets. Its main value is that it makes different mistakes from the shared public models.

**2C. Rank normalization**
- AUC is scored separately for each target, so normalizing across targets has no effect on the score.
- Ranking matters only when blending models with different calibrations, and that's worth about 0.0005–0.002.
- The bigger gains are:
  - **Rare targets:** pick the best model for Fracture, MCL and Baker's. Each is 1/12 of the score but has few positives, so its AUC swings a lot.
  - **A small second-stage model using findings that tend to occur together:** ACL tear with pivot-shift bone bruise, the ACL + MCL + medial meniscus triad, a meniscus tear with OA in the same compartment, and effusion with synovitis.

**3. Exp 4, as three submissions**
- **4a:** reproduce the public 0.946 blend unchanged. It must score at least 0.945 before we add anything.
- **4b:** keep 4a at 70–80% of the blend and add our DINOv2, RadImageNet and CoAtNet models at 20–30%. Choose weights only from honest held-out predictions (the `m_f*` fold models scored on their own held-out folds); where we don't have those, use equal weights.
- **4c:** 4b plus the orientation-safe TTA and the second-stage model.
- For the final two picks, submit the best public score and a simple equal-weight blend as protection against the private leaderboard reshuffling.

Separately, Slack in claude.ai isn't authorized yet. Connect it in your claude.ai connector settings if you want to use it.

Should I start with 4a, reproducing the public 0.946 notebook?