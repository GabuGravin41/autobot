I read the competition README, the Exp 10 code and its Kaggle log, and the leaderboard. The main finding: the gap is not about model architecture. **Exp 10 scored 0.0597 on our own site-grouped validation (OOF) but 0.205 on the public leaderboard, and the leader's 0.046 is better than our own validation score.** The leaders are using information we ignore; a bigger sequence model won't close that.

## Corrections to the brief
These change the plan, so they come first (from `competitions/ieee_ai_emulation/README.md`):
- **What's scored:** only **height and above-ground biomass (agb) at year 40**. NPP, Rh and soil carbon are not scored.
- **Scenarios:** only **2** (ssp126 and ssp585), not 4.
- **Metric:** mean squared error on values divided by their global mean (height ÷ 10.90645, agb ÷ 3.39567). It's not an error averaged over the whole trajectory.
- **Loss split:** biomass is about **72% of the loss** (0.087 vs 0.033 for height in the Exp 10 log). Most of the gain is in biomass.
- **Unused labels:** the training files hold **monthly values for all 7 variables over 41 years**. Exp 10 trains on only the final height and biomass value.

## 1. Why we're stuck at 0.205
- **The ceiling is the same for every model.** Exp 9 (linear) got 0.207 and Exp 10 (neural) got 0.205. When models that different hit the same score, the limit is a systematic offset or information we're not using.
- **The gap is shift, not noise.** Our validation only ever holds out *sites* under *historical* climate. The test set is *future* climate: in Exp 4, 100% of test rows had at least one input outside the training range. Every model we've built learns absolute levels, and absolute levels are what break under a climate shift.
- **We're fitting each site from scratch.** Most of the 0.06 validation error is site-to-site variation. Each test row comes with 15 initial states per site (one per stand age), which already encode that site's behaviour. Exp 10 uses them only as plain inputs.

## 2. Ideas, ranked by expected gain
1. **Match test sites to training sites (highest gain, not yet verified).** The 6,171 test sites are probably a subset of the 54,152 training sites. If their constant soil/biome inputs match exactly, we already have each test site's historical 40-year outcome for all 15 ages. We would then only predict how much the future scenario shifts that outcome, and the site-specific error cancels out. This is the most plausible way anyone gets below our own 0.06.
2. **Use each site's own growth curve as the baseline.** The 15 initial ages are the simulator's own growth curve for that site. As a first guess, a stand aged *a* after 40 years looks like the curve's value at age *a*+40, interpolated. The model then only learns the climate-change correction, which is much smaller, so extrapolation errors shrink with it.
3. **Check for a definition mismatch.** The README says "targets are annual", but Exp 10 trains on the last month of year 40. The initial state may also be defined differently from how we build it. A mismatch like that would cap every model at the same score, which is exactly what we see.
4. **Physics-shaped features and a mass balance** (your Avenues 1 and 2):
   - CO₂ fertilization enters as a log of CO₂ relative to a reference, so it extrapolates the way the simulator does.
   - Growing degree days and a monthly drought index (SPEI-like), measured as anomalies against each site's own baseline.
   - A carbon-balance penalty on biomass: change ≈ α·NPP − k·biomass, using the NPP labels we already have.
   - Height derived from predicted biomass using the site's own height-to-biomass curve.
   
   These are real gains, but they come second, after ideas 1 and 2.
5. **Architecture** (your Avenue 3). I'd skip TFT, Mamba and graph attention: the sequence is only 40 steps and capacity isn't the limit. Instead, use a small yearly GRU fed monthly inputs, trained on its **own** full 40-year rollout rather than the true previous state, with a loss on every year. That removes the error build-up that sank Exp 7 (5-year steps).

## 3. Recommended experiments
The full spec is in the plan file.
- **Exp 11a — data checks.** CPU only, about 20 minutes, no submission. It answers five questions, all computed on the training data:
  - Do test sites match training sites?
  - Is the initial state defined the way we assume?
  - Is the target the last month or the annual mean?
  - What validation score does the growth-curve baseline get with no model at all?
  - How far outside the training range do CO₂ and temperature go in each scenario?
- **Exp 11b — baseline-plus-correction model.**
  - Baseline: the site's historical outcome if sites match, otherwise the growth-curve value.
  - A yearly GRU predicts the log-ratio to that baseline from climate *anomalies* plus the physics features, which also keeps predictions positive.
  - It's trained on the full 40-year rollout: the competition loss at year 40, plus an extra loss on all 7 variables every year and the carbon-balance penalty.
  - Height comes from a 50/50 blend of a direct prediction and the site's height-to-biomass curve.
  - **Model selection** uses a new holdout: the 10% of sites with the fastest warming and CO₂ rise, since plain site-grouped validation has been misleading all along.
- **Cheap leaderboard probes** (each needs your approval before submitting):
  - P1: the growth-curve baseline alone.
  - P2: the matched-site historical outcome alone.
  - Each costs one submission with no training, and tells us whether the idea behind it holds.

Nothing has been run or submitted. The plan is saved at `C:\Users\User 1\.claude\plans\you-are-claude-opus-linear-cascade.md`. I'd start with Exp 11a: it's cheap, and whether sites match decides which version of 11b we build.