I read `competitions/ieee_ai_emulation/exp5_tight_rollout_step10/main.py` and based this on what it actually does. Going from STEP=10 to STEP=5 will probably help somewhat, but the gains have been shrinking with each halving (0.330 → 0.270 → 0.222). Getting from 0.22 to the top 10's 0.05–0.08 will take structural fixes, not a shorter step.

## 1. Predict the change, not the next state, and in log space

Exp 5 trains on the absolute next state (`y_all = state_t1`, line 197).
- **The problem:** a tree model can only output values it saw in training. Asking it for the absolute next state of an unusual stand pulls the answer back toward the training mean. Repeat that over 8 hops and it compounds.
- **Why a change target works better:** 5-year changes are small, smoother, and more similar across sites. When the climate input is outside the training range, the tree gives the change from the edge of that range. Adding that onto the current state still moves it in a sensible direction.
- **Use the log change:** `Δ = log1p(s_{t+5}) − log1p(s_t)`, rebuilt as `expm1(log1p(s_t) + Δ)`. States can never go negative, young and old stands get handled on the same scale, and tree-growth changes are roughly multiplicative anyway.

**Probably the biggest lever:** `test_y0` gives the same site at 15 seed ages (1, 10, …, 500) at t=0. Exp 5 only uses one row of that per prediction.
- Interpolate across those ages (in log-age) to get the site's own growth curve under current climate.
- Add the growth-curve change `chrono(age+t+5) − chrono(age+t)` as a feature.
- Train on the difference between the actual change and that growth-curve change.

The model then only has to learn how climate bends a curve the site already shows, and the same data exists in training (`Y{age}[:, 0]`). I'd guess this is how the top teams get to 0.05, but I haven't confirmed it.

## 2. Handling climate outside the training range in LightGBM

When a feature is outside the training range, LightGBM uses the edge bin, so it effectively treats the extreme value as the most extreme one it saw. Exp 5 found 100% of test rows had at least one out-of-range feature. Four ways to reduce that:
- **Measure climate as the change from the site's own baseline:** the window's climate minus the site's own climate in the first window, instead of global z-scores. The shift relative to training shrinks a lot, and the model learns sensitivities rather than memorising absolute levels.
- **Adversarial check:** train a classifier to tell training climate windows from SSP585 windows. Drop or coarsen the features that separate them most easily, which are likely CO2 and the temperature extremes.
- **Monotone constraints:** +1 on the `state_*` features and the growth-curve features, and on CO2 if it's one of the 136 channels (check). This stops non-physical reversals in regions the model never saw.
- **Allow some extrapolation:** try `linear_tree=True`, or a ridge model on the climate changes plus a GBM on what's left. Either lets the model follow a trend beyond the training range instead of flattening.

## 3. Keeping errors from growing across 8 hops

1. **Clip after every hop, not just at the end.** Exp 5 only applies the 1e-3 floor after the last hop (line 393), so a negative or implausible state goes straight into the next hop (line 385). At each hop:
   - clip the change to the 0.1–99.9 percentile range for its age group;
   - clip the state to that age group's plausible range (at most 1.05× the training max);
   - keep agb within ±3σ of what the predicted height implies.
2. **Train on the model's own predictions (DAgger).** Exp 5 only trains on true states, so it never learns to handle its own errors, and that gets worse with 8 hops. Refit hops 1–7 on rows built from the chained fold predictions, and keep the true-state rows as well.
3. **Give the model more than the current state.** The 4 state variables don't capture everything the underlying forest model tracks, so the next state isn't fully determined by them. Add the initial state `s_0` and the growth-curve features as fixed per-site inputs.
4. **Check drift before spending a submission.** Plot the mean and spread of predicted states against the growth curve at each hop, and the share of states hitting the clip limits per hop. If the error grows geometrically, clipping alone won't save the 8-hop version.

## Suggested ablation order

1. Log-change target plus clipping at every hop. Low risk, and it probably fixes most of the drift.
2. Add the growth-curve (chronosequence) prior. Likely the largest single gain.
3. Add DAgger retraining.
4. Add baseline-relative climate features and monotone constraints.

The full plan, including the verification steps, is in `C:\Users\User 1\.claude\plans\you-are-an-expert-sharded-nebula.md`. Plan mode is on, so I haven't written any code. Once you approve, I'd create `exp6_step5_residual_rollout/` from exp5's `main.py`.

One more thing: the Slack connector needs authorising in your claude.ai connector settings before I can use it.