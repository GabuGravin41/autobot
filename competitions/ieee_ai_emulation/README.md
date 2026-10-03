# CarbonGlobe Challenge - Test Data (IEEE Big Data Cup 2026)

Forecast the **future** forest carbon cycle. Train on the CarbonGlobe present-day dataset, then for
each **site** and each **seed forest age** in the two future-climate scenarios, roll your model out
**40 annual steps** from the provided initial state and report the **year-40 (final) state** of two
variables: **height** and **agb** (the long-horizon *cumulative* outcome).

## Files in this folder
| file | description |
|---|---|
| `test_ssp126.npz`, `test_ssp585.npz` | model inputs per scenario (`test_x`, `test_y0`) |
| `sites_ssp.csv` | site index for each axis-0 row (shared by both scenarios) |
| `sample_submission.csv` | the exact rows/order to submit (height & agb zero-filled) |
| `README.md` | this file |

## Scenarios (both scored; one submission covers both)
- `ssp126` - SSP1, low emissions    - `ssp585` - SSP5, high emissions
Each covers **6171 globally-distributed sites** and **15 seed ages**. A hidden random split of the
sites forms the **Public** (live) and **Private** (final) leaderboards; your live score previews the
final one. Climate inputs use the real 12 monthly values; targets are annual.

## Public inputs - `public/test_{ssp126,ssp585}.npz`
| array | shape | meaning |
|---|---|---|
| `test_x`  | (6171, 40, 12, 136) | environmental forcing: site x year x month x 136 drivers |
| `test_y0` | (6171, 15, 7)       | initial ecosystem state (year 0) per site x seed age |

Axis-0 site order matches `sites_ssp.csv`. Seed ages (axis 1 of `test_y0`): [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500].
Target order (last axis of `test_y0`): ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh'].

## What to submit - `submission.csv`
One row per **(scenario, site, seed age)** giving the **year-40** height & agb, **each divided by
its global mean** (height/10.90645, agb/3.39567):
```
id,height,agb
ssp126_0_1,0.83,1.12
ssp585_6170_500,0.61,1.43
```
`id = "{scenario}_{site}_{age}"`: `scenario` in {ssp126, ssp585}; `site` from `sites_ssp.csv`;
`age` in [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500] (185,130 rows total). See `sample_submission.csv` for the exact rows/order.

> **Scale before you submit.** Divide your predicted year-40 `height` by 10.90645 and `agb` by 3.39567.
> The solution is stored pre-scaled, so an **unscaled** submission scores as if it were completely
> wrong. `tutorials/03_submission_tutorial.ipynb` shows the one-line step.

## Metric
**Mean squared error (MSE)** over the two pre-scaled columns -- Kaggle's built-in MSE (lower is
better). Because values are already /mean, this is the mean of squared *relative* errors at year 40:
`score = 1/2 * ( MSE_height + MSE_agb )`, where each variable's error is divided by its global mean
(height/10.90645, agb/3.39567). `sqrt(score)` is the relative RMSE.
