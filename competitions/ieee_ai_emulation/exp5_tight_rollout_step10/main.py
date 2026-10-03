# ==============================================================================
# Autobot AI Emulation Exp 5: Tighter Autoregressive Rollout (STEP=10, 4 hops)
# Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)
#
# WHY THIS EXPERIMENT EXISTS
# ---------------------------------------------------------------------------
# exp4 (STEP=20, 2 hops: 0->20->40) tested the hypothesis that windowing
# climate into per-hop chunks instead of one 40-year aggregate reduces how
# far the model has to extrapolate on the future-SSP test climate. It found
# real, OOF-independent evidence for this: exp4's height clip-floor
# (positivity-clamp) hit rate on the real test set dropped to 6.24%, vs.
# exp2's 46.23% for the same direct-40-year-aggregate architecture. But
# exp4's own "rollout OOF" metric (0.034731 composite scaled MSE) was
# assessed as NOT a trustworthy LB predictor -- it's computed via
# GroupKFold-by-site on HISTORICAL climate only, chained through hop
# predictions, so it never sees the actual out-of-training-range future
# climate the real test set has (test-time diagnostics showed 100% of test
# rows / ~19.2% of all feature cells were out-of-range for exp4, with ZERO
# analog of that exposure anywhere in how the rollout-OOF number itself was
# computed).
#
# This experiment (exp5) is the cheap, submission-free next step recommended
# in THINKING_AND_DECISIONS.md section 10.3: halve STEP again, from 20 to
# 10 years (4 hops: 0->10->20->30->40 instead of 2), which should roughly
# halve the climate-extrapolation distance the model has to generalize
# across at each individual hop, moving closer to the README's prescribed
# full annual (STEP=1) rollout. The readout that matters BEFORE spending a
# real submission is whether the same OOF-independent signals exp4 showed
# (clip-floor hit rate, %-of-cells-out-of-range) keep improving in the same
# direction as exp2->exp4 did, not whether the (still structurally
# untrustworthy) rollout-OOF number goes up or down.
#
# All model/feature-engineering logic below is a direct generalization of
# exp4's code from a hardcoded 2-hop chain to an arbitrary N-hop chain
# (STEP_STARTS = [0, STEP, 2*STEP, ...]) -- same climate-window feature
# function, same fixed LightGBM hyperparameters, same GroupKFold-by-site
# validation, same "chain the fold model's own OOF prediction into the next
# hop" honesty principle for the rollout-OOF metric, same test-time
# extrapolation-range diagnostic. Only STEP/STEP_STARTS changed and the
# hardcoded-2-hop chaining logic (rollout-OOF section + rollout_scenario())
# was rewritten as a general loop over hops.
# ==============================================================================
import os
import sys
import gc
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=== AUTOBOT AI EMULATION EXP 5: TIGHTER AUTOREGRESSIVE ROLLOUT (STEP=10, 4 HOPS) ===")
print(f"Python Runtime: {sys.version.split()[0]}")
t0_global = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Path Resolution (same robust find_file pattern as exp3/exp4)
# ------------------------------------------------------------------------------
def find_file(name, search_dirs=('/kaggle/input', '.')):
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            if name in files:
                return os.path.join(root, name)
    raise FileNotFoundError(f'Cannot find {name}')

glob_x_path = find_file('glob_X_fea.npy')
BASE = os.path.dirname(os.path.dirname(glob_x_path))
print('Resolved CarbonGlobe BASE:', BASE)

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
AGE_FILES = {age: f"glob_Y{age:03d}.npy" for age in SEED_AGES}
ALL_TARGET_NAMES = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']
SCALE = {'height': 10.90645, 'agb': 3.39567}

# Only propagate the 4 "stock" state variables recursively (height, agb, soil,
# lai) -- same rationale as exp4: gpp/npp/rh are annual flux/rate diagnostics,
# dropped from both the input-state features and prediction targets to keep
# the state vector self-consistent.
STATE_VARS = ['height', 'agb', 'soil', 'lai']
STATE_IDX = [ALL_TARGET_NAMES.index(v) for v in STATE_VARS]  # [0, 1, 2, 3]

STEP = 10                              # years per rollout step (half of exp4's 20)
STEP_STARTS = list(range(0, 40, STEP))  # -> [0, 10, 20, 30]: 4 hops, still spans 0->40 exactly
N_HOPS = len(STEP_STARTS)
print(f'STEP={STEP}, STEP_STARTS={STEP_STARTS}, N_HOPS={N_HOPS} (exp4 used STEP=20, N_HOPS=2)')

SITE_SUBSAMPLE_FRAC = 0.20   # same site pool fraction as exp4 for an apples-to-apples comparison
NUM_FOLDS = 3                 # GroupKFold by site_id; kept small to bound CPU runtime
EPS = 1e-6

X = np.load(f'{BASE}/data_global/glob_X_fea.npy', mmap_mode='r')  # (54152, 40, 12, 136)
print('X shape:', X.shape)

train_sites = pd.read_csv(f'{BASE}/train.csv', header=None).iloc[:, 0].astype(int).values
val_sites = pd.read_csv(f'{BASE}/val.csv', header=None).iloc[:, 0].astype(int).values
test_sites_bench = pd.read_csv(f'{BASE}/test.csv', header=None).iloc[:, 0].astype(int).values
print('benchmark split sizes -- train/val/test:', len(train_sites), len(val_sites), len(test_sites_bench))

all_sites = np.unique(np.concatenate([train_sites, val_sites, test_sites_bench]))
rng = np.random.RandomState(RANDOM_SEED)
n_keep = int(len(all_sites) * SITE_SUBSAMPLE_FRAC)
all_sites = rng.choice(all_sites, size=n_keep, replace=False)
print('final labeled-site pool used for training/CV:', len(all_sites),
      f'({len(all_sites) / 54152:.1%} of all sites)')

stats = np.load(f'{BASE}/data_stats/data_stats.npz')
x_mean, x_std = stats['x_mean'], stats['x_std']


# ------------------------------------------------------------------------------
# 2. Feature engineering for one climate window [t0, t0+STEP) -- unchanged
#    from exp4, just called with a narrower STEP this round.
# ------------------------------------------------------------------------------
def climate_window_feats(raw_window):
    """raw_window: (n, STEP, 12, 136) raw-unit monthly climate. Returns (n, 4*136)."""
    raw_annual_mean = raw_window.mean(axis=2)                       # (n, STEP, 136) raw units
    Xn = (raw_annual_mean - x_mean) / (x_std + 1e-8)                 # z-scored annual means

    feat_mean = Xn.mean(axis=1)
    feat_std = Xn.std(axis=1)

    years = np.arange(raw_window.shape[1])
    years_c = years - years.mean()
    denom = (years_c ** 2).sum()
    feat_trend = (Xn * years_c[None, :, None]).sum(axis=1) / denom

    half = raw_window.shape[1] // 2
    raw_first = raw_annual_mean[:, :half, :].mean(axis=1)
    raw_second = raw_annual_mean[:, half:, :].mean(axis=1)
    rel_change = np.clip((raw_second - raw_first) / (np.abs(raw_first) + EPS), -10, 10)

    return np.concatenate([feat_mean, feat_std, feat_trend, rel_change], axis=1).astype(np.float32)


def build_feature_names():
    names = []
    for tag in ['mean', 'std', 'trend', 'relchange']:
        names += [f'clim_{tag}_{i}' for i in range(136)]
    names += [f'state_{v}' for v in STATE_VARS]
    names += ['stand_age']
    return names


FEATURE_NAMES = build_feature_names()
N_FEAT = len(FEATURE_NAMES)
print('n_features (per step-transition row):', N_FEAT)


# ------------------------------------------------------------------------------
# 3. Assemble the pooled step-transition training set
#    Rows = site_pool x seed_age x {0->10, 10->20, 20->30, 30->40} hops
#    X_all is a single stationary transition function (state, climate_window,
#    stand_age) -> next_state, applied identically at every rollout hop.
#    This section is architecturally identical to exp4's -- it already
#    generalizes to any STEP_STARTS list without changes.
# ------------------------------------------------------------------------------
print(f'\nReading climate block once for {len(all_sites)} sites (shared across all seed ages)...')
t_x = time.time()
Xf_full = np.array(X[all_sites], dtype=np.float32)  # (n, 40, 12, 136)
print(f'  climate block: {Xf_full.shape} ({Xf_full.nbytes / (1024**3):.2f} GB) in {time.time()-t_x:.1f}s')

window_feats_cache = {}
for t0 in STEP_STARTS:
    window_feats_cache[t0] = climate_window_feats(Xf_full[:, t0:t0 + STEP, :, :])
    print(f'  window[{t0}:{t0+STEP}] climate features ready: {window_feats_cache[t0].shape}')
del Xf_full
gc.collect()

n_sites = len(all_sites)
n_rows = n_sites * len(SEED_AGES) * N_HOPS
X_all = np.empty((n_rows, N_FEAT), dtype=np.float32)
y_all = np.empty((n_rows, len(STATE_VARS)), dtype=np.float32)
sid_all = np.empty(n_rows, dtype=np.int64)
hopstart_all = np.empty(n_rows, dtype=np.int32)  # which STEP_STARTS value this row belongs to

print(f'\nBuilding {n_rows:,} step-transition rows from {len(SEED_AGES)} seed-age files x {N_HOPS} hops...')
row = 0
for age in SEED_AGES:
    t_age = time.time()
    Y = np.load(f'{BASE}/data_global/{AGE_FILES[age]}', mmap_mode='r')  # (54152, 41, 12, 7)
    Yf = np.array(Y[all_sites], dtype=np.float32)  # (n, 41, 12, 7)
    for t0 in STEP_STARTS:
        t1 = t0 + STEP
        state_t = Yf[:, t0, -1, STATE_IDX]    # (n, 4) state at end-of-year t0 (Dec)
        state_t1 = Yf[:, t1, -1, STATE_IDX]   # (n, 4) target state at end-of-year t1
        stand_age = np.full((n_sites, 1), age + t0, dtype=np.float32)
        feats = np.concatenate([window_feats_cache[t0], state_t, stand_age], axis=1).astype(np.float32)

        lo, hi = row, row + n_sites
        X_all[lo:hi] = feats
        y_all[lo:hi] = state_t1
        sid_all[lo:hi] = all_sites
        hopstart_all[lo:hi] = t0
        row = hi
        del state_t, state_t1, feats
    del Yf
    gc.collect()
    print(f'  assembled age={age} in {time.time()-t_age:.1f}s ({row:,}/{n_rows:,} rows so far)')

del window_feats_cache
gc.collect()
print(f'X_all: {X_all.shape}  ({X_all.nbytes / (1024**3):.2f} GB)')


# ------------------------------------------------------------------------------
# 4. Train one LightGBM regressor per propagated state variable, GroupKFold
#    by site_id. Fixed, reasonably-regularized hyperparameters (no grid
#    search this round -- same as exp4, keeps CPU runtime bounded despite
#    2x the step-transition rows from doubling N_HOPS).
# ------------------------------------------------------------------------------
LGB_PARAMS = dict(
    objective='regression', metric='rmse', learning_rate=0.04,
    num_leaves=31, min_data_in_leaf=60, lambda_l1=0.5, lambda_l2=1.0,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
)
NUM_BOOST_ROUND = 2000
EARLY_STOP = 100

gkf = GroupKFold(n_splits=NUM_FOLDS)
fold_indices = list(gkf.split(X_all, groups=sid_all))

models = {v: [] for v in STATE_VARS}
oof_pred = {v: np.zeros(len(y_all), dtype=np.float32) for v in STATE_VARS}

print(f'\n=== Training {len(STATE_VARS)} state-transition targets x {NUM_FOLDS} folds ===')
for vi, var in enumerate(STATE_VARS):
    y_t = y_all[:, vi]
    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
        t_fold = time.time()
        train_set = lgb.Dataset(X_all[tr_idx], label=y_t[tr_idx], feature_name=FEATURE_NAMES)
        val_set = lgb.Dataset(X_all[va_idx], label=y_t[va_idx], reference=train_set)
        model = lgb.train(LGB_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND,
                           valid_sets=[val_set], valid_names=['val'],
                           callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(X_all[va_idx])
        oof_pred[var][va_idx] = pred
        models[var].append(model)
        rmse = mean_squared_error(y_t[va_idx], pred) ** 0.5
        print(f'  [{var} | Fold {fold_i+1}/{NUM_FOLDS}] best_iter={model.best_iteration} '
              f'val RMSE={rmse:.4f} ({time.time()-t_fold:.1f}s)')
    gc.collect()

for var in STATE_VARS:
    rmse = mean_squared_error(y_all[:, STATE_VARS.index(var)], oof_pred[var]) ** 0.5
    r2 = r2_score(y_all[:, STATE_VARS.index(var)], oof_pred[var])
    print(f'{var}: single-step OOF RMSE={rmse:.4f}  R2={r2:.4f}')


# ------------------------------------------------------------------------------
# 5. Honest "rollout OOF": chain the fold model's OWN predicted intermediate
#    states through ALL N_HOPS hops (not just 2, generalized from exp4),
#    using ground truth only for the very first hop's input state (state_0,
#    which is genuinely known at real inference time too), and compare the
#    resulting predicted FINAL state (t=40) to the real state_40 for
#    held-out sites. Same honesty principle as exp4, same caveat noted in
#    THINKING_AND_DECISIONS.md section 10.2: this still never exposes any
#    fold to out-of-training-range (future-SSP) climate, so it is not a
#    trustworthy absolute LB predictor -- it isolates compounding/chaining
#    error, not climate-distribution-shift error.
# ------------------------------------------------------------------------------
print('\n=== Rollout OOF (chains fold model\'s own OOF predictions through all hops) ===')

df_idx = pd.DataFrame({
    'sid': sid_all, 'hop': hopstart_all, 'row': np.arange(len(sid_all)),
    'stand_age': X_all[:, FEATURE_NAMES.index('stand_age')],
})
df_idx['seed_age'] = df_idx['stand_age'] - df_idx['hop']

chain_series = (df_idx.sort_values('hop')
                .groupby(['sid', 'seed_age'])['row']
                .apply(list))
chain_series = chain_series[chain_series.apply(len) == N_HOPS]  # defensive: keep only full chains
chain_arr = np.array(chain_series.tolist())                      # (n_groups, N_HOPS), hop-ordered
chain_sid = np.array([sid for sid, seed_age in chain_series.index])
print(f'paired rollout chains for eval: {len(chain_arr):,} (x{N_HOPS} hops each, '
      f'final target = state at t={STEP_STARTS[-1] + STEP})')

state_col_start = FEATURE_NAMES.index('state_height')
state_col_end = state_col_start + len(STATE_VARS)

rollout_true_h = y_all[chain_arr[:, -1], STATE_VARS.index('height')]
rollout_true_a = y_all[chain_arr[:, -1], STATE_VARS.index('agb')]
rollout_pred_h = np.zeros(len(chain_arr), dtype=np.float32)
rollout_pred_a = np.zeros(len(chain_arr), dtype=np.float32)

for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
    va_sites = set(sid_all[va_idx].tolist())
    fold_mask = np.isin(chain_sid, list(va_sites))
    if fold_mask.sum() == 0:
        continue
    fold_chains = chain_arr[fold_mask]

    pred_state = None
    cur_feats = None
    for h in range(N_HOPS):
        cur_feats = X_all[fold_chains[:, h]].copy()
        if h > 0:
            cur_feats[:, state_col_start:state_col_end] = pred_state  # substitute chained prediction
        pred_state = np.column_stack([
            models[v][fold_i].predict(cur_feats) for v in STATE_VARS
        ])
    # after the loop, pred_state holds the predicted state at the FINAL hop's target (t=40)
    rollout_pred_h[fold_mask] = pred_state[:, STATE_VARS.index('height')]
    rollout_pred_a[fold_mask] = pred_state[:, STATE_VARS.index('agb')]

scaled_mse_h = mean_squared_error(rollout_true_h / SCALE['height'], rollout_pred_h / SCALE['height'])
scaled_mse_a = mean_squared_error(rollout_true_a / SCALE['agb'], rollout_pred_a / SCALE['agb'])
rollout_scaled_mse = 0.5 * (scaled_mse_h + scaled_mse_a)
print(f'Rollout OOF scaled MSE (height): {scaled_mse_h:.6f}')
print(f'Rollout OOF scaled MSE (agb):    {scaled_mse_a:.6f}')
print('=======================================================')
print(f'ROLLOUT OOF COMPOSITE METRIC (SCALED MSE): {rollout_scaled_mse:.6f}')
print(f'Estimated Relative RMSE: {rollout_scaled_mse ** 0.5:.6f}')
print('(compare against exp1 OOF 0.0299/LB 0.274, exp2 OOF 0.0177/LB 0.330,')
print(' exp4 (STEP=20, 2 hops) rollout-OOF 0.034731 (not yet submitted as of exp5) --')
print(' per THINKING_AND_DECISIONS.md section 10.2, this number is NOT expected to be a')
print(' trustworthy LB predictor either way; the diagnostic that matters here is the')
print(' test-time clip-floor / out-of-range-cell stats below, same as exp2->exp4.)')
print('=======================================================')


# ------------------------------------------------------------------------------
# 6. Test inference: real autoregressive rollout on future SSP climate,
#    chained through all N_HOPS hops. Generalized from exp4's hardcoded
#    2-hop rollout_scenario() to loop over STEP_STARTS.
#    Ensembled by averaging predictions across the NUM_FOLDS fold models.
# ------------------------------------------------------------------------------
sites_ssp_file = find_file('sites_ssp.csv')
TEST_BASE = os.path.dirname(sites_ssp_file)
print('\nResolved TEST_BASE:', TEST_BASE)

sites_ssp = pd.read_csv(sites_ssp_file)
site_ids = sites_ssp.iloc[:, 0].astype(int).values
print(f'Loaded {len(site_ids)} test sites from sites_ssp.csv')

STATE_VARS_ALL_IDX = [ALL_TARGET_NAMES.index(v) for v in STATE_VARS]

train_min = X_all.min(axis=0)
train_max = X_all.max(axis=0)


def extrapolation_report(test_feats, scenario_name):
    below = test_feats < train_min[None, :]
    above = test_feats > train_max[None, :]
    out_of_range = below | above
    frac_rows_flagged = (out_of_range.any(axis=1)).mean()
    frac_cells_flagged = out_of_range.mean()
    worst = pd.Series(out_of_range.mean(axis=0), index=FEATURE_NAMES).sort_values(ascending=False).head(8)
    print(f'--- {scenario_name} ---')
    print(f'{frac_rows_flagged:.1%} of rows have >=1 out-of-training-range feature')
    print(f'{frac_cells_flagged:.2%} of all (row, feature) cells are out of range')
    print('most-affected features:')
    print(worst)
    print()


def rollout_scenario(npz_path, scenario_name):
    d = np.load(npz_path)
    test_x, test_y0 = d['test_x'].astype(np.float32), d['test_y0'].astype(np.float32)  # (n,40,12,136) / (n,15,7)
    n_test = test_x.shape[0]

    clim_windows = [climate_window_feats(test_x[:, t0:t0 + STEP, :, :]) for t0 in STEP_STARTS]

    all_feats_for_report = []
    rows = []
    for age_i, age in enumerate(SEED_AGES):
        state = test_y0[:, age_i, STATE_VARS_ALL_IDX]  # (n_test, 4) TRUE initial state at t=0

        for h, t0 in enumerate(STEP_STARTS):
            stand_age_h = np.full((n_test, 1), age + t0, dtype=np.float32)
            feats_h = np.concatenate([clim_windows[h], state, stand_age_h], axis=1).astype(np.float32)
            all_feats_for_report.append(feats_h)
            state = np.column_stack([
                np.mean([models[v][f].predict(feats_h) for f in range(NUM_FOLDS)], axis=0)
                for v in STATE_VARS
            ])
        # after the loop, `state` holds the predicted FINAL (t=40) state, chained through all hops

        pred_h = state[:, STATE_VARS.index('height')]
        pred_a = state[:, STATE_VARS.index('agb')]
        pred_h_scaled = np.clip(pred_h / SCALE['height'], 1e-3, None)
        pred_a_scaled = np.clip(pred_a / SCALE['agb'], 1e-3, None)

        rows.append(pd.DataFrame({'site': site_ids, 'age': age, 'height': pred_h_scaled, 'agb': pred_a_scaled}))

    all_feats = np.concatenate(all_feats_for_report, axis=0)
    return pd.concat(rows, ignore_index=True), all_feats


print('\n>>> Commencing Test Set Inference (autoregressive rollout) on Future Scenarios...')

df_126, feats_126 = rollout_scenario(find_file('test_ssp126.npz'), 'SSP126')
df_126['id'] = 'ssp126_' + df_126['site'].astype(str) + '_' + df_126['age'].astype(str)
extrapolation_report(feats_126, 'ssp126 (low emissions)')

df_585, feats_585 = rollout_scenario(find_file('test_ssp585.npz'), 'SSP585')
df_585['id'] = 'ssp585_' + df_585['site'].astype(str) + '_' + df_585['age'].astype(str)
extrapolation_report(feats_585, 'ssp585 (high emissions)')

submission = pd.concat([df_126, df_585], ignore_index=True)[['id', 'height', 'agb']]

sample_sub = pd.read_csv(find_file('sample_submission.csv'))
submission = sample_sub[['id']].merge(submission, on='id', how='left')

out_path = Path('/kaggle/working/submission.csv') if Path('/kaggle/working').exists() else Path('submission.csv')
submission.to_csv(out_path, index=False)
print(f'Successfully generated {out_path} ({out_path.stat().st_size / (1024**2):.2f} MB)')

# --- Integrity Contracts (same as exp2/exp3/exp4) ---
print('\n--- Integrity Diagnostic Summary ---')
assert submission.shape[0] == 185130, f'Expected 185,130 rows, got {submission.shape[0]}'
assert not submission[['height', 'agb']].isna().any().any(), 'Fatal: Null values found in predictions!'
assert (submission['height'] > 0).all(), 'Fatal: Negative or zero height predictions found!'
assert (submission['agb'] > 0).all(), 'Fatal: Negative or zero agb predictions found!'
print(f'Total Rows: {submission.shape[0]:,}')
print(f"Height Stats -> Mean: {submission['height'].mean():.4f}, Std: {submission['height'].std():.4f}, "
      f"Min: {submission['height'].min():.4f}, Max: {submission['height'].max():.4f}")
print(f"AGB Stats    -> Mean: {submission['agb'].mean():.4f}, Std: {submission['agb'].std():.4f}, "
      f"Min: {submission['agb'].min():.4f}, Max: {submission['agb'].max():.4f}")
# Explicit clip-floor hit-rate readout -- this is the OOF-independent signal
# THINKING_AND_DECISIONS.md section 10.3 said to track before spending a submission.
clip_h = int(np.isclose(submission['height'], 1e-3, atol=1e-9).sum())
clip_a = int(np.isclose(submission['agb'], 1e-3, atol=1e-9).sum())
print(f"Height clip-floor (1e-3) hit rate: {clip_h:,} / {len(submission):,} ({clip_h/len(submission)*100:.2f}%) "
      f"[exp2: 46.23%, exp4 (STEP=20): 6.24%]")
print(f"AGB clip-floor (1e-3) hit rate:    {clip_a:,} / {len(submission):,} ({clip_a/len(submission)*100:.2f}%) "
      f"[exp2: 0.00%, exp4 (STEP=20): 3.24%]")
print('\nFirst 5 Submission Rows:')
print(submission.head())
print(f'\n=== PIPELINE EXECUTION SUCCESSFUL: ALL CONTRACTS SATISFIED ({time.time()-t0_global:.1f}s) ===')
