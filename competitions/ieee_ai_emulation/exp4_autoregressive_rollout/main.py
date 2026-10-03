# ==============================================================================
# Autobot AI Emulation Exp 4: Autoregressive Rollout (Stepwise State Dynamics)
# Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)
#
# WHY THIS EXPERIMENT EXISTS
# ---------------------------------------------------------------------------
# exp1/exp2/exp3 (and the one public notebook available for this competition)
# all take the SAME shortcut: summarize the full 40-year climate record into
# aggregate statistics (mean/std/trend/decade means/...), concatenate the
# YEAR-0 initial state, and regress DIRECTLY onto the YEAR-40 final state in
# one shot. The competition's own README says otherwise:
#
#   "roll your model out 40 annual steps from the provided initial state and
#    report the year-40 (final) state ... watch long-horizon cumulative drift"
#
# The direct-regression approach's own numbers make the problem visible:
#   exp1 OOF scaled-MSE 0.0299  -> real LB score 0.274   (~9x worse)
#   exp2 OOF scaled-MSE 0.0177  -> real LB score 0.330   (~19x worse)
# and exp2's submitted height predictions hit a hard floor clip (1e-3) for a
# non-trivial share of rows -- i.e. the raw model went negative. Trees can't
# extrapolate past the range of the (historical-climate) features they were
# trained on, and the test set is FUTURE ssp126/ssp585 climate, which is
# guaranteed to drift outside that historical range the longer the
# aggregation window is. A 40-year-aggregate feature is about as far outside
# the training distribution as you can get.
#
# This experiment instead trains a STATE-TRANSITION model:
#   (state_t, climate over [t, t+STEP))  ->  state_{t+STEP}
# pooled across many (site, seed_age, step-start) examples, then RUNS IT
# RECURSIVELY at inference time -- exactly the rollout the README describes.
# Each individual step only has to generalize across a STEP-year climate
# window instead of a 40-year one, which should be a smaller extrapolation
# jump, and the model only ever has to learn one "local" dynamics function
# instead of memorizing a distant point-to-point mapping.
#
# We also build an honest "rollout OOF" metric (chaining the fold model's own
# OOF-predicted intermediate state into the next step, not ground truth) so
# we get an estimate of the compounding-drift error the README warns about,
# rather than just the (very optimistic, per exp1-3's evidence) single-step
# OOF number.
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

print("=== AUTOBOT AI EMULATION EXP 4: AUTOREGRESSIVE ROLLOUT (STEPWISE DYNAMICS) ===")
print(f"Python Runtime: {sys.version.split()[0]}")
t0_global = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Path Resolution (same robust find_file pattern as exp3)
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
# lai). gpp/npp/rh are annual flux/rate diagnostics; modeling+propagating them
# too would triple the model count for uncertain benefit at this iteration,
# so they're dropped from both the input-state features and prediction
# targets to keep the state vector self-consistent (nothing references a
# variable we don't produce).
STATE_VARS = ['height', 'agb', 'soil', 'lai']
STATE_IDX = [ALL_TARGET_NAMES.index(v) for v in STATE_VARS]  # [0, 1, 2, 3]

STEP = 20            # years per rollout step
STEP_STARTS = [0, 20]  # -> two hops: 0->20, 20->40 (matches the 40-yr horizon exactly)
N_HOPS = len(STEP_STARTS)

SITE_SUBSAMPLE_FRAC = 0.20   # ~10,830 labeled sites x 15 ages x 2 hops ~= 325k rows
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
# 2. Feature engineering for one climate window [t0, t0+STEP)
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
#    Rows = site_pool x seed_age x {0->20, 20->40} hops
#    X_all is a single stationary transition function (state, climate_window,
#    stand_age) -> next_state, applied identically at every rollout hop.
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
hopstart_all = np.empty(n_rows, dtype=np.int32)  # 0 or 20 -- which hop this row belongs to

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
#    search this round -- keeps CPU runtime bounded; a natural exp5 follow-up
#    if this architecture beats exp1-3 on the real leaderboard).
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
# 5. Honest "rollout OOF": chain the fold model's OWN predicted state_20 (not
#    ground truth) into the second-hop model, exactly mirroring how test
#    inference will work, and compare the resulting predicted state_40 to the
#    real state_40 for held-out sites. This is the number that should be
#    compared against exp1-3's OOF (0.0299 / 0.0177) and their real LB scores
#    (0.274 / 0.330) -- it's expected to be worse than single-step OOF (some
#    drift is real) but the open question this experiment tests is whether
#    it's dramatically closer to the eventual LB score than exp1-3's OOF was.
# ------------------------------------------------------------------------------
print('\n=== Rollout OOF (chained hop-1 prediction -> hop-2 input) ===')
hop0_mask = (hopstart_all == 0)
hop20_mask = (hopstart_all == 20)

# hop-0 rows and hop-20 rows for the same (site, seed_age) are adjacent in
# construction order (assembled together per age/hop loop) but let's align
# them robustly via (site_id, stand_age relationship) instead of assuming
# order. hop-0 row's stand_age = seed_age; hop-20 row's stand_age = seed_age+20.
df_idx = pd.DataFrame({
    'sid': sid_all, 'hop': hopstart_all, 'row': np.arange(len(sid_all)),
    'stand_age': X_all[:, FEATURE_NAMES.index('stand_age')],
})
df_idx['seed_age'] = np.where(df_idx['hop'] == 0, df_idx['stand_age'], df_idx['stand_age'] - 20)

hop0 = df_idx[df_idx['hop'] == 0][['sid', 'seed_age', 'row']].rename(columns={'row': 'row0'})
hop20 = df_idx[df_idx['hop'] == 20][['sid', 'seed_age', 'row']].rename(columns={'row': 'row20'})
pairs = hop0.merge(hop20, on=['sid', 'seed_age'], how='inner')
print(f'paired hop0/hop20 rows for rollout eval: {len(pairs):,}')

rollout_pred_h = np.zeros(len(pairs), dtype=np.float32)
rollout_pred_a = np.zeros(len(pairs), dtype=np.float32)
rollout_true_h = y_all[pairs['row20'].values, STATE_VARS.index('height')]
rollout_true_a = y_all[pairs['row20'].values, STATE_VARS.index('agb')]

# hop-20 feature columns for climate/age; only the 'state_*' block needs to be
# swapped out for the hop-0 model's OOF-predicted state.
state_col_start = FEATURE_NAMES.index('state_height')
state_col_end = state_col_start + len(STATE_VARS)

for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
    va_sites = set(sid_all[va_idx].tolist())
    fold_pairs_mask = pairs['sid'].isin(va_sites).values
    if fold_pairs_mask.sum() == 0:
        continue
    row0_fold = pairs.loc[fold_pairs_mask, 'row0'].values
    row20_fold = pairs.loc[fold_pairs_mask, 'row20'].values

    # predicted state_20 (all 4 vars) for these held-out sites, from fold's hop-0-capable models
    pred_state20 = np.column_stack([models[v][fold_i].predict(X_all[row0_fold]) for v in STATE_VARS])

    feats_hop2 = X_all[row20_fold].copy()
    feats_hop2[:, state_col_start:state_col_end] = pred_state20  # substitute predicted state

    pred_h = models['height'][fold_i].predict(feats_hop2)
    pred_a = models['agb'][fold_i].predict(feats_hop2)
    rollout_pred_h[fold_pairs_mask] = pred_h
    rollout_pred_a[fold_pairs_mask] = pred_a

scaled_mse_h = mean_squared_error(rollout_true_h / SCALE['height'], rollout_pred_h / SCALE['height'])
scaled_mse_a = mean_squared_error(rollout_true_a / SCALE['agb'], rollout_pred_a / SCALE['agb'])
rollout_scaled_mse = 0.5 * (scaled_mse_h + scaled_mse_a)
print(f'Rollout OOF scaled MSE (height): {scaled_mse_h:.6f}')
print(f'Rollout OOF scaled MSE (agb):    {scaled_mse_a:.6f}')
print('=======================================================')
print(f'ROLLOUT OOF COMPOSITE METRIC (SCALED MSE): {rollout_scaled_mse:.6f}')
print(f'Estimated Relative RMSE: {rollout_scaled_mse ** 0.5:.6f}')
print('(compare against exp1 OOF 0.0299 / LB 0.274, exp2 OOF 0.0177 / LB 0.330 --')
print(' this rollout number should sit closer to the eventual LB score if the')
print(' hypothesis -- that the OOF/LB gap is driven by extrapolation on 40-yr')
print(' aggregate features -- is correct.)')
print('=======================================================')


# ------------------------------------------------------------------------------
# 6. Test inference: real autoregressive rollout on future SSP climate.
#    hop 1: state_0 (from test_y0) + climate[0:20]  -> predicted state_20
#    hop 2: predicted state_20     + climate[20:40] -> predicted state_40
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

    clim_hop1 = climate_window_feats(test_x[:, 0:STEP, :, :])
    clim_hop2 = climate_window_feats(test_x[:, STEP:2 * STEP, :, :])

    all_feats_for_report = []
    rows = []
    for age_i, age in enumerate(SEED_AGES):
        state0 = test_y0[:, age_i, STATE_VARS_ALL_IDX]  # (n_test, 4) initial state at this seed age

        # --- hop 1: state_0 -> predicted state_20 ---
        stand_age1 = np.full((n_test, 1), age, dtype=np.float32)
        feats1 = np.concatenate([clim_hop1, state0, stand_age1], axis=1).astype(np.float32)
        all_feats_for_report.append(feats1)
        pred_state20 = np.column_stack([
            np.mean([models[v][f].predict(feats1) for f in range(NUM_FOLDS)], axis=0)
            for v in STATE_VARS
        ])

        # --- hop 2: predicted state_20 -> predicted state_40 ---
        stand_age2 = np.full((n_test, 1), age + STEP, dtype=np.float32)
        feats2 = np.concatenate([clim_hop2, pred_state20, stand_age2], axis=1).astype(np.float32)
        all_feats_for_report.append(feats2)
        pred_h = np.mean([models['height'][f].predict(feats2) for f in range(NUM_FOLDS)], axis=0)
        pred_a = np.mean([models['agb'][f].predict(feats2) for f in range(NUM_FOLDS)], axis=0)

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

# --- Integrity Contracts (same as exp2/exp3) ---
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
print('\nFirst 5 Submission Rows:')
print(submission.head())
print(f'\n=== PIPELINE EXECUTION SUCCESSFUL: ALL CONTRACTS SATISFIED ({time.time()-t0_global:.1f}s) ===')
