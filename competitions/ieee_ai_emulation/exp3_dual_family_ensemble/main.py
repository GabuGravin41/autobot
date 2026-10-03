# 0. Setup

import gc
import itertools
import random
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
print('CatBoost available:', HAS_CATBOOST, '(optional — only used if present, nothing breaks without it)')

import os
from pathlib import Path

def find_file(name, search_dirs=['/kaggle/input', '.']):
    for d in search_dirs:
        for root, dirs, files in os.walk(d):
            if name in files:
                return os.path.join(root, name)
    raise FileNotFoundError(f'Cannot find {name}')

glob_x_path = find_file('glob_X_fea.npy')
BASE = os.path.dirname(os.path.dirname(glob_x_path))
print('Resolved CarbonGlobe BASE:', BASE)

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
AGE_FILES = {age: f"glob_Y{age:03d}.npy" for age in SEED_AGES}
TARGET_NAMES = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']
SCALE = {'height': 10.90645, 'agb': 3.39567}

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- knobs to trade compute for score; defaults chase leaderboard score ----
USE_ALL_LABELED_SITES = True   # False = reproduce v3's conservative train-only behavior
SITE_SUBSAMPLE_FRAC = 0.5      # 27,000 sites x 15 ages = 405,000 training instances
NUM_FOLDS = 5                  # folds for both hyperparam CV and final bagging
HPO_SEARCH_SITES = 4000        # sites used for the (fast) hyperparameter search only


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'glob_X_fea' in filename:
            print(os.path.join(dirname, filename))


X = np.load(f'{BASE}/data_global/glob_X_fea.npy', mmap_mode='r')
print('X shape:', X.shape)

train_sites = pd.read_csv(f'{BASE}/train.csv', header=None).iloc[:, 0].astype(int).values
val_sites   = pd.read_csv(f'{BASE}/val.csv',   header=None).iloc[:, 0].astype(int).values
test_sites  = pd.read_csv(f'{BASE}/test.csv',  header=None).iloc[:, 0].astype(int).values
print('benchmark split sizes -- train/val/test:', len(train_sites), len(val_sites), len(test_sites))

if USE_ALL_LABELED_SITES:
    all_sites = np.concatenate([train_sites, val_sites, test_sites])
else:
    all_sites = train_sites.copy()

all_sites = np.unique(all_sites)
if SITE_SUBSAMPLE_FRAC < 1.0:
    rng = np.random.RandomState(RANDOM_SEED)
    n_keep = int(len(all_sites) * SITE_SUBSAMPLE_FRAC)
    all_sites = rng.choice(all_sites, size=n_keep, replace=False)

print('final labeled-site pool used for training/CV:', len(all_sites),
      f'({len(all_sites) / 54152:.1%} of all sites)')

stats = np.load(f'{BASE}/data_stats/data_stats.npz')
x_mean, x_std = stats['x_mean'], stats['x_std']


EPS = 1e-6

def extract_climate_features_for_sites(site_idx_array):
    print(f'Precomputing climate feature matrix for {len(site_idx_array)} sites...')
    t_c = time.time()
    Xf = np.array(X[site_idx_array], dtype=np.float32)
    raw_annual_mean = Xf.mean(axis=2)   # (n, 40, 136)
    annual_mean = (raw_annual_mean - x_mean) / (x_std + 1e-8)

    feat_mean  = annual_mean.mean(axis=1)
    feat_std   = annual_mean.std(axis=1)
    feat_last5 = annual_mean[:, -5:, :].mean(axis=1)

    years = np.arange(40)
    years_c = years - years.mean()
    denom = (years_c ** 2).sum()
    feat_trend = (annual_mean * years_c[None, :, None]).sum(axis=1) / denom

    decade_feats = np.concatenate(
        [annual_mean[:, d*10:(d+1)*10, :].mean(axis=1) for d in range(4)], axis=1
    )

    raw_early = raw_annual_mean[:, :5, :].mean(axis=1)
    raw_late  = raw_annual_mean[:, -5:, :].mean(axis=1)
    rel_change = (raw_late - raw_early) / (np.abs(raw_early) + EPS)
    rel_change = np.clip(rel_change, -10, 10)

    climate_feats = np.concatenate(
        [feat_mean, feat_std, feat_trend, feat_last5, decade_feats, rel_change], axis=1
    ).astype(np.float32)
    del Xf, raw_annual_mean, annual_mean
    gc.collect()
    print(f'Climate feature matrix ready: {climate_feats.shape} ({climate_feats.nbytes / (1024**2):.1f} MB) in {time.time()-t_c:.1f}s')
    return climate_feats


def build_feature_names():
    names = [f'x_mean_{i}' for i in range(136)]
    names += [f'x_std_{i}' for i in range(136)]
    names += [f'x_trend_{i}' for i in range(136)]
    names += [f'x_last5_{i}' for i in range(136)]
    for d in range(4):
        names += [f'x_decade{d}_{i}' for i in range(136)]
    names += [f'x_relchange_{i}' for i in range(136)]
    names += [f'init_{t}' for t in TARGET_NAMES]
    names += [f'initrate_{t}' for t in TARGET_NAMES]
    names += ['seed_age']
    return names

FEATURE_NAMES = build_feature_names()
print('n_features:', len(FEATURE_NAMES))


def assemble(site_idx_array, ages=SEED_AGES):
    n_sites = len(site_idx_array)
    n_rows = n_sites * len(ages)
    n_feat = len(FEATURE_NAMES)

    climate_feats = extract_climate_features_for_sites(site_idx_array)

    X_out = np.empty((n_rows, n_feat), dtype=np.float32)
    y_out = np.empty((n_rows, 2), dtype=np.float32)
    sid_out = np.empty(n_rows, dtype=np.int64)

    for i, age in enumerate(ages):
        Y = np.load(f'{BASE}/data_global/{AGE_FILES[age]}', mmap_mode='r')
        Yf = np.array(Y[site_idx_array], dtype=np.float32)
        init_state = Yf[:, 0, -1, :]
        init_rate = init_state / (age + EPS)
        age_col = np.full((n_sites, 1), age, dtype=np.float32)

        feats = np.concatenate([climate_feats, init_state, init_rate, age_col], axis=1).astype(np.float32)
        target = Yf[:, 40, -1, :2]

        lo, hi = i * n_sites, (i + 1) * n_sites
        X_out[lo:hi] = feats
        y_out[lo:hi] = target
        sid_out[lo:hi] = site_idx_array
        del Yf, init_state, init_rate, feats, target
        gc.collect()
        print(f'  assembled age={age} ({i + 1}/{len(ages)})')

    del climate_feats
    gc.collect()
    return X_out, y_out, sid_out

print('Building features for', len(all_sites), 'sites x', len(SEED_AGES), 'ages'
      f' -> {len(all_sites) * len(SEED_AGES):,} rows x {len(FEATURE_NAMES)} features'
      ' (this is the slow, memory-heavy step)')
X_all, y_all, sid_all = assemble(all_sites)
print('X_all:', X_all.shape)


rng = np.random.RandomState(RANDOM_SEED)
if len(all_sites) > HPO_SEARCH_SITES:
    hpo_site_pool = rng.choice(all_sites, size=HPO_SEARCH_SITES, replace=False)
else:
    hpo_site_pool = all_sites
hpo_mask = np.isin(sid_all, hpo_site_pool)
X_hpo, y_hpo, sid_hpo = X_all[hpo_mask], y_all[hpo_mask], sid_all[hpo_mask]
print('HPO subsample:', X_hpo.shape)

LGB_GRID = [
    dict(num_leaves=nl, min_data_in_leaf=md_, lambda_l1=l1, lambda_l2=l2,
         feature_fraction=ff, max_depth=depth)
    for nl, md_, l1, l2, ff, depth in itertools.product(
        [15, 31], [30, 80], [0.0, 0.5], [0.5, 1.0], [0.6, 0.8], [-1, 6])
]
LGB_GRID_STRONG = [
    dict(num_leaves=nl, min_data_in_leaf=md_, lambda_l1=l1, lambda_l2=l2,
         feature_fraction=ff, max_depth=depth)
    for nl, md_, l1, l2, ff, depth in itertools.product(
        [7, 15], [150, 300], [1.0, 2.0], [2.0, 5.0], [0.5, 0.6], [4, 6])
]

XGB_GRID = [
    dict(max_depth=depth, min_child_weight=mcw, reg_alpha=a, reg_lambda=l, subsample=ss, colsample_bytree=cs)
    for depth, mcw, a, l, ss, cs in itertools.product(
        [4, 6], [5, 20], [0.0, 0.5], [1.0, 2.0], [0.7, 0.8], [0.6, 0.8])
]
XGB_GRID_STRONG = [
    dict(max_depth=depth, min_child_weight=mcw, reg_alpha=a, reg_lambda=l, subsample=ss, colsample_bytree=cs)
    for depth, mcw, a, l, ss, cs in itertools.product(
        [2, 3], [40, 80], [1.0, 2.0], [3.0, 5.0], [0.6, 0.7], [0.5, 0.6])
]

def lgb_cv_score(X_tr, y_tr, groups, params, n_splits=4, num_boost_round=1200):
    gkf = GroupKFold(n_splits=n_splits)
    fold_rmses = []
    for tr_idx, va_idx in gkf.split(X_tr, groups=groups):
        train_set = lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx])
        val_set = lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx], reference=train_set)
        full_params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                            bagging_fraction=0.8, bagging_freq=1, verbose=-1, **params)
        model = lgb.train(full_params, train_set, num_boost_round=num_boost_round,
                           valid_sets=[val_set],
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(X_tr[va_idx])
        fold_rmses.append(mean_squared_error(y_tr[va_idx], pred) ** 0.5)
    return np.mean(fold_rmses)

def xgb_cv_score(X_tr, y_tr, groups, params, n_splits=4, num_boost_round=1200):
    gkf = GroupKFold(n_splits=n_splits)
    fold_rmses = []
    for tr_idx, va_idx in gkf.split(X_tr, groups=groups):
        dtrain = xgb.DMatrix(X_tr[tr_idx], label=y_tr[tr_idx])
        dval = xgb.DMatrix(X_tr[va_idx], label=y_tr[va_idx])
        full_params = dict(objective='reg:squarederror', eta=0.05, eval_metric='rmse', tree_method='hist', **params)
        model = xgb.train(full_params, dtrain, num_boost_round=num_boost_round,
                           evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        fold_rmses.append(mean_squared_error(y_tr[va_idx], pred) ** 0.5)
    return np.mean(fold_rmses)

def search_best(cv_fn, grid_moderate, grid_strong, target_idx, target_name, n_moderate, n_strong):
    grid = random.sample(grid_moderate, min(n_moderate, len(grid_moderate)))
    if target_name == 'agb':
        grid += random.sample(grid_strong, min(n_strong, len(grid_strong)))
    print(f'  searching {len(grid)} combos for {target_name}')
    scores = []
    for params in grid:
        s = cv_fn(X_hpo, y_hpo[:, target_idx], sid_hpo, params)
        scores.append((s, params))
    scores.sort(key=lambda t: t[0])
    print(f'  BEST {target_name}: CV RMSE {scores[0][0]:.4f} | {scores[0][1]}')
    return scores[0][1]

best_lgb_params, best_xgb_params = {}, {}
for target_idx, target_name in enumerate(['height', 'agb']):
    print(f'=== LightGBM: tuning {target_name} ===')
    best_lgb_params[target_name] = search_best(lgb_cv_score, LGB_GRID, LGB_GRID_STRONG,
                                                target_idx, target_name, n_moderate=8, n_strong=8)
    print(f'=== XGBoost: tuning {target_name} ===')
    best_xgb_params[target_name] = search_best(xgb_cv_score, XGB_GRID, XGB_GRID_STRONG,
                                                target_idx, target_name, n_moderate=8, n_strong=8)

del X_hpo, y_hpo, sid_hpo
gc.collect()


gkf = GroupKFold(n_splits=NUM_FOLDS)
fold_indices = list(gkf.split(X_all, groups=sid_all))

lgb_models = {'height': [], 'agb': []}
xgb_models = {'height': [], 'agb': []}
oof_lgb = {'height': np.zeros(len(y_all)), 'agb': np.zeros(len(y_all))}
oof_xgb = {'height': np.zeros(len(y_all)), 'agb': np.zeros(len(y_all))}

for target_idx, target_name in enumerate(['height', 'agb']):
    y_t = y_all[:, target_idx]
    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
        # LightGBM
        train_set = lgb.Dataset(X_all[tr_idx], label=y_t[tr_idx], feature_name=FEATURE_NAMES)
        val_set = lgb.Dataset(X_all[va_idx], label=y_t[va_idx], reference=train_set)
        lgb_params = dict(objective='regression', metric='rmse', learning_rate=0.03,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                           **best_lgb_params[target_name])
        model_lgb = lgb.train(lgb_params, train_set, num_boost_round=4000,
                               valid_sets=[val_set], valid_names=['val'],
                               callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[target_name][va_idx] = model_lgb.predict(X_all[va_idx])
        lgb_models[target_name].append(model_lgb)

        # XGBoost
        dtrain = xgb.DMatrix(X_all[tr_idx], label=y_t[tr_idx])
        dval = xgb.DMatrix(X_all[va_idx], label=y_t[va_idx])
        xgb_params = dict(objective='reg:squarederror', eta=0.03, eval_metric='rmse', tree_method='hist',
                           **best_xgb_params[target_name])
        model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=4000,
                               evals=[(dval, 'val')], early_stopping_rounds=150, verbose_eval=False)
        oof_xgb[target_name][va_idx] = model_xgb.predict(
            dval, iteration_range=(0, model_xgb.best_iteration + 1))
        xgb_models[target_name].append(model_xgb)

        print(f'{target_name} fold {fold_i + 1}/{NUM_FOLDS} done '
              f'(lgb best_iter={model_lgb.best_iteration}, xgb best_iter={model_xgb.best_iteration})')

    gc.collect()


def check_overfit_oof(fold_models, X_full, y_full, fold_indices, oof_pred, name):
    tr_rmses = []
    for (tr_idx, _), model in zip(fold_indices, fold_models):
        pred_tr = model.predict(X_full[tr_idx]) if hasattr(model, 'predict') and not isinstance(model, xgb.Booster) \
            else model.predict(xgb.DMatrix(X_full[tr_idx]))
        tr_rmses.append(mean_squared_error(y_full[tr_idx], pred_tr) ** 0.5)
    tr_rmse = np.mean(tr_rmses)
    oof_rmse = mean_squared_error(y_full, oof_pred) ** 0.5
    ratio = oof_rmse / (tr_rmse + 1e-9)
    print(f'{name}: mean train RMSE {tr_rmse:.4f} | OOF RMSE {oof_rmse:.4f} | OOF/train ratio {ratio:.2f}'
          + ('  <-- still overfitting, consider more regularization' if ratio > 3 else '  OK'))

for target_name in ['height', 'agb']:
    check_overfit_oof(lgb_models[target_name], X_all, y_all[:, ['height', 'agb'].index(target_name)],
                       fold_indices, oof_lgb[target_name], f'LGB {target_name}')
    check_overfit_oof(xgb_models[target_name], X_all, y_all[:, ['height', 'agb'].index(target_name)],
                       fold_indices, oof_xgb[target_name], f'XGB {target_name}')

def blend_and_score_oof(target_idx, target_name):
    lgb_pred = oof_lgb[target_name]
    xgb_pred = oof_xgb[target_name]
    y_t = y_all[:, target_idx]

    best_w, best_mse = 0.5, np.inf
    for w in np.linspace(0, 1, 41):
        blend = w * lgb_pred + (1 - w) * xgb_pred
        scaled_mse = mean_squared_error(y_t / SCALE[target_name], blend / SCALE[target_name])
        if scaled_mse < best_mse:
            best_mse, best_w = scaled_mse, w

    blend = best_w * lgb_pred + (1 - best_w) * xgb_pred
    rmse = mean_squared_error(y_t, blend) ** 0.5
    mae = mean_absolute_error(y_t, blend)
    r2 = r2_score(y_t, blend)
    print(f'--- {target_name} OOF blend (lgb weight={best_w:.2f}) ---')
    print(f'RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}  scaled MSE: {best_mse:.6f}')
    return blend, best_w, best_mse

blend_height, w_height, mse_h = blend_and_score_oof(0, 'height')
blend_agb, w_agb, mse_a = blend_and_score_oof(1, 'agb')

final_score = 0.5 * (mse_h + mse_a)
print(f'\nEstimated (honest, OOF) competition-style score: {final_score:.6f}'
      f'  (relative RMSE {final_score ** 0.5:.6f})')


def plot_importance_avg(models, title, top_n=20):
    imp = np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    imp = pd.Series(imp, index=FEATURE_NAMES).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    imp[::-1].plot(kind='barh')
    plt.title(f'Top {top_n} features — {title} (mean gain across {len(models)} folds)')
    plt.xlabel('Gain')
    plt.tight_layout()
    plt.show()

plot_importance_avg(lgb_models['height'], 'height')
plot_importance_avg(lgb_models['agb'], 'agb')


age_col_idx = FEATURE_NAMES.index('seed_age')
all_ages_col = X_all[:, age_col_idx]
results = []
for age in SEED_AGES:
    mask = all_ages_col == age
    if mask.sum() == 0:
        continue
    mse_h_age = mean_squared_error(y_all[mask, 0] / SCALE['height'], blend_height[mask] / SCALE['height'])
    mse_a_age = mean_squared_error(y_all[mask, 1] / SCALE['agb'], blend_agb[mask] / SCALE['agb'])
    results.append((age, mask.sum(), 0.5 * (mse_h_age + mse_a_age)))

res_df = pd.DataFrame(results, columns=['seed_age', 'n_samples', 'scaled_mse'])
print(res_df)

plt.figure(figsize=(7, 4))
plt.plot(res_df['seed_age'], res_df['scaled_mse'], marker='o')
plt.xlabel('Seed forest age (years)')
plt.ylabel('Scaled MSE (OOF blend)')
plt.title('OOF error vs. seed forest age')
plt.tight_layout()
plt.show()


train_min = X_all.min(axis=0)
train_max = X_all.max(axis=0)

def extrapolation_report(test_feats, scenario_name):
    below = test_feats < train_min[None, :]
    above = test_feats > train_max[None, :]
    out_of_range = below | above
    frac_rows_flagged = (out_of_range.any(axis=1)).mean()
    frac_cells_flagged = out_of_range.mean()
    worst_features = pd.Series(out_of_range.mean(axis=0), index=FEATURE_NAMES) \
        .sort_values(ascending=False).head(10)
    print(f'--- {scenario_name} ---')
    print(f'{frac_rows_flagged:.1%} of rows have at least one out-of-training-range feature')
    print(f'{frac_cells_flagged:.2%} of all (row, feature) cells are out of range')
    print('most-affected features:')
    print(worst_features)
    print()


sites_ssp_file = find_file('sites_ssp.csv')
TEST_BASE = os.path.dirname(sites_ssp_file)
print('Resolved TEST_BASE:', TEST_BASE)

sites_ssp = pd.read_csv(sites_ssp_file)
print(sites_ssp.head())
site_ids = sites_ssp.iloc[:, 0].astype(int).values

def featurize_test(npz_path):
    d = np.load(npz_path)
    test_x, test_y0 = d['test_x'], d['test_y0']
    n_sites = test_x.shape[0]

    Xn = (test_x - x_mean) / (x_std + 1e-8)
    raw_annual_mean = test_x.mean(axis=2)
    annual_mean = Xn.mean(axis=2)

    feat_mean = annual_mean.mean(axis=1)
    feat_std = annual_mean.std(axis=1)
    feat_last5 = annual_mean[:, -5:, :].mean(axis=1)
    years = np.arange(40); years_c = years - years.mean(); denom = (years_c ** 2).sum()
    feat_trend = (annual_mean * years_c[None, :, None]).sum(axis=1) / denom
    decade_feats = np.concatenate(
        [annual_mean[:, d_*10:(d_+1)*10, :].mean(axis=1) for d_ in range(4)], axis=1)
    raw_early = raw_annual_mean[:, :5, :].mean(axis=1)
    raw_late = raw_annual_mean[:, -5:, :].mean(axis=1)
    rel_change = np.clip((raw_late - raw_early) / (np.abs(raw_early) + EPS), -10, 10)

    rows = []
    all_feats_for_report = []
    for age_i, age in enumerate(SEED_AGES):
        init_state = test_y0[:, age_i, :]
        init_rate = init_state / (age + EPS)
        age_col = np.full((n_sites, 1), age, dtype=np.float32)
        feats = np.concatenate(
            [feat_mean, feat_std, feat_trend, feat_last5, decade_feats,
             rel_change, init_state, init_rate, age_col], axis=1
        ).astype(np.float32)
        all_feats_for_report.append(feats)

        lgb_h = np.mean([m.predict(feats) for m in lgb_models['height']], axis=0)
        xgb_h = np.mean([m.predict(xgb.DMatrix(feats)) for m in xgb_models['height']], axis=0)
        lgb_a = np.mean([m.predict(feats) for m in lgb_models['agb']], axis=0)
        xgb_a = np.mean([m.predict(xgb.DMatrix(feats)) for m in xgb_models['agb']], axis=0)

        pred_h = (w_height * lgb_h + (1 - w_height) * xgb_h) / SCALE['height']
        pred_a = (w_agb * lgb_a + (1 - w_agb) * xgb_a) / SCALE['agb']

        rows.append(pd.DataFrame({'site': site_ids, 'age': age, 'height': pred_h, 'agb': pred_a}))

    return pd.concat(rows, ignore_index=True), np.concatenate(all_feats_for_report, axis=0)

df_126, feats_126 = featurize_test(find_file('test_ssp126.npz'))
df_126['id'] = 'ssp126_' + df_126['site'].astype(str) + '_' + df_126['age'].astype(str)
extrapolation_report(feats_126, 'ssp126 (low emissions)')

df_585, feats_585 = featurize_test(find_file('test_ssp585.npz'))
df_585['id'] = 'ssp585_' + df_585['site'].astype(str) + '_' + df_585['age'].astype(str)
extrapolation_report(feats_585, 'ssp585 (high emissions)')

submission = pd.concat([df_126, df_585], ignore_index=True)[['id', 'height', 'agb']]

sample_sub = pd.read_csv(find_file('sample_submission.csv'))
submission = sample_sub[['id']].merge(submission, on='id', how='left')

assert submission.shape[0] == 185130, f'expected 185130 rows, got {submission.shape[0]}'
assert submission[['height', 'agb']].isna().sum().sum() == 0, 'missing predictions for some ids -- id string mismatch!'

submission.to_csv('submission.csv', index=False)
print('submission.csv saved successfully! Head:')
print(submission.head())


