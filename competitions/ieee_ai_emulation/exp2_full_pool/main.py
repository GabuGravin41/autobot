# ==============================================================================
# Autobot Exp 1: LightGBM Multi-Decadal Climate & Growth Dynamics Baseline
# Competition: IEEE BigData Cup 2026 - AI Emulation Challenge
# Task: Global-Scale Land Ecosystem Forecasting (CarbonGlobe)
# ==============================================================================
import os
import sys
import gc
import time
import glob
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"=== AUTOBOT AI EMULATION EXP 2: FULL-POOL LIGHTGBM SOTA ===")
print(f"Python Runtime: {sys.version.split()[0]}")
t0 = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Path Resolution
# ------------------------------------------------------------------------------
def locate_path(candidates, desc="path"):
    for c in candidates:
        p = Path(c)
        if p.exists():
            print(f"Located {desc}: {p}")
            return p
    glob_candidates = []
    for base in ['/kaggle/input', '.']:
        if Path(base).exists():
            for m in Path(base).rglob('*'):
                if m.is_dir() and any(c.lower() in m.name.lower() for c in ['carbonglobe', 'emulation']):
                    glob_candidates.append(m)
    if glob_candidates:
        print(f"Located {desc} via fallback glob: {glob_candidates[0]}")
        return glob_candidates[0]
    raise FileNotFoundError(f"Could not locate {desc} from candidates: {candidates}")

CARB_CANDIDATES = [
    '/kaggle/input/carbonglobe',
    '/kaggle/input/datasets/zhihaow/carbonglobe',
    '/kaggle/input/zhihaow/carbonglobe',
    'carbonglobe',
    '.'
]

TEST_CANDIDATES = [
    '/kaggle/input/ieee-bigdata-cup-2026-ai-emulation-challenge',
    '/kaggle/input/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge',
    'ieee-bigdata-cup-2026-ai-emulation-challenge',
    '.'
]

CARB_BASE = locate_path(CARB_CANDIDATES, "CarbonGlobe Training Dataset")
TEST_BASE = locate_path(TEST_CANDIDATES, "AI Emulation Test Dataset")

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
AGE_FILES = {age: f"glob_Y{age:03d}.npy" for age in SEED_AGES}
TARGET_NAMES = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']
SCALE = {'height': 10.90645, 'agb': 3.39567}
EPS = 1e-6

# ------------------------------------------------------------------------------
# 2. Normalization Statistics & Site Indexing
# ------------------------------------------------------------------------------
stats_path = list(CARB_BASE.rglob('data_stats.npz'))
if not stats_path:
    raise FileNotFoundError(f"Could not locate data_stats.npz in {CARB_BASE}")
stats = np.load(stats_path[0])
x_mean, x_std = stats['x_mean'].astype(np.float32), stats['x_std'].astype(np.float32)
print(f"Loaded driver normalizer: x_mean shape={x_mean.shape}, x_std shape={x_std.shape}")

# Find glob_X_fea.npy
x_fea_path = list(CARB_BASE.rglob('glob_X_fea.npy'))
if not x_fea_path:
    raise FileNotFoundError(f"Could not locate glob_X_fea.npy in {CARB_BASE}")
print(f"Mapping environmental drivers from {x_fea_path[0]} via mmap...")
X_mmap = np.load(x_fea_path[0], mmap_mode='r')
print(f"Driver mmap shape: {X_mmap.shape}")

# Pool available benchmark sites
site_files = {
    'train': list(CARB_BASE.rglob('train.csv')),
    'val': list(CARB_BASE.rglob('val.csv')),
    'test': list(CARB_BASE.rglob('test.csv'))
}

site_pools = []
for split_name, flist in site_files.items():
    if flist:
        df_split = pd.read_csv(flist[0], header=None)
        site_pools.append(df_split.iloc[:, 0].astype(int).values)
        print(f"Found {split_name}.csv: {len(site_pools[-1])} sites")

if site_pools:
    all_candidate_sites = np.unique(np.concatenate(site_pools))
else:
    all_candidate_sites = np.arange(min(len(X_mmap), 10000))

print(f"Total unique labeled sites: {len(all_candidate_sites)}")

# Select representative site pool for optimal cloud memory & execution speed
MAX_TRAIN_SITES = 35000
if len(all_candidate_sites) > MAX_TRAIN_SITES:
    rng = np.random.RandomState(RANDOM_SEED)
    selected_sites = np.sort(rng.choice(all_candidate_sites, size=MAX_TRAIN_SITES, replace=False))
else:
    selected_sites = all_candidate_sites

print(f"Selected {len(selected_sites)} training sites across {len(SEED_AGES)} seed ages "
      f"({len(selected_sites) * len(SEED_AGES):,} samples)")

# ------------------------------------------------------------------------------
# 3. Biophysical Feature Engineering Pipeline
# ------------------------------------------------------------------------------
def build_feature_names():
    names = []
    # 40-year multi-decadal mean
    names += [f'x_mean_{i}' for i in range(136)]
    # Inter-annual climate variability
    names += [f'x_std_{i}' for i in range(136)]
    # 40-year linear warming / drying trend
    names += [f'x_trend_{i}' for i in range(136)]
    # Late period climate (last 5 years)
    names += [f'x_last5_{i}' for i in range(136)]
    # Decadal trajectories (4 decades x 136)
    for d in range(4):
        names += [f'x_decade{d}_{i}' for i in range(136)]
    # Relative climate shift (late vs early)
    names += [f'x_relchange_{i}' for i in range(136)]
    # Initial ecosystem state variables (t=0)
    names += [f'init_{t}' for t in TARGET_NAMES]
    # Initial growth rate (state / age)
    names += [f'initrate_{t}' for t in TARGET_NAMES]
    # Stand seed age
    names += ['seed_age']
    return names

FEATURE_NAMES = build_feature_names()
N_FEATURES = len(FEATURE_NAMES)
print(f"Engineered Feature Vector Dimension: {N_FEATURES}")

def extract_features_from_block(X_raw, Y0_raw, seed_age, site_ids_arr):
    """
    Extracts multi-decadal climate summary statistics and ecosystem state features.
    X_raw: (n_sites, 40, 12, 136) in raw units
    Y0_raw: (n_sites, 7) initial ecosystem state at year 0
    seed_age: scalar integer
    """
    n_sites = X_raw.shape[0]
    
    # Standardize climate drivers
    Xn = (X_raw - x_mean) / (x_std + 1e-8)
    annual_mean = Xn.mean(axis=2)                     # (n_sites, 40, 136) z-scored
    raw_annual_mean = X_raw.mean(axis=2)             # (n_sites, 40, 136) raw
    
    # 1. 40-year multi-decadal mean
    feat_mean = annual_mean.mean(axis=1)             # (n_sites, 136)
    
    # 2. Inter-annual variability (std)
    feat_std = annual_mean.std(axis=1)               # (n_sites, 136)
    
    # 3. 40-year linear trend slope
    years = np.arange(40, dtype=np.float32)
    years_c = years - years.mean()
    denom = (years_c ** 2).sum()
    feat_trend = (annual_mean * years_c[None, :, None]).sum(axis=1) / denom # (n_sites, 136)
    
    # 4. Late period climate (last 5 years: years 35-40)
    feat_last5 = annual_mean[:, -5:, :].mean(axis=1) # (n_sites, 136)
    
    # 5. Decadal means (4 x 10 years)
    decade_feats = np.concatenate(
        [annual_mean[:, d*10:(d+1)*10, :].mean(axis=1) for d in range(4)], axis=1
    ) # (n_sites, 544)
    
    # 6. Relative climate shift
    raw_early = raw_annual_mean[:, :5, :].mean(axis=1)
    raw_late = raw_annual_mean[:, -5:, :].mean(axis=1)
    rel_change = (raw_late - raw_early) / (np.abs(raw_early) + EPS)
    rel_change = np.clip(rel_change, -10.0, 10.0)    # (n_sites, 136)
    
    # 7. Initial ecosystem state & rate of growth
    init_state = Y0_raw                              # (n_sites, 7)
    init_rate = init_state / (float(seed_age) + EPS) # (n_sites, 7)
    age_col = np.full((n_sites, 1), float(seed_age), dtype=np.float32)
    
    feats = np.concatenate(
        [feat_mean, feat_std, feat_trend, feat_last5, decade_feats,
         rel_change, init_state, init_rate, age_col], axis=1
    ).astype(np.float32)
    
    return feats

def assemble_training_data(sites, ages=SEED_AGES):
    n_sites = len(sites)
    n_rows = n_sites * len(ages)
    
    X_mat = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    y_mat = np.empty((n_rows, 2), dtype=np.float32)   # [height, agb]
    site_arr = np.empty(n_rows, dtype=np.int64)
    
    print(f"\nMaterializing environmental drivers for {n_sites} sites from disk...")
    t_mat = time.time()
    X_sites_raw = np.array(X_mmap[sites], dtype=np.float32)
    print(f"Loaded driver slice ({X_sites_raw.shape}, {X_sites_raw.nbytes / (1024**2):.1f} MB) in {time.time()-t_mat:.1f}s")
    
    for i, age in enumerate(ages):
        t_age = time.time()
        y_file_name = AGE_FILES[age]
        y_path = list(CARB_BASE.rglob(y_file_name))
        if not y_path:
            raise FileNotFoundError(f"Missing {y_file_name} in {CARB_BASE}")
        
        # Load age-specific Y slice
        Y_mmap = np.load(y_path[0], mmap_mode='r')
        Y_sites_raw = np.array(Y_mmap[sites], dtype=np.float32)
        
        # Initial state at year 0 (month 11)
        Y0 = Y_sites_raw[:, 0, -1, :]
        # Target state at year 40 (month 11) for height & agb (indices 0 & 1)
        Y_target = Y_sites_raw[:, 40, -1, :2]
        
        feats = extract_features_from_block(X_sites_raw, Y0, age, sites)
        
        lo, hi = i * n_sites, (i + 1) * n_sites
        X_mat[lo:hi] = feats
        y_mat[lo:hi] = Y_target
        site_arr[lo:hi] = sites
        
        del Y_sites_raw, Y0, Y_target, feats
        gc.collect()
        print(f"  [Age {age:03d} | {i+1:02d}/{len(ages):02d}] Extracted {n_sites} rows in {time.time()-t_age:.1f}s")
        
    del X_sites_raw
    gc.collect()
    print(f"Complete Training Matrix: {X_mat.shape}, Targets: {y_mat.shape}")
    return X_mat, y_mat, site_arr

X_train, y_train, site_train = assemble_training_data(selected_sites)

# ------------------------------------------------------------------------------
# 4. GroupKFold Cross-Validation & LightGBM Training
# ------------------------------------------------------------------------------
N_FOLDS = 4
gkf = GroupKFold(n_splits=N_FOLDS)
fold_splits = list(gkf.split(X_train, y_train, groups=site_train))

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 45,
    'max_depth': 7,
    'min_child_samples': 50,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'n_jobs': -1,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

models = {'height': [], 'agb': []}
oof_preds = {
    'height': np.zeros(len(y_train), dtype=np.float32),
    'agb': np.zeros(len(y_train), dtype=np.float32)
}

print(f"\n>>> Commencing 5-Fold GroupKFold Cross-Validation by Site...")
for target_idx, target_name in enumerate(['height', 'agb']):
    print(f"\n--- Training Target: {target_name.upper()} (Scale Constant: {SCALE[target_name]}) ---")
    y_target = y_train[:, target_idx]
    
    fold_rmses = []
    fold_scaled_mses = []
    
    for fold, (train_idx, val_idx) in enumerate(fold_splits, 1):
        X_tr, y_tr = X_train[train_idx], y_target[train_idx]
        X_va, y_va = X_train[val_idx], y_target[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        model = lgb.train(
            LGB_PARAMS,
            train_data,
            num_boost_round=1500,
            valid_sets=[train_data, val_data],
            callbacks=callbacks
        )
        
        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[target_name][val_idx] = val_pred
        models[target_name].append(model)
        
        rmse = np.sqrt(mean_squared_error(y_va, val_pred))
        scaled_mse = mean_squared_error(y_va / SCALE[target_name], val_pred / SCALE[target_name])
        fold_rmses.append(rmse)
        fold_scaled_mses.append(scaled_mse)
        
        print(f"  [Fold {fold}/{N_FOLDS}] Best Iter: {model.best_iteration:4d} | "
              f"Val RMSE: {rmse:.4f} | Scaled MSE: {scaled_mse:.6f}")
        
    mean_rmse = np.mean(fold_rmses)
    mean_scaled_mse = np.mean(fold_scaled_mses)
    r2 = r2_score(y_target, oof_preds[target_name])
    print(f"  --> {target_name.upper()} 5-Fold Mean RMSE: {mean_rmse:.4f} | "
          f"Scaled MSE: {mean_scaled_mse:.6f} | R2: {r2:.4f}")

# Overall Composite OOF Metric
overall_scaled_mse = 0.5 * (
    mean_squared_error(y_train[:, 0] / SCALE['height'], oof_preds['height'] / SCALE['height']) +
    mean_squared_error(y_train[:, 1] / SCALE['agb'], oof_preds['agb'] / SCALE['agb'])
)
print(f"\n=======================================================")
print(f"OFFICIAL OOF COMPOSITE METRIC (SCALED MSE): {overall_scaled_mse:.6f}")
print(f"Estimated Relative RMSE: {np.sqrt(overall_scaled_mse):.6f}")
print(f"=======================================================")

# Free training memory before test inference
del X_train, y_train, site_train
gc.collect()

# ------------------------------------------------------------------------------
# 5. Test Inference Across SSP126 and SSP585
# ------------------------------------------------------------------------------
print(f"\n>>> Commencing Test Set Inference on Future Scenarios...")

# Locate test input files
sites_ssp_file = list(TEST_BASE.rglob('sites_ssp.csv'))
if not sites_ssp_file:
    raise FileNotFoundError("Missing sites_ssp.csv in test dataset")
sites_ssp = pd.read_csv(sites_ssp_file[0])
test_site_ids = sites_ssp.iloc[:, 0].astype(int).values
n_test_sites = len(test_site_ids)
print(f"Loaded {n_test_sites} test sites from {sites_ssp_file[0].name}")

scenario_files = {
    'ssp126': list(TEST_BASE.rglob('test_ssp126.npz')),
    'ssp585': list(TEST_BASE.rglob('test_ssp585.npz'))
}

scenario_dfs = []
for scenario, npz_list in scenario_files.items():
    if not npz_list:
        raise FileNotFoundError(f"Missing test_{scenario}.npz in test dataset")
    npz_path = npz_list[0]
    print(f"\n--- Loading and Predicting Scenario: {scenario.upper()} from {npz_path.name} ---")
    t_scen = time.time()
    
    test_data = np.load(npz_path)
    test_x = test_data['test_x'].astype(np.float32)     # (6171, 40, 12, 136)
    test_y0 = test_data['test_y0'].astype(np.float32)   # (6171, 15, 7)
    
    scen_rows = []
    for age_idx, age in enumerate(SEED_AGES):
        t_scen_age = time.time()
        Y0_age = test_y0[:, age_idx, :]                  # (6171, 7)
        
        feats_test = extract_features_from_block(test_x, Y0_age, age, test_site_ids)
        
        # 5-Fold Ensemble Average
        pred_h_raw = np.mean([m.predict(feats_test) for m in models['height']], axis=0)
        pred_a_raw = np.mean([m.predict(feats_test) for m in models['agb']], axis=0)
        
        # Pre-scale strictly as required by Kaggle solution contract
        # Biological positive floor clipping to eliminate zero failure mode
        pred_h_scaled = np.clip(pred_h_raw / SCALE['height'], 1e-3, None)
        pred_a_scaled = np.clip(pred_a_raw / SCALE['agb'], 1e-3, None)
        
        # Build scenario-site-age dataframe
        df_age = pd.DataFrame({
            'scenario': scenario,
            'site': test_site_ids,
            'age': age,
            'height': pred_h_scaled,
            'agb': pred_a_scaled
        })
        scen_rows.append(df_age)
        print(f"  [{scenario} | Age {age:03d}] Inferred {n_test_sites} sites in {time.time()-t_scen_age:.1f}s")
        
    df_scen = pd.concat(scen_rows, ignore_index=True)
    df_scen['id'] = df_scen['scenario'] + '_' + df_scen['site'].astype(str) + '_' + df_scen['age'].astype(str)
    scenario_dfs.append(df_scen)
    print(f"Scenario {scenario.upper()} complete: {len(df_scen):,} rows in {time.time()-t_scen:.1f}s")
    
    del test_data, test_x, test_y0
    gc.collect()

all_preds_df = pd.concat(scenario_dfs, ignore_index=True)

# ------------------------------------------------------------------------------
# 6. Submission Alignment & Strict Integrity Checks
# ------------------------------------------------------------------------------
print(f"\n>>> Aligning Predictions to sample_submission.csv...")
sample_sub_file = list(TEST_BASE.rglob('sample_submission.csv'))
if not sample_sub_file:
    raise FileNotFoundError("Missing sample_submission.csv in test dataset")
sample_sub = pd.read_csv(sample_sub_file[0])

# Match exact row sequence of sample_submission.csv
submission = sample_sub[['id']].merge(all_preds_df[['id', 'height', 'agb']], on='id', how='left')

out_path = Path('/kaggle/working/submission.csv') if Path('/kaggle/working').exists() else Path('submission.csv')
submission.to_csv(out_path, index=False)
print(f"Successfully generated {out_path} ({out_path.stat().st_size / (1024**2):.2f} MB)")

# --- Integrity Contracts ---
print("\n--- Integrity Diagnostic Summary ---")
assert submission.shape[0] == 185130, f"Expected 185,130 rows, got {submission.shape[0]}"
assert not submission[['height', 'agb']].isna().any().any(), "Fatal: Null values found in predictions!"
assert (submission['height'] > 0).all(), "Fatal: Negative or zero height predictions found!"
assert (submission['agb'] > 0).all(), "Fatal: Negative or zero AGB predictions found!"
assert submission['height'].std() > 0.05, "Fatal: Height predictions collapsed to constant!"
assert submission['agb'].std() > 0.05, "Fatal: AGB predictions collapsed to constant!"

print(f"Total Rows: {len(submission):,}")
print(f"Height Stats -> Mean: {submission['height'].mean():.4f}, Std: {submission['height'].std():.4f}, Min: {submission['height'].min():.4f}, Max: {submission['height'].max():.4f}")
print(f"AGB Stats    -> Mean: {submission['agb'].mean():.4f}, Std: {submission['agb'].std():.4f}, Min: {submission['agb'].min():.4f}, Max: {submission['agb'].max():.4f}")
print(f"\nFirst 5 Submission Rows:")
print(submission.head())

print(f"\n=== PIPELINE EXECUTION SUCCESSFUL: ALL CONTRACTS SATISFIED ({time.time()-t0:.1f}s) ===")
