"""
Autobot AI Emulation Exp 15: Stand-Age Spline Anchor & Climate Residual GBDT SOTA
Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)
Target Hardware: Kaggle Cloud 4-Core CPU (0 GPU quota consumed, ~3 min runtime)

SCIENTIFIC ARCHITECTURE:
1. Stand-Age Growth Equilibrium Anchor:
   - Exp 11 & 13 proved that test_y0 contains the site's own biological growth trajectory.
   - For seed age 10: Stand reaches age 50 (index 4 of y0) -> zero unperturbed error.
   - For seed age 30: Stand reaches age 70 (index 5 of y0) -> zero unperturbed error.
   - For seed age >= 450: Stand reaches equilibrium canopy carrying capacity (index 14).
   - Log-log PCHIP interpolation handles intermediate ages, eliminating tree height/biomass
     prediction scale errors across global biomes.
2. Residual Formulation on Real Simulation Dynamics:
   - Rather than attempting to predict absolute forest biomass (which fails when features extrapolate),
     models predict ONLY the climate residual delta:
       Delta_h = (Y40_h - Spline_h) / SCALE_H
       Delta_a = (Y40_a - Spline_a) / SCALE_A
   - Trained on 4,000 diverse global sites from zhihaow/carbonglobe using GroupKFold by site.
   - Features: annual climate mean, trend, late-early delta, CO2 trajectory, static soil hydraulics,
     and initial ecological fluxes (GPP, NPP, LAI, Rh).
3. Additive Physics-Consistent Reconstruction:
   - Y_final = max(1e-4, Spline_scaled + Delta_GBDT)
   - Guaranteed stability: even if out-of-range climate causes tree prediction to step,
     the stand-age anchor protects the prediction from catastrophic collapse or blowup.
"""

import os
import sys
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

print("=== AUTOBOT AI EMULATION EXP 15: RESIDUAL GBDT SOTA ===")
t0 = time.time()

# ------------------------------------------------------------------------------
# 1. Path Resolution
# ------------------------------------------------------------------------------
def find_file(name, search_dirs=('/kaggle/input', '.')):
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            if name in files:
                p = os.path.join(root, name)
                print(f"Found {name}: {p}")
                return p
    raise FileNotFoundError(f"Cannot find {name}")

SCALE_H = 10.90645
SCALE_A = 3.39567
SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
AGES_ARR = np.array(SEED_AGES, dtype=float)
LOG_AGES = np.log(AGES_ARR)

sample_sub_path = find_file('sample_submission.csv')
sample_sub = pd.read_csv(sample_sub_path)
print(f"Loaded sample_submission.csv: {sample_sub.shape[0]:,} rows.")

sites_ssp_path = find_file('sites_ssp.csv')
sites_ssp = pd.read_csv(sites_ssp_path)
n_test_sites = len(sites_ssp)
print(f"Loaded sites_ssp.csv: {n_test_sites:,} sites.")

# ------------------------------------------------------------------------------
# 2. Stand-Age Spline Interpolation Engine
# ------------------------------------------------------------------------------
def compute_stand_age_splines(y0_array):
    """
    Computes allometric stand-age baseline for each site and seed age.
    y0_array: (n_sites, 15, 7)
    Returns:
      h_spline: (n_sites, 15) in raw meters
      a_spline: (n_sites, 15) in raw kg C/m^2
    """
    n_sites = y0_array.shape[0]
    h_curves = y0_array[:, :, 0]
    a_curves = y0_array[:, :, 1]
    
    h_spline = np.zeros((n_sites, 15), dtype=np.float32)
    a_spline = np.zeros((n_sites, 15), dtype=np.float32)
    
    for age_idx, seed_age in enumerate(SEED_AGES):
        target_age = float(seed_age + 40)
        log_target_age = np.log(target_age)
        
        if seed_age == 10:
            h_spline[:, age_idx] = h_curves[:, 4].copy()
            a_spline[:, age_idx] = a_curves[:, 4].copy()
        elif seed_age == 30:
            h_spline[:, age_idx] = h_curves[:, 5].copy()
            a_spline[:, age_idx] = a_curves[:, 5].copy()
        elif target_age >= 490.0:
            h_spline[:, age_idx] = h_curves[:, -1].copy()
            a_spline[:, age_idx] = a_curves[:, -1].copy()
        else:
            for s in range(n_sites):
                h_vals = h_curves[s, :]
                a_vals = a_curves[s, :]
                
                # Barren / desert site check (no woody biomass)
                if np.max(h_vals) < 0.05:
                    h_spline[s, age_idx] = 0.0
                    a_spline[s, age_idx] = 0.0
                    continue
                
                eps = 1e-4
                log_h = np.log(np.maximum(h_vals, eps))
                log_a = np.log(np.maximum(a_vals, eps))
                
                spline_log_h = PchipInterpolator(LOG_AGES, log_h)
                spline_log_a = PchipInterpolator(LOG_AGES, log_a)
                
                pred_h_interp = float(np.exp(spline_log_h(log_target_age)))
                pred_a_interp = float(np.exp(spline_log_a(log_target_age)))
                
                h_spline[s, age_idx] = max(h_vals[age_idx], pred_h_interp)
                a_spline[s, age_idx] = max(a_vals[age_idx], pred_a_interp)
                
    return h_spline, a_spline

# ------------------------------------------------------------------------------
# 3. Training Data Assembly & Climate Feature Extraction
# ------------------------------------------------------------------------------
print("\n--- Resolving Training Data Pool ---")
glob_x_path = find_file('glob_X_fea.npy')
BASE = os.path.dirname(os.path.dirname(glob_x_path))
print(f"CarbonGlobe training BASE resolved: {BASE}")

# Load normalization statistics
stats = np.load(f'{BASE}/data_stats/data_stats.npz')
x_mean = stats['x_mean'].astype(np.float32)
x_std = stats['x_std'].astype(np.float32)

train_sites = pd.read_csv(f'{BASE}/train.csv', header=None).iloc[:, 0].astype(int).values
val_sites = pd.read_csv(f'{BASE}/val.csv', header=None).iloc[:, 0].astype(int).values
test_sites_bench = pd.read_csv(f'{BASE}/test.csv', header=None).iloc[:, 0].astype(int).values
all_train_sites = np.unique(np.concatenate([train_sites, val_sites, test_sites_bench]))
print(f"Total available labeled sites: {len(all_train_sites):,}")

# Subsample 4,000 diverse sites for fast, high-quality residual training
N_SUB = 4000
rng = np.random.RandomState(42)
sub_sites = rng.choice(all_train_sites, size=N_SUB, replace=False)
sub_sites.sort()
print(f"Selected {len(sub_sites):,} sites for residual calibration.")

# Load raw X memmap: (54152, 40, 12, 136)
X_raw = np.load(glob_x_path, mmap_mode='r')
train_x_chunk = np.array(X_raw[sub_sites], dtype=np.float32)  # (4000, 40, 12, 136)

# Extract climate features
def extract_climate_features(x_array):
    """
    x_array: (n_sites, 40, 12, 136)
    Returns: (n_sites, n_clim_feats)
    """
    annual = x_array.mean(axis=2)  # (n_sites, 40, 136)
    
    # Selected drivers:
    # 0: Temperature, 1: Precipitation, 2: CO2, 3: Shortwave Radiation
    # 110:136: 26 static soil & biome properties
    t_mean = annual[:, :, 0].mean(axis=1, keepdims=True)
    t_std = annual[:, :, 0].std(axis=1, keepdims=True)
    t_early = annual[:, :5, 0].mean(axis=1, keepdims=True)
    t_late = annual[:, -5:, 0].mean(axis=1, keepdims=True)
    t_delta = t_late - t_early
    
    p_mean = annual[:, :, 1].mean(axis=1, keepdims=True)
    p_std = annual[:, :, 1].std(axis=1, keepdims=True)
    p_early = annual[:, :5, 1].mean(axis=1, keepdims=True)
    p_late = annual[:, -5:, 1].mean(axis=1, keepdims=True)
    p_rel = (p_late - p_early) / (np.abs(p_early) + 1e-4)
    p_rel = np.clip(p_rel, -5.0, 5.0)
    
    co2_first = np.maximum(annual[:, 0, 2:3], 100.0)
    co2_last = np.maximum(annual[:, -1, 2:3], 100.0)
    co2_log_ratio = np.log(co2_last / co2_first)
    
    rad_mean = annual[:, :, 3].mean(axis=1, keepdims=True)
    
    # Static soil & biome properties
    static_feats = annual[:, 0, 110:136]  # (n_sites, 26)
    
    clim_mat = np.column_stack([
        t_mean, t_std, t_delta,
        p_mean, p_std, p_rel,
        co2_log_ratio,
        rad_mean,
        static_feats
    ])
    return clim_mat.astype(np.float32)

print("Extracting climate features for training sites...")
train_clim_feats = extract_climate_features(train_x_chunk)  # (4000, 34)
del train_x_chunk
gc.collect()

# Load initial state y0 and target y40 for all 15 ages
print("Loading y0 and y40 target matrices across 15 seed ages...")
train_y0 = np.empty((N_SUB, 15, 7), dtype=np.float32)
train_y40 = np.empty((N_SUB, 15, 2), dtype=np.float32)

for i, age in enumerate(SEED_AGES):
    Y_raw = np.load(f'{BASE}/data_global/glob_Y{age:03d}.npy', mmap_mode='r')
    Y_slice = np.array(Y_raw[sub_sites], dtype=np.float32)  # (4000, 41, 12, 7)
    train_y0[:, i, :] = Y_slice[:, 0, -1, :]
    train_y40[:, i, :] = Y_slice[:, 40, -1, :2]
    del Y_slice
    gc.collect()

# Compute stand-age spline baseline on training set
print("Computing stand-age splines on training pool...")
h_spline_train, a_spline_train = compute_stand_age_splines(train_y0)

# Compute true scaled targets and baseline scaled predictions
y40_h_scaled = train_y40[:, :, 0] / SCALE_H
y40_a_scaled = train_y40[:, :, 1] / SCALE_A
spline_h_scaled = h_spline_train / SCALE_H
spline_a_scaled = a_spline_train / SCALE_A

baseline_mse_h = mean_squared_error(y40_h_scaled.ravel(), spline_h_scaled.ravel())
baseline_mse_a = mean_squared_error(y40_a_scaled.ravel(), spline_a_scaled.ravel())
baseline_total_mse = 0.5 * (baseline_mse_h + baseline_mse_a)
print(f"=== Baseline Spline CV MSE on Training Pool ===")
print(f"Height MSE: {baseline_mse_h:.6f} | AGB MSE: {baseline_mse_a:.6f} | Composite MSE: {baseline_total_mse:.6f}")

# Target residuals to predict
target_res_h = y40_h_scaled - spline_h_scaled  # (4000, 15)
target_res_a = y40_a_scaled - spline_a_scaled  # (4000, 15)

# Assemble tabular training matrix
print("Assembling tabular feature matrix for residual models...")
rows_X = []
rows_res_h = []
rows_res_a = []
rows_site = []

for age_idx, seed_age in enumerate(SEED_AGES):
    # Features for this age:
    age_val = np.full((N_SUB, 1), seed_age, dtype=np.float32)
    log_age = np.full((N_SUB, 1), np.log(float(seed_age) + 1.0), dtype=np.float32)
    
    # Initial state (7 variables: height, agb, soil, lai, gpp, npp, rh)
    init_state = train_y0[:, age_idx, :]  # (4000, 7)
    init_rates = init_state / (float(seed_age) + 1.0)  # (4000, 7)
    
    spline_h_col = spline_h_scaled[:, age_idx:age_idx+1]
    spline_a_col = spline_a_scaled[:, age_idx:age_idx+1]
    biomass_ratio = spline_a_col / (spline_h_col + 1e-4)
    
    X_age = np.column_stack([
        train_clim_feats,
        age_val, log_age,
        init_state, init_rates,
        spline_h_col, spline_a_col, biomass_ratio
    ])
    
    rows_X.append(X_age)
    rows_res_h.append(target_res_h[:, age_idx])
    rows_res_a.append(target_res_a[:, age_idx])
    rows_site.append(sub_sites)

X_train_full = np.concatenate(rows_X, axis=0)       # (60000, n_feats)
y_res_h_full = np.concatenate(rows_res_h, axis=0)   # (60000,)
y_res_a_full = np.concatenate(rows_res_a, axis=0)   # (60000,)
site_groups = np.concatenate(rows_site, axis=0)     # (60000,)

print(f"Feature Matrix Shape: {X_train_full.shape}")
print(f"Residual H Stats -> Mean: {y_res_h_full.mean():.5f}, Std: {y_res_h_full.std():.5f}, Min: {y_res_h_full.min():.5f}, Max: {y_res_h_full.max():.5f}")
print(f"Residual A Stats -> Mean: {y_res_a_full.mean():.5f}, Std: {y_res_a_full.std():.5f}, Min: {y_res_a_full.min():.5f}, Max: {y_res_a_full.max():.5f}")

# ------------------------------------------------------------------------------
# 4. Train GroupKFold LightGBM Residual Models
# ------------------------------------------------------------------------------
print("\n--- Training 5-Fold GroupKFold LightGBM Residual Models ---")
gkf = GroupKFold(n_splits=5)
lgb_models_h = []
lgb_models_a = []
oof_pred_res_h = np.zeros_like(y_res_h_full)
oof_pred_res_a = np.zeros_like(y_res_a_full)

lgb_params_h = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.04,
    'num_leaves': 24,
    'max_depth': 5,
    'min_child_samples': 50,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'verbose': -1,
    'n_jobs': 4,
    'random_state': 42
}

lgb_params_a = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.04,
    'num_leaves': 24,
    'max_depth': 5,
    'min_child_samples': 50,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 1.0,
    'lambda_l2': 2.0,
    'verbose': -1,
    'n_jobs': 4,
    'random_state': 42
}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train_full, groups=site_groups)):
    # Height Model
    ds_tr_h = lgb.Dataset(X_train_full[tr_idx], label=y_res_h_full[tr_idx])
    ds_va_h = lgb.Dataset(X_train_full[va_idx], label=y_res_h_full[va_idx], reference=ds_tr_h)
    m_h = lgb.train(
        lgb_params_h, ds_tr_h, num_boost_round=800,
        valid_sets=[ds_va_h],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_pred_res_h[va_idx] = m_h.predict(X_train_full[va_idx])
    lgb_models_h.append(m_h)
    
    # AGB Model
    ds_tr_a = lgb.Dataset(X_train_full[tr_idx], label=y_res_a_full[tr_idx])
    ds_va_a = lgb.Dataset(X_train_full[va_idx], label=y_res_a_full[va_idx], reference=ds_tr_a)
    m_a = lgb.train(
        lgb_params_a, ds_tr_a, num_boost_round=800,
        valid_sets=[ds_va_a],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_pred_res_a[va_idx] = m_a.predict(X_train_full[va_idx])
    lgb_models_a.append(m_a)
    
    print(f"Fold {fold+1}/5 complete: Best Iter H={m_h.best_iteration}, A={m_a.best_iteration}")

# Calculate honest OOF improvements
base_h_flat = np.concatenate([spline_h_scaled[:, a] for a in range(15)])
base_a_flat = np.concatenate([spline_a_scaled[:, a] for a in range(15)])
true_h_flat = np.concatenate([y40_h_scaled[:, a] for a in range(15)])
true_a_flat = np.concatenate([y40_a_scaled[:, a] for a in range(15)])

boosted_h = np.maximum(1e-4, base_h_flat + oof_pred_res_h)
boosted_a = np.maximum(1e-4, base_a_flat + oof_pred_res_a)

boosted_mse_h = mean_squared_error(true_h_flat, boosted_h)
boosted_mse_a = mean_squared_error(true_a_flat, boosted_a)
boosted_total_mse = 0.5 * (boosted_mse_h + boosted_mse_a)

print(f"\n=======================================================")
print(f"Baseline Spline OOF MSE: {baseline_total_mse:.6f}")
print(f"Residual-Boosted OOF MSE: {boosted_total_mse:.6f}")
print(f"RELATIVE GAIN: -{(baseline_total_mse - boosted_total_mse) / baseline_total_mse * 100:.2f}% MSE")
print(f"=======================================================")

# Clean memory before test inference
del train_clim_feats, train_y0, train_y40, X_train_full, y_res_h_full, y_res_a_full
gc.collect()

# ------------------------------------------------------------------------------
# 5. Multi-Scenario Test Inference Engine
# ------------------------------------------------------------------------------
print("\n--- Generating Predictions for Test Scenarios ---")
dfs = []
scenarios = ['ssp126', 'ssp585']

for sc in scenarios:
    t_sc = time.time()
    npz_path = find_file(f'test_{sc}.npz')
    data = np.load(npz_path)
    
    test_x = data['test_x']    # (6171, 40, 12, 136)
    test_y0 = data['test_y0']  # (6171, 15, 7)
    print(f"\nScenario {sc}: test_x={test_x.shape}, test_y0={test_y0.shape}")
    
    # Extract test climate features
    test_clim_feats = extract_climate_features(test_x)  # (6171, 34)
    
    # Compute test stand-age spline baseline
    test_h_spline, test_a_spline = compute_stand_age_splines(test_y0)
    test_h_spline_scaled = test_h_spline / SCALE_H
    test_a_spline_scaled = test_a_spline / SCALE_A
    
    sc_rows = []
    
    for age_idx, seed_age in enumerate(SEED_AGES):
        age_val = np.full((n_test_sites, 1), seed_age, dtype=np.float32)
        log_age = np.full((n_test_sites, 1), np.log(float(seed_age) + 1.0), dtype=np.float32)
        
        init_state = test_y0[:, age_idx, :]
        init_rates = init_state / (float(seed_age) + 1.0)
        
        spline_h_col = test_h_spline_scaled[:, age_idx:age_idx+1]
        spline_a_col = test_a_spline_scaled[:, age_idx:age_idx+1]
        biomass_ratio = spline_a_col / (spline_h_col + 1e-4)
        
        X_test_age = np.column_stack([
            test_clim_feats,
            age_val, log_age,
            init_state, init_rates,
            spline_h_col, spline_a_col, biomass_ratio
        ])
        
        # Bagged prediction across 5 folds
        pred_res_h = np.mean([m.predict(X_test_age) for m in lgb_models_h], axis=0)
        pred_res_a = np.mean([m.predict(X_test_age) for m in lgb_models_a], axis=0)
        
        # Additive combination with safety clipping
        pred_h_final = np.clip(test_h_spline_scaled[:, age_idx] + pred_res_h, 1e-4, 4.0)
        pred_a_final = np.clip(test_a_spline_scaled[:, age_idx] + pred_res_a, 1e-4, 15.0)
        
        sc_rows.append(pd.DataFrame({
            'site': np.arange(n_test_sites),
            'age': seed_age,
            'height': pred_h_final,
            'agb': pred_a_final
        }))
        
    df_sc = pd.concat(sc_rows, ignore_index=True)
    df_sc['id'] = f"{sc}_" + df_sc['site'].astype(str) + "_" + df_sc['age'].astype(str)
    dfs.append(df_sc)
    print(f"Scenario {sc} generated in {time.time() - t_sc:.1f}s.")

# ------------------------------------------------------------------------------
# 6. Assembly & Verification Contract
# ------------------------------------------------------------------------------
print("\n=== Assembling Submission Contract ===")
submission = pd.concat(dfs, ignore_index=True)[['id', 'height', 'agb']]
submission = sample_sub[['id']].merge(submission, on='id', how='left')

out_path = Path('/kaggle/working/submission.csv') if Path('/kaggle/working').exists() else Path('submission.csv')
submission.to_csv(out_path, index=False)
print(f"Submission saved to {out_path} ({out_path.stat().st_size / (1024**2):.2f} MB)")

# Triple-Gate Verification
EXPECTED_ROWS = 185130
assert submission.shape[0] == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {submission.shape[0]}"
assert not submission[['height', 'agb']].isna().any().any(), "Fatal: NaNs found in predictions!"
assert (submission['height'] > 0).all(), "Fatal: Non-positive height values found!"
assert (submission['agb'] > 0).all(), "Fatal: Non-positive AGB values found!"

print(f"Contract Verified: {submission.shape[0]:,} rows, 0 NaNs.")
print(f"Height Stats -> Mean: {submission['height'].mean():.4f}, Std: {submission['height'].std():.4f}, Min: {submission['height'].min():.4f}, Max: {submission['height'].max():.4f}")
print(f"AGB Stats    -> Mean: {submission['agb'].mean():.4f}, Std: {submission['agb'].std():.4f}, Min: {submission['agb'].min():.4f}, Max: {submission['agb'].max():.4f}")

print("\nSample Predictions:")
print(submission.head(10))
print(f"\nTotal Pipeline Execution Time: {time.time() - t0:.1f}s")
print("=== END AUTOBOT AI EMULATION EXP 15 RESIDUAL SOTA ===")
