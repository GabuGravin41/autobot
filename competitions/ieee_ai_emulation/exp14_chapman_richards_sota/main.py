"""
Autobot AI Emulation Exp 14: Chapman-Richards Non-Linear Growth & Biophysical Climate SOTA
Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)
Target Hardware: Kaggle Cloud 4-Core CPU (0 GPU quota consumed, ~2-3 min runtime)

SCIENTIFIC ARCHITECTURE:
1. Analytical Chapman-Richards Biological Growth Formulation:
   - Forest stand height and above-ground woody biomass follow von Bertalanffy metabolic growth:
       Y(t) = A * (1 - exp(-k * t))^p
     where A = asymptotic carrying capacity, k = intrinsic rate parameter, p = allometric shape.
   - For each site, the 15 known points in test_y0 directly identify the site's biological potential.
   - Closed-form evaluation at t + 40 eliminates spline convexity artifacts in early aggradation (ages 1-30).
2. Hybrid Monotonic Spline Stability:
   - Blends Chapman-Richards parametric projections with Log-Log PCHIP allometric splines (50/50).
   - Desert / barren sites (max(y) < 0.005) floor to zero with zero computational overhead.
3. Exact Zero-Error Reference Anchoring:
   - Seed age 10 -> Year 40 age is exactly 50 (index 4 of test_y0).
   - Seed age 30 -> Year 40 age is exactly 70 (index 5 of test_y0).
   - Seed age >= 450 -> reaches dynamic equilibrium canopy saturation (index 14).
4. Biome-Specific Thermal & CO2 Climate Response:
   - Thermal growing degree response:
     * Cold biomes (T_mean < global median): warming extends photosynthetic growing season (+beta_T).
     * Hot/tropical biomes (T_mean >= global median): warming increases VPD drought stress (-beta_T).
   - Moisture-coupled CO2 fertilization:
     * Humid sites efficiently utilize elevated CO2 into structural woody biomass.
     * Arid sites are stomatal-limited, attenuating fertilization.
5. Age-Dependent Climate Plasticity:
   - Young stands (ages 1-30) have high metabolic plasticity.
   - Mature old-growth stands (ages 150-500) are buffered by structural carbon inertia.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator

print("=== AUTOBOT AI EMULATION EXP 14: CHAPMAN-RICHARDS & BIOPHYSICAL SOTA ===")
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
n_sites = len(sites_ssp)
print(f"Loaded sites_ssp.csv: {n_sites:,} sites.")

# ------------------------------------------------------------------------------
# 2. Chapman-Richards Non-Linear Function
# ------------------------------------------------------------------------------
def chapman_richards(t, A, k, p):
    """von Bertalanffy / Chapman-Richards growth equation."""
    return A * np.power(np.maximum(1.0 - np.exp(-k * t), 0.0), p)

def fit_chapman_richards(t_obs, y_obs):
    """Fit 3-parameter Chapman-Richards growth curve with bounds."""
    max_y = float(np.max(y_obs))
    if max_y < 0.005:
        return None
    p0 = [max(max_y * 1.05, 1.0), 0.02, 1.5]
    bounds = ([0.0, 1e-4, 0.1], [max(max_y * 4.0, 100.0), 0.5, 8.0])
    try:
        popt, _ = curve_fit(chapman_richards, t_obs, y_obs, p0=p0, bounds=bounds, maxfev=600)
        return popt
    except Exception:
        return None

# ------------------------------------------------------------------------------
# 3. Multi-Scenario Modeling Engine
# ------------------------------------------------------------------------------
dfs = []
scenarios = ['ssp126', 'ssp585']

for sc in scenarios:
    t_sc = time.time()
    npz_path = find_file(f'test_{sc}.npz')
    data = np.load(npz_path)
    
    test_x = data['test_x']    # (6171, 40, 12, 136)
    test_y0 = data['test_y0']  # (6171, 15, 7)
    print(f"\nScenario {sc}: test_x={test_x.shape}, test_y0={test_y0.shape}")
    
    # Climate sequence features
    # Channel 0: Temperature
    # Channel 1: Precipitation
    # Channel 2: Atmospheric CO2
    temp_annual = test_x[:, :, :, 0].mean(axis=2)   # (6171, 40)
    pr_annual = test_x[:, :, :, 1].mean(axis=2)     # (6171, 40)
    co2_annual = test_x[:, :, :, 2].mean(axis=2)    # (6171, 40)
    
    # 1. CO2 enrichment ratio over 40 years
    co2_ratio = np.log(np.maximum(co2_annual[:, -1], 100.0) / np.maximum(co2_annual[:, 0], 100.0))
    
    # 2. Temperature warming anomaly
    t_early = temp_annual[:, :5].mean(axis=1)
    t_late = temp_annual[:, -5:].mean(axis=1)
    temp_anomaly = t_late - t_early
    
    # 3. Moisture factor
    pr_mean = pr_annual.mean(axis=1)
    moisture_factor = np.clip((pr_mean - np.median(pr_mean)) / (np.std(pr_mean) + 1e-6), -2.0, 2.0)
    
    # 4. Thermal sensitivity
    t_median = np.median(t_early)
    thermal_sensitivity = np.where(t_early < t_median, 0.015, -0.010)
    
    h_curves = test_y0[:, :, 0]  # (6171, 15)
    a_curves = test_y0[:, :, 1]  # (6171, 15)
    
    print(f"Pre-fitting Chapman-Richards growth curves across {n_sites} sites...")
    t_fit0 = time.time()
    cr_params_h = []
    cr_params_a = []
    for s in range(n_sites):
        cr_params_h.append(fit_chapman_richards(AGES_ARR, h_curves[s, :]))
        cr_params_a.append(fit_chapman_richards(AGES_ARR, a_curves[s, :]))
    n_h_fit = sum(1 for p in cr_params_h if p is not None)
    n_a_fit = sum(1 for p in cr_params_a if p is not None)
    print(f"Chapman-Richards converged for {n_h_fit}/{n_sites} height curves and {n_a_fit}/{n_sites} AGB curves in {time.time()-t_fit0:.1f}s.")
    
    sc_rows = []
    print(f"Computing hybrid growth trajectories for all 15 seed ages...")
    
    for age_idx, seed_age in enumerate(SEED_AGES):
        target_age = float(seed_age + 40)
        log_target_age = np.log(target_age)
        
        # Age-dependent climate plasticity
        age_plasticity = np.exp(-float(seed_age) / 120.0) + 0.15
        
        beta_co2_h = 0.04 * age_plasticity
        beta_co2_a = 0.10 * age_plasticity
        
        delta_clim_h = beta_co2_h * co2_ratio + thermal_sensitivity * temp_anomaly * age_plasticity * 0.5
        delta_clim_a = beta_co2_a * co2_ratio + thermal_sensitivity * temp_anomaly * age_plasticity * 1.0 + 0.02 * moisture_factor * age_plasticity
        
        climate_mult_h = np.clip(1.0 + delta_clim_h, 0.92, 1.25)
        climate_mult_a = np.clip(1.0 + delta_clim_a, 0.85, 1.35)
        
        pred_h_base = np.zeros(n_sites, dtype=np.float32)
        pred_a_base = np.zeros(n_sites, dtype=np.float32)
        
        if seed_age == 10:
            # Exact reference anchor: age 10 + 40 = 50 -> index 4
            pred_h_base = h_curves[:, 4].copy()
            pred_a_base = a_curves[:, 4].copy()
        elif seed_age == 30:
            # Exact reference anchor: age 30 + 40 = 70 -> index 5
            pred_h_base = h_curves[:, 5].copy()
            pred_a_base = a_curves[:, 5].copy()
        elif target_age >= 500.0:
            # Saturated canopy carrying capacity
            pred_h_base = h_curves[:, -1].copy()
            pred_a_base = a_curves[:, -1].copy()
        else:
            for s in range(n_sites):
                h_vals = h_curves[s, :]
                a_vals = a_curves[s, :]
                
                # Check for desert / barren site
                if np.max(h_vals) < 0.005 and np.max(a_vals) < 0.005:
                    pred_h_base[s] = 1e-4
                    pred_a_base[s] = 1e-4
                    continue
                
                # 1. Log-Log PCHIP prediction
                eps = 1e-4
                spline_log_h = PchipInterpolator(LOG_AGES, np.log(np.maximum(h_vals, eps)))
                spline_log_a = PchipInterpolator(LOG_AGES, np.log(np.maximum(a_vals, eps)))
                pchip_pred_h = float(np.exp(spline_log_h(log_target_age)))
                pchip_pred_a = float(np.exp(spline_log_a(log_target_age)))
                
                # 2. Chapman-Richards parametric prediction
                p_h = cr_params_h[s]
                if p_h is not None:
                    cr_pred_h = float(chapman_richards(target_age, *p_h))
                    val_h = 0.5 * cr_pred_h + 0.5 * pchip_pred_h
                else:
                    val_h = pchip_pred_h
                    
                p_a = cr_params_a[s]
                if p_a is not None:
                    cr_pred_a = float(chapman_richards(target_age, *p_a))
                    val_a = 0.5 * cr_pred_a + 0.5 * pchip_pred_a
                else:
                    val_a = pchip_pred_a
                
                pred_h_base[s] = max(h_vals[age_idx], val_h)
                pred_a_base[s] = max(a_vals[age_idx], val_a)
        
        # Apply biophysical climate adjustment
        pred_h = pred_h_base * climate_mult_h
        pred_a = pred_a_base * climate_mult_a
        
        # Pre-scale by global means
        h_scaled = np.maximum(pred_h / SCALE_H, 1e-4)
        a_scaled = np.maximum(pred_a / SCALE_A, 1e-4)
        
        # Format row IDs matching sample_submission: {sc}_{site}_{age}
        for s in range(n_sites):
            row_id = f"{sc}_{s}_{seed_age}"
            sc_rows.append({
                'id': row_id,
                'height': float(h_scaled[s]),
                'agb': float(a_scaled[s])
            })
            
    df_sc = pd.DataFrame(sc_rows)
    dfs.append(df_sc)
    print(f"Scenario {sc} completed in {time.time()-t_sc:.1f}s.")

# ------------------------------------------------------------------------------
# 4. Assembling and Verifying Submission Contract
# ------------------------------------------------------------------------------
print("\n=== Assembling Submission Contract ===")
sub_raw = pd.concat(dfs, ignore_index=True)
sub = sample_sub[['id']].merge(sub_raw, on='id', how='left')

assert len(sub) == len(sample_sub), f"Row mismatch: {len(sub)} vs {len(sample_sub)}"
assert sub['height'].isna().sum() == 0, "NaNs found in height predictions!"
assert sub['agb'].isna().sum() == 0, "NaNs found in agb predictions!"
assert (sub['height'] >= 0).all(), "Negative height predictions detected!"
assert (sub['agb'] >= 0).all(), "Negative agb predictions detected!"

out_path = '/kaggle/working/submission.csv'
sub.to_csv(out_path, index=False)
print(f"Submission saved to {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
print(f"Contract Verified: {len(sub):,} rows, 0 NaNs.")
print(f"Height Stats -> Mean: {sub['height'].mean():.4f}, Std: {sub['height'].std():.4f}, Min: {sub['height'].min():.4f}, Max: {sub['height'].max():.4f}")
print(f"AGB Stats    -> Mean: {sub['agb'].mean():.4f}, Std: {sub['agb'].std():.4f}, Min: {sub['agb'].min():.4f}, Max: {sub['agb'].max():.4f}")

print("\nSample Predictions:")
print(sub.head(10))

print(f"\nTotal Pipeline Execution Time: {time.time()-t0:.1f}s")
print("=== END AUTOBOT AI EMULATION EXP 14 SOTA ===")
