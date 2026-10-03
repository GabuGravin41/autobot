"""
Autobot AI Emulation Exp 17: Vectorized Allometric PCHIP Spline & Multi-Driver Biophysical SOTA
Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)
Target Hardware: Kaggle Cloud 4-Core CPU (0 GPU quota consumed, ~30s runtime)

SCIENTIFIC HYPOTHESES & ARCHITECTURAL ADVANCES:
1. Vectorized Allometric PCHIP Spline Backbone:
   - Anchored on Exp 13's proven champion Log-Log PCHIP Spline (0.102, Rank #42 worldwide).
   - Fully vectorized via PchipInterpolator(LOG_AGES, log_y, axis=1), executing across all
     6,171 sites simultaneously in < 0.1 seconds per age.
   - Smooth C^1 cubic Hermite polynomials properly round biological stand maturation without
     Runge oscillations or piecewise kinks.
2. Exact Golden Anchors (Preserved from Exp 13):
   - Seed age 10 -> Year 40 age is exactly 50 (index 4 of test_y0).
   - Seed age 30 -> Year 40 age is exactly 70 (index 5 of test_y0).
   - Old growth (target_age >= 500) -> Dynamic equilibrium carrying capacity (index 14).
   - Biological growth floor: pred >= initial_state.
3. Multi-Driver Biophysical Refinements:
   - Hydraulic Height Regulation: Water availability directly limits xylem turgor and vertical
     elongation. Moisture factor now modulates height growth in addition to biomass (+0.010).
   - Photosynthetic Radiation Coupling: Channel 3 provides incoming shortwave radiation.
     Sites with abundant solar irradiance utilize elevated CO2 with higher carboxylation efficiency.
   - Stomatal-Coupled CO2 Fertilization: Multiplies Farquhar CO2 sensitivity by local moisture index,
     accounting for stomatal closure during drought.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

print("=== AUTOBOT AI EMULATION EXP 17: VECTORIZED PCHIP & MULTI-DRIVER SOTA ===")
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
LOG_AGES = np.log(np.array(SEED_AGES, dtype=float))

sample_sub_path = find_file('sample_submission.csv')
sample_sub = pd.read_csv(sample_sub_path)
print(f"Loaded sample_submission.csv: {sample_sub.shape[0]:,} rows.")

sites_ssp_path = find_file('sites_ssp.csv')
sites_ssp = pd.read_csv(sites_ssp_path)
n_sites = len(sites_ssp)
print(f"Loaded sites_ssp.csv: {n_sites:,} sites.")

# ------------------------------------------------------------------------------
# 2. Multi-Scenario Modeling Engine
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
    # Channel 0: Temperature (deg C or K)
    # Channel 1: Precipitation (mm/day or mm/month)
    # Channel 2: Atmospheric CO2 (ppm)
    # Channel 3: Shortwave Radiation (W/m2)
    temp_annual = test_x[:, :, :, 0].mean(axis=2)   # (6171, 40)
    pr_annual = test_x[:, :, :, 1].mean(axis=2)     # (6171, 40)
    co2_annual = test_x[:, :, :, 2].mean(axis=2)    # (6171, 40)
    rad_annual = test_x[:, :, :, 3].mean(axis=2)    # (6171, 40)
    
    # 1. CO2 enrichment ratio over 40 years
    co2_ratio = np.log(np.maximum(co2_annual[:, -1], 100.0) / np.maximum(co2_annual[:, 0], 100.0))  # (6171,)
    
    # 2. Temperature warming anomaly (deg C warming over 40 years)
    t_early = temp_annual[:, :5].mean(axis=1)
    t_late = temp_annual[:, -5:].mean(axis=1)
    temp_anomaly = t_late - t_early                 # (6171,)
    
    # 3. Moisture stress: normalized precipitation
    pr_mean = pr_annual.mean(axis=1)
    moisture_factor = np.clip((pr_mean - np.median(pr_mean)) / (np.std(pr_mean) + 1e-6), -2.0, 2.0)
    
    # 4. Solar radiation: normalized energy availability
    rad_mean = rad_annual.mean(axis=1)
    rad_factor = np.clip((rad_mean - np.median(rad_mean)) / (np.std(rad_mean) + 1e-6), -2.0, 2.0)
    
    # 5. Biome thermal response:
    t_median = np.median(t_early)
    thermal_sensitivity = np.where(t_early < t_median, 0.015, -0.010)
    
    h_curves = test_y0[:, :, 0]  # (6171, 15)
    a_curves = test_y0[:, :, 1]  # (6171, 15)
    
    eps = 1e-4
    log_h_all = np.log(np.maximum(h_curves, eps))
    log_a_all = np.log(np.maximum(a_curves, eps))
    
    # Pre-fit vectorized PCHIP interpolators across all sites (axis=1)
    spline_h = PchipInterpolator(LOG_AGES, log_h_all, axis=1)
    spline_a = PchipInterpolator(LOG_AGES, log_a_all, axis=1)
    
    sc_rows = []
    print(f"Executing Vectorized Allometric Stand-Age Spline across {n_sites} sites...")
    
    for age_idx, seed_age in enumerate(SEED_AGES):
        target_age = float(seed_age + 40)
        log_target_age = np.log(target_age)
        
        # Age-dependent climate plasticity
        age_plasticity = np.exp(-float(seed_age) / 120.0) + 0.15
        
        # Stomatal moisture-coupled CO2 fertilization
        stomatal_co2_factor = np.clip(1.0 + 0.15 * moisture_factor, 0.7, 1.3)
        beta_co2_h = 0.040 * age_plasticity * stomatal_co2_factor
        beta_co2_a = 0.100 * age_plasticity * stomatal_co2_factor
        
        delta_clim_h = (beta_co2_h * co2_ratio + 
                        thermal_sensitivity * temp_anomaly * age_plasticity * 0.5 +
                        0.010 * moisture_factor * age_plasticity)
                        
        delta_clim_a = (beta_co2_a * co2_ratio + 
                        thermal_sensitivity * temp_anomaly * age_plasticity * 1.0 + 
                        0.020 * moisture_factor * age_plasticity +
                        0.008 * rad_factor * age_plasticity)
        
        climate_mult_h = np.clip(1.0 + delta_clim_h, 0.92, 1.25)
        climate_mult_a = np.clip(1.0 + delta_clim_a, 0.85, 1.35)
        
        # Stand-Age Growth Formulation:
        if seed_age == 10:
            # Exact reference point: 10 + 40 = 50 -> index 4
            pred_h_base = h_curves[:, 4].copy()
            pred_a_base = a_curves[:, 4].copy()
        elif seed_age == 30:
            # Exact reference point: 30 + 40 = 70 -> index 5
            pred_h_base = h_curves[:, 5].copy()
            pred_a_base = a_curves[:, 5].copy()
        elif target_age >= 500.0:
            # Carrying capacity saturation -> index 14
            pred_h_base = h_curves[:, -1].copy()
            pred_a_base = a_curves[:, -1].copy()
        else:
            # Vectorized PCHIP spline evaluation
            pred_log_h = spline_h(log_target_age)
            pred_log_a = spline_a(log_target_age)
            pred_h_interp = np.exp(pred_log_h)
            pred_a_interp = np.exp(pred_log_a)
            
            # Growth floor constraint: stand at t+40 cannot be shorter than stand at t
            pred_h_base = np.maximum(h_curves[:, age_idx], pred_h_interp)
            pred_a_base = np.maximum(a_curves[:, age_idx], pred_a_interp)
            
        pred_h = pred_h_base * climate_mult_h
        pred_a = pred_a_base * climate_mult_a
        
        pred_h_scaled = np.clip(pred_h / SCALE_H, eps, None)
        pred_a_scaled = np.clip(pred_a / SCALE_A, eps, None)
        
        sc_rows.append(pd.DataFrame({
            'site': np.arange(n_sites),
            'age': seed_age,
            'height': pred_h_scaled,
            'agb': pred_a_scaled,
        }))
        
    df_sc = pd.concat(sc_rows, ignore_index=True)
    df_sc['id'] = f"{sc}_" + df_sc['site'].astype(str) + "_" + df_sc['age'].astype(str)
    dfs.append(df_sc)
    print(f"Scenario {sc} completed in {time.time() - t_sc:.2f}s.")

# ------------------------------------------------------------------------------
# 3. Assembly & Submission Verification Contract
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
print(f"\nTotal Pipeline Execution Time: {time.time() - t0:.2f}s")
print("=== END AUTOBOT AI EMULATION EXP 17 SOTA ===")
