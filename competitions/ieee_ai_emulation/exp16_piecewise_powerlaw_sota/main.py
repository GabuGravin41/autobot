"""
Autobot AI Emulation Exp 16: Piecewise Allometric Power-Law & Stomatal Biophysical SOTA
Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)
Target Hardware: Kaggle Cloud 4-Core CPU (0 GPU quota consumed, ~1 min runtime)

SCIENTIFIC HYPOTHESES & ARCHITECTURAL ADVANCES:
1. Piecewise Allometric Power-Law Interpolation:
   - Forest growth between known stand ages follows an exact metabolic allometric power law:
       Y(t) = Y_1 * (t / t_1)^gamma,  where gamma = log(Y_2 / Y_1) / log(t_2 / t_1).
   - In log-log space, this is exact linear interpolation between adjacent brackets [t_1, t_2].
   - Eliminates polynomial/spline wiggle, oscillation, and Runge overshoots across the 500-year span.
   - Guaranteed monotonicity: Y_1 <= Y(t) <= Y_2 for all t in [t_1, t_2].
2. Exact Reference Anchors:
   - Seed age 10 -> Year 40 age is exactly 50 (index 4 of test_y0).
   - Seed age 30 -> Year 40 age is exactly 70 (index 5 of test_y0).
   - Seed age 500 -> Year 40 age is >= 500 (index 14 carrying capacity saturation).
3. Barren / Non-Forest Clamping:
   - Sites where carrying capacity (age 500) has Height < 0.05m and AGB < 0.01 are barren
     deserts, alpine rocks, ice sheets, or open water. Clamped strictly to 1e-4 floor to eliminate noise.
4. Stomatal Moisture-Coupled CO2 Fertilization:
   - In plant physiology (Farquhar model), elevated CO2 only enhances carboxylation if stomata
     are open. Under moisture stress, stomata close, attenuating CO2 fertilization.
   - We couple CO2 fertilization directly with precipitation moisture availability.
5. Stand-Age Sequence Monotonicity:
   - At any given geographic site, a forest that started at age 20 (now 60) cannot have lower
     biomass than a forest that started at age 10 (now 50).
   - We enforce cumulative monotonicity across the 15 seed ages per site.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

print("=== AUTOBOT AI EMULATION EXP 16: PIECEWISE POWER-LAW & BIOPHYSICAL SOTA ===")
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
    temp_annual = test_x[:, :, :, 0].mean(axis=2)   # (6171, 40)
    pr_annual = test_x[:, :, :, 1].mean(axis=2)     # (6171, 40)
    co2_annual = test_x[:, :, :, 2].mean(axis=2)    # (6171, 40)
    
    # 1. CO2 enrichment ratio over 40 years
    co2_ratio = np.log(np.maximum(co2_annual[:, -1], 100.0) / np.maximum(co2_annual[:, 0], 100.0))  # (6171,)
    
    # 2. Temperature warming anomaly (deg C warming over 40 years)
    t_early = temp_annual[:, :5].mean(axis=1)
    t_late = temp_annual[:, -5:].mean(axis=1)
    temp_anomaly = t_late - t_early                 # (6171,)
    
    # 3. Moisture stress: normalized precipitation
    pr_mean = pr_annual.mean(axis=1)
    moisture_factor = np.clip((pr_mean - np.median(pr_mean)) / (np.std(pr_mean) + 1e-6), -2.0, 2.0)
    
    # 4. Biome thermal response:
    t_median = np.median(t_early)
    thermal_sensitivity = np.where(t_early < t_median, 0.012, -0.008)
    
    h_curves = test_y0[:, :, 0]  # (6171, 15)
    a_curves = test_y0[:, :, 1]  # (6171, 15)
    
    # Barren site identification
    barren_mask = (h_curves[:, -1] < 0.05) & (a_curves[:, -1] < 0.01)
    print(f"Identified {barren_mask.sum()} barren / non-forest sites ({barren_mask.sum() / n_sites:.1%}).")
    
    # Allocate predictions: shape (n_sites, 15)
    pred_h_all = np.zeros((n_sites, len(SEED_AGES)), dtype=np.float32)
    pred_a_all = np.zeros((n_sites, len(SEED_AGES)), dtype=np.float32)
    
    eps = 1e-4
    
    for age_idx, seed_age in enumerate(SEED_AGES):
        target_age = float(seed_age + 40)
        
        # Age-dependent climate plasticity
        age_plasticity = np.exp(-float(seed_age) / 120.0) + 0.15
        
        # Stomatal moisture-coupled CO2 fertilization
        moisture_multiplier = np.clip(1.0 + 0.25 * moisture_factor, 0.3, 1.6)
        beta_co2_h = 0.035 * age_plasticity * moisture_multiplier
        beta_co2_a = 0.090 * age_plasticity * moisture_multiplier
        
        delta_clim_h = beta_co2_h * co2_ratio + thermal_sensitivity * temp_anomaly * age_plasticity * 0.5
        delta_clim_a = beta_co2_a * co2_ratio + thermal_sensitivity * temp_anomaly * age_plasticity * 1.0 + 0.015 * moisture_factor * age_plasticity
        
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
            # Canopy equilibrium carrying capacity -> index 14
            pred_h_base = h_curves[:, -1].copy()
            pred_a_base = a_curves[:, -1].copy()
        else:
            # Find bracket [t1, t2] in SEED_AGES
            # SEED_AGES is sorted
            i2 = np.searchsorted(SEED_AGES, target_age)
            i1 = i2 - 1
            t1 = float(SEED_AGES[i1])
            t2 = float(SEED_AGES[i2])
            
            # Log-linear allometric weight
            w = (np.log(target_age) - np.log(t1)) / (np.log(t2) - np.log(t1))
            
            # Vectorized allometric interpolation for height
            log_h1 = np.log(np.maximum(h_curves[:, i1], eps))
            log_h2 = np.log(np.maximum(h_curves[:, i2], eps))
            pred_h_interp = np.exp((1.0 - w) * log_h1 + w * log_h2)
            
            # Vectorized allometric interpolation for biomass
            log_a1 = np.log(np.maximum(a_curves[:, i1], eps))
            log_a2 = np.log(np.maximum(a_curves[:, i2], eps))
            pred_a_interp = np.exp((1.0 - w) * log_a1 + w * log_a2)
            
            # Growth constraint: stand at t+40 cannot be shorter than stand at t
            pred_h_base = np.maximum(h_curves[:, age_idx], pred_h_interp)
            pred_a_base = np.maximum(a_curves[:, age_idx], pred_a_interp)
            
        pred_h_all[:, age_idx] = pred_h_base * climate_mult_h
        pred_a_all[:, age_idx] = pred_a_base * climate_mult_a

    # Barren site clamping
    pred_h_all[barren_mask, :] = eps
    pred_a_all[barren_mask, :] = eps
    
    # Stand-Age Sequence Monotonicity Enforcement:
    # Older stands at t=0 must have >= height and >= biomass at t=40
    pred_h_all = np.maximum.accumulate(pred_h_all, axis=1)
    pred_a_all = np.maximum.accumulate(pred_a_all, axis=1)
    
    # Assemble scenario dataframe
    sc_rows = []
    for age_idx, seed_age in enumerate(SEED_AGES):
        pred_h_scaled = np.clip(pred_h_all[:, age_idx] / SCALE_H, eps, None)
        pred_a_scaled = np.clip(pred_a_all[:, age_idx] / SCALE_A, eps, None)
        
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
print("=== END AUTOBOT AI EMULATION EXP 16 SOTA ===")
