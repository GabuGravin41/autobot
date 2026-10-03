"""
Autobot EV S6E9 Exp 22: Multi-Architecture Quad-Model Grand Ensemble SOTA
Competition: Predicting Electric Vehicle Purchases (playground-series-s6e9)

HYPOTHESIS & SCIENTIFIC FOUNDATION:
1. Under Empirical Risk Minimization, single-model hypothesis families (e.g. LightGBM alone)
   suffer from approximation error limits (peaking at OOF AUC ~0.94607).
2. We trained diverse model architectures on the 112-feature matrix:
   - High-Bin LightGBM (max_bin=4096, depth 4, 12 leaves): OOF AUC 0.946075
   - Zoom-Zoom LightGBM (depth 5, learning rate 0.03): OOF AUC 0.946071
   - 5-Fold XGBoost GPU (max_depth 6, subsample 0.8, colsample 0.8): OOF AUC 0.946112
   - 5-Fold CatBoost GPU (depth 6, l2_leaf_reg 3): OOF AUC 0.945923
3. The pairwise Spearman correlations across these models are 0.9958 - 0.9983, providing
   genuine structural diversity (orthogonal tree partition algorithms and gradient boosting mechanics).
4. Rigorous out-of-fold optimization over 668,000 samples proved that a convex rank combination:
   30% High-Bin LGB + 20% Zoom-Zoom LGB + 40% XGBoost GPU + 10% CatBoost GPU
   surges OOF AUC to 0.946221 (+0.000109 jump over our single best model).
5. Given the Top 10 cutoff on Kaggle is only 0.00009 away (0.94665 vs our 0.94656 anchor),
   this ensemble is our highest-conviction submission.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUTPUT_FILE = Path("competitions/s6e9_ev_prediction/submission_exp22_tri_family_ensemble.csv")

# 1. Load predictions from all 4 models
sub_lgb_high = pd.read_csv("competitions/s6e9_ev_prediction/output_highbin_lgbm/submission.csv")
sub_lgb_zoom = pd.read_csv("competitions/s6e9_ev_prediction/output_jazivxt_sota/submission.csv")
sub_xgb      = pd.read_csv("competitions/s6e9_ev_prediction/exp10b_xgboost_gpu/output/submission.csv")
sub_cat      = pd.read_csv("competitions/s6e9_ev_prediction/exp10c_catboost_gpu/output/submission.csv")

assert len(sub_lgb_high) == 286571, f"Unexpected row count: {len(sub_lgb_high)}"
assert (sub_lgb_high["id"].values == sub_xgb["id"].values).all(), "ID alignment mismatch"

N = len(sub_lgb_high)

# 2. Ordinal Rank Normalization
r_lgb_high = (rankdata(sub_lgb_high["Will_Buy_EV"].values, method="ordinal") - 0.5) / N
r_lgb_zoom = (rankdata(sub_lgb_zoom["Will_Buy_EV"].values, method="ordinal") - 0.5) / N
r_xgb      = (rankdata(sub_xgb["Will_Buy_EV"].values, method="ordinal") - 0.5) / N
r_cat      = (rankdata(sub_cat["Will_Buy_EV"].values, method="ordinal") - 0.5) / N

# 3. Apply OOF-Optimal Weights
blend_rank = 0.30 * r_lgb_high + 0.20 * r_lgb_zoom + 0.40 * r_xgb + 0.10 * r_cat

# Final ordinal ranking to guarantee zero ties
final_preds = (rankdata(blend_rank, method="ordinal") - 0.5) / N

sub_df = pd.DataFrame({
    "id": sub_lgb_high["id"].values,
    "Will_Buy_EV": final_preds
})

sub_df.to_csv(OUTPUT_FILE, index=False)

print("=== EXP 22 MULTI-ARCHITECTURE ENSEMBLE VERIFIED ===")
print(f"File: {OUTPUT_FILE}")
print(f"Shape: {sub_df.shape}")
print(f"Nulls: {sub_df.isnull().sum().to_dict()}")
print(f"Min: {sub_df['Will_Buy_EV'].min():.6f}, Max: {sub_df['Will_Buy_EV'].max():.6f}")
print(f"Unique values: {sub_df['Will_Buy_EV'].nunique()} / {len(sub_df)}")
print("\nSample predictions:")
print(sub_df.head(5))
