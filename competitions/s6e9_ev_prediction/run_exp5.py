import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

print("=== Executing Experiment 5: Physics Boundaries + CatBoost Zero-Tie Ranking ===")

# Load test set and predictions
test = pd.read_csv("competitions/s6e9_ev_prediction/data/test.csv")
p_lgb = pd.read_csv("competitions/s6e9_ev_prediction/output_exp2/submission.csv")["Will_Buy_EV"].values
p_cat = pd.read_csv("competitions/s6e9_ev_prediction/output_exp4_catboost/submission.csv")["Will_Buy_EV"].values

print(f"Test shape: {test.shape}, LGBM preds: {len(p_lgb)}, CatBoost preds: {len(p_cat)}")

def rk01(x):
    return (rankdata(x) - 0.5) / len(x)

def lexsort_zero_ties(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    order = np.lexsort((secondary, primary))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks - 0.5) / len(ranks)

# 1. Pre-rank both diverse models into uniform (0, 1) ranks
r_lgb = rk01(p_lgb)
r_cat = rk01(p_cat)

# 2. Weighted blend (82% LightGBM + 18% CatBoost)
blend = 0.82 * r_lgb + 0.18 * r_cat

# 3. Apply the 4 Deterministic Physics/Synthetic Domain Shifts
income = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").values
commute = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").values
subsidy = (test["Subsidy_Available"].astype(str).str.lower() == "no").values
env1 = (pd.to_numeric(test["Environmental_Concern_Level"], errors="coerce") == 1.0).values
anx_high = test["Range_Anxiety_Level"].isin(["Medium", "High"]).values

# Boundary 1: Millionaire saturation
blend[income >= 170537.0] += 10.0
# Boundary 2: Dead income zone
blend[(income >= 31004.0) & (income <= 41970.0)] -= 10.0
# Boundary 3: Extreme commute range anxiety
blend[commute >= 83.0] -= 5.0
# Boundary 4: Subsidy-less anxiety spike at $30k
blend[(income == 30000.0) & subsidy & (env1 | anx_high)] -= 5.0

# 4. Final Continuous Zero-Tie Ranking using CatBoost as fine-grained secondary ranker
final_ranks = lexsort_zero_ties(blend, r_cat)

out_dir = "competitions/s6e9_ev_prediction/output_exp5"
os.makedirs(out_dir, exist_ok=True)
sub_path = os.path.join(out_dir, "submission.csv")

sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": final_ranks})
sub.to_csv(sub_path, index=False)

print(f"Saved Exp 5 submission to {sub_path}")
print(f"Total rows: {len(sub):,}")
print(f"Unique predictions: {sub['Will_Buy_EV'].nunique():,} / {len(sub):,} (PERFECT ZERO TIES!)")
print(f"Prediction range: [{sub['Will_Buy_EV'].min():.6f}, {sub['Will_Buy_EV'].max():.6f}]")
print(f"Nulls: {sub['Will_Buy_EV'].isna().sum()}")
print("\nFirst 10 predictions:")
print(sub.head(10))
print("=== Experiment 5 Generation Complete! ===")
