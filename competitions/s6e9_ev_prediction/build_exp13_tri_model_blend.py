import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("=== Building Experiment 13: Tri-Model Symmetrical Master Blend ===")
print("Combining LightGBM (Multi-Seed), XGBoost (GPU), and CatBoost (GPU) on Top 10% Features")

p_lgb = "competitions/s6e9_ev_prediction/output_exp10a_lgbm/submission.csv"
p_xgb = "competitions/s6e9_ev_prediction/output_exp10b_xgboost/submission.csv"
p_cat = "competitions/s6e9_ev_prediction/output_exp10c_catboost/submission.csv"

assert os.path.exists(p_lgb), f"Exp 10A not found: {p_lgb}"
assert os.path.exists(p_xgb), f"Exp 10B not found: {p_xgb}"
assert os.path.exists(p_cat), f"Exp 10C not found: {p_cat}"

df_lgb = pd.read_csv(p_lgb)
df_xgb = pd.read_csv(p_xgb)
df_cat = pd.read_csv(p_cat)

n = len(df_lgb)
assert len(df_xgb) == n and len(df_cat) == n == 286571, "Row count mismatch!"
assert (df_lgb["id"] == df_xgb["id"]).all() and (df_lgb["id"] == df_cat["id"]).all(), "ID alignment mismatch!"

r_lgb = rankdata(df_lgb["Will_Buy_EV"].values, method="average") / n
r_xgb = rankdata(df_xgb["Will_Buy_EV"].values, method="average") / n
r_cat = rankdata(df_cat["Will_Buy_EV"].values, method="average") / n

# 45% Multi-Seed LGBM + 35% XGBoost GPU + 20% CatBoost GPU
blend_rank = 0.45 * r_lgb + 0.35 * r_xgb + 0.20 * r_cat

# Ordinal ranking to guarantee zero ties across all 286,571 test rows
final_rank = rankdata(blend_rank, method="ordinal") / n

print(f"Total test rows: {n}")
print(f"Unique prediction ranks: {len(np.unique(final_rank))}")
print(f"Tied predictions: {n - len(np.unique(final_rank))}")

out_dir = "competitions/s6e9_ev_prediction/output_exp13_tri_blend"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "submission.csv")

sub = pd.DataFrame({
    "id": df_lgb["id"],
    "Will_Buy_EV": final_rank
})
sub.to_csv(out_file, index=False)
print(f"Saved Tri-Model Master Blend to {out_file}")