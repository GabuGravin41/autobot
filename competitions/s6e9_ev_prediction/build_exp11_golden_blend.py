import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("=== Building Experiment 11: Golden Symmetric Blend (Exp 9 LGBM 0.50 + Exp 10B XGBoost 0.50) ===")

p_lgb = "competitions/s6e9_ev_prediction/output_exp9_lgbm/submission.csv"
p_xgb = "competitions/s6e9_ev_prediction/output_exp10b_xgboost/submission.csv"

df_lgb = pd.read_csv(p_lgb)
df_xgb = pd.read_csv(p_xgb)

assert len(df_lgb) == len(df_xgb) == 286571, "Test size mismatch!"
assert (df_lgb["id"] == df_xgb["id"]).all(), "ID alignment mismatch!"

n = len(df_lgb)

# Normalized percentile ranks [0, 1]
r_lgb = rankdata(df_lgb["Will_Buy_EV"].values, method="average") / n
r_xgb = rankdata(df_xgb["Will_Buy_EV"].values, method="average") / n

# 50/50 symmetric blend of two 0.9463+ models
blend_rank = 0.50 * r_lgb + 0.50 * r_xgb

# Final ordinal ranking to guarantee zero ties across all 286,571 test rows
final_rank = rankdata(blend_rank, method="ordinal") / n

print(f"Total test rows: {n}")
print(f"Number of unique values in final rank: {len(np.unique(final_rank))}")
print(f"Number of tied predictions: {n - len(np.unique(final_rank))}")

out_dir = "competitions/s6e9_ev_prediction/output_exp11_golden_blend"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "submission.csv")

sub = pd.DataFrame({
    "id": df_lgb["id"],
    "Will_Buy_EV": final_rank
})
sub.to_csv(out_file, index=False)
print(f"Saved clean Golden Blend submission to {out_file}")
