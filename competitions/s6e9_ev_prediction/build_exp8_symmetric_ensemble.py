import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("=== Building Experiment 8: Symmetric Equal-Footing Ensemble (LGBM 0.50 + XGBoost GPU 0.50) ===")

p_lgb = "competitions/s6e9_ev_prediction/output_exp2/submission.csv"
p_xgb = "competitions/s6e9_ev_prediction/output_exp7_xgboost/submission.csv"

df_lgb = pd.read_csv(p_lgb)
df_xgb = pd.read_csv(p_xgb)

assert len(df_lgb) == len(df_xgb) == 286571, "Test size mismatch!"
assert (df_lgb["id"] == df_xgb["id"]).all(), "ID alignment mismatch!"

n = len(df_lgb)

# Normalized percentile ranks [0, 1]
r_lgb = rankdata(df_lgb["Will_Buy_EV"].values, method="average") / n
r_xgb = rankdata(df_xgb["Will_Buy_EV"].values, method="average") / n

# 50/50 symmetric blend
blend_rank = 0.50 * r_lgb + 0.50 * r_xgb

# Final ordinal ranking to guarantee zero ties across all 286,571 predictions
final_rank = rankdata(blend_rank, method="ordinal") / n

print(f"Total test rows: {n}")
print(f"Number of unique values in final rank: {len(np.unique(final_rank))}")
print(f"Number of tied predictions: {n - len(np.unique(final_rank))}")

out_dir = "competitions/s6e9_ev_prediction/output_exp8_symmetric_ensemble"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "submission.csv")

sub = pd.DataFrame({
    "id": df_lgb["id"],
    "Will_Buy_EV": final_rank
})
sub.to_csv(out_file, index=False)
print(f"Saved clean symmetric ensemble submission to {out_file}")
