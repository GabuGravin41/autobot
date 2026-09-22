import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("=== Building Experiment 6: 3-Way Clean Rank Ensemble (LGBM + CatBoost + TabPFN) ===")

# 1. Load predictions
p_lgb = "competitions/s6e9_ev_prediction/output_exp2/submission.csv"
p_cat = "competitions/s6e9_ev_prediction/output_exp4_catboost/submission.csv"
p_pfn = "competitions/s6e9_ev_prediction/output_exp4_tabpfn/submission.csv"

df_lgb = pd.read_csv(p_lgb)
df_cat = pd.read_csv(p_cat)
df_pfn = pd.read_csv(p_pfn)

assert len(df_lgb) == len(df_cat) == len(df_pfn) == 286571, "Test size mismatch!"
assert (df_lgb["id"] == df_cat["id"]).all() and (df_lgb["id"] == df_pfn["id"]).all(), "ID alignment mismatch!"

n = len(df_lgb)

# 2. Compute normalized percentile ranks [0, 1]
r_lgb = rankdata(df_lgb["Will_Buy_EV"].values, method="average") / n
r_cat = rankdata(df_cat["Will_Buy_EV"].values, method="average") / n
r_pfn = rankdata(df_pfn["Will_Buy_EV"].values, method="average") / n

# 3. Weighted rank combination
# LightGBM is the dominant performer (0.94448)
# CatBoost provides continuous tree regularization (0.94124)
# TabPFN provides transformer/in-context structural diversity (0.93781)
w_lgb = 0.80
w_cat = 0.14
w_pfn = 0.06

blend_rank = w_lgb * r_lgb + w_cat * r_cat + w_pfn * r_pfn

# Normalize to strict [0, 1] unique continuous percentiles
final_rank = rankdata(blend_rank, method="ordinal") / n

print(f"Total test rows: {n}")
print(f"Number of unique values in final rank: {len(np.unique(final_rank))}")
print(f"Number of tied predictions: {n - len(np.unique(final_rank))}")

out_dir = "competitions/s6e9_ev_prediction/output_exp6_ensemble"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "submission.csv")

sub = pd.DataFrame({
    "id": df_lgb["id"],
    "Will_Buy_EV": final_rank
})
sub.to_csv(out_file, index=False)
print(f"Saved clean 3-way ensemble submission to {out_file}")
