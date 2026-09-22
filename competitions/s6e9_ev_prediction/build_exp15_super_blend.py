import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

print("=" * 70)
print("=== Autobot Exp 15: SOTA Grandmaster Multi-Paradigm Super-Blend ===")
print("=" * 70)

p_mega = "competitions/s6e9_ev_prediction/output_megayak_094656/submission.csv"
p_berat = "competitions/s6e9_ev_prediction/output_berat_094656/submission.csv"
p_subodh = "competitions/s6e9_ev_prediction/output_subodh_094651/submission_blend.csv"
p_10a = "competitions/s6e9_ev_prediction/output_exp10a_lgbm/submission.csv"
p_10b = "competitions/s6e9_ev_prediction/output_exp10b_xgboost/submission.csv"

for p in [p_mega, p_berat, p_subodh, p_10a, p_10b]:
    assert os.path.exists(p), f"Missing file: {p}"

df_mega = pd.read_csv(p_mega).sort_values("id").reset_index(drop=True)
df_berat = pd.read_csv(p_berat).sort_values("id").reset_index(drop=True)
df_subodh = pd.read_csv(p_subodh).sort_values("id").reset_index(drop=True)
df_10a = pd.read_csv(p_10a).sort_values("id").reset_index(drop=True)
df_10b = pd.read_csv(p_10b).sort_values("id").reset_index(drop=True)

N = len(df_mega)
assert N == 286571, f"Unexpected row count: {N}"
assert (df_mega["id"] == df_berat["id"]).all() and (df_mega["id"] == df_10a["id"]).all(), "ID mismatch!"

r_mega = rankdata(df_mega["Will_Buy_EV"].values, method="average") / N
r_berat = rankdata(df_berat["Will_Buy_EV"].values, method="average") / N
r_subodh = rankdata(df_subodh["Will_Buy_EV"].values, method="average") / N
r_10a = rankdata(df_10a["Will_Buy_EV"].values, method="average") / N
r_10b = rankdata(df_10b["Will_Buy_EV"].values, method="average") / N

blend_scores = (
    0.35 * r_mega +
    0.35 * r_berat +
    0.15 * r_subodh +
    0.10 * r_10a +
    0.05 * r_10b
)


final_ranks = (rankdata(blend_scores, method="ordinal") - 0.5) / N

out_dir = "competitions/s6e9_ev_prediction/output_exp15_super_blend"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "submission.csv")

sub = pd.DataFrame({
    "id": df_mega["id"],
    "Will_Buy_EV": final_ranks
})
sub.to_csv(out_file, index=False)

print(f"Total rows: {len(sub)}")
print(f"Unique predictions: {sub['Will_Buy_EV'].nunique()}")
print(f"Min prediction: {sub['Will_Buy_EV'].min():.8f}")
print(f"Max prediction: {sub['Will_Buy_EV'].max():.8f}")
print(f"Spearman vs Megayak (0.94656): {spearmanr(final_ranks, r_mega).statistic:.6f}")
print(f"Spearman vs Berat (0.94656):   {spearmanr(final_ranks, r_berat).statistic:.6f}")
print(f"Spearman vs Subodh (0.94651):  {spearmanr(final_ranks, r_subodh).statistic:.6f}")
print(f"Spearman vs Exp10A (0.94640):  {spearmanr(final_ranks, r_10a).statistic:.6f}")
print(f"Saved to {out_file}")
