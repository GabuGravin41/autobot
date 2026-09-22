import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

def rk(v):
    return rankdata(v, method="average") / len(v)

def lexrank(primary, secondary):
    N = len(primary)
    o = np.lexsort((secondary, primary))
    r = np.empty(N)
    r[o] = np.arange(1, N + 1)
    return (r - 0.5) / N

# 1. Load Submissions & Test Features
base_sub = pd.read_csv("/kaggle/input/datasets/jazivxt/s6e9-zoom-zoom-baseline/submission_latest_best.csv").sort_values("id")
lasso_test = pd.read_csv("/kaggle/input/notebooks/sometimessubodh/s6e9-6-model-meta-stacking-engine/lasso_test_predictions.csv").sort_values("id")
test_df = pd.read_csv("/kaggle/input/competitions/playground-series-s6e9/test.csv").sort_values("id").reset_index(drop=True)

ids = test_df["id"].to_numpy()
base_preds = base_sub["Will_Buy_EV"].to_numpy(dtype=float)
lasso_preds = lasso_test["lasso_test"].to_numpy(dtype=float)

# 2. Extract Feature Masks for 4 Boundary Rules
inc = pd.to_numeric(test_df.Annual_Income_USD, errors="coerce").to_numpy()
km = pd.to_numeric(test_df.Daily_Commute_km, errors="coerce").to_numpy()
env1 = pd.to_numeric(test_df.Environmental_Concern_Level, errors="coerce").to_numpy() == 1
no_sub = test_df.Subsidy_Available.astype(str).to_numpy() == "No"
anx_mh = test_df.Range_Anxiety_Level.isin(["Medium", "High"]).to_numpy()

# 3. Calculate Shift Adjustments
shift = np.zeros(len(test_df))
shift += np.where(inc >= 170537, +10.0, 0.0)                                    # High income (+10)
shift += np.where((inc >= 31004) & (inc <= 41970), -10.0, 0.0)                 # Dead-income zone (-10)
shift += np.where(km >= 83, -5.0, 0.0)                                          # High commute (-5)
shift += np.where((inc == 30000) & no_sub & (env1 | anx_mh), -5.0, 0.0)          # Low income / no subsidy (-5)

# 4. Blend 5% Lasso + Post-Processing + Tie-breaking
# Rank blend primary base with 5% lasso
primary_blend = rk(base_preds) + 0.05 * rk(lasso_preds) + shift

# Tie-break using lasso_preds as the secondary sorter
final_ranks = lexrank(primary_blend, lasso_preds)

# 5. Output Submission
sub = pd.DataFrame({"id": ids, "Will_Buy_EV": final_ranks})
sub.to_csv("submission_blend.csv", index=False)

print(f"Spearman against base anchor: {spearmanr(final_ranks, base_preds).statistic:.6f}")
print("Saved submission_blend.csv successfully.")