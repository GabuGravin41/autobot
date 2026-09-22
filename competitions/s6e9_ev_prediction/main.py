import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb

print("=== Starting Autobot Experiment 2: Advanced Feature Engineering + Multi-Seed LGBM ===")

# 1. Locate dataset
data_dir = None
for p in ["/kaggle/input/playground-series-s6e9", "/kaggle/input/competitions/playground-series-s6e9"]:
    if os.path.exists(os.path.join(p, "train.csv")):
        data_dir = p
        break

if not data_dir:
    for root, _, files in os.walk("/kaggle/input"):
        if "train.csv" in files and "test.csv" in files:
            data_dir = root
            break

print(f"Data directory: {data_dir}")
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
print(f"Train raw shape: {train.shape}, Test raw shape: {test.shape}")

target_col = "Will_Buy_EV"
id_col = "id"

y = train[target_col].values
test_ids = test[id_col].values

X = train.drop(columns=[id_col, target_col]).copy()
X_test = test.drop(columns=[id_col]).copy()

# 2. Base categorical encoding
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Categorical features: {cat_cols}")
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
X_test[cat_cols] = oe.transform(X_test[cat_cols].astype(str))

# 3. Static feature engineering (domain + synthetic artifacts)
def static_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inc = df["Annual_Income_USD"].to_numpy(dtype=np.int64)
    km = df["Daily_Commute_km"].to_numpy(dtype=float)
    km_x10 = np.round(km * 10).astype(np.int64)

    # Domain interaction features
    df["Income_x_Age"] = df["Annual_Income_USD"] * df["Age"]
    df["Income_x_Subsidy"] = df["Annual_Income_USD"] * df["Subsidy_Available"]
    df["City_x_HomeCharging"] = df["City_Type"] * df["Charging_Stations_Near_Home"]
    df["City_x_WorkCharging"] = df["City_Type"] * df["Charging_Stations_Near_Work"]
    df["Total_Charging"] = df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    df["Effective_Charging"] = df["Total_Charging"] * df["Home_Charging_Possible"]
    df["Commute_Income_Ratio"] = df["Daily_Commute_km"] / (df["Annual_Income_USD"] + 1000.0)

    # Synthetic data generation artifact features (digits & modulos)
    df["inc_d0"] = (inc % 10).astype("int8")
    df["inc_d1"] = (inc // 10 % 10).astype("int8")
    df["inc_d2"] = (inc // 100 % 10).astype("int8")
    df["inc_d3"] = (inc // 1000 % 10).astype("int8")
    df["inc_d4"] = (inc // 10000 % 10).astype("int8")
    df["inc_mod100"] = (inc % 100).astype("int16")
    df["inc_mod1000"] = (inc % 1000).astype("int16")

    df["km_d0"] = (km_x10 % 10).astype("int8")
    df["km_mod100"] = (km_x10 % 100).astype("int8")

    # Quantile bins & divisors
    for divisor in (50, 100, 250, 500, 1000, 2500, 5000):
        df[f"inc_q{divisor}"] = (inc // divisor).astype("int32")
    for divisor in (5, 10, 25, 50):
        df[f"km_q{divisor}"] = (km_x10 // divisor).astype("int32")

    # Known boundary flags
    df["is_30k_spike"] = (inc == 30000).astype("int8")
    df["is_millionaire_cliff"] = (inc >= 170537).astype("int8")
    df["is_dead_zone"] = ((inc >= 38000) & (inc <= 42000)).astype("int8")
    df["is_env_hater"] = (df["Environmental_Concern_Level"] == 1).astype("int8")

    return df

print("Applying static feature engineering...")
X = static_feature_engineering(X)
X_test = static_feature_engineering(X_test)
print(f"Features after static engineering: {X.shape[1]}")

# 4. Multi-Seed Stratified K-Fold Training
n_splits = 5
seeds = [42, 2026]
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_total = np.zeros(len(train))
test_preds_total = np.zeros(len(test))

lgb_base_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 31,
    "feature_fraction": 0.35,
    "min_data_in_leaf": 10,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.08,
    "lambda_l2": 2.0,
    "max_bins": 1024,
    "n_estimators": 3500,
    "n_jobs": -1,
    "verbose": -1,
}

for seed_idx, seed in enumerate(seeds, 1):
    print(f"\n--- Running Seed {seed_idx}/{len(seeds)} (seed={seed}) ---")
    seed_oof = np.zeros(len(train))
    seed_test = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx].copy()
        y_va = y[val_idx]
        X_te = X_test.copy()

        # Dynamic fold-level transforms (no leakage):
        # 1. High-granularity quantile discretizer
        kb = KBinsDiscretizer(n_bins=600, encode="ordinal", strategy="quantile")
        X_tr["inc_bin_600"] = kb.fit_transform(X_tr[["Annual_Income_USD"]]).ravel().astype("int32")
        X_va["inc_bin_600"] = kb.transform(X_va[["Annual_Income_USD"]]).ravel().astype("int32")
        X_te["inc_bin_600"] = kb.transform(X_te[["Annual_Income_USD"]]).ravel().astype("int32")

        # 2. Frequency encodings based on train fold
        freq_targets = ["inc_q100", "km_q10", "City_Type", "Current_Car_Type", "Total_Charging"]
        for col in freq_targets:
            freq = X_tr[col].value_counts(normalize=True)
            X_tr[f"{col}_fe"] = X_tr[col].map(freq).fillna(0).astype("float32")
            X_va[f"{col}_fe"] = X_va[col].map(freq).fillna(0).astype("float32")
            X_te[f"{col}_fe"] = X_te[col].map(freq).fillna(0).astype("float32")

        params = lgb_base_params.copy()
        params["random_state"] = seed + fold * 13

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)],
        )

        val_pred = model.predict_proba(X_va)[:, 1]
        seed_oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        print(f"  [Seed {seed}] Fold {fold} - Best Iter: {model.best_iteration_} - AUC: {fold_auc:.5f}")

        # Accumulate fold test prediction
        seed_test += model.predict_proba(X_te)[:, 1] / n_splits

    seed_auc = roc_auc_score(y, seed_oof)
    print(f"-> Seed {seed} Overall OOF AUC: {seed_auc:.5f}")

    # Rank-average to blend seeds without distortion
    oof_preds_total += rankdata(seed_oof) / len(seed_oof) / len(seeds)
    test_preds_total += rankdata(seed_test) / len(seed_test) / len(seeds)

final_cv_auc = roc_auc_score(y, oof_preds_total)
print("\n" + "=" * 60)
print(f"★ FINAL ENSEMBLE OOF CROSS-VALIDATION ROC AUC: {final_cv_auc:.5f}")
print("=" * 60)

# Build submission
output_dir = "/kaggle/working"
os.makedirs(output_dir, exist_ok=True)
sub_file = os.path.join(output_dir, "submission.csv")

submission = pd.DataFrame({
    id_col: test_ids,
    target_col: test_preds_total
})

submission.to_csv(sub_file, index=False)
print(f"Saved submission to {sub_file}")
print(f"Submission shape: {submission.shape}")
print(f"Nulls: {submission[target_col].isna().sum()}")
print(f"Sample predictions:\n{submission.head()}")
print("=== Autobot Experiment 2 Completed Successfully! ===")
