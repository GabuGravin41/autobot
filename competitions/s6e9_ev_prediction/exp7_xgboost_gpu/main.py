import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

print(f"=== Starting Autobot Experiment 7: 10-Fold XGBoost GPU on Enriched Feature Matrix ===")
print(f"XGBoost version: {xgb.__version__}")

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

y = (train[target_col] == "Yes").astype(int).values
print(f"Target distribution: {np.bincount(y)} (positive mean = {np.mean(y):.4f})")
test_ids = test[id_col].values

X = train.drop(columns=[id_col, target_col]).copy()
X_test = test.drop(columns=[id_col]).copy()

# 2. Base categorical encoding
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Categorical features: {cat_cols}")
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
X_test[cat_cols] = oe.transform(X_test[cat_cols].astype(str))

# 3. Static feature engineering (matched with Exp 2 LightGBM matrix)
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

# 4. Determine GPU backend compatibility
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.40,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "min_child_weight": 10,
    "max_bin": 512,
    "n_estimators": 3500,
    "early_stopping_rounds": 100,
}

try:
    test_m = xgb.XGBClassifier(n_estimators=2, tree_method="hist", device="cuda")
    test_m.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))
    xgb_params["tree_method"] = "hist"
    xgb_params["device"] = "cuda"
    print("XGBoost GPU verified: tree_method='hist', device='cuda'")
except Exception as e:
    print(f"device='cuda' check failed ({e}), trying tree_method='gpu_hist'...")
    try:
        test_m = xgb.XGBClassifier(n_estimators=2, tree_method="gpu_hist")
        test_m.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))
        xgb_params["tree_method"] = "gpu_hist"
        print("XGBoost GPU verified: tree_method='gpu_hist'")
    except Exception as e2:
        print(f"gpu_hist failed ({e2}), falling back to CPU hist...")
        xgb_params["tree_method"] = "hist"
        xgb_params["n_jobs"] = -1

# 5. 10-Fold Stratified K-Fold Training
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test.copy()

    # Dynamic fold-level transforms:
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

    fold_params = xgb_params.copy()
    fold_params["random_state"] = 42 + fold * 37

    model = xgb.XGBClassifier(**fold_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    val_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_va, val_pred)
    best_iter = getattr(model, "best_iteration", "N/A")
    print(f"Fold {fold}/{n_splits} - Best Iter: {best_iter} - AUC: {fold_auc:.5f}")

    test_preds += model.predict_proba(X_te)[:, 1] / n_splits

overall_oof_auc = roc_auc_score(y, oof_preds)
print("\n" + "=" * 60)
print(f"★ 10-FOLD XGBOOST GPU OVERALL OOF ROC AUC: {overall_oof_auc:.5f}")
print("=" * 60)

# Save artifacts
output_dir = "/kaggle/working"
os.makedirs(output_dir, exist_ok=True)

sub_file = os.path.join(output_dir, "submission.csv")
sub = pd.DataFrame({
    id_col: test_ids,
    target_col: test_preds
})
sub.to_csv(sub_file, index=False)
print(f"Saved submission to {sub_file}")

oof_file = os.path.join(output_dir, "oof_preds.npy")
np.save(oof_file, oof_preds)
print(f"Saved OOF predictions to {oof_file}")
print("=== Experiment 7 Completed Successfully! ===")
