import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=== Starting Experiment 4: Standalone 10-Fold CatBoost on Kaggle GPU ===")

# Locate data
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
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

target_col = "Will_Buy_EV"
id_col = "id"

y = train[target_col].values
test_ids = test[id_col].values

X = train.drop(columns=[id_col, target_col]).copy()
X_test = test.drop(columns=[id_col]).copy()

# Fill missing categoricals as string "Missing" for CatBoost native support
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].fillna("Missing").astype(str)
    X_test[col] = X_test[col].fillna("Missing").astype(str)

print(f"Native categorical features for CatBoost ({len(cat_cols)}): {cat_cols}")

# 10-Fold Stratified K-Fold
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_aucs = []

cb_params = {
    "iterations": 3500,
    "learning_rate": 0.04,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": 42,
    "task_type": "GPU",
    "verbose": 0,
}

print(f"\nTraining 10-Fold CatBoost on GPU...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model = CatBoostClassifier(**cb_params, cat_features=cat_cols)
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=100,
        verbose=False,
    )

    val_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_va, val_pred)
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold}/{n_splits} - Best Iter: {model.get_best_iteration()} - Val AUC: {fold_auc:.5f}")

    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

cv_auc = roc_auc_score(y, oof_preds)
print("\n" + "=" * 60)
print(f"★ STANDALONE 10-FOLD CATBOOST OOF ROC AUC: {cv_auc:.5f}")
print("=" * 60)

# Build submission
output_dir = "/kaggle/working"
os.makedirs(output_dir, exist_ok=True)
sub_file = os.path.join(output_dir, "submission.csv")

submission = pd.DataFrame({
    id_col: test_ids,
    target_col: test_preds
})

submission.to_csv(sub_file, index=False)
print(f"Saved CatBoost submission to {sub_file}")
print(f"Submission shape: {submission.shape}")
print(f"Nulls: {submission[target_col].isna().sum()}")
print(f"Sample predictions:\n{submission.head()}")
print("=== CatBoost Standalone Completed! ===")
