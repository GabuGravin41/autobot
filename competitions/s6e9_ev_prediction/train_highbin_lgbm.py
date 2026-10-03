"""
Autobot S6E9: High-Bin (max_bin=4096) LightGBM on 112 Features
Hypothesis:
Georgy Mamarin proved in 'Income Needs 1782 Bins, LightGBM Gives 255' that
Annual_Income_USD has 13,214 distinct values. Default max_bin=255 compresses
critical decision boundaries. Raising max_bin to 4096 resolves the fine-grained
income distribution directly.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.insert(0, ".")
from competitions.s6e9_ev_prediction.train_jazivxt_sota import (
    build_row_local_features,
    prepare_full_features,
    income_asymmetric_full
)

print("=== AUTOBOT S6E9: HIGH-BIN (max_bin=4096) LIGHTGBM ===")
t0 = time.time()

data_dir = "competitions/s6e9_ev_prediction/data"
print("Reading train and test data...")
train = pd.read_csv(f"{data_dir}/train.csv")
test = pd.read_csv(f"{data_dir}/test.csv")

y = (
    train["Will_Buy_EV"].map({"No": 0, "Yes": 1})
    if not pd.api.types.is_numeric_dtype(train["Will_Buy_EV"])
    else train["Will_Buy_EV"]
).to_numpy(np.uint8)

X_train, train_keys = build_row_local_features(train.drop(columns=["id", "Will_Buy_EV"]))
X_test, test_keys = build_row_local_features(test.drop(columns="id"))

print("Engineering full features...")
X_train, X_test = prepare_full_features(X_train, train_keys, X_test, test_keys, y)

income = train["Annual_Income_USD"].to_numpy()
test_income = test["Annual_Income_USD"].to_numpy()

for resolution in (8192, 16384):
    train_block, test_block = income_asymmetric_full(income, y, test_income, resolution)
    for j in range(1 if resolution == 16384 else 0, train_block.shape[1]):
        name = f"own_income_{resolution}_{j}"
        X_train[name] = train_block[:, j]
        X_test[name] = test_block[:, j]

print(f"Total Features: {X_train.shape[1]} (verified exact match to 112)")
assert X_train.shape[1] == 112, f"Expected 112 features, got {X_train.shape[1]}"

# Stratified 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train), dtype=np.float32)
test_preds = np.zeros(len(test), dtype=np.float32)

print("\n--- Training 5-Fold Stratified High-Bin LightGBM (max_bin=4096) ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=12,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.40,
        reg_alpha=0.1,
        reg_lambda=3.0,
        max_bin=4096,
        feature_pre_filter=False,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
        metric="auc"
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"  Fold {fold+1}/5 AUC: {fold_auc:.6f}")

    test_preds += model.predict_proba(X_test)[:, 1] / 5.0

overall_auc = roc_auc_score(y, oof_preds)
print(f"\n===> OVERALL 5-FOLD OOF AUC: {overall_auc:.6f} (baseline was 0.946071) <===")

out_dir = "competitions/s6e9_ev_prediction/output_highbin_lgbm"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/oof.npy", oof_preds)
np.save(f"{out_dir}/test_pred.npy", test_preds)

sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": test_preds})
sub.to_csv(f"{out_dir}/submission.csv", index=False)
print(f"Saved submission to {out_dir}/submission.csv ({len(sub)} rows)")
print(f"Total pipeline elapsed: {time.time() - t0:.1f}s")
