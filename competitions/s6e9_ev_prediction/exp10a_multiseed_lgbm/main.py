import os
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

print("=== Starting Autobot Experiment 10A: Multi-Seed LightGBM on Top 10% Feature Matrix ===")

def find_file(name_keyword):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if name_keyword.lower() in f.lower() and f.endswith(".csv"):
                return os.path.join(root, f)
    return None

train_path = find_file("train.csv")
test_path = find_file("test.csv")

orig_path = None
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if ("ev_adoption" in f.lower() or "anxiety" in f.lower() or "adoption" in f.lower()) and f.endswith(".csv"):
            if "train" not in f.lower() and "test" not in f.lower() and "sample" not in f.lower():
                orig_path = os.path.join(root, f)
                break
    if orig_path:
        break

print(f"Train path: {train_path}")
print(f"Test path: {test_path}")
print(f"Original dataset path: {orig_path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

TARGET = "Will_Buy_EV"
ID_COL = "id"

test_ids = test[ID_COL].values

train[TARGET] = (train[TARGET] == "Yes").astype(int)
train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan

combined = pd.concat([train, test], ignore_index=True)
combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")

cat_cols = combined.select_dtypes(include=["object", "string", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c not in [ID_COL, "is_train", TARGET]]
num_cols = [c for c in combined.columns if c not in cat_cols + [ID_COL, "is_train", TARGET]]

# 1. Full-Spectrum Digit Decomposition from 10^-4 to 10^3
print("Extracting full-spectrum digits (10^-4 to 10^3)...")
digit_features = []
for c in num_cols:
    for k in range(-4, 4):
        col_name = f"{c}_digit{k}"
        combined[col_name] = (combined[c].fillna(0) // (10**k) % 10).astype("int8")
        digit_features.append(col_name)

num_cols.extend(digit_features)

# 2. Map Original Real-World Dataset Target Means
if orig_path and os.path.exists(orig_path):
    print(f"Mapping ground truth priors from original dataset: {orig_path}")
    orig = pd.read_csv(orig_path)
    orig[TARGET] = (orig[TARGET] == "Yes").astype(int)
    orig_global_mean = orig[TARGET].mean()
    for col in cat_cols + num_cols:
        if col in orig.columns:
            real_world_stats = orig.groupby(col, observed=False)[TARGET].mean()
            combined[f"{col}_org_mean"] = combined[col].map(real_world_stats).fillna(orig_global_mean).astype(float)

# 3. Numeric to String Category & Global Frequency Encoding
print("Computing frequency encodings...")
num_to_cat_cols = []
for col in num_cols:
    cat_name = f"{col}_cat"
    combined[cat_name] = combined[col].fillna("NaN").astype(str)
    num_to_cat_cols.append(cat_name)

all_cats = cat_cols + num_to_cat_cols
for col in all_cats:
    freq_mapping = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_fe"] = combined[col].map(freq_mapping).astype(float).fillna(0.0)

# 4. Known Boundaries & Smooth Keys
combined["is_30k_spike"] = (combined["Annual_Income_USD"] == 30000.0).astype("int8")
combined["is_millionaire_cliff"] = (combined["Annual_Income_USD"] >= 170537.0).astype("int8")
combined["is_dead_zone"] = ((combined["Annual_Income_USD"] >= 38000.0) & (combined["Annual_Income_USD"] <= 42000.0)).astype("int8")
combined["is_env_hater"] = (combined["Environmental_Concern_Level"] == 1).astype("int8")

combined["income_exact_int"] = np.floor(combined["Annual_Income_USD"]).astype(str)
combined["income100_floor"] = np.floor(combined["Annual_Income_USD"] / 100.0).astype(str)
combined["income1000_floor"] = np.floor(combined["Annual_Income_USD"] / 1000.0).astype(str)
combined["commute_integer"] = np.floor(combined["Daily_Commute_km"]).astype(str)
all_cats.extend(["income_exact_int", "income100_floor", "income1000_floor", "commute_integer"])

train_df = combined[combined["is_train"] == 1].drop(columns=["is_train"])
test_df = combined[combined["is_train"] == 0].drop(columns=["is_train", TARGET])

# 5. Feature Pruning
eval_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET] and pd.api.types.is_numeric_dtype(train_df[c])]
corr_matrix = train_df[eval_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]
to_drop_const = [c for c in train_df.columns if train_df[c].nunique() == 1] + [c for c in test_df.columns if test_df[c].nunique() == 1]
DROP = list(set(to_drop_corr).union(set(to_drop_const)) - {ID_COL, TARGET})

if len(DROP) > 0:
    print(f"Dropping {len(DROP)} redundant/constant features")
    train_df.drop(columns=DROP, inplace=True, errors="ignore")
    test_df.drop(columns=DROP, inplace=True, errors="ignore")

FEATURES = [c for c in test_df.columns if c != ID_COL]
TARGET_ENCODE_COLS = [c for c in all_cats if c not in DROP]

print(f"Total features: {len(FEATURES)}, Target Encode columns: {len(TARGET_ENCODE_COLS)}")

# 6. Multi-Seed Stratified K-Fold Training
Folds = 5
seeds = [42, 2026]
print(f"\nTraining Multi-Seed LightGBM ({len(seeds)} Seeds x {Folds} Folds = {len(seeds)*Folds} Models)...")

X = train_df[FEATURES]
y = train_df[TARGET].values
X_test = test_df[FEATURES]

oof_preds_total = np.zeros(len(train_df))
test_preds_total = np.zeros(len(test_df))

for seed_idx, seed in enumerate(seeds, 1):
    print(f"\n==================== SEED {seed_idx}/{len(seeds)} (seed={seed}) ====================")
    skf = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train_df))
    seed_test = np.zeros(len(test_df))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx].copy(), y[valid_idx]
        X_test_fold = X_test.copy()

        te_auto = TargetEncoder(shuffle=True, cv=Folds, smooth="auto", random_state=seed + fold)
        te_10 = TargetEncoder(shuffle=True, cv=Folds, smooth=10.0, random_state=seed + fold)
        te_100 = TargetEncoder(shuffle=True, cv=Folds, smooth=100.0, random_state=seed + fold)

        X_train_enc_auto = te_auto.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
        X_valid_enc_auto = te_auto.transform(X_valid[TARGET_ENCODE_COLS])
        X_test_enc_auto = te_auto.transform(X_test_fold[TARGET_ENCODE_COLS])

        X_train_enc_10 = te_10.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
        X_valid_enc_10 = te_10.transform(X_valid[TARGET_ENCODE_COLS])
        X_test_enc_10 = te_10.transform(X_test_fold[TARGET_ENCODE_COLS])

        X_train_enc_100 = te_100.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
        X_valid_enc_100 = te_100.transform(X_valid[TARGET_ENCODE_COLS])
        X_test_enc_100 = te_100.transform(X_test_fold[TARGET_ENCODE_COLS])

        for i, col in enumerate(TARGET_ENCODE_COLS):
            X_train[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype("float32")
            X_valid[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype("float32")
            X_test_fold[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype("float32")

            X_train[f"{col}_TE_10"] = X_train_enc_10[:, i].astype("float32")
            X_valid[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype("float32")
            X_test_fold[f"{col}_TE_10"] = X_test_enc_10[:, i].astype("float32")

            X_train[f"{col}_TE_100"] = X_train_enc_100[:, i].astype("float32")
            X_valid[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype("float32")
            X_test_fold[f"{col}_TE_100"] = X_test_enc_100[:, i].astype("float32")

            X_train.drop(columns=[col], inplace=True)
            X_valid.drop(columns=[col], inplace=True)
            X_test_fold.drop(columns=[col], inplace=True)

        clf = lgb.LGBMClassifier(
            n_estimators=20000,
            learning_rate=0.02,
            max_depth=5,
            num_leaves=32,
            min_child_samples=10,
            subsample=0.812763,
            colsample_bytree=0.30293,
            reg_alpha=0.07094,
            reg_lambda=2.03303,
            max_bin=1024,
            random_state=seed + fold * 19,
            feature_pre_filter=False,
            metric="auc",
            n_jobs=-1,
            verbose=-1,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, verbose=False),
                lgb.log_evaluation(period=1000),
            ],
        )

        valid_probs = clf.predict_proba(X_valid)[:, 1]
        seed_oof[valid_idx] = valid_probs
        seed_test += clf.predict_proba(X_test_fold)[:, 1] / Folds

        fold_auc = roc_auc_score(y_valid, valid_probs)
        best_iter = getattr(clf, "best_iteration_", "N/A")
        print(f"  [Seed {seed}] Fold {fold}/{Folds} Best Tree: {best_iter} | ROC-AUC: {fold_auc:.5f}")

    seed_auc = roc_auc_score(y, seed_oof)
    print(f"★ Seed {seed} Overall OOF ROC-AUC: {seed_auc:.5f}")

    # Accumulate rank percentiles to eliminate tie distortion
    oof_preds_total += rankdata(seed_oof, method="average") / (len(train_df) * len(seeds))
    test_preds_total += rankdata(seed_test, method="average") / (len(test_df) * len(seeds))

final_oof_auc = roc_auc_score(y, oof_preds_total)
print("\n" + "=" * 65)
print(f"🏆 EXPERIMENT 10A MULTI-SEED FINAL ENSEMBLE OOF ROC-AUC: {final_oof_auc:.5f}")
print("=" * 65)

# Convert to ordinal ranks [0, 1] to guarantee zero ties
final_submission_ranks = rankdata(test_preds_total, method="ordinal") / len(test_df)
print(f"Final submission unique values: {len(np.unique(final_submission_ranks))} / {len(test_df)}")

output_dir = "/kaggle/working"
os.makedirs(output_dir, exist_ok=True)

sub_file = os.path.join(output_dir, "submission.csv")
sub = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: final_submission_ranks,
})
sub.to_csv(sub_file, index=False)
print(f"Saved submission to {sub_file}")

oof_file = os.path.join(output_dir, "oof_preds.npy")
np.save(oof_file, oof_preds_total)
print("=== Experiment 10A Completed Successfully! ===")
