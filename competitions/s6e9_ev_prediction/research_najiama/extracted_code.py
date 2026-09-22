import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
import warnings
warnings.filterwarnings('ignore')
# --- CELL ---
TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e9/train.csv'
TEST_PATH  = '/kaggle/input/competitions/playground-series-s6e9/test.csv'
SUB_PATH   = '/kaggle/input/competitions/playground-series-s6e9/sample_submission.csv'
ORIG_PATH  = '/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv' # 

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
orig = pd.read_csv(ORIG_PATH)
submission = pd.read_csv(SUB_PATH)
# --- CELL ---
TARGET = 'Will_Buy_EV'
train[TARGET] = train[TARGET].map({'Yes': 1, 'No': 0})
orig[TARGET] = orig[TARGET].map({'Yes': 1, 'No': 0})

train['is_train'] = 1
test['is_train'] = 0
test[TARGET] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined.drop(columns=['Number_of_Cars_Owned'], inplace=True, errors='ignore')

cat_cols = combined.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = [c for c in combined.columns if c not in cat_cols + ['id', 'is_train', TARGET]]

# Extract digits from the 10^-4 place up to the 10^3 place
digit_features = []
for c in num_cols:
    for k in range(-4, 4):
        col_name = f"{c}_digit{k}"
        combined[col_name] = (combined[c].fillna(0) // (10**k) % 10).astype('int8')
        digit_features.append(col_name)

# Add the new digit features so they get processed by your frequency/target encoders
num_cols.extend(digit_features)

# Map Original Dataset Target Means
orig_global_mean = orig[TARGET].mean()
for col in cat_cols + num_cols:
    if col in orig.columns:
        real_world_stats = orig.groupby(col, observed=False)[TARGET].mean()
        combined[f"{col}_org_mean"] = combined[col].map(real_world_stats).fillna(orig_global_mean).astype(float)

# Convert Numerics to String Categories
num_to_cat_cols = []
for col in num_cols:
    cat_name = f"{col}_cat"
    combined[cat_name] = combined[col].fillna('NaN').astype(str)
    num_to_cat_cols.append(cat_name)

# Global Frequency Encoding
all_cats = cat_cols + num_to_cat_cols
for col in all_cats:
    freq_mapping = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_fe"] = combined[col].map(freq_mapping).astype(float).fillna(0.0)

# The Mode Collapse Spike
combined['is_30k_spike'] = (combined['Annual_Income_USD'] == 30000.0).astype('int8')

# The Millionaire Cliff (100% buy rate region)
combined['is_millionaire_cliff'] = (combined['Annual_Income_USD'] >= 170537.0).astype('int8')

# The Dead Zone (0% buy rate region)
combined['is_dead_zone'] = ((combined['Annual_Income_USD'] >= 38000.0) & (combined['Annual_Income_USD'] <= 42000.0)).astype('int8')

# Environmental Concern Extremes
combined['is_env_hater'] = (combined['Environmental_Concern_Level'] == 1).astype('int8')

# ==========================================
# Markus's "Smooth Keys" (Binned Numerics)
# ==========================================
print("🔑 Adding Smooth Keys (Income/Commute Bins)...")
combined['income_exact_int'] = np.floor(combined['Annual_Income_USD']).astype(str)
combined['income100_floor']  = np.floor(combined['Annual_Income_USD'] / 100.0).astype(str)
combined['income1000_floor'] = np.floor(combined['Annual_Income_USD'] / 1000.0).astype(str)
combined['commute_integer']  = np.floor(combined['Daily_Commute_km']).astype(str)
# Adding these 4 new string columns to all_cats so they get Frequency and Target Encoded
all_cats.extend(['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'])

train = combined[combined['is_train'] == 1].drop(columns=['is_train'])
test = combined[combined['is_train'] == 0].drop(columns=['is_train', TARGET])

#  FEATURE DROPPING
# Identify numeric columns to evaluate for correlatio (ignore strings/objects because .corr() will fail on them)
eval_cols = [c for c in train.columns if c not in ['id', TARGET] and pd.api.types.is_numeric_dtype(train[c])]

# Find perfectly correlated features (1.0 correlation)
corr_matrix = train[eval_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]

# Find constant features (only 1 unique value) in train or test
to_drop_const = [c for c in train.columns if train[c].nunique() == 1] + \
                [c for c in test.columns if test[c].nunique() == 1]

# Combine all bad features into a set to drop
DROP = set(to_drop_corr).union(set(to_drop_const))
DROP = [c for c in DROP if c not in ['id', TARGET]] # Safety check

if len(DROP) > 0:
    print(f"Dropping {len(DROP)} redundant/constant features")
    # print(f"Dropped features: {{DROP}}")
    train.drop(columns=DROP, inplace=True, errors='ignore')
    test.drop(columns=DROP, inplace=True, errors='ignore')
else:
    print("   -> No redundant features found.")
# ==========================================

FEATURES = [c for c in test.columns if c != 'id']
# Safely remove dropped columns from target encoding list
TARGET_ENCODE_COLS = [c for c in all_cats if c not in DROP] 

print(f"✅ Total Features: {len(FEATURES)}")
print(f"✅ Columns to Target Encode: {len(TARGET_ENCODE_COLS)}")
# --- CELL ---
Folds = 5
print(f"\n🚀 Training  LIGHTGBM with {Folds} Folds...")

X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]

skf = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx].copy(), y.iloc[valid_idx]
    X_test_fold = X_test.copy()

    
    # Triple Sklearn Target Encoders (Auto, Strict 10, and Massive 100)
    te_auto = TargetEncoder(shuffle=True, cv=Folds, smooth='auto', random_state=42)
    te_10   = TargetEncoder(shuffle=True, cv=Folds, smooth=10.0, random_state=42)
    te_100  = TargetEncoder(shuffle=True, cv=Folds, smooth=100.0, random_state=42)
    
    X_train_enc_auto = te_auto.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
    X_valid_enc_auto = te_auto.transform(X_valid[TARGET_ENCODE_COLS])
    X_test_enc_auto  = te_auto.transform(X_test_fold[TARGET_ENCODE_COLS])

    X_train_enc_10 = te_10.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
    X_valid_enc_10 = te_10.transform(X_valid[TARGET_ENCODE_COLS])
    X_test_enc_10  = te_10.transform(X_test_fold[TARGET_ENCODE_COLS])

    X_train_enc_100 = te_100.fit_transform(X_train[TARGET_ENCODE_COLS], y_train)
    X_valid_enc_100 = te_100.transform(X_valid[TARGET_ENCODE_COLS])
    X_test_enc_100  = te_100.transform(X_test_fold[TARGET_ENCODE_COLS])
    
    for i, col in enumerate(TARGET_ENCODE_COLS):
        # Auto smoothing TE
        X_train[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
        X_valid[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype('float32')
        X_test_fold[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')
        
        # Strict (10.0) smoothing TE
        X_train[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
        X_valid[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype('float32')
        X_test_fold[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')

        # Massive (100.0) smoothing TE (Markus's method)
        X_train[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
        X_valid[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype('float32')
        X_test_fold[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')
        
        # Drop the original string column
        X_train.drop(columns=[col], inplace=True)
        X_valid.drop(columns=[col], inplace=True)
        X_test_fold.drop(columns=[col], inplace=True)

    # --- LIGHTGBM MODEL ---
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
        random_state=42,
        feature_pre_filter=False,
        metric='auc',
        n_jobs=-1,
        verbose=-1
    )        
    
    clf.fit(
        X_train, y_train, 
        eval_set=[(X_valid, y_valid)], 
        callbacks=[
            lgb.early_stopping(stopping_rounds=500, verbose=False),
            lgb.log_evaluation(period=1000)
        ]
    )
    
    valid_probs = clf.predict_proba(X_valid)[:, 1]
    oof_preds[valid_idx] = valid_probs
    test_preds += clf.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    
    fold_auc = roc_auc_score(y_valid, valid_probs)
    print(f"   --> Fold {fold} CONVERGED at Tree #{clf.best_iteration_} | ROC-AUC: {fold_auc:.5f}")
# --- CELL ---
final_cv_score = roc_auc_score(y, oof_preds)
print("\n" + "="*45)
print(f"🏆   LIGHTGBM FINAL OOF ROC-AUC: {final_cv_score:.5f}")
print("="*45)

MODEL_NAME = "LIGHTGBM"

# Save Kaggle Submission (Averaged Test Predictions)
submission[TARGET] = test_preds
submission.to_csv(f'submission_{MODEL_NAME}.csv', index=False)
print(f"💾 Saved 'submission_{MODEL_NAME}.csv'")

# Save Raw Test Predictions (for ensembling)
test_df = pd.DataFrame({'id': test['id'], TARGET: test_preds})
test_df.to_csv(f'test_{MODEL_NAME}.csv', index=False)
print(f"💾 Saved 'test_{MODEL_NAME}.csv'")

# Save OOF Predictions (for ensembling meta-model)
oof_df = pd.DataFrame({'id': train['id'], 'OOF_Pred': oof_preds})
oof_df.to_csv(f'oof_{MODEL_NAME}.csv', index=False)
print(f"💾 Saved 'oof_{MODEL_NAME}.csv'")