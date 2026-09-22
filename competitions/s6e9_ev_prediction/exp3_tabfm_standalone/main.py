import os
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

print("=== Starting Experiment 3: Standalone Google TabFM on Kaggle GPU ===")

# Force PyTorch backend by removing incompatible Flax version
print("Setting up pure PyTorch environment for tabfm...")
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "flax", "--quiet"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tabfm", "huggingface_hub", "--quiet"])

from tabfm import TabFMClassifier

# Locate dataset
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

# Encode target properly as integer 0/1
le = LabelEncoder()
y = le.fit_transform(train[target_col].astype(str))
print(f"Target distribution: {np.bincount(y)}")

test_ids = test[id_col].values
X = train.drop(columns=[id_col, target_col]).copy()
X_test = test.drop(columns=[id_col]).copy()

# Stratified Context Subsampling
n_contexts = 8
context_size = 5000
print(f"Running {n_contexts} diverse TabFM in-context estimators (context_size={context_size})...")

val_size = 20000
np.random.seed(42)
val_idx = np.random.choice(len(train), size=val_size, replace=False)
train_pool_idx = np.setdiff1d(np.arange(len(train)), val_idx)

X_val = X.iloc[val_idx].copy()
y_val = y[val_idx]
X_pool = X.iloc[train_pool_idx].copy()
y_pool = y[train_pool_idx]

val_preds = np.zeros(len(X_val))
test_preds = np.zeros(len(X_test))
batch_size = 10000

pos_mean = float(np.mean(y_pool))

for i in range(n_contexts):
    print(f"\n--- Context Iteration {i+1}/{n_contexts} ---")
    np.random.seed(100 + i * 37)
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    
    n_pos = int(context_size * pos_mean)
    n_neg = context_size - n_pos
    pos_sample = np.random.choice(pos_idx, size=n_pos, replace=False)
    neg_sample = np.random.choice(neg_idx, size=n_neg, replace=False)
    ctx_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(ctx_idx)

    X_ctx = X_pool.iloc[ctx_idx]
    y_ctx = y_pool[ctx_idx]

    clf = TabFMClassifier()
    clf.fit(X_ctx, y_ctx)

    val_p = clf.predict_proba(X_val)[:, 1]
    iter_val_auc = roc_auc_score(y_val, val_p)
    print(f"  Context {i+1} Holdout AUC: {iter_val_auc:.5f}")
    val_preds += val_p / n_contexts

    curr_test_p = []
    for b in range(0, len(X_test), batch_size):
        chunk = X_test.iloc[b:b+batch_size]
        p = clf.predict_proba(chunk)[:, 1]
        curr_test_p.append(p)
    
    test_preds += np.concatenate(curr_test_p) / n_contexts

final_tabfm_auc = roc_auc_score(y_val, val_preds)
print("\n" + "=" * 60)
print(f"★ STANDALONE TABFM ENSEMBLE HOLDOUT ROC AUC: {final_tabfm_auc:.5f}")
print("=" * 60)

output_dir = "/kaggle/working"
os.makedirs(output_dir, exist_ok=True)
sub_file = os.path.join(output_dir, "submission.csv")

submission = pd.DataFrame({
    id_col: test_ids,
    target_col: test_preds
})

submission.to_csv(sub_file, index=False)
print(f"Saved standalone TabFM submission to {sub_file}")
print("=== Standalone TabFM Run Completed! ===")
