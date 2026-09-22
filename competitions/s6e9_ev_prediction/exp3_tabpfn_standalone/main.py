import os
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score

print("=== Starting Experiment: Standalone TabPFN on Kaggle GPU ===")

# 1. Locate checkpoint from mounted datasets
ckpt_path = None
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if "classifier" in f and f.endswith(".ckpt"):
            ckpt_path = os.path.join(root, f)
            print(f"--> Found pre-mounted TabPFN checkpoint: {ckpt_path}")
            break
    if ckpt_path:
        break

if ckpt_path:
    os.environ["TABPFN_MODEL_CACHE_DIR"] = os.path.dirname(ckpt_path)

# 2. Install official TabPFN package
print("Installing tabpfn...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn", "--quiet"])

from tabpfn import TabPFNClassifier

# 3. Locate data
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

le = LabelEncoder()
y = le.fit_transform(train[target_col].astype(str))
test_ids = test[id_col].values

X = train.drop(columns=[id_col, target_col]).copy()
X_test = test.drop(columns=[id_col]).copy()

# Ordinal encode categoricals
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
X_test[cat_cols] = oe.transform(X_test[cat_cols].astype(str))

# Create holdout validation set (15,000 samples)
val_size = 15000
np.random.seed(42)
val_idx = np.random.choice(len(train), size=val_size, replace=False)
train_pool_idx = np.setdiff1d(np.arange(len(train)), val_idx)

X_val = X.iloc[val_idx].copy()
y_val = y[val_idx]
X_pool = X.iloc[train_pool_idx].copy()
y_pool = y[train_pool_idx]

# TabPFN in-context learning over stratified subsets
n_subsamples = 5
subsample_size = 4000  # 4k diverse samples per TabPFN context
print(f"Running {n_subsamples} TabPFN estimators on GPU (subsample_size={subsample_size})...")

val_preds = np.zeros(len(X_val))
test_preds = np.zeros(len(X_test))
batch_size = 10000

pos_mean = float(np.mean(y_pool))

for i in range(n_subsamples):
    print(f"\n--- Subsample Iteration {i+1}/{n_subsamples} ---")
    np.random.seed(2026 + i * 47)
    
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    
    n_pos = int(subsample_size * pos_mean)
    n_neg = subsample_size - n_pos
    pos_sample = np.random.choice(pos_idx, size=n_pos, replace=False)
    neg_sample = np.random.choice(neg_idx, size=n_neg, replace=False)
    ctx_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(ctx_idx)

    X_ctx = X_pool.iloc[ctx_idx].values
    y_ctx = y_pool[ctx_idx]

    kwargs = {"device": "cuda"}
    if ckpt_path:
        kwargs["model_path"] = ckpt_path

    try:
        clf = TabPFNClassifier(**kwargs)
    except Exception as e:
        print(f"CUDA initialization fallback ({e}), trying CPU...")
        kwargs["device"] = "cpu"
        clf = TabPFNClassifier(**kwargs)

    clf.fit(X_ctx, y_ctx)

    # Predict validation holdout in batches
    val_iter_preds = []
    for b in range(0, len(X_val), batch_size):
        chunk = X_val.iloc[b:b+batch_size].values
        p = clf.predict_proba(chunk)[:, 1]
        val_iter_preds.append(p)
    val_iter_p = np.concatenate(val_iter_preds)
    iter_auc = roc_auc_score(y_val, val_iter_p)
    print(f"  Iteration {i+1} Holdout AUC: {iter_auc:.5f}")
    val_preds += val_iter_p / n_subsamples

    # Predict test set in batches
    test_iter_preds = []
    for b in range(0, len(X_test), batch_size):
        chunk = X_test.iloc[b:b+batch_size].values
        p = clf.predict_proba(chunk)[:, 1]
        test_iter_preds.append(p)
    test_preds += np.concatenate(test_iter_preds) / n_subsamples

final_tabpfn_auc = roc_auc_score(y_val, val_preds)
print("\n" + "=" * 60)
print(f"★ STANDALONE TABPFN ENSEMBLE HOLDOUT ROC AUC: {final_tabpfn_auc:.5f}")
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
print(f"Saved standalone TabPFN submission to {sub_file}")
print("=== Standalone TabPFN Run Completed Successfully! ===")
