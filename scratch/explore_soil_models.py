import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

# 1. Load data
DATA_DIR = Path("competitions/soil_grain_size_photos/data")
train_labels = pd.read_csv(DATA_DIR / "Training_labels_updated.csv")
sample_sub = pd.read_csv(Path("competitions/soil_grain_size_photos/output_exp4/submission.csv"))

npz = np.load("competitions/soil_grain_size_photos/output_nomannic/dinov2_physical_features.npz", allow_pickle=True)
features = npz["features"]       # (162, 5, 384)
sample_ids = npz["sample_ids"]   # 162
crop_names = npz["crop_names"]

train_ids = train_labels["sample_id"].tolist()
test_ids = sample_sub["sample_id"].tolist()

target_cols = [c for c in train_labels.columns if c != "sample_id"]
diameters = np.array([float(c) for c in target_cols])
targets = train_labels[target_cols].values

# Sqrt-mass transformation
def encode_sqrt_mass(y):
    # y shape (N, 11)
    mass = np.diff(np.column_stack([np.zeros(len(y)), y]), axis=1) # (N, 11)
    return np.sqrt(np.maximum(0, mass) / 100.0)

def decode_sqrt_mass(raw):
    # raw shape (N, 11)
    mass = np.square(np.maximum(0, raw)) + 1e-8
    mass = 100.0 * mass / mass.sum(axis=1, keepdims=True)
    cdf = np.cumsum(mass, axis=1)
    cdf = np.maximum.accumulate(np.clip(cdf, 0, 100), axis=1)
    cdf[:, -1] = 100.0
    return cdf

def emd_score(y_true, y_pred):
    # Earth Mover's Distance under log-spacing
    # Integral of |CDF_true - CDF_pred| over log(d)
    log_d = np.log10(diameters)
    # trapezoidal integration weights
    weights = np.diff(log_d)
    # intervals
    diff = np.abs(y_true - y_pred) # (N, 11)
    # trapezoid: 0.5 * (diff[:, i] + diff[:, i+1]) * weights[i]
    trapezoid = 0.5 * (diff[:, :-1] + diff[:, 1:]) * weights
    emd_per_sample = trapezoid.sum(axis=1)
    return emd_per_sample.mean()

# Pool features per sample_id
# Let's pool crops [1, 2, 3, 4] mean + std
selected_crops = [1, 2, 3, 4]
feature_dim = features.shape[2]

sample_features = {}
for sid in np.unique(sample_ids):
    mask = (sample_ids == sid)
    sample_imgs = features[mask][:, selected_crops].reshape(-1, feature_dim)
    feat = np.concatenate([sample_imgs.mean(axis=0), sample_imgs.std(axis=0)])
    sample_features[sid] = feat

X_train = np.vstack([sample_features[sid] for sid in train_ids])
X_test = np.vstack([sample_features[sid] for sid in test_ids])
Y_train_encoded = encode_sqrt_mass(targets)

print(f"X_train shape: {X_train.shape}, Y_train shape: {targets.shape}")

# Leave-One-Out Cross Validation
N = len(train_ids)

def run_loocv(model_fn):
    oof_encoded = np.zeros_like(Y_train_encoded)
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        val_idx = [i]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[train_idx])
        X_va = scaler.transform(X_train[val_idx])
        
        model = model_fn()
        model.fit(X_tr, Y_train_encoded[train_idx])
        oof_encoded[val_idx] = model.predict(X_va)
        
    oof_cdf = decode_sqrt_mass(oof_encoded)
    return emd_score(targets, oof_cdf)

# Test Ridge with various alphas
print("\n--- Evaluating Ridge Models (LOOCV EMD) ---")
for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    score = run_loocv(lambda: Ridge(alpha=alpha, solver="lsqr"))
    print(f"Ridge alpha={alpha:<6}: LOOCV EMD = {score:.4f}")

# Test Bayesian Ridge
print("\n--- Evaluating Bayesian Ridge ---")
from sklearn.multioutput import MultiOutputRegressor
score_br = run_loocv(lambda: MultiOutputRegressor(BayesianRidge()))
print(f"Bayesian Ridge: LOOCV EMD = {score_br:.4f}")

# Test SVR (RBF kernel)
print("\n--- Evaluating SVR RBF ---")
for C in [0.1, 1.0, 5.0, 10.0]:
    for gamma in ["scale", 0.001, 0.01]:
        score_svr = run_loocv(lambda: MultiOutputRegressor(SVR(C=C, gamma=gamma)))
        print(f"SVR C={C}, gamma={gamma}: LOOCV EMD = {score_svr:.4f}")
