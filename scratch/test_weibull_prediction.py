import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("competitions/soil_grain_size_photos/data")
train_labels = pd.read_csv(DATA_DIR / "Training_labels_updated.csv")
sample_sub = pd.read_csv(Path("competitions/soil_grain_size_photos/output_exp4/submission.csv"))

npz = np.load("competitions/soil_grain_size_photos/output_nomannic/dinov2_physical_features.npz", allow_pickle=True)
features = npz["features"]       # (162, 5, 384)
sample_ids = npz["sample_ids"]   # 162

train_ids = train_labels["sample_id"].tolist()
test_ids = sample_sub["sample_id"].tolist()

target_cols = [c for c in train_labels.columns if c != "sample_id"]
diameters = np.array([float(c) for c in target_cols])
targets = train_labels[target_cols].values

def weibull_cdf(d, d0, n):
    return 100.0 * (1.0 - np.exp(- (d / d0) ** n))

def emd_score(y_true, y_pred):
    log_d = np.log10(diameters)
    weights = np.diff(log_d)
    diff = np.abs(y_true - y_pred)
    trapezoid = 0.5 * (diff[:, :-1] + diff[:, 1:]) * weights
    return trapezoid.sum(axis=1).mean()

# Fit Weibull (d0, n) to all training samples
fit_params = []
for i in range(len(targets)):
    y = targets[i]
    d50_idx = np.searchsorted(y, 50.0)
    d0_init = diameters[min(d50_idx, len(diameters)-1)]
    popt, _ = curve_fit(weibull_cdf, diameters, y, p0=[d0_init, 1.0], bounds=([1e-4, 0.1], [100.0, 10.0]))
    fit_params.append([np.log(popt[0]), np.log(popt[1])]) # predict in log-space

Y_params = np.array(fit_params) # (24, 2) [log_d0, log_n]

# Pool features
selected_crops = [1, 2, 3, 4]
feature_dim = features.shape[2]

sample_features = {}
for sid in np.unique(sample_ids):
    mask = (sample_ids == sid)
    sample_imgs = features[mask][:, selected_crops].reshape(-1, feature_dim)
    feat = np.concatenate([sample_imgs.mean(axis=0), sample_imgs.std(axis=0)])
    sample_features[sid] = feat

X_train = np.vstack([sample_features[sid] for sid in train_ids]) # (24, 768)
X_test = np.vstack([sample_features[sid] for sid in test_ids])   # (10, 768)

N = len(train_ids)

print("--- LOOCV Evaluation: Direct Prediction of Physical Weibull (log_d0, log_n) ---")
for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0]:
    oof_params = np.zeros_like(Y_params)
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        val_idx = [i]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[train_idx])
        X_va = scaler.transform(X_train[val_idx])
        
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(X_tr, Y_params[train_idx])
        oof_params[val_idx] = model.predict(X_va)
        
    # Reconstruct CDF from predicted (d0, n)
    d0_pred = np.exp(oof_params[:, 0])
    n_pred = np.exp(oof_params[:, 1])
    
    oof_cdf = np.array([weibull_cdf(diameters, d0_pred[i], n_pred[i]) for i in range(N)])
    oof_cdf = np.maximum.accumulate(np.clip(oof_cdf, 0, 100), axis=1)
    oof_cdf[:, -1] = 100.0
    
    score = emd_score(targets, oof_cdf)
    print(f"Weibull Ridge alpha={alpha:<5}: LOOCV EMD = {score:.4f}")
