import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load data
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

def encode_sqrt_mass(y):
    mass = np.diff(np.column_stack([np.zeros(len(y)), y]), axis=1)
    return np.sqrt(np.maximum(0, mass) / 100.0)

def decode_sqrt_mass(raw):
    mass = np.square(np.maximum(0, raw)) + 1e-8
    mass = 100.0 * mass / mass.sum(axis=1, keepdims=True)
    cdf = np.cumsum(mass, axis=1)
    cdf = np.maximum.accumulate(np.clip(cdf, 0, 100), axis=1)
    cdf[:, -1] = 100.0
    return cdf

def emd_score(y_true, y_pred):
    log_d = np.log10(diameters)
    weights = np.diff(log_d)
    diff = np.abs(y_true - y_pred)
    trapezoid = 0.5 * (diff[:, :-1] + diff[:, 1:]) * weights
    return trapezoid.sum(axis=1).mean()

# Pool features per sample_id
selected_crops = [1, 2, 3, 4]
feature_dim = features.shape[2]

sample_features_mean = {}
sample_features_full = {}
for sid in np.unique(sample_ids):
    mask = (sample_ids == sid)
    sample_imgs = features[mask][:, selected_crops].reshape(-1, feature_dim)
    # L2 normalize each patch representation
    norms = np.linalg.norm(sample_imgs, axis=1, keepdims=True)
    sample_imgs_norm = sample_imgs / np.maximum(norms, 1e-8)
    
    m = sample_imgs_norm.mean(axis=0)
    m = m / np.linalg.norm(m) # unit vector
    sample_features_mean[sid] = m
    sample_features_full[sid] = np.concatenate([sample_imgs.mean(axis=0), sample_imgs.std(axis=0)])

X_mean = np.vstack([sample_features_mean[sid] for sid in train_ids]) # (24, 384) unit vectors
X_full = np.vstack([sample_features_full[sid] for sid in train_ids]) # (24, 768)
Y_train_encoded = encode_sqrt_mass(targets)

N = len(train_ids)

# 1. Test Cosine Kernel Ridge
print("--- 1. Cosine Kernel Ridge on Normalized DINOv2 Embeddings ---")
for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    oof_encoded = np.zeros_like(Y_train_encoded)
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        val_idx = [i]
        
        # Kernel matrix is cosine similarity: K = X @ X.T
        K_tr = X_mean[train_idx] @ X_mean[train_idx].T
        K_val = X_mean[val_idx] @ X_mean[train_idx].T
        
        kr = KernelRidge(alpha=alpha, kernel="precomputed")
        kr.fit(K_tr, Y_train_encoded[train_idx])
        oof_encoded[val_idx] = kr.predict(K_val)
        
    oof_cdf = decode_sqrt_mass(oof_encoded)
    score = emd_score(targets, oof_cdf)
    print(f"Cosine Kernel Ridge alpha={alpha:<6}: LOOCV EMD = {score:.4f}")

# 2. Test PCA + Ridge
print("\n--- 2. PCA + Ridge on 768-d Features ---")
for n_comp in [2, 3, 4, 5, 6, 8, 10, 12, 15]:
    for alpha in [0.1, 1.0, 10.0]:
        oof_encoded = np.zeros_like(Y_train_encoded)
        for i in range(N):
            train_idx = [j for j in range(N) if j != i]
            val_idx = [i]
            
            scaler = StandardScaler()
            pca = PCA(n_components=n_comp)
            
            X_tr = pca.fit_transform(scaler.fit_transform(X_full[train_idx]))
            X_val = pca.transform(scaler.transform(X_full[val_idx]))
            
            model = Ridge(alpha=alpha)
            model.fit(X_tr, Y_train_encoded[train_idx])
            oof_encoded[val_idx] = model.predict(X_val)
            
        oof_cdf = decode_sqrt_mass(oof_encoded)
        score = emd_score(targets, oof_cdf)
        if score < 36.0:
            print(f"PCA({n_comp:02d}) + Ridge(alpha={alpha:<4}): LOOCV EMD = {score:.4f}")

# 3. Test PLSRegression on Sqrt-Mass Targets
print("\n--- 3. PLS Regression on Sqrt-Mass Targets ---")
for n_comp in [1, 2, 3, 4, 5, 6]:
    oof_encoded = np.zeros_like(Y_train_encoded)
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        val_idx = [i]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_full[train_idx])
        X_val = scaler.transform(X_full[val_idx])
        
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_tr, Y_train_encoded[train_idx])
        oof_encoded[val_idx] = pls.predict(X_val)
        
    oof_cdf = decode_sqrt_mass(oof_encoded)
    score = emd_score(targets, oof_cdf)
    print(f"PLSRegression(n_components={n_comp}): LOOCV EMD = {score:.4f}")
