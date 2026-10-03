import os, cv2, random, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
from skimage.feature import local_binary_pattern
from skimage.morphology import disk, opening

warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATA_DIR = "/kaggle/input/competitions/soil-grain-size-from-photos"
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "Training_labels_updated.csv")
PPM_PATH = os.path.join(DATA_DIR, "ppm_updated.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "Training-All_Photos_updated")
TEST_IMG_DIR = os.path.join(DATA_DIR, "Test_All_Photos")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

def build_image_map(directory):
    img_map = {}
    if not os.path.exists(directory): return img_map
    for root, _, files in os.walk(directory):
        for file in files:
            name_idx = file.rfind('.')
            if name_idx != -1:
                sample_id = file[:name_idx].lower()
                img_map.setdefault(sample_id, []).append(os.path.join(root, file))
    return img_map

train_img_map = build_image_map(TRAIN_IMG_DIR)
test_img_map = build_image_map(TEST_IMG_DIR)

effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
extractor_eff = nn.Sequential(effnet.features, effnet.avgpool, nn.Flatten()).to(device).eval()

resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
extractor_res = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten()).to(device).eval()

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def normalize_string(s):
    s = str(s).lower().strip()
    s = s.replace('\u00fc', 'ue').replace('\u00f6', 'oe').replace('\u00e4', 'ae').replace('\u00df', 'ss')
    s = s.replace(',', '_').replace('-', '_').replace(' ', '_')
    while '__' in s: s = s.replace('__', '_')
    return s

def compute_granulometry(gray, radii=(1, 2, 3, 5, 8, 12, 18, 25)):
    total = float(gray.astype(np.float64).sum())
    if total == 0: return np.zeros(len(radii))
    feats = []
    for r in radii:
        opened = opening(gray, disk(r))
        feats.append(1.0 - float(opened.astype(np.float64).sum()) / total)
    return np.array(feats)

def compute_glcm(gray, distances=(1, 3, 5, 10, 20)):
    g = (gray // 4).astype(np.uint8)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    mat = graycomatrix(g, list(distances), angles, levels=64, symmetric=True, normed=True)
    feats = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        v = graycoprops(mat, prop)
        feats.extend([float(v.mean()), float(v.std())])
    return np.array(feats)

def compute_lbp(gray):
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    h, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)
    return h.astype(np.float64)

def compute_color(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    feats = []
    for ci in range(3):
        ch = hsv[:, :, ci].ravel()
        m, s = ch.mean(), ch.std()
        feats.append(m)
        feats.append(s)
        feats.append(0.0 if s < 1e-10 else float(np.mean(((ch - m) / s) ** 3)))
        feats.append(0.0 if s < 1e-10 else float(np.mean(((ch - m) / s) ** 4) - 3))
    return np.array(feats)

def compute_multiscale_edges(gray, sigmas=(1, 3, 5, 9, 15)):
    feats = []
    for s in sigmas:
        bl = cv2.GaussianBlur(gray, (0, 0), s)
        feats.append(float(np.mean(cv2.Canny(bl, 30, 100) > 0)))
    return np.array(feats)

# ═════════════════════════════════════════════════════════════════════════════
# EXTENSIVE TTA FEATURE EXTRACTION (Sample-Level Averaging)
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(df, img_map, ppm_lookup, global_mean_ppm, set_name="Dataset"):
    features_list = []
    TARGET_PPM = 15.0
    print(f"\nExtracting CNN + Texture features (Extensive TTA) for: {set_name}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["sample_id"])
        norm_sample_id = normalize_string(sample_id)

        matching_paths = []
        for k, paths in img_map.items():
            if (norm_sample_id in normalize_string(k) or 
                normalize_string(k) in norm_sample_id or 
                norm_sample_id.replace('hpc_', '') in normalize_string(k)):
                matching_paths.extend(paths)

        feat = {"sample_id": row["sample_id"]}

        if matching_paths:
            eff_embs, res_embs = [], []
            all_edges = []
            all_granulo, all_glcm, all_lbp_h, all_color, all_medge = [], [], [], [], []

            for img_path in matching_paths:
                img = cv2.imread(img_path)
                if img is None: continue

                h, w = img.shape[:2]
                current_ppm = ppm_lookup.get((w, h), global_mean_ppm)
                scale_factor = TARGET_PPM / current_ppm
                img_scaled = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
                sh, sw = img_scaled.shape[:2]

                if sh < 224 or sw < 224:
                    img_scaled = cv2.resize(img_scaled, (max(224, sw), max(224, sh)))
                    sh, sw = img_scaled.shape[:2]

                gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
                all_edges.append(np.mean(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100) > 0))

                # ── Texture features on center crop (max 512x512) ──
                csz = min(512, sh, sw)
                cy_t, cx_t = sh // 2, sw // 2
                cs = csz // 2
                y0, y1_t = max(0, cy_t - cs), min(sh, cy_t + cs)
                x0, x1_t = max(0, cx_t - cs), min(sw, cx_t + cs)
                
                all_granulo.append(compute_granulometry(gray[y0:y1_t, x0:x1_t]))
                all_glcm.append(compute_glcm(gray[y0:y1_t, x0:x1_t]))
                all_lbp_h.append(compute_lbp(gray[y0:y1_t, x0:x1_t]))
                all_color.append(compute_color(img_scaled[y0:y1_t, x0:x1_t]))
                all_medge.append(compute_multiscale_edges(gray[y0:y1_t, x0:x1_t]))

                # ── CNN patches (5 inner shifts + TTA Flips) ──
                cy, cx = sh // 2, sw // 2
                inner_shifts = [(cy, cx), (cy-45, cx), (cy+45, cx), (cy, cx-45), (cy, cx+45)]
                img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)

                for y_c, x_c in inner_shifts:
                    y1 = max(0, min(sh - 224, y_c - 112))
                    x1 = max(0, min(sw - 224, x_c - 112))
                    
                    patch = img_rgb[y1:y1+224, x1:x1+224]
                    
                    # Generate TTA variants: Original, H-Flip, V-Flip
                    patch_h = cv2.flip(patch, 1)
                    patch_v = cv2.flip(patch, 0)

                    for p_variant in (patch, patch_h, patch_v):
                        tensor_patch = tensor_transform(Image.fromarray(p_variant)).unsqueeze(0).to(device)

                        with torch.no_grad():
                            emb_eff = extractor_eff(tensor_patch).cpu().numpy().flatten()
                            emb_res = extractor_res(tensor_patch).cpu().numpy().flatten()
                        eff_embs.append(emb_eff)
                        res_embs.append(emb_res)

            if eff_embs:
                feat["edge_density"] = np.mean(all_edges)
                feat["ppm"] = current_ppm

                # Averaged CNN embeddings (15 patches per image instead of 5)
                avg_eff = np.mean(eff_embs, axis=0)
                avg_res = np.mean(res_embs, axis=0)
                for i, val in enumerate(avg_eff): feat[f"eff_{i}"] = val
                for i, val in enumerate(avg_res): feat[f"res_{i}"] = val

                if all_granulo:
                    for prefix, arr in [("gr", all_granulo), ("glcm", all_glcm), 
                                        ("lbp", all_lbp_h), ("clr", all_color), 
                                        ("med", all_medge)]:
                        avg = np.mean(arr, axis=0)
                        for i, v in enumerate(avg): feat[f"{prefix}_{i}"] = v
            else:
                feat["edge_density"] = 0.0
                feat["ppm"] = global_mean_ppm
        else:
            feat["edge_density"] = 0.0
            feat["ppm"] = global_mean_ppm

        features_list.append(feat)
    return pd.DataFrame(features_list)

df_train_labels = pd.read_csv(TRAIN_LABELS_PATH)
df_sub = pd.read_csv(SAMPLE_SUB_PATH)
df_ppm = pd.read_csv(PPM_PATH)

ppm_lookup = df_ppm.set_index(["width", "height"])["ppm"].to_dict()
global_mean_ppm = df_ppm["ppm"].mean()

X_train_df = extract_features(df_train_labels, train_img_map, ppm_lookup, global_mean_ppm, "Train")
X_test_df  = extract_features(df_sub,          test_img_map,  ppm_lookup, global_mean_ppm, "Test")


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, MultiTaskElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import median_absolute_error

target_cols = [col for col in df_train_labels.columns if col != "sample_id"]

common_cols = list(set(X_train_df.columns) & set(X_test_df.columns))
feature_cols = [col for col in common_cols if col != "sample_id"]

X = X_train_df[feature_cols].fillna(0)
Y = df_train_labels[target_cols].values
X_test = X_test_df[feature_cols].fillna(0)

SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
LOG_W = np.diff(np.log10(SUPPORT))

def emd_score(y_true, y_pred):
    return float(np.mean(np.sum(np.abs(y_true[:, :10] - y_pred[:, :10]) * LOG_W, axis=1)))

# 30-Seed High-Stability Blend
SEEDS = list(range(42, 72))  # Increased from 10 to 30 for max stability
ridge_alphas = np.logspace(-1, 5, 40)
N_PLS = min(10, len(X) - 2, len(feature_cols))

acc_oof_ridge = np.zeros(Y.shape)
acc_oof_enet  = np.zeros(Y.shape)
acc_oof_pls   = np.zeros(Y.shape)
acc_test_ridge = np.zeros((len(X_test), Y.shape[1]))
acc_test_enet  = np.zeros((len(X_test), Y.shape[1]))
acc_test_pls   = np.zeros((len(X_test), Y.shape[1]))

print(f"Training 30-Seed Tri-Engine Ensemble...")

for seed in SEEDS:
    best_alpha, best_score = None, float('inf')
    
    # Quick Alpha search
    for alpha in ridge_alphas:
        oof_temp = np.zeros(Y.shape)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(X, Y):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X.iloc[train_idx])
            X_va_s = sc.transform(X.iloc[val_idx])
            m = Ridge(alpha=alpha, random_state=seed)
            m.fit(X_tr_s, Y[train_idx])
            oof_temp[val_idx] = m.predict(X_va_s)
        score = median_absolute_error(Y, np.sort(np.clip(oof_temp, 0, 100), axis=1))
        if score < best_score:
            best_score, best_alpha = score, alpha

    oof_r = np.zeros(Y.shape)
    oof_e = np.zeros(Y.shape)
    oof_p = np.zeros(Y.shape)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for train_idx, val_idx in kf.split(X, Y):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X.iloc[train_idx])
        X_va_s = sc.transform(X.iloc[val_idx])

        ridge = Ridge(alpha=best_alpha, random_state=seed)
        ridge.fit(X_tr_s, Y[train_idx])
        oof_r[val_idx] = ridge.predict(X_va_s)

        enet = MultiTaskElasticNet(alpha=0.5, l1_ratio=0.1, max_iter=2000, random_state=seed)
        enet.fit(X_tr_s, Y[train_idx])
        oof_e[val_idx] = enet.predict(X_va_s)

        n_c = min(N_PLS, X_tr_s.shape[0] - 1)
        pls = PLSRegression(n_components=n_c, scale=False)
        pls.fit(X_tr_s, Y[train_idx])
        oof_p[val_idx] = pls.predict(X_va_s)

    acc_oof_ridge += oof_r / len(SEEDS)
    acc_oof_enet  += oof_e / len(SEEDS)
    acc_oof_pls   += oof_p / len(SEEDS)

    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    X_te_s = sc.transform(X_test)

    f_r = Ridge(alpha=best_alpha, random_state=seed); f_r.fit(X_s, Y)
    acc_test_ridge += f_r.predict(X_te_s) / len(SEEDS)

    f_e = MultiTaskElasticNet(alpha=0.5, l1_ratio=0.1, max_iter=2000, random_state=seed)
    f_e.fit(X_s, Y)
    acc_test_enet += f_e.predict(X_te_s) / len(SEEDS)

    n_c = min(N_PLS, X_s.shape[0] - 1)
    f_p = PLSRegression(n_components=n_c, scale=False); f_p.fit(X_s, Y)
    acc_test_pls += f_p.predict(X_te_s) / len(SEEDS)
    
    if seed % 10 == 0:
        blend_oof = np.sort(np.clip(0.5 * oof_r + 0.5 * oof_e, 0, 100), axis=1)
        print(f"  Seed {seed}  Ridge alpha={best_alpha:.2f}  MedAE={median_absolute_error(Y, blend_oof):.4f}")

def proc(p): return np.sort(np.clip(p, 0, 100), axis=1)

print(f"\n30-Seed OOF EMD (lower = better):")
print(f"  Ridge:       {emd_score(Y, proc(acc_oof_ridge)):.4f}")
print(f"  ElasticNet:  {emd_score(Y, proc(acc_oof_enet)):.4f}")
print(f"  PLS:         {emd_score(Y, proc(acc_oof_pls)):.4f}")

best_emd, best_w = 1e9, (0.5, 0.5, 0.0)
for wa in np.arange(0, 1.05, 0.05):
    for wb in np.arange(0, 1.05 - wa, 0.05):
        wc = round(1.0 - wa - wb, 2)
        blend = proc(wa * acc_oof_ridge + wb * acc_oof_enet + wc * acc_oof_pls)
        e = emd_score(Y, blend)
        if e < best_emd:
            best_emd, best_w = e, (round(wa, 2), round(wb, 2), round(wc, 2))

print(f"\n  Optimal:  Ridge={best_w[0]}  ENet={best_w[1]}  PLS={best_w[2]}")
print(f"  Blend EMD = {best_emd:.4f}")

wa, wb, wc = best_w
final = wa * acc_test_ridge + wb * acc_test_enet + wc * acc_test_pls
final = np.clip(final, 0.0, 100.0)
final = np.sort(final, axis=1)

submission = pd.DataFrame(final, columns=target_cols)
submission.insert(0, "sample_id", df_sub["sample_id"])
submission["200"] = 100.0
submission = submission[["sample_id"] + target_cols]
submission.to_csv("submission.csv", index=False)

print(f"\nsubmission.csv saved ({len(submission)} rows)")


