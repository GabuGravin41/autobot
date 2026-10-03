# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')



import os, re, glob
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.linear_model import Ridge, ElasticNet, MultiTaskElasticNet, MultiTaskLasso, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from collections import Counter

import torch
import torchvision.transforms as T
from PIL import Image

DATA_DIR = "/kaggle/input/competitions/soil-grain-size-from-photos"
TRAIN_IMG_DIR = f"{DATA_DIR}/Training-All_Photos_updated/Training-All_Photos_updated"
TEST_IMG_DIR = f"{DATA_DIR}/Test_All_Photos/Test_All_Photos"

SUPPORT_MM = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
SUPPORT_COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
LOG_X = np.log10(SUPPORT_MM)
LOG_WEIGHTS = np.diff(LOG_X)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



def emd_log_score(y_true, y_pred):
    y_true = np.atleast_2d(y_true); y_pred = np.atleast_2d(y_pred)
    abs_diff = np.abs(y_true[:, :-1] - y_pred[:, :-1])
    return (abs_diff * LOG_WEIGHTS).sum(axis=1).mean()

def enforce_monotonic_and_bounds(pred_row):
    p = np.clip(np.asarray(pred_row, dtype=float), 0, 100)
    p = np.maximum.accumulate(p); p[-1] = 100.0; p = np.minimum(p, 100.0)
    return np.maximum.accumulate(p)

class CumulativeTransform:
    name = 'cumulative'
    def encode(self, Y): return Y.copy()
    def decode(self, Yt): return np.array([enforce_monotonic_and_bounds(r) for r in Yt])

class DeltaTransform:
    name = 'delta'
    def encode(self, Y): return np.diff(Y, axis=1, prepend=0)
    def decode(self, Dt):
        D = np.clip(Dt, 0, None)
        s = D.sum(axis=1, keepdims=True); s = np.where(s <= 1e-9, 1.0, s)
        F = np.cumsum(D / s * 100.0, axis=1)
        return np.array([enforce_monotonic_and_bounds(r) for r in F])

TRANSFORMS = [CumulativeTransform(), DeltaTransform()]

def parse_train_filename(path):
    base = os.path.basename(path)
    m = re.search(r'([A-Za-z]\d{3})_(\d+)\.(jpg|jpeg)$', base, flags=re.IGNORECASE)
    if not m: return None, None
    return base[:m.start()].rstrip('_'), m.group(1)

def normalize_sample_id(s):
    s = s.replace('Ü','Ue').replace('ü','ue').replace('Ä','Ae').replace('ä','ae')
    s = s.replace('Ö','Oe').replace('ö','oe').replace('ß','ss').replace(',', '_')
    return s.strip()

def parse_test_filename(path):
    base = os.path.basename(path)
    n = re.sub(r'\.(jpg|jpeg)$', '', base, flags=re.IGNORECASE)
    n = re.sub(r'\s*\(\d+\)\s*$', '', n)
    idx = n.find('HPC')
    if idx == -1: return 'unknown', normalize_sample_id(n)
    return n[:idx].rstrip('_').rstrip(), normalize_sample_id(n[idx:])

def _canon(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

def load_ppm_table():
    df = pd.read_csv(f"{DATA_DIR}/ppm_updated.csv")
    print("ppm_updated.csv columns:", list(df.columns)); print(df.to_string())
    return df

def pick_ppm_columns(ppm_df):
    cand_match = [c for c in ppm_df.columns if any(k in c.lower() for k in ('cam','phone','device','model'))]
    match_col = cand_match[0] if cand_match else ppm_df.columns[0]
    cand_ppm = [c for c in ppm_df.columns if 'ppm' in c.lower() or 'pixel' in c.lower()]
    ppm_col = cand_ppm[0] if cand_ppm else ppm_df.columns[1]
    print(f"Guessed columns -> camera: '{match_col}', ppm: '{ppm_col}'")
    return match_col, ppm_col

def lookup_ppm(camera_name, ppm_df, match_col, ppm_col):
    norm_name = _canon(camera_name)
    for _, row in ppm_df.iterrows():
        row_name = _canon(row[match_col])
        if row_name and (row_name in norm_name or norm_name in row_name):
            return float(row[ppm_col])
    return float(ppm_df[ppm_col].median())

def fft_dominant_scale_mm(gray_eq, ppm):
    f = np.fft.fft2(gray_eq.astype(np.float32)); fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2
    h, w = gray_eq.shape; cy, cx = h // 2, w // 2
    y, x = np.indices((h, w)); r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int)
    radial = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    freqs = np.arange(len(radial)); valid = freqs > 3
    if valid.sum() == 0: return 0.0
    idx = freqs[valid][np.argmax(radial[valid])]
    if idx == 0: return 0.0
    return float((max(h, w) / idx) / ppm)

def extract_texture_features(gray_raw, gray_eq, ppm):
    feats = {}
    feats['lap_var'] = float(cv2.Laplacian(gray_eq, cv2.CV_64F).var())
    img = gray_eq.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    feats['grad_mean'] = float(gmag.mean()); feats['grad_std'] = float(gmag.std())
    feats['grad_p90'] = float(np.percentile(gmag, 90))
    radius = max(1, int(round(ppm * 0.5)))
    lbp = local_binary_pattern(gray_eq, 8, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for i, h in enumerate(hist): feats[f'lbp_{i}'] = float(h)
    step = max(1, int(round(ppm * 1.0)))
    glcm = graycomatrix(gray_eq, distances=[step], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
    feats['glcm_contrast'] = float(graycoprops(glcm, 'contrast').mean())
    feats['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity').mean())
    feats['glcm_energy'] = float(graycoprops(glcm, 'energy').mean())
    feats['glcm_correlation'] = float(graycoprops(glcm, 'correlation').mean())
    for mm in [0.2, 0.63, 2.0, 6.3, 20.0]:
        ksize = max(1, int(round(ppm * mm))); ksize = ksize+1 if ksize % 2 == 0 else ksize
        max_k = min(gray_eq.shape); max_k = max_k-1 if max_k % 2 == 0 else max_k
        ksize = max(1, min(ksize, max_k))
        blurred = cv2.GaussianBlur(gray_eq, (ksize, ksize), 0)
        edges = cv2.Canny(blurred, 50, 150)
        feats[f'edge_density_{mm}mm'] = float(edges.mean() / 255.0)
    _, thresh = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) / (ppm**2) for c in contours if cv2.contourArea(c) > 0]
    if areas:
        diam = 2 * np.sqrt(np.array(areas) / np.pi)
        feats['blob_diam_mean_mm'] = float(diam.mean()); feats['blob_diam_median_mm'] = float(np.median(diam))
        feats['blob_diam_p90_mm'] = float(np.percentile(diam, 90))
        feats['blob_count_per_mm2'] = float(len(diam) / (gray_eq.shape[0]*gray_eq.shape[1] / ppm**2))
    else:
        feats['blob_diam_mean_mm'] = feats['blob_diam_median_mm'] = 0.0
        feats['blob_diam_p90_mm'] = feats['blob_count_per_mm2'] = 0.0
    feats['fft_dominant_scale_mm'] = fft_dominant_scale_mm(gray_eq, ppm)
    feats['intensity_mean'] = float(gray_raw.mean()); feats['intensity_std'] = float(gray_raw.std())
    return feats

def load_and_preprocess(path, target_long_side=1024):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    h, w = img.shape[:2]; scale = target_long_side / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_raw)
    return gray_raw, gray_eq, scale

# ---------------------------------------------------------------------
# NEW: DINOv2 embedding extraction
# ---------------------------------------------------------------------

_DINO_MODEL = None
_DINO_TRANSFORM = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dino_model():
    global _DINO_MODEL
    if _DINO_MODEL is None:
        # skip_validation=True works around a repo-fork-check 403 I hit in
        # testing -- may also help on Kaggle's shared IPs against GitHub's API.
        _DINO_MODEL = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                      skip_validation=True, trust_repo=True)
        _DINO_MODEL.eval()
    return _DINO_MODEL

def extract_dino_embedding(path, model):
    img = Image.open(path).convert('RGB')
    x = _DINO_TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(x)
    return emb.squeeze(0).numpy()  # (384,)

def build_dino_feature_df(paths_and_ids, model):
    """paths_and_ids: list of (path, sample_id) tuples."""
    rows = []
    for path, sample_id in paths_and_ids:
        try:
            emb = extract_dino_embedding(path, model)
        except Exception as e:
            print(f"WARNING: DINOv2 failed on {os.path.basename(path)}: {e}")
            continue
        row = {f'dino_{i}': v for i, v in enumerate(emb)}
        row['sample_id'] = sample_id
        rows.append(row)
    return pd.DataFrame(rows)


def list_photos(img_dir, is_train):
    paths = sorted(set(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.JPG") +
                        glob.glob(f"{img_dir}/*.jpeg") + glob.glob(f"{img_dir}/*.JPEG")))
    out = []
    for path in paths:
        if is_train:
            camera, sample_id = parse_train_filename(path)
            if sample_id is None:
                print(f"WARNING: could not parse TRAIN filename {os.path.basename(path)}, skipping")
                continue
        else:
            camera, sample_id = parse_test_filename(path)
        out.append((path, sample_id, camera))
    return out

def build_handcrafted_feature_df(photo_list, ppm_df, match_col, ppm_col):
    rows = []
    for path, sample_id, camera in photo_list:
        result = load_and_preprocess(path)
        if result is None:
            print(f"WARNING: could not read image {os.path.basename(path)}, skipping")
            continue
        gray_raw, gray_eq, resize_scale = result
        eff_ppm = lookup_ppm(camera, ppm_df, match_col, ppm_col) * resize_scale
        feats = extract_texture_features(gray_raw, gray_eq, eff_ppm)
        feats['sample_id'] = sample_id
        rows.append(feats)
    return pd.DataFrame(rows)



class YScaledModel:
    def __init__(self, base_model_fn, transform):
        self.base_model_fn = base_model_fn; self.transform = transform
    def fit(self, X, Y_raw):
        Yt = self.transform.encode(Y_raw)
        self.y_scaler = StandardScaler(); Yt_s = self.y_scaler.fit_transform(Yt)
        self.model = self.base_model_fn(); self.model.fit(X, Yt_s)
        return self
    def predict(self, X):
        pred_s = np.asarray(self.model.predict(X))
        if pred_s.ndim == 1: pred_s = pred_s.reshape(1, -1)
        return self.transform.decode(self.y_scaler.inverse_transform(pred_s))

def build_candidates():
    return {
        'ridge_a15': lambda: Ridge(alpha=15.0),
        'ridge_a30': lambda: Ridge(alpha=30.0),
        'ridge_a50': lambda: Ridge(alpha=50.0),
        'enet_a0.1_l05': lambda: MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000, tol=1e-3)),
        'enet_a0.15_l04': lambda: MultiOutputRegressor(ElasticNet(alpha=0.15, l1_ratio=0.4, max_iter=50000, tol=1e-3)),
        'pls_2': lambda: PLSRegression(n_components=2),
        'pls_3': lambda: PLSRegression(n_components=3),
        'mtenet_a0.05': lambda: MultiTaskElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=50000, tol=1e-3),
        'mtlasso_a0.05': lambda: MultiTaskLasso(alpha=0.05, max_iter=50000, tol=1e-3),
        'bayesridge': lambda: MultiOutputRegressor(BayesianRidge()),
    }

def fit_predict_one(X_tr, Y_tr, X_te, model_fn, transform):
    """X_te must be exactly ONE row (used inside LOOCV/nested-CV loops)."""
    m = YScaledModel(model_fn, transform); m.fit(X_tr, Y_tr)
    return m.predict(X_te)[0]

def fit_predict_all(X_tr, Y_tr, X_te, model_fn, transform):
    """X_te can be any number of rows (used for the final test-set prediction)."""
    m = YScaledModel(model_fn, transform); m.fit(X_tr, Y_tr)
    return m.predict(X_te)

def loocv_score_combo(X, Y, model_fn, transform):
    loo = LeaveOneOut(); preds = np.zeros_like(Y)
    for tr, te in loo.split(X):
        preds[te[0]] = fit_predict_one(X[tr], Y[tr], X[te], model_fn, transform)
    return emd_log_score(Y, preds), preds

def best_single_via_loocv(X, Y, combos):
    scores = {k: loocv_score_combo(X, Y, mf, tf)[0] for k, (mf, tf) in combos.items()}
    best = min(scores, key=scores.get)
    return best, scores[best], scores

def nested_cv_single_best(X, Y, combos):
    n = X.shape[0]; outer = np.zeros_like(Y); chosen = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_in, Y_in = X[mask], Y[mask]
        inner = {k: loocv_score_combo(X_in, Y_in, mf, tf)[0] for k, (mf, tf) in combos.items()}
        bk = min(inner, key=inner.get); chosen.append(bk)
        mf, tf = combos[bk]
        outer[i] = fit_predict_one(X_in, Y_in, X[i:i+1], mf, tf)
    return emd_log_score(Y, outer), chosen



def main():
    print("="*70, "\nLoading PPM table...\n", "="*70, sep="")
    ppm_df = load_ppm_table()
    match_col, ppm_col = pick_ppm_columns(ppm_df)

    train_photos = list_photos(TRAIN_IMG_DIR, is_train=True)
    test_photos = list_photos(TEST_IMG_DIR, is_train=False)
    print(f"\nTrain photos: {len(train_photos)}, test photos: {len(test_photos)}")

    print("\n" + "="*70, "\nHand-crafted features (proven, from v4)...\n", "="*70, sep="")
    train_hc = build_handcrafted_feature_df(train_photos, ppm_df, match_col, ppm_col)
    test_hc = build_handcrafted_feature_df(test_photos, ppm_df, match_col, ppm_col)

    print("\n" + "="*70, "\nDINOv2 embeddings (downloads pretrained weights -- needs internet "
          "ON for this notebook)...\n", "="*70, sep="")
    dino_model = get_dino_model()
    train_dino = build_dino_feature_df([(p, sid) for p, sid, _ in train_photos], dino_model)
    test_dino = build_dino_feature_df([(p, sid) for p, sid, _ in test_photos], dino_model)
    print(f"DINOv2 train rows: {len(train_dino)}, test rows: {len(test_dino)}")

    labels_df = pd.read_csv(f"{DATA_DIR}/Training_labels_updated.csv")
    sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

    def aggregate(df, cols):
        agg_mean = df.groupby('sample_id')[cols].mean().add_suffix('_mean')
        agg_std = df.groupby('sample_id')[cols].std().add_suffix('_std').fillna(0.0)
        return pd.concat([agg_mean, agg_std], axis=1)

    hc_cols = [c for c in train_hc.columns if c != 'sample_id']
    dino_cols = [c for c in train_dino.columns if c != 'sample_id']

    train_hc_agg = aggregate(train_hc, hc_cols).reset_index()
    test_hc_agg = aggregate(test_hc, hc_cols).reset_index()
    # DINOv2: mean only (std would double 384 dims -- too extreme for n=24)
    train_dino_agg = train_dino.groupby('sample_id')[dino_cols].mean().reset_index()
    test_dino_agg = test_dino.groupby('sample_id')[dino_cols].mean().reset_index()

    train_merged = train_hc_agg.merge(train_dino_agg, on='sample_id', suffixes=('', '_dino')) \
                                .merge(labels_df, on='sample_id', how='inner')
    test_merged = test_hc_agg.merge(test_dino_agg, on='sample_id', suffixes=('', '_dino'))
    print(f"\nMerged train samples: {len(train_merged)} (should be 24)")

    hc_feat_cols = [c for c in train_hc_agg.columns if c != 'sample_id']
    dino_feat_cols = [c for c in train_dino_agg.columns if c != 'sample_id']

    Y_train = train_merged[SUPPORT_COLS].values.astype(float)
    X_test_ids = test_merged['sample_id'].values

    feature_sets = {
        'handcrafted_only': hc_feat_cols,
        'dino_only': dino_feat_cols,
        'combined': hc_feat_cols + dino_feat_cols,
    }

    candidates = build_candidates()
    print(f"\n{'='*70}\nComparing 3 feature sets, each with the SAME {len(candidates)*2}-combo "
          f"grid from v4 (not re-expanding the search)\n{'='*70}")

    results = {}
    for fs_name, cols in feature_sets.items():
        X_train = train_merged[cols].values
        X_test = test_merged[cols].values
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        combos = {f"{name}__{t.name}": (fn, t) for name, fn in candidates.items() for t in TRANSFORMS}
        best_key, best_score, _ = best_single_via_loocv(X_train_s, Y_train, combos)
        results[fs_name] = dict(X_train_s=X_train_s, X_test_s=X_test_s, combos=combos,
                                 best_key=best_key, best_score=best_score)
        print(f"  {fs_name:18s}: best={best_key:28s} LOOCV EMD={best_score:.4f}")

    winner_name = min(results, key=lambda k: results[k]['best_score'])
    winner = results[winner_name]
    print(f"\n=> Winning feature set: {winner_name} (LOOCV EMD = {winner['best_score']:.4f})")

    print(f"\n{'='*70}\nNested CV honesty check on the winner (~60-90s)...\n{'='*70}")
    nested_score, chosen = nested_cv_single_best(winner['X_train_s'], Y_train, winner['combos'])
    gap = nested_score - winner['best_score']
    print(f"Naive LOOCV: {winner['best_score']:.4f}   Nested honest: {nested_score:.4f}   Gap: {gap:+.4f}")
    print(f"{'Small gap -- trustworthy.' if gap < 3 else 'Notable gap -- treat this number with caution.'}")
    print(f"Selection stability across folds: {Counter(chosen).most_common(3)}")

    if winner_name == 'handcrafted_only' and results['handcrafted_only']['best_score'] > 42:
        print("\nNote: if this doesn't beat your existing v2/v4 hand-crafted LOOCV number, "
              "keep using that submission -- this script re-derives hand-crafted features "
              "independently and minor numeric differences vs, e.g., v4's run are expected.")

    mf, tf = winner['combos'][winner['best_key']]
    final_pred = fit_predict_all(winner['X_train_s'], Y_train, winner['X_test_s'], mf, tf)
    final_pred = np.asarray(final_pred).reshape(len(X_test_ids), 11)
    final_pred = np.array([enforce_monotonic_and_bounds(r) for r in final_pred])

    submission = pd.DataFrame(final_pred, columns=SUPPORT_COLS)
    submission.insert(0, 'sample_id', X_test_ids)
    submission = sample_sub[['sample_id']].merge(submission, on='sample_id', how='left')

    missing = submission[SUPPORT_COLS].isna().any(axis=1)
    if missing.any():
        print(f"\n!!! {missing.sum()} test rows missing -- filling with training-mean fallback")
        fallback = enforce_monotonic_and_bounds(Y_train.mean(axis=0))
        for col, val in zip(SUPPORT_COLS, fallback):
            submission.loc[missing, col] = val

    assert submission[SUPPORT_COLS].isna().sum().sum() == 0
    assert (submission[SUPPORT_COLS].diff(axis=1).iloc[:, 1:] >= -1e-9).all().all()
    assert np.allclose(submission['200'], 100.0)

    submission.to_csv('submission.csv', index=False)
    print(f"\n{'='*70}\nSaved submission.csv using '{winner_name}' / {winner['best_key']} "
          f"(naive={winner['best_score']:.2f}, nested={nested_score:.2f})\n{'='*70}")
    print(submission.head(10))

if __name__ == "__main__":
    main()

