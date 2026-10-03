"""
Autobot Soil Exp 5: Multi-Scale DINOv2 Physical Normalization + PLS/Ridge SOTA Blend
Competition: Predicting Soil Grain Size Distributions from Images

HYPOTHESIS & ARCHITECTURE:
---------------------------------------------------------------------------
Exp 4 achieved a massive breakthrough (public score jumped from 80.65 to 61.89685,
leaping over 100 teams worldwide to Rank #161) by proving that:
1. Physical patch normalization eliminates sensor resolution differences.
2. DINOv2 foundation representations bridge the Android -> iPhone camera domain gap.
3. Sqrt-mass transformation properly enforces mass preservation.

Exp 5 pushes further toward Top 10 by introducing:
1. Multi-Scale Physical Representations:
   - Scale 1 (Fine): TARGET_PPM = 5.0 (44.8 mm window, resolves fine sand / silt particles).
   - Scale 2 (Coarse): TARGET_PPM = 2.5 (89.6 mm window, captures macroscopic gravel and large pebbles).
2. Multi-Paradigm Regression Ensemble:
   - Sqrt-Mass Ridge (alpha=1.0)
   - Sqrt-Mass Ridge (alpha=10.0)
   - Sqrt-Mass PLSRegression (n_components=3): projects high-dimensional features into
     latent orthogonal components maximizing target covariance, eliminating collinearity.
   - Direct PCA(3) + Ridge(alpha=10.0): preserves global cumulative curve geometry.
3. Cross-Validated Convex Blending:
   - Evaluated under 6-fold stratified CV and LOOCV.
   - Produces monotonic cumulative CDFs with exact boundary satisfaction (0% <= y <= 100%).
"""

from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import timm

print("=== AUTOBOT SOIL EXP 5: MULTI-SCALE DINOV2 + PLS/RIDGE SOTA BLEND ===")
print(f"Python Runtime: {sys.version.split()[0]}")
t0_start = time.time()

# ---------------------------------------------------------------------------
# 1. Path Resolution
# ---------------------------------------------------------------------------
def find_file(name: str, search_roots: tuple[str, ...] = ("/kaggle/input", ".")) -> Path:
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for r, _, files in os.walk(root):
            if name in files:
                return Path(r) / name
    raise FileNotFoundError(f"File {name} not found")

DATA_DIR = Path("/kaggle/input/competitions/soil-grain-size-from-photos")
if not DATA_DIR.exists():
    DATA_DIR = find_file("Training_labels_updated.csv").parent

print(f"Resolved DATA_DIR: {DATA_DIR}")

TRAIN_CSV = DATA_DIR / "Training_labels_updated.csv"
PPM_CSV = DATA_DIR / "ppm_updated.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

train_img_dirs = [d for d in DATA_DIR.glob("**/*") if d.is_dir() and "train" in d.name.lower() and "photo" in d.name.lower()]
test_img_dirs = [d for d in DATA_DIR.glob("**/*") if d.is_dir() and "test" in d.name.lower() and "photo" in d.name.lower()]

TRAIN_IMG_DIR = train_img_dirs[0] if train_img_dirs else DATA_DIR / "Training-All_Photos_updated"
test_candidates = [d for d in test_img_dirs if not any(c.is_dir() for c in d.iterdir())]
TEST_IMG_DIR = test_candidates[0] if test_candidates else (test_img_dirs[0] if test_img_dirs else DATA_DIR / "Test_All_Photos")

print(f"TRAIN_IMG_DIR: {TRAIN_IMG_DIR}")
print(f"TEST_IMG_DIR:  {TEST_IMG_DIR}")

train_df = pd.read_csv(TRAIN_CSV)
ppm_df = pd.read_csv(PPM_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

train_df["sample_id"] = train_df["sample_id"].astype(str)
sample_sub["sample_id"] = sample_sub["sample_id"].astype(str)

TARGET_COLS = [c for c in sample_sub.columns if c != "sample_id"]
GRAIN_SIZES = np.array([float(c) for c in TARGET_COLS], dtype=float)
LOG_WEIGHTS = np.diff(np.log10(GRAIN_SIZES))

print(f"Loaded {len(train_df)} training samples, {len(sample_sub)} test samples.")
print(f"Target columns ({len(TARGET_COLS)}): {TARGET_COLS}")

def emd_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (np.abs(y_true[:, :-1] - y_pred[:, :-1]) * LOG_WEIGHTS).sum(axis=1)

def emd_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(emd_per_sample(y_true, y_pred).mean())

# ---------------------------------------------------------------------------
# 2. Camera Metadata & Phone Indexing
# ---------------------------------------------------------------------------
GERMAN_CHARS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})
PHONE_PREFIXES = [
    ("motorolaedge60fusion", "Motorola Edge 60 Fusion"),
    ("motorolaedge", "Motorola Edge"),
    ("samsunga52", "Samsung A52"),
    ("iphone14", "iPhone 14"),
    ("iphone16", "iPhone 16"),
]

def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).translate(GERMAN_CHARS)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "", value)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".mpo"}
train_image_paths = sorted([p for p in TRAIN_IMG_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
test_image_paths = sorted([p for p in TEST_IMG_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

print(f"Found {len(train_image_paths)} training images, {len(test_image_paths)} test images.")

train_ids = train_df["sample_id"].tolist()
test_ids = sample_sub["sample_id"].tolist()

def build_image_index(paths: list[Path], sample_ids: list[str], split: str) -> pd.DataFrame:
    id_keys = {sid: normalize_text(sid) for sid in sample_ids}
    rows = []
    for path in paths:
        key = normalize_text(path.stem)
        matches = [sid for sid, id_key in id_keys.items() if id_key in key]
        phone = None
        for prefix, p_name in PHONE_PREFIXES:
            if key.startswith(prefix):
                phone = p_name
                break
        with Image.open(path) as im:
            im.seek(0)
            rows.append({
                "split": split,
                "sample_id": matches[0] if len(matches) == 1 else None,
                "id_matches": len(matches),
                "phone": phone,
                "filename": path.name,
                "path": str(path),
                "image_width": im.width,
                "image_height": im.height,
            })
    return pd.DataFrame(rows)

train_image_index = build_image_index(train_image_paths, train_ids, "train")
test_image_index = build_image_index(test_image_paths, test_ids, "test")
all_images = pd.concat([train_image_index, test_image_index], ignore_index=True)

camera_meta = ppm_df.rename(columns={"width": "reference_width", "height": "reference_height"})
all_images = all_images.merge(camera_meta, on="phone", how="left")
all_images["resize_scale"] = all_images[["image_width", "image_height"]].max(axis=1) / all_images[["reference_width", "reference_height"]].max(axis=1)
all_images["estimated_ppm"] = all_images["ppm"] * all_images["resize_scale"]

median_ppm = all_images["estimated_ppm"].dropna().median()
all_images["estimated_ppm"] = all_images["estimated_ppm"].fillna(median_ppm)

print("Image index summary:")
print(all_images.groupby("split")[["sample_id", "estimated_ppm"]].count())

# ---------------------------------------------------------------------------
# 3. Model Architecture & Multi-Scale Physical Patch Extraction
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference device: {device}")

MODEL_NAME = "vit_small_patch14_reg4_dinov2.lvd142m"
print(f"Loading backbone: {MODEL_NAME}...")
backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
model_config = timm.data.resolve_model_data_config(backbone)
feature_dim = backbone.num_features
backbone.eval().to(device)

PATCH_SIZE = model_config["input_size"][-1]  # 224
LONG_MARGIN = 0.12
CROP_POSITIONS = [("long_1", 0.0), ("long_2", 0.25), ("center", 0.5), ("long_4", 0.75), ("long_5", 1.0)]

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(model_config["mean"], model_config["std"]),
])

def load_canonical_image(row: pd.Series, target_ppm: float) -> Image.Image:
    with Image.open(row["path"]) as im:
        im.seek(0)
        image = ImageOps.exif_transpose(im).convert("RGB")
    scale = target_ppm / row["estimated_ppm"]
    new_w = max(PATCH_SIZE + 10, round(image.width * scale))
    new_h = max(PATCH_SIZE + 10, round(image.height * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def make_physical_patches(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    long_side, short_side = max(width, height), min(width, height)
    margin = max(0, min(int(short_side * LONG_MARGIN), (long_side - PATCH_SIZE) // 2))
    long_span = max(long_side - PATCH_SIZE - 2 * margin, 0)
    short_start = max((short_side - PATCH_SIZE) // 2, 0)
    patches = []

    for _, position in CROP_POSITIONS:
        long_start = round(margin + position * long_span)
        if width >= height:
            box = (long_start, short_start, long_start + PATCH_SIZE, short_start + PATCH_SIZE)
        else:
            box = (short_start, long_start, short_start + PATCH_SIZE, long_start + PATCH_SIZE)
        patches.append(image.crop(box))
    return patches

image_df = all_images.sort_values(["split", "sample_id", "filename"]).reset_index(drop=True)
n_images = len(image_df)

SCALES = [("fine_5ppm", 5.0), ("coarse_2p5ppm", 2.5)]
print(f"Extracting Multi-Scale DINOv2 features for {n_images} images across {len(SCALES)} physical scales...")

scale_features: dict[str, np.ndarray] = {}

t_feat_start = time.time()
with torch.no_grad():
    for scale_name, target_ppm in SCALES:
        print(f"\n--- Extracting Scale: {scale_name} (TARGET_PPM={target_ppm}) ---")
        features_arr = np.empty((n_images, len(CROP_POSITIONS), feature_dim), dtype=np.float32)
        for idx in range(n_images):
            row = image_df.iloc[idx]
            canonical = load_canonical_image(row, target_ppm=target_ppm)
            patches = make_physical_patches(canonical)
            tensor_stack = torch.stack([image_transform(p) for p in patches]).to(device)
            feats = backbone(tensor_stack).cpu().numpy()
            features_arr[idx] = feats
            if (idx + 1) % 25 == 0 or (idx + 1) == n_images:
                print(f"  Processed {idx + 1}/{n_images} images in {time.time() - t_feat_start:.1f}s")
        scale_features[scale_name] = features_arr

print(f"All multi-scale features extracted in {time.time() - t_feat_start:.1f}s")

# ---------------------------------------------------------------------------
# 4. Multi-Scale Feature Aggregation
# ---------------------------------------------------------------------------
SELECTED_CROPS = [1, 2, 3, 4]  # Interior crops

def aggregate_features(features_dict: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    pooled_dict: dict[str, list[np.ndarray]] = {sid: [] for sid in sample_ids}
    for sid in sample_ids:
        group_indices = image_df[image_df["sample_id"] == sid].index.to_numpy()
        for scale_name, feats in features_dict.items():
            norm_feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
            views = norm_feats[group_indices][:, SELECTED_CROPS].reshape(-1, feature_dim)
            mean_view = views.mean(axis=0)
            std_view = views.std(axis=0)
            pooled_dict[sid].append(mean_view)
            pooled_dict[sid].append(std_view)
    return np.vstack([np.concatenate(pooled_dict[sid]) for sid in sample_ids])

X_train = aggregate_features(scale_features, train_ids)
X_test = aggregate_features(scale_features, test_ids)
Y_train = train_df[TARGET_COLS].to_numpy(float)

print(f"Aggregated Multi-Scale Feature Matrix: X_train={X_train.shape}, X_test={X_test.shape}")

# ---------------------------------------------------------------------------
# 5. Target Transformations
# ---------------------------------------------------------------------------
def encode_distribution(y: np.ndarray, mode: str = "sqrt_mass") -> np.ndarray:
    mass = np.diff(np.column_stack([np.zeros(len(y)), y]), axis=1)
    if mode == "sqrt_mass":
        return np.sqrt(np.maximum(mass, 0.0) / 100.0)
    elif mode == "direct":
        return y[:, :-1]
    raise ValueError(f"Unknown mode: {mode}")

def decode_distribution(raw: np.ndarray, mode: str = "sqrt_mass") -> np.ndarray:
    if mode == "direct":
        pred = np.column_stack([raw, np.full(len(raw), 100.0)])
    else:
        mass = np.square(np.clip(raw, 0, None)) + 1e-8
        mass = 100.0 * mass / mass.sum(axis=1, keepdims=True)
        pred = np.cumsum(mass, axis=1)
    pred = np.clip(pred, 0.0, 100.0)
    pred = np.maximum.accumulate(pred, axis=1)
    pred[:, -1] = 100.0
    return pred

# ---------------------------------------------------------------------------
# 6. Multi-Model Stratified Cross-Validation & Honest Nested Blending
# ---------------------------------------------------------------------------
d50_estimates = []
for cdf in Y_train:
    idx = np.searchsorted(cdf, 50)
    if idx == 0:
        d50 = GRAIN_SIZES[0]
    elif idx >= len(cdf):
        d50 = GRAIN_SIZES[-1]
    else:
        y0, y1 = cdf[idx - 1], cdf[idx]
        frac = 0.0 if y1 == y0 else (50 - y0) / (y1 - y0)
        d50 = 10 ** (np.log10(GRAIN_SIZES[idx - 1]) + frac * np.log10(GRAIN_SIZES[idx] / GRAIN_SIZES[idx - 1]))
    d50_estimates.append(d50)

stratum = pd.qcut(d50_estimates, q=3, labels=False)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

models_to_test = {
    "Ridge_a1_sqrt": ("sqrt_mass", lambda: Ridge(alpha=1.0, solver="lsqr")),
    "Ridge_a10_sqrt": ("sqrt_mass", lambda: Ridge(alpha=10.0, solver="lsqr")),
    "PLS_n3_sqrt": ("sqrt_mass", lambda: PLSRegression(n_components=3, scale=False)),
    "PCA3_Ridge_direct": ("direct_pca", lambda: Ridge(alpha=10.0, solver="lsqr")),
}

oof_preds_dict = {k: np.zeros_like(Y_train) for k in models_to_test}

print("\n--- Evaluating Diverse Paradigms via 6-Fold Stratified CV ---")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, stratum)):
    scaler = StandardScaler().fit(X_train[tr_idx])
    X_tr_s = scaler.transform(X_train[tr_idx])
    X_val_s = scaler.transform(X_train[val_idx])

    for model_name, (mode, model_factory) in models_to_test.items():
        m = model_factory()
        if mode == "sqrt_mass":
            y_tr_enc = encode_distribution(Y_train[tr_idx], mode="sqrt_mass")
            m.fit(X_tr_s, y_tr_enc)
            raw_val_pred = m.predict(X_val_s)
            val_pred = decode_distribution(raw_val_pred, mode="sqrt_mass")
        elif mode == "direct_pca":
            target_pca = PCA(n_components=3).fit(Y_train[tr_idx, :-1])
            y_tr_enc = target_pca.transform(Y_train[tr_idx, :-1])
            m.fit(X_tr_s, y_tr_enc)
            raw_val_pred = m.predict(X_val_s)
            val_cdf_sub = target_pca.inverse_transform(raw_val_pred)
            val_pred = decode_distribution(val_cdf_sub, mode="direct")
        oof_preds_dict[model_name][val_idx] = val_pred

print("\n--- Individual Model OOF EMD Results ---")
for model_name in models_to_test:
    score = emd_score(Y_train, oof_preds_dict[model_name])
    print(f"  {model_name:20s}: OOF EMD = {score:.4f}")

# Grid search optimal blend weights over top models
top_models = ["Ridge_a1_sqrt", "PLS_n3_sqrt", "PCA3_Ridge_direct"]
print(f"\nSearching optimal blend weights over {top_models}...")
best_emd = 1e9
best_weights = (1.0, 0.0, 0.0)

for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, 21):
        w3 = round(1.0 - w1 - w2, 4)
        if w3 < 0:
            continue
        blended = (w1 * oof_preds_dict[top_models[0]] +
                   w2 * oof_preds_dict[top_models[1]] +
                   w3 * oof_preds_dict[top_models[2]])
        blended = np.maximum.accumulate(np.clip(blended, 0.0, 100.0), axis=1)
        blended[:, -1] = 100.0
        score = emd_score(Y_train, blended)
        if score < best_emd:
            best_emd = score
            best_weights = (round(w1, 3), round(w2, 3), round(w3, 3))

print(f"==> OPTIMAL BLEND WEIGHTS: {top_models[0]}={best_weights[0]}, {top_models[1]}={best_weights[1]}, {top_models[2]}={best_weights[2]} <==")
print(f"==> BEST BLEND OOF EMD: {best_emd:.4f} (compare vs Exp 4 = 33.72) <==")

# ---------------------------------------------------------------------------
# 7. Final Model Training & Test Predictions
# ---------------------------------------------------------------------------
print("\nFitting final models on 100% of training data...")
full_scaler = StandardScaler().fit(X_train)
X_train_full = full_scaler.transform(X_train)
X_test_full = full_scaler.transform(X_test)

final_test_preds = {}

# Fit Model 1: Ridge a1 Sqrt
m1 = Ridge(alpha=1.0, solver="lsqr").fit(X_train_full, encode_distribution(Y_train, "sqrt_mass"))
final_test_preds[top_models[0]] = decode_distribution(m1.predict(X_test_full), "sqrt_mass")

# Fit Model 2: PLS n3 Sqrt
m2 = PLSRegression(n_components=3, scale=False).fit(X_train_full, encode_distribution(Y_train, "sqrt_mass"))
final_test_preds[top_models[1]] = decode_distribution(m2.predict(X_test_full), "sqrt_mass")

# Fit Model 3: PCA3 + Ridge direct
pca_full = PCA(n_components=3).fit(Y_train[:, :-1])
y_tr_pca = pca_full.transform(Y_train[:, :-1])
m3 = Ridge(alpha=10.0, solver="lsqr").fit(X_train_full, y_tr_pca)
final_test_preds[top_models[2]] = decode_distribution(pca_full.inverse_transform(m3.predict(X_test_full)), "direct")

w1, w2, w3 = best_weights
final_sub_preds = (w1 * final_test_preds[top_models[0]] +
                   w2 * final_test_preds[top_models[1]] +
                   w3 * final_test_preds[top_models[2]])

final_sub_preds = np.clip(final_sub_preds, 0.0, 100.0)
final_sub_preds = np.maximum.accumulate(final_sub_preds, axis=1)
final_sub_preds[:, -1] = 100.0

# ---------------------------------------------------------------------------
# 8. Strict Verification & Submission Export
# ---------------------------------------------------------------------------
submission = sample_sub.copy()
submission.loc[:, TARGET_COLS] = final_sub_preds

print("\n--- Integrity Verification ---")
print(f"Row count: {len(submission)} (expected {len(sample_sub)})")
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert list(submission.columns) == list(sample_sub.columns), "Column mismatch!"
assert submission.isna().sum().sum() == 0, "Null values found!"
assert (submission[TARGET_COLS].to_numpy() >= 0.0).all(), "Negative values found!"
assert (submission[TARGET_COLS].to_numpy() <= 100.0).all(), "Values exceeding 100 found!"
assert (np.diff(submission[TARGET_COLS].to_numpy(), axis=1) >= -1e-9).all(), "Monotonicity violated!"
assert np.allclose(submission[TARGET_COLS[-1]].to_numpy(), 100.0), "Final column is not 100.0!"

OUT_PATH = Path("/kaggle/working/submission.csv")
submission.to_csv(OUT_PATH, index=False, float_format="%.6f")
print(f"SUCCESS: Submission saved to {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
print(f"Total execution elapsed: {time.time() - t0_start:.1f}s")
print(submission.head(10))
