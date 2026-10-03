"""
Autobot Soil Exp 4: DINOv2 Physical-Patch Normalization + Sqrt-Mass Ridge SOTA
Competition: Predicting Soil Grain Size Distributions from Images

WHY THIS EXPERIMENT EXISTS
---------------------------------------------------------------------------
In exp2 and exp3, our submission scored 80.65 - 80.96 on the public leaderboard.
The diagnostic audit revealed 100% camera disjointness between training and test sets:
- Training photos were taken with Android phones (Samsung A52, Motorola Edge).
- Test photos were taken with Apple iPhones (iPhone 14, iPhone 16).
Unnormalized RGB/HSV texture features, uncalibrated edge magnitudes, and generic
ImageNet features completely failed across this sensor and ISP domain gap.

Nomannic's breakthrough in public SOTA solves this via two key mechanisms:
1. Physical patch normalization: Each photo is resized so that 1 pixel = 1/5 mm
   (TARGET_PPM = 5.0) using the exact camera PPM lookup. This guarantees that an
   extracted 224x224 patch covers exactly the same physical millimeter window (44.8 mm)
   regardless of camera resolution, aspect ratio, or sensor format.
2. DINOv2 foundation representations: Extracted with vit_small_patch14_reg4_dinov2.
   Patches are L2-normalized per patch to eliminate camera exposure, brightness,
   and color tone shifts across phone models.
3. Sqrt-Mass target transformation:
   Cumulative distributions are differenced into interval masses, square-root transformed,
   regressed with regularized Ridge (alpha=1.0), and then projected back to valid
   monotonic cumulative CDFs with exact 100% mass preservation.
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import timm

print("=== AUTOBOT SOIL EXP 4: DINOV2 PHYSICAL PATCH SQRT-MASS SOTA ===")
print(f"Python Runtime: {sys.version.split()[0]}")
t0_start = time.time()

# ---------------------------------------------------------------------------
# 1. Path Resolution
# ---------------------------------------------------------------------------
def find_dir(name: str, search_roots: tuple[str, ...] = ("/kaggle/input", ".")) -> Path:
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for r, dirs, _ in os.walk(root):
            if name in dirs:
                return Path(r) / name
    raise FileNotFoundError(f"Directory {name} not found")

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
    # Fallback search
    DATA_DIR = find_file("Training_labels_updated.csv").parent

print(f"Resolved DATA_DIR: {DATA_DIR}")

TRAIN_CSV = DATA_DIR / "Training_labels_updated.csv"
PPM_CSV = DATA_DIR / "ppm_updated.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

# Locate training and test photo directories
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

# Metric function
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
        # Match phone
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

# Impute any missing estimated_ppm with median
median_ppm = all_images["estimated_ppm"].dropna().median()
all_images["estimated_ppm"] = all_images["estimated_ppm"].fillna(median_ppm)

print("Image index summary:")
print(all_images.groupby("split")[["sample_id", "estimated_ppm"]].count())

# ---------------------------------------------------------------------------
# 3. Model Architecture & Physical Patch Extraction
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference device: {device}")

MODEL_NAME = "vit_small_patch14_reg4_dinov2.lvd142m"
print(f"Loading backbone: {MODEL_NAME}...")
backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
model_config = timm.data.resolve_model_data_config(backbone)
feature_dim = backbone.num_features
backbone.eval().to(device)

TARGET_PPM = 5.0
PATCH_SIZE = model_config["input_size"][-1]  # 224
LONG_MARGIN = 0.12
CROP_POSITIONS = [("long_1", 0.0), ("long_2", 0.25), ("center", 0.5), ("long_4", 0.75), ("long_5", 1.0)]

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(model_config["mean"], model_config["std"]),
])

def load_canonical_image(row: pd.Series) -> Image.Image:
    with Image.open(row["path"]) as im:
        im.seek(0)
        image = ImageOps.exif_transpose(im).convert("RGB")
    scale = TARGET_PPM / row["estimated_ppm"]
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

# Extract features image by image
image_df = all_images.sort_values(["split", "sample_id", "filename"]).reset_index(drop=True)
n_images = len(image_df)
print(f"Extracting DINOv2 features for {n_images} images (5 physical patches each)...")

physical_features = np.empty((n_images, len(CROP_POSITIONS), feature_dim), dtype=np.float32)

t_feat_start = time.time()
with torch.no_grad():
    for idx in range(n_images):
        row = image_df.iloc[idx]
        canonical = load_canonical_image(row)
        patches = make_physical_patches(canonical)
        tensor_stack = torch.stack([image_transform(p) for p in patches]).to(device)
        feats = backbone(tensor_stack).cpu().numpy()
        physical_features[idx] = feats
        if (idx + 1) % 25 == 0 or (idx + 1) == n_images:
            print(f"  Processed {idx + 1}/{n_images} images in {time.time() - t_feat_start:.1f}s")

print(f"Feature extraction complete in {time.time() - t_feat_start:.1f}s")

# ---------------------------------------------------------------------------
# 4. Feature Aggregation & Normalization
# ---------------------------------------------------------------------------
# L2 normalize each patch feature vector to cancel exposure/lighting differences
normalized_features = physical_features / np.linalg.norm(physical_features, axis=-1, keepdims=True)

# Select interior crops (indices 1, 2, 3, 4)
SELECTED_CROPS = [1, 2, 3, 4]

sample_feature_dict: dict[str, np.ndarray] = {}
for sid, group in image_df.groupby("sample_id", sort=False):
    pos = group.index.to_numpy()
    views = normalized_features[pos][:, SELECTED_CROPS].reshape(-1, feature_dim)
    mean_view = views.mean(axis=0)
    std_view = views.std(axis=0)
    sample_feature_dict[sid] = np.concatenate([mean_view, std_view])

X_train = np.vstack([sample_feature_dict[sid] for sid in train_ids])
X_test = np.vstack([sample_feature_dict[sid] for sid in test_ids])
Y_train = train_df[TARGET_COLS].to_numpy(float)

print(f"Aggregated feature shapes: X_train={X_train.shape}, X_test={X_test.shape}")

# ---------------------------------------------------------------------------
# 5. Target Transformation & Stratified Cross-Validation
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
    # Enforce monotonicity and boundary constraints
    pred = np.clip(pred, 0.0, 100.0)
    pred = np.maximum.accumulate(pred, axis=1)
    pred[:, -1] = 100.0
    return pred

# 6-Fold Stratified CV based on coarse/fine median grain size
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

print("\n--- Cross-Validation Evaluation (6 Folds) ---")
oof_preds = np.zeros_like(Y_train)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, stratum)):
    scaler = StandardScaler().fit(X_train[tr_idx])
    X_tr_s = scaler.transform(X_train[tr_idx])
    X_val_s = scaler.transform(X_train[val_idx])

    y_tr_enc = encode_distribution(Y_train[tr_idx], mode="sqrt_mass")
    model = Ridge(alpha=1.0, solver="lsqr")
    model.fit(X_tr_s, y_tr_enc)

    raw_val_pred = model.predict(X_val_s)
    oof_preds[val_idx] = decode_distribution(raw_val_pred, mode="sqrt_mass")
    fold_emd = emd_score(Y_train[val_idx], oof_preds[val_idx])
    print(f"  Fold {fold + 1}/6 EMD: {fold_emd:.4f}")

oof_total_emd = emd_score(Y_train, oof_preds)
print(f"==> OVERALL OOF EMD: {oof_total_emd:.4f} (compare vs exp2=46.94, exp3=42.72) <==")

# ---------------------------------------------------------------------------
# 6. Full Model Fit & Test Inference
# ---------------------------------------------------------------------------
print("\nFitting full model on all 24 training samples...")
full_scaler = StandardScaler().fit(X_train)
X_train_full = full_scaler.transform(X_train)
X_test_full = full_scaler.transform(X_test)

Y_train_enc = encode_distribution(Y_train, mode="sqrt_mass")
final_model = Ridge(alpha=1.0, solver="lsqr")
final_model.fit(X_train_full, Y_train_enc)

test_raw = final_model.predict(X_test_full)
test_preds = decode_distribution(test_raw, mode="sqrt_mass")

# ---------------------------------------------------------------------------
# 7. Integrity Contracts & Submission Formatting
# ---------------------------------------------------------------------------
submission = sample_sub.copy()
submission.loc[:, TARGET_COLS] = test_preds

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
