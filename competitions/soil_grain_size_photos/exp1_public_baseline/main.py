# ==============================================================================
# Autobot Soil Grain Size Exp 1: Public Baseline
# Competition: soil-grain-size-from-photos (EU project GRID)
# Task: Predict cumulative grain size mass fraction (%) at 11 DIN EN ISO
#       14688-1 sieve diameters from soil surface photographs.
#
# ATTRIBUTION: This is Autobot's Phase 0 SOTA-discovery baseline. The
# modeling pipeline (deterministic PPM-scaled CV texture/edge features +
# frozen ResNet-34 embeddings -> per-fold Ridge/SVR ensemble -> isotonic-
# style monotonic post-processing) is adapted, with minimal changes, from
# the best-scoring public kernel found via
# `kaggle kernels list --competition soil-grain-size-from-photos
# --sort-by scoreDescending`:
#   avikdas567/soil-grain-size-prediction-calibrated-cv-ensemble
#   ("Soil Grain Size Prediction: Calibrated CV Ensemble", 20 votes, #1 by
#   score among all public kernels found for this competition as of
#   2026-09-24). The original notebook is archived unmodified at
#   competitions/soil_grain_size_photos/public_nb/ for reference.
#
# Changes made here vs. the original notebook:
#   - Converted from .ipynb (with EDA/plotting cells) to a linear .py
#     script (this repo's kernel_type=script convention) -- plotting cells
#     dropped since a script kernel has nowhere to render them.
#   - enable_gpu=false: the original requested a T4 GPU for the frozen
#     ResNet-34 forward pass, but with a ~25-sample training pool (and a
#     similarly small test pool) that inference is a handful of images and
#     runs fine on CPU in well under a minute -- keeping this off the GPU
#     preserves the account's shared 2-slot GPU ledger for competitions
#     that actually need it (see kaggle_watchdog.py).
#   - Added multi-candidate path resolution (this repo's existing
#     locate_path() convention from other competitions/*/exp*/main.py
#     scripts) since the original hardcoded a single
#     /kaggle/input/competitions/... path shape that may not match every
#     kernel-attachment mode.
#   - Logic (feature extraction, fold structure, model hyperparameters,
#     post-processing) is otherwise unchanged from the source notebook.
# ==============================================================================
import os
import re
import sys
import glob
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== AUTOBOT SOIL GRAIN SIZE EXP 1: PUBLIC BASELINE (calibrated CV+ResNet ensemble) ===")
print(f"Python Runtime: {sys.version.split()[0]}")
print(f"System Execution Device: {device}")
t0 = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Path Resolution (robust: multi-candidate + glob fallback)
# ------------------------------------------------------------------------------
def locate_path(candidates, desc="path", must_exist=True):
    for c in candidates:
        p = Path(c)
        if p.exists():
            print(f"Located {desc}: {p}")
            return p
    if must_exist:
        raise FileNotFoundError(f"Could not locate {desc} from candidates: {candidates}")
    return None


INPUT_ROOTS = [
    "/kaggle/input/competitions/soil-grain-size-from-photos",
    "/kaggle/input/soil-grain-size-from-photos",
    "soil-grain-size-from-photos",
    ".",
]
COMP_ROOT = locate_path(INPUT_ROOTS, "competition data root")

TRAIN_DIR = locate_path(
    [
        # Current dataset (verified via `kaggle competitions files` on 2026-09-24):
        # the competition data was updated at some point after the source public
        # notebook was published -- the old "_without_H374" files/dirs it hardcoded
        # no longer exist, replaced by "_updated" names (H374 is back in the pool).
        COMP_ROOT / "Training-All_Photos_updated" / "Training-All_Photos_updated",
        COMP_ROOT / "Training-All_Photos_updated",
        # Old dataset names (source notebook), kept as a fallback in case the
        # competition data gets swapped back or differs by kernel-attach mode.
        COMP_ROOT / "Training-All_Photos_without_H374" / "Training-All_Photos_without_H374",
        COMP_ROOT / "Training-All_Photos_without_H374",
    ],
    "training photos directory",
)
TEST_DIR = locate_path(
    [
        COMP_ROOT / "Test_All_Photos" / "Test_All_Photos",
        COMP_ROOT / "Test_All_Photos",
    ],
    "test photos directory",
)
SAMPLE_SUB_PATH = locate_path([COMP_ROOT / "sample_submission.csv"], "sample_submission.csv")
PPM_PATH = locate_path(
    [COMP_ROOT / "ppm_updated.csv", COMP_ROOT / "ppm.csv"], "ppm csv"
)
TRAIN_LABELS_PATH = locate_path(
    [
        COMP_ROOT / "Training_labels_updated.csv",
        COMP_ROOT / "Training_labels_without_H374.csv",
    ],
    "training labels csv",
)

# ------------------------------------------------------------------------------
# 2. Competition Constants
# ------------------------------------------------------------------------------
SUPPORT_DIAMETERS = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
log_diameters = np.log10(SUPPORT_DIAMETERS)
interval_widths = np.diff(log_diameters)

train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
ppm_df = pd.read_csv(PPM_PATH)
sample_sub_df = pd.read_csv(SAMPLE_SUB_PATH)

# TARGET_COLUMNS is derived from sample_submission.csv's own header (every
# column except sample_id) rather than hand-typed, per this project's
# submission-format rule -- avoids a silent '2' vs '2.0' style mismatch.
TARGET_COLUMNS = [c for c in sample_sub_df.columns if c != "sample_id"]
assert len(TARGET_COLUMNS) == 11, f"expected 11 target columns, got {len(TARGET_COLUMNS)}: {TARGET_COLUMNS}"
missing_in_labels = [c for c in TARGET_COLUMNS if c not in train_labels_df.columns]
assert not missing_in_labels, (
    f"sample_submission.csv columns {missing_in_labels} not found in training labels "
    f"(training labels has: {list(train_labels_df.columns)}) -- header formatting mismatch."
)
print(f"TARGET_COLUMNS (from sample_submission.csv): {TARGET_COLUMNS}")

print(f"Training Targets: {train_labels_df.shape[0]} samples, {train_labels_df.shape[1]} columns.")
print(f"Calibration Meta-Database: {ppm_df.shape[0]} sensor configurations.")
print(f"Test submission rows required: {sample_sub_df.shape[0]}")

is_monotonic = train_labels_df[TARGET_COLUMNS].apply(lambda x: x.is_monotonic_increasing, axis=1)
print(f"Training label monotonicity check: {(is_monotonic.sum() / len(train_labels_df)) * 100:.2f}% valid.")

# ------------------------------------------------------------------------------
# 3. Image <-> sample_id mapping + PPM calibration lookup
#
# CHANGE vs. the source notebook: added a fuzzy fallback pass after the
# original two literal-substring passes. Verified against the live
# competition data (`kaggle competitions files`) that one test location's
# filename contains a mangled/non-UTF8 diacritic --
# "iPhone14_HPC_M?nster_BS6_9,0-10m (1).JPG" for sample_id
# "HPC_Muenster_BS6_9_0-10m" -- so neither of the source notebook's literal
# passes ever matches "muenster" against "m?nster", and that test row would
# silently fall back to the trivial equal-fraction baseline (a real hit
# with only 10 test rows total).
#
# The fallback is deliberately NOT a plain token-overlap-with-short-token-
# filtering scheme (an earlier draft tried that and was caught by a local
# dry run before ever being pushed to Kaggle: filtering out short numeric
# tokens like "2", "3", "18" made all four distinct "HPC_Kleinkummerfeld
# 2-2/2-3/9-4/18-3" test locations collapse onto a single match, since only
# "hpc"+"kleinkummerfeld" survived the filter for every one of them). The
# fix keeps the original exact-substring passes untouched (proven correct,
# and they already resolve every non-mangled filename including the
# Kleinkummerfeld/Airbus-BS numeric-suffix cases) and only reaches the
# fuzzy pass -- with UNFILTERED tokens and an "all tokens but at most one"
# threshold, tight enough to need every distinguishing digit/suffix token
# except the specific one a mangled byte can break -- when both literal
# passes come up empty.
# ------------------------------------------------------------------------------
def _tokenize(s):
    return [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t]


def fuzzy_match_sample_id(filename, label_sample_ids):
    """Last-resort match: allows exactly one of a sample_id's tokens (incl.
    short/numeric ones) to fail to appear in the filename, so a single
    mangled character doesn't lose the whole match -- but still requires
    every *other* token (including the numeric suffixes that distinguish
    e.g. Kleinkummerfeld 2-2 from 18-3), so it can't collapse distinct
    locations together the way filtering short tokens out would."""
    fn_clean = re.sub(r"[^a-z0-9]+", "", filename.lower())
    best_id, best_matched, best_total = None, -1, 0
    for sample_id in label_sample_ids:
        tokens = _tokenize(sample_id)
        if len(tokens) < 3:
            continue  # too few tokens for an "all but one" rule to mean anything
        matched = sum(1 for t in tokens if t in fn_clean)
        if matched >= len(tokens) - 1 and matched > best_matched:
            best_matched, best_total, best_id = matched, len(tokens), sample_id
    return best_id


def match_sample_id(filename, label_sample_ids):
    fn_lower = filename.lower()
    for sample_id in label_sample_ids:
        if str(sample_id).lower() in fn_lower:
            return sample_id
    cleaned_filename = fn_lower.replace(" ", "")
    for sample_id in label_sample_ids:
        cleaned_sid = str(sample_id).lower().replace(" ", "")
        if cleaned_sid in cleaned_filename or cleaned_filename in cleaned_sid:
            return sample_id
    return fuzzy_match_sample_id(filename, label_sample_ids)


def build_image_mapping_database(directory_path, label_sample_ids):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    all_files = []
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(str(directory_path), ext)))

    records = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        matched_sample_id = match_sample_id(filename, label_sample_ids)

        if matched_sample_id is not None:
            fn_lower = filename.lower().replace(" ", "")
            if "iphone14" in fn_lower:
                detected_camera = "iPhone 14"
            elif "iphone16" in fn_lower:
                detected_camera = "iPhone 16"
            elif "60fusion" in fn_lower or ("60" in fn_lower and "fusion" in fn_lower):
                # CHANGE vs. source notebook: must be checked before the generic
                # motorola/edge branch below. ppm_updated.csv lists two distinct
                # Motorola models -- "motorola edge 20" (ppm=11.492) and
                # "Motorola Edge 60 fusion" (ppm=12.465) -- and training filenames
                # include both ("Motorola_Edge_60_fusion_H374_0*.jpg" vs.
                # "Motorola_Edge_H030_0*.jpg" etc). The source notebook's original
                # ordering (bare "motorola"/"edge" check first) would bucket every
                # "60 fusion" image under the wrong PPM, silently mis-calibrating
                # the physical-scale Sobel features for those images.
                detected_camera = "Motorola Edge 60 fusion"
            elif "motorola" in fn_lower or "edge" in fn_lower:
                detected_camera = "motorola edge 20"
            elif "samsung" in fn_lower or "a52" in fn_lower or "sm-a525f" in fn_lower:
                detected_camera = "SM-A525F"
            else:
                detected_camera = "unknown"
                for _, row in ppm_df.iterrows():
                    cam_spec = str(row["camera"]).lower()
                    phone_spec = str(row["phone"]).lower()
                    if cam_spec in filename.lower() or phone_spec in filename.lower():
                        detected_camera = row["camera"]
                        break

            records.append(
                {
                    "filepath": filepath,
                    "filename": filename,
                    "sample_id": matched_sample_id,
                    "camera_key": detected_camera,
                }
            )
    return pd.DataFrame(records)


train_sample_list = train_labels_df["sample_id"].unique().tolist()
test_sample_list = sample_sub_df["sample_id"].unique().tolist()

train_mapping_database = build_image_mapping_database(TRAIN_DIR, train_sample_list)
test_mapping_database = build_image_mapping_database(TEST_DIR, test_sample_list)

print(f"Mapped {len(train_mapping_database)} training images to sample IDs.")
print(f"Mapped {len(test_mapping_database)} test images to sample IDs.")

if len(train_mapping_database) == 0 or len(test_mapping_database) == 0:
    raise RuntimeError(
        "Image mapping produced zero rows for train or test set -- path resolution or "
        "filename matching is broken; check COMP_ROOT/TRAIN_DIR/TEST_DIR above."
    )


def inject_ppm_parameters(mapping_df, specs_df):
    default_ppm = specs_df["ppm"].mean()
    resolved = []
    for _, row in mapping_df.iterrows():
        cam_key = row["camera_key"]
        matched_ppm = specs_df[specs_df["camera"] == cam_key]["ppm"].values
        if len(matched_ppm) == 0:
            matched_ppm = specs_df[specs_df["phone"] == cam_key]["ppm"].values
        final_ppm = matched_ppm[0] if len(matched_ppm) > 0 else default_ppm
        record = row.to_dict()
        record["ppm"] = final_ppm
        resolved.append(record)
    return pd.DataFrame(resolved)


train_mapping_database = inject_ppm_parameters(train_mapping_database, ppm_df)
test_mapping_database = inject_ppm_parameters(test_mapping_database, ppm_df)

# ------------------------------------------------------------------------------
# 4. Deterministic, PPM-scale-invariant CV feature extraction
# ------------------------------------------------------------------------------
def extract_deterministic_surface_features(image_path, ppm_ratio):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return np.zeros(25)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    feature_vector = []

    for channel in range(3):
        ch_data = image_rgb[:, :, channel].astype(float)
        feature_vector.append(np.mean(ch_data))
        feature_vector.append(np.std(ch_data))
        feature_vector.append(skew(ch_data.flatten()))

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        ch_data = image_hsv[:, :, channel].astype(float)
        feature_vector.append(np.mean(ch_data))
        feature_vector.append(np.std(ch_data))

    base_kernel_sizes = [1.0, 3.0, 5.0]
    for target_mm in base_kernel_sizes:
        computed_pixels = int(round(target_mm * ppm_ratio))
        if computed_pixels % 2 == 0:
            computed_pixels += 1
        computed_pixels = max(3, min(computed_pixels, 31))

        blurred = cv2.GaussianBlur(image_gray, (computed_pixels, computed_pixels), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        feature_vector.append(np.mean(magnitude))
        feature_vector.append(np.std(magnitude))

    local_kernel = 7
    mean_filter = cv2.blur(image_gray.astype(float), (local_kernel, local_kernel))
    sq_mean_filter = cv2.blur(image_gray.astype(float) ** 2, (local_kernel, local_kernel))
    local_variance = np.clip(sq_mean_filter - mean_filter ** 2, 0, None)
    local_std = np.sqrt(local_variance)

    feature_vector.append(np.mean(local_std))
    feature_vector.append(np.std(local_std))
    feature_vector.append(kurtosis(local_std.flatten()))

    return np.array(feature_vector)


def execute_feature_extraction_loop(mapping_df):
    return np.array(
        [
            extract_deterministic_surface_features(row["filepath"], row["ppm"])
            for _, row in mapping_df.iterrows()
        ]
    )


print("Extracting deterministic CV features (train)...")
train_cv_features = execute_feature_extraction_loop(train_mapping_database)
print("Extracting deterministic CV features (test)...")
test_cv_features = execute_feature_extraction_loop(test_mapping_database)
print(f"CV feature shapes: train={train_cv_features.shape}, test={test_cv_features.shape}")

# ------------------------------------------------------------------------------
# 5. Frozen ResNet-34 deep embeddings (CPU inference -- small dataset)
# ------------------------------------------------------------------------------
class GeotechnicalImageDataset(Dataset):
    def __init__(self, mapping_df, transform=None):
        self.mapping_df = mapping_df
        self.transform = transform

    def __len__(self):
        return len(self.mapping_df)

    def __getitem__(self, idx):
        row = self.mapping_df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx


deep_transformations = transforms.Compose(
    [
        transforms.CenterCrop(1024),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = GeotechnicalImageDataset(train_mapping_database, transform=deep_transformations)
test_dataset = GeotechnicalImageDataset(test_mapping_database, transform=deep_transformations)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))


class DeepFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.feature_layer = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return torch.flatten(self.feature_layer(x), 1)


deep_extractor = DeepFeatureExtractor().to(device)
deep_extractor.eval()


def extract_deep_embeddings(dataloader, model):
    all_embeddings = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    embeddings = model(images)
            else:
                embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


print("Extracting deep ResNet-34 embeddings (train)...")
train_deep_features = extract_deep_embeddings(train_loader, deep_extractor)
print("Extracting deep ResNet-34 embeddings (test)...")
test_deep_features = extract_deep_embeddings(test_loader, deep_extractor)
print(f"Deep feature shapes: train={train_deep_features.shape}, test={test_deep_features.shape}")

# ------------------------------------------------------------------------------
# 6. Sample-level aggregation (mean pool over images per sample)
# ------------------------------------------------------------------------------
def consolidate_sample_features(mapping_df, cv_feat, deep_feat, unique_sample_list):
    sample_feature_map = {}
    for idx, row in mapping_df.iterrows():
        sid = row["sample_id"]
        combined = np.concatenate([cv_feat[idx], deep_feat[idx]])
        sample_feature_map.setdefault(sid, []).append(combined)

    consolidated_features, valid_samples = [], []
    for sid in unique_sample_list:
        if sid in sample_feature_map:
            consolidated_features.append(np.mean(sample_feature_map[sid], axis=0))
            valid_samples.append(sid)
    return np.array(consolidated_features), valid_samples


train_mapping_database = train_mapping_database.reset_index(drop=True)
test_mapping_database = test_mapping_database.reset_index(drop=True)

train_features_aggregated, train_ids_aggregated = consolidate_sample_features(
    train_mapping_database, train_cv_features, train_deep_features, train_sample_list
)
test_features_aggregated, test_ids_aggregated = consolidate_sample_features(
    test_mapping_database, test_cv_features, test_deep_features, test_sample_list
)

print(f"Aggregated feature shapes: train={train_features_aggregated.shape}, test={test_features_aggregated.shape}")

labels_master_dict = train_labels_df.set_index("sample_id")[TARGET_COLUMNS].to_dict(orient="index")
Y_train_raw = np.array([list(labels_master_dict[sid].values()) for sid in train_ids_aggregated])
print(f"Target matrix shape: {Y_train_raw.shape}")

# ------------------------------------------------------------------------------
# 7. Competition metric (log-weighted EMD) + 5-fold Ridge/SVR ensemble
# ------------------------------------------------------------------------------
def compute_log_weighted_emd(y_true, y_pred, widths=interval_widths):
    absolute_errors = np.abs(y_true - y_pred)
    interval_errors = (absolute_errors[:, :-1] + absolute_errors[:, 1:]) / 2.0
    weighted_intervals = interval_errors * widths
    return np.sum(weighted_intervals, axis=1)


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_ridge_predictions = np.zeros_like(Y_train_raw)
oof_svr_predictions = np.zeros_like(Y_train_raw)
test_ridge_predictions_accum = np.zeros((len(test_features_aggregated), 11))
test_svr_predictions_accum = np.zeros((len(test_features_aggregated), 11))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_features_aggregated)):
    X_tr, Y_tr = train_features_aggregated[train_idx], Y_train_raw[train_idx]
    X_va = train_features_aggregated[val_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(test_features_aggregated)

    ridge_regressor = Ridge(alpha=150.0, random_state=SEED)
    ridge_regressor.fit(X_tr_scaled, Y_tr)
    oof_ridge_predictions[val_idx] = ridge_regressor.predict(X_va_scaled)
    test_ridge_predictions_accum += ridge_regressor.predict(X_te_scaled) / FOLDS

    for target_idx in range(11):
        svr_model = SVR(C=10.0, epsilon=0.1, kernel="rbf")
        svr_model.fit(X_tr_scaled, Y_tr[:, target_idx])
        oof_svr_predictions[val_idx, target_idx] = svr_model.predict(X_va_scaled)
        test_svr_predictions_accum[:, target_idx] += svr_model.predict(X_te_scaled) / FOLDS

    print(f"Fold {fold + 1}/{FOLDS} done.")

ridge_emd_scores = compute_log_weighted_emd(Y_train_raw, oof_ridge_predictions)
svr_emd_scores = compute_log_weighted_emd(Y_train_raw, oof_svr_predictions)
blended_oof_predictions = 0.5 * oof_ridge_predictions + 0.5 * oof_svr_predictions
blended_emd_scores = compute_log_weighted_emd(Y_train_raw, blended_oof_predictions)

print(f"OOF EMD - Ridge: {np.mean(ridge_emd_scores):.4f}")
print(f"OOF EMD - SVR: {np.mean(svr_emd_scores):.4f}")
print(f"OOF EMD - Blend: {np.mean(blended_emd_scores):.4f}")

# ------------------------------------------------------------------------------
# 8. Physical constraint post-processing (structural validity guarantee)
# ------------------------------------------------------------------------------
def apply_geotechnical_constraints(predicted_matrix):
    bounded = np.clip(predicted_matrix, 0.0, 100.0)
    for col_idx in range(1, bounded.shape[1]):
        bounded[:, col_idx] = np.maximum(bounded[:, col_idx], bounded[:, col_idx - 1])
    bounded[:, -1] = 100.00
    return bounded


optimized_oof_predictions = apply_geotechnical_constraints(blended_oof_predictions)
optimized_emd_scores = compute_log_weighted_emd(Y_train_raw, optimized_oof_predictions)
print(f"OOF EMD - Blend + constraints: {np.mean(optimized_emd_scores):.4f}")

is_oof_monotonic = all(
    np.all(np.diff(optimized_oof_predictions[i, :]) >= 0) for i in range(optimized_oof_predictions.shape[0])
)
print(f"Post-processed OOF monotonicity check: {is_oof_monotonic}")
print(f"Post-processed OOF 200mm==100 check: {np.all(optimized_oof_predictions[:, -1] == 100.0)}")

# ------------------------------------------------------------------------------
# 9. Test inference + submission export
# ------------------------------------------------------------------------------
blended_test_predictions = 0.5 * test_ridge_predictions_accum + 0.5 * test_svr_predictions_accum
optimized_test_predictions = apply_geotechnical_constraints(blended_test_predictions)

submission_mapping_dict = {
    sid: optimized_test_predictions[idx] for idx, sid in enumerate(test_ids_aggregated)
}

FALLBACK_ROW = [9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.00]

final_rows = []
n_fallback = 0
for _, row in sample_sub_df.iterrows():
    sid = row["sample_id"]
    if sid in submission_mapping_dict:
        final_rows.append([sid] + list(submission_mapping_dict[sid]))
    else:
        final_rows.append([sid] + FALLBACK_ROW)
        n_fallback += 1

if n_fallback:
    print(f"WARNING: {n_fallback} test sample_id(s) had no matched image -- used equal-fraction fallback row.")

submission_df = pd.DataFrame(final_rows, columns=["sample_id"] + TARGET_COLUMNS)
for col in TARGET_COLUMNS:
    submission_df[col] = submission_df[col].astype(float)
submission_df["200"] = 100.00

# ------------------------------------------------------------------------------
# 10. Local structural validation before writing the file (fail loud, not late)
# ------------------------------------------------------------------------------
vals = submission_df[TARGET_COLUMNS].values
assert vals.min() >= 0.0 and vals.max() <= 100.0, "values out of [0, 100] range"
assert np.all(np.diff(vals, axis=1) >= -1e-9), "rows are not non-decreasing"
assert np.allclose(vals[:, -1], 100.00), "200mm column is not exactly 100.00"
assert submission_df["sample_id"].is_unique, "duplicate sample_id in submission"
assert set(submission_df["sample_id"]) == set(sample_sub_df["sample_id"]), "sample_id set mismatch vs sample_submission.csv"
assert list(submission_df.columns) == list(sample_sub_df.columns), "column names/order mismatch vs sample_submission.csv"
print("Local structural validation: PASSED (range, monotonic, 200mm==100, ids match, columns match).")

submission_df.to_csv("submission.csv", index=False)
print(f"Submission saved: submission.csv ({submission_df.shape[0]} rows).")
print(f"Total runtime: {time.time() - t0:.1f}s")
print(submission_df.head(10).to_string())
