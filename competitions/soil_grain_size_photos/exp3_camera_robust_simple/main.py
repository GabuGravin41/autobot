# ==============================================================================
# Autobot Soil Grain Size Exp 3: Camera-robust, simplified (no deep features)
# Competition: soil-grain-size-from-photos (EU project GRID)
#
# ROOT CAUSE (see THINKING_AND_DECISIONS.md Section 10 for full writeup, and
# diag1_test_image_audit/ for the raw evidence this is based on): exp2
# validated at OOF EMD 46.94 (LOO) but scored 80.96 on the real public
# leaderboard -- worse than even a single un-blended model's OOF number
# (Ridge 53.96 / PLS 53.96 / SVR 56.48 individually). A diagnostic kernel
# that audited every train/test image's filename -> sample_id match and
# detected camera revealed why:
#
#   ALL 127 training images were shot on 1 of 3 Android devices (motorola
#   edge 20: 69 images/13 samples, SM-A525F: 55 images/10 samples, Motorola
#   Edge 60 fusion: 3 images/1 sample). ALL 35 test images were shot on
#   iPhone 14 or iPhone 16. There is ZERO camera-model overlap between
#   train and test.
#
# This means exp1/exp2's LOO-CV OOF estimate -- however rigorously computed
# -- can only ever measure in-domain (Android-to-Android) generalization,
# because every training image (and therefore every LOO fold, no matter how
# the fold is drawn) is Android-only. It structurally cannot measure
# cross-camera generalization, which is exactly the axis the real test set
# hinges on 100% of the time. No amount of "more careful OOF" (more folds,
# nested CV, blend-weight discipline) fixes this -- the training pool
# simply contains no iPhone-shot images to hold out. This is the same
# *category* of gap already found today in ieee_ai_emulation: an OOF/CV
# scheme that never tests generalization along the one axis the real test
# set actually varies on.
#
# Given that, the fix has to be at the feature/model level, not the CV
# level:
#   1. DROP the frozen ImageNet ResNet-34 embedding block entirely. A
#      generic ImageNet backbone's conv features are well known to carry
#      low-level, camera-specific signal (sensor color response, JPEG
#      compression signature, lens sharpening/distortion) entangled with
#      genuine content -- exactly the kind of shortcut a small-sample
#      regularized model could latch onto if it happens to correlate with
#      the training labels, and exactly the kind of signal that provides
#      zero transfer (or actively misleads) once every test image comes
#      from cameras the model has never seen a single frame from. This also
#      removes the 512->10 per-fold PCA compression exp2 used, which was
#      already flagged (Section 8, hypothesis 2) as being closer to
#      memorization than compression at n<=23 -- moot now since there's no
#      deep block left to compress.
#   2. Keep ONLY the 24 deterministic, PPM-scaled CV features (color
#      moments, physical-unit-scaled Sobel edge density, local-variance
#      roughness) -- these were already this project's own stated
#      philosophy (README: "physically grounded, low-dimensional features
#      matter more than raw model capacity" at this sample size) and the
#      Sobel/roughness features in particular are computed in PPM-corrected
#      physical units specifically so they're comparable across cameras.
#      Not modified further here (no evidence collected yet that the raw
#      color-moment features themselves are the problem -- flagged as a
#      candidate for exp4 if this still underperforms, not guessed at now).
#   3. FIX a real, confirmed (independent) bug the audit also caught: the
#      camera-detection heuristic checks for the literal substring
#      "iphone14"/"iphone16" in the filename with spaces stripped, but ONE
#      of the 35 test filenames is `iPhone_16_HPC_Airbus BS10-4bis7
#      (4).JPG` -- an underscore between "iPhone" and "16" that the
#      space-only stripping doesn't remove, so this image silently fell
#      through to "unknown" camera / mean-imputed PPM instead of iPhone
#      16's real ppm=19.525. Minor on its own (1 of 35 images, averaged
#      into a sample that also has 2 correctly-detected images) but free
#      and cheap to fix, and confirmed by direct evidence (diag1's full
#      per-image audit table), not a guess.
#   4. SIMPLIFY the model per this session's own guidance and the n=24
#      regime: no OOF-searched blend weight (exp2's blend search was
#      already flagged, Section 7/8, as being tuned against the same 24
#      OOF points it was scored on -- a real, if small, leakage-by-
#      iteration risk on top of the much bigger camera-shift problem). exp3
#      uses a FIXED, un-searched 50/50 average of Ridge and PLS -- two
#      different, independently well-motivated regularized linear model
#      families for a high-colinearity/low-n regime (exp2 already showed
#      they land within 0.003 of each other on OOF, a sign both are near a
#      genuine regularized floor rather than one overfitting more than the
#      other) -- with the blend weight fixed a priori, not tuned. SVR is
#      dropped: exp2 already showed it as the weakest of the three even
#      after PCA compression, and RBF-SVR has no obvious reason to be more
#      camera-robust than Ridge/PLS, so it isn't worth the added free
#      parameter (11 independent per-target C=10 fits) at n=24.
#   5. ADD a Leave-One-Camera-Group-Out (LOCGO) validation alongside the
#      standard LOO-CV -- NOT used for any model/feature selection (that
#      would repeat exactly the "tuned against the same small OOF" pattern
#      this writeup is criticizing), purely as an honest internal sanity
#      check: does the pipeline generalize at all across distinct physical
#      camera hardware, using only camera splits that exist in the training
#      data (motorola edge 20 vs. SM-A525F; Motorola Edge 60 fusion is a
#      single sample, excluded as its own fold). This can't fully stand in
#      for the real Android->iPhone shift (both LOCGO groups are still
#      Android), but if LOCGO EMD is substantially worse than standard LOO
#      EMD, that's concrete, reproducible internal evidence -- not
#      speculation -- that this pipeline's error grows specifically under
#      camera-family shift, corroborating the real leaderboard gap's cause.
#
# Unchanged from exp1/exp2 (already proven correct against live data): path
# resolution w/ multi-candidate fallback, TARGET_COLUMNS-from-
# sample_submission.csv, the exact-substring/exact-cleaned/fuzzy sample_id
# matching cascade (diag1 confirmed the fuzzy pass's one real hit --
# Muenster -- resolves correctly, 5/6 tokens matched, the missing token
# being exactly the diacritic-mangled one -- so this is NOT a bug, ruling
# out Section 8's hypothesis 3), the Motorola "60 fusion" PPM fix, the
# PPM-scaled deterministic CV feature extraction itself, the monotonic-
# constraint post-processing, and the local structural validation block.
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
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import cv2

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


SEED = 42
seed_everything(SEED)

print("=== AUTOBOT SOIL GRAIN SIZE EXP 3: camera-robust, simplified (no deep features) ===")
print(f"Python Runtime: {sys.version.split()[0]}")
t0 = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Path Resolution (unchanged from exp1/exp2 -- proven correct)
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
        COMP_ROOT / "Training-All_Photos_updated" / "Training-All_Photos_updated",
        COMP_ROOT / "Training-All_Photos_updated",
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
PPM_PATH = locate_path([COMP_ROOT / "ppm_updated.csv", COMP_ROOT / "ppm.csv"], "ppm csv")
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
#    Matching cascade UNCHANGED from exp1/exp2 (diag1 confirmed it's
#    correct -- 0 unmatched images in either split, and the one fuzzy hit
#    resolves to the right sample_id). ONLY change here: camera-detection
#    now also normalizes away underscores (not just spaces) before checking
#    for "iphone14"/"iphone16", fixing the confirmed "iPhone_16_..." miss
#    (diag1 found this filename fell through to "unknown"/mean-imputed PPM).
# ------------------------------------------------------------------------------
def _tokenize(s):
    return [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t]


def fuzzy_match_sample_id(filename, label_sample_ids):
    fn_clean = re.sub(r"[^a-z0-9]+", "", filename.lower())
    best_id, best_matched = None, -1
    for sample_id in label_sample_ids:
        tokens = _tokenize(sample_id)
        if len(tokens) < 3:
            continue
        matched = sum(1 for t in tokens if t in fn_clean)
        if matched >= len(tokens) - 1 and matched > best_matched:
            best_matched, best_id = matched, sample_id
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
            # CHANGE vs. exp1/exp2: also strip underscores (not just spaces)
            # before the iPhone substring checks -- diag1_test_image_audit
            # found "iPhone_16_HPC_Airbus BS10-4bis7 (4).JPG" falls through
            # to "unknown" under the old space-only stripping because of the
            # underscore between "iPhone" and "16".
            fn_norm = filename.lower().replace(" ", "").replace("_", "")
            if "iphone14" in fn_norm:
                detected_camera = "iPhone 14"
            elif "iphone16" in fn_norm:
                detected_camera = "iPhone 16"
            elif "60fusion" in fn_norm or ("60" in fn_norm and "fusion" in fn_norm):
                # Must be checked before the generic motorola/edge branch --
                # ppm_updated.csv lists two distinct Motorola models
                # ("motorola edge 20" ppm=11.492 vs. "Motorola Edge 60
                # fusion" ppm=12.465); unchanged from exp1/exp2.
                detected_camera = "Motorola Edge 60 fusion"
            elif "motorola" in fn_norm or "edge" in fn_norm:
                detected_camera = "motorola edge 20"
            elif "samsung" in fn_norm or "a52" in fn_norm or "sma525f" in fn_norm:
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

n_unknown_camera_test = (test_mapping_database["camera_key"] == "unknown").sum()
print(f"Test images with 'unknown' camera (mean-imputed PPM): {n_unknown_camera_test}")


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

# Camera distribution report (this is the diagnostic finding that drove
# this experiment -- printed here too so it's visible in this kernel's own
# log, not just diag1's separate one).
print("\nCamera distribution (per image), train vs test:")
tr_cam = train_mapping_database["camera_key"].value_counts()
te_cam = test_mapping_database["camera_key"].value_counts()
for cam in sorted(set(tr_cam.index) | set(te_cam.index)):
    print(f"  {cam:<26} train={int(tr_cam.get(cam, 0)):>4}   test={int(te_cam.get(cam, 0)):>4}")
train_cams = set(train_mapping_database["camera_key"].unique())
test_cams = set(test_mapping_database["camera_key"].unique())
print(f"Camera overlap between train and test: {train_cams & test_cams or '(NONE)'}")

# ------------------------------------------------------------------------------
# 4. Deterministic, PPM-scale-invariant CV feature extraction (UNCHANGED
#    from exp1/exp2 -- this is now the ONLY feature block, no deep ResNet).
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


print("\nExtracting deterministic CV features (train)...")
train_cv_features = execute_feature_extraction_loop(train_mapping_database)
print("Extracting deterministic CV features (test)...")
test_cv_features = execute_feature_extraction_loop(test_mapping_database)
print(f"CV feature shapes: train={train_cv_features.shape}, test={test_cv_features.shape}")

# ------------------------------------------------------------------------------
# 5. Sample-level aggregation (mean pool over images per sample) -- NO deep
#    features this time, so this is just the 24-dim CV block.
# ------------------------------------------------------------------------------
def consolidate_sample_features(mapping_df, cv_feat, unique_sample_list):
    sample_feature_map = {}
    for idx, row in mapping_df.iterrows():
        sid = row["sample_id"]
        sample_feature_map.setdefault(sid, []).append(cv_feat[idx])

    consolidated_features, valid_samples = [], []
    for sid in unique_sample_list:
        if sid in sample_feature_map:
            consolidated_features.append(np.mean(sample_feature_map[sid], axis=0))
            valid_samples.append(sid)
    return np.array(consolidated_features), valid_samples


train_mapping_database = train_mapping_database.reset_index(drop=True)
test_mapping_database = test_mapping_database.reset_index(drop=True)

train_features_aggregated, train_ids_aggregated = consolidate_sample_features(
    train_mapping_database, train_cv_features, train_sample_list
)
test_features_aggregated, test_ids_aggregated = consolidate_sample_features(
    test_mapping_database, test_cv_features, test_sample_list
)

# Camera-group label per aggregated training sample (for LOCGO validation,
# section 7 below) -- majority camera across that sample's images.
train_sample_camera = (
    train_mapping_database.groupby("sample_id")["camera_key"]
    .agg(lambda x: x.mode().iloc[0])
    .to_dict()
)
train_camera_groups = np.array([train_sample_camera[sid] for sid in train_ids_aggregated])

print(f"Aggregated feature shapes: train={train_features_aggregated.shape}, test={test_features_aggregated.shape}")

labels_master_dict = train_labels_df.set_index("sample_id")[TARGET_COLUMNS].to_dict(orient="index")
Y_train_raw = np.array([list(labels_master_dict[sid].values()) for sid in train_ids_aggregated])
print(f"Target matrix shape: {Y_train_raw.shape}")

# ------------------------------------------------------------------------------
# 6. Competition metric + LOO Ridge/PLS, FIXED (not searched) 50/50 blend.
# ------------------------------------------------------------------------------
def compute_log_weighted_emd(y_true, y_pred, widths=interval_widths):
    absolute_errors = np.abs(y_true - y_pred)
    interval_errors = (absolute_errors[:, :-1] + absolute_errors[:, 1:]) / 2.0
    weighted_intervals = interval_errors * widths
    return np.sum(weighted_intervals, axis=1)


N_PLS = 5  # small, fixed a priori (24 CV features, <=23 LOO training rows) --
           # NOT chosen by comparing multiple values against OOF, to avoid
           # repeating the "tuned against the same n=24 OOF" pattern this
           # experiment's own writeup criticizes in exp2's blend search.
RIDGE_ALPHA = 150.0  # unchanged from exp1/exp2/source notebook -- deliberately
                      # not re-tuned for the smaller (24-dim, no-deep-block)
                      # feature set here, for the same reason.

loo = LeaveOneOut()
N_FOLDS = len(train_features_aggregated)

oof_ridge_predictions = np.zeros_like(Y_train_raw)
oof_pls_predictions = np.zeros_like(Y_train_raw)
test_ridge_predictions_accum = np.zeros((len(test_features_aggregated), 11))
test_pls_predictions_accum = np.zeros((len(test_features_aggregated), 11))

for fold, (train_idx, val_idx) in enumerate(loo.split(train_features_aggregated)):
    X_tr, Y_tr = train_features_aggregated[train_idx], Y_train_raw[train_idx]
    X_va = train_features_aggregated[val_idx]
    X_te = test_features_aggregated

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_te)

    ridge_regressor = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    ridge_regressor.fit(X_tr_scaled, Y_tr)
    oof_ridge_predictions[val_idx] = ridge_regressor.predict(X_va_scaled)
    test_ridge_predictions_accum += ridge_regressor.predict(X_te_scaled) / N_FOLDS

    n_pls = min(N_PLS, X_tr_scaled.shape[0] - 1, X_tr_scaled.shape[1])
    pls_model = PLSRegression(n_components=n_pls)
    pls_model.fit(X_tr_scaled, Y_tr)
    oof_pls_predictions[val_idx] = pls_model.predict(X_va_scaled)
    test_pls_predictions_accum += pls_model.predict(X_te_scaled) / N_FOLDS

    if (fold + 1) % 6 == 0 or (fold + 1) == N_FOLDS:
        print(f"LOO fold {fold + 1}/{N_FOLDS} done.")

ridge_emd_scores = compute_log_weighted_emd(Y_train_raw, oof_ridge_predictions)
pls_emd_scores = compute_log_weighted_emd(Y_train_raw, oof_pls_predictions)
print(f"OOF EMD - Ridge (alpha={RIDGE_ALPHA}): {np.mean(ridge_emd_scores):.4f}")
print(f"OOF EMD - PLS (n_components={N_PLS}): {np.mean(pls_emd_scores):.4f}")

# FIXED 50/50 blend -- not searched against OOF (see module docstring).
W_RIDGE, W_PLS = 0.5, 0.5
blended_oof_predictions = W_RIDGE * oof_ridge_predictions + W_PLS * oof_pls_predictions
blended_emd_scores = compute_log_weighted_emd(Y_train_raw, blended_oof_predictions)
print(f"OOF EMD - Fixed 50/50 Ridge+PLS blend (pre-constraints): {np.mean(blended_emd_scores):.4f}")

# ------------------------------------------------------------------------------
# 7. Leave-One-Camera-Group-Out (LOCGO) validation -- DIAGNOSTIC ONLY, not
#    used to pick features/models/weights. See module docstring point 5.
# ------------------------------------------------------------------------------
print("\n--- Leave-One-Camera-Group-Out (LOCGO) validation (diagnostic only) ---")
camera_group_counts = pd.Series(train_camera_groups).value_counts()
print(f"Training sample camera groups: {camera_group_counts.to_dict()}")
locgo_groups = [g for g in camera_group_counts.index if camera_group_counts[g] >= 5]
print(f"LOCGO folds (groups with >=5 samples, held out one at a time): {locgo_groups}")

for held_out_group in locgo_groups:
    tr_mask = train_camera_groups != held_out_group
    va_mask = ~tr_mask
    X_tr, Y_tr = train_features_aggregated[tr_mask], Y_train_raw[tr_mask]
    X_va, Y_va = train_features_aggregated[va_mask], Y_train_raw[va_mask]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    ridge_regressor = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    ridge_regressor.fit(X_tr_scaled, Y_tr)
    ridge_va_pred = ridge_regressor.predict(X_va_scaled)

    n_pls = min(N_PLS, X_tr_scaled.shape[0] - 1, X_tr_scaled.shape[1])
    pls_model = PLSRegression(n_components=n_pls)
    pls_model.fit(X_tr_scaled, Y_tr)
    pls_va_pred = pls_model.predict(X_va_scaled)

    blend_va_pred = 0.5 * ridge_va_pred + 0.5 * pls_va_pred
    locgo_emd = np.mean(compute_log_weighted_emd(Y_va, blend_va_pred))
    print(
        f"  Held out camera group '{held_out_group}' "
        f"(n_val={va_mask.sum()}, n_train={tr_mask.sum()}): blend EMD = {locgo_emd:.4f}"
    )

print(
    "NOTE: LOCGO folds are still Android-vs-Android (no iPhone data exists in "
    "training to hold out), so this under-states the real Android->iPhone gap "
    "if anything -- it's a lower bound on the cross-camera generalization risk, "
    "not a full replica of it. Used for documentation only; does not affect "
    "the model/weights used for the actual test predictions below."
)

# ------------------------------------------------------------------------------
# 8. Physical constraint post-processing (structural validity guarantee) --
#    unchanged from exp1/exp2.
# ------------------------------------------------------------------------------
def apply_geotechnical_constraints(predicted_matrix):
    bounded = np.clip(predicted_matrix, 0.0, 100.0)
    for col_idx in range(1, bounded.shape[1]):
        bounded[:, col_idx] = np.maximum(bounded[:, col_idx], bounded[:, col_idx - 1])
    bounded[:, -1] = 100.00
    return bounded


optimized_oof_predictions = apply_geotechnical_constraints(blended_oof_predictions)
optimized_emd_scores = compute_log_weighted_emd(Y_train_raw, optimized_oof_predictions)
print(f"\nOOF EMD - Blend + constraints: {np.mean(optimized_emd_scores):.4f}")

is_oof_monotonic = all(
    np.all(np.diff(optimized_oof_predictions[i, :]) >= 0) for i in range(optimized_oof_predictions.shape[0])
)
print(f"Post-processed OOF monotonicity check: {is_oof_monotonic}")
print(f"Post-processed OOF 200mm==100 check: {np.all(optimized_oof_predictions[:, -1] == 100.0)}")

# ------------------------------------------------------------------------------
# 9. Test inference + submission export
# ------------------------------------------------------------------------------
blended_test_predictions = W_RIDGE * test_ridge_predictions_accum + W_PLS * test_pls_predictions_accum
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
# 10. Local structural validation before writing the file (unchanged)
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
