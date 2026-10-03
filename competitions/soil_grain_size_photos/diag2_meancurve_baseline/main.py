# ==============================================================================
# Autobot Soil Grain Size -- Diagnostic 2: mean-training-curve baseline
# Competition: soil-grain-size-from-photos (EU project GRID)
#
# NOT a modeling experiment -- no images touched, no photos read. exp3
# (camera-robust, deep-ResNet-features dropped) scored 80.65 on the real
# leaderboard, barely better than exp2's 80.96 (which used the ImageNet
# ResNet-34 block exp3 removed). Since removing the single biggest,
# most-obviously-domain-brittle feature block barely moved the real score,
# that means the camera-domain-shift damage (Section 11: 100% train/test
# camera-model disjointness, Android train vs. iPhone test) isn't confined
# to the deep block -- the remaining 24 deterministic CV features (raw
# color moments + PPM-scaled edge/texture stats, all computed directly
# from unnormalized pixel values with no camera-invariant treatment) are
# apparently *also* failing to transfer.
#
# This diagnostic checks a specific, cheap, and important question before
# any further feature engineering: is image-conditioned prediction net
# HELPING or net HURTING right now, given the domain shift? Computes a
# pure, image-free LOO baseline -- for each held-out training sample,
# predict the *mean curve of the other 23 samples* (no photo used at all)
# -- and reports its OOF EMD. If this image-free baseline's OOF EMD is
# anywhere close to (or better than) exp2/exp3's per-image-feature OOF
# EMD (42.7-46.9), that's a strong signal the photo-conditioned models
# aren't actually earning their complexity relative to just guessing the
# average curve, which would be important context for deciding exp4's
# direction (more feature engineering vs. a much more conservative,
# heavily-shrunk-toward-the-mean model).
# ==============================================================================
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

print("=== AUTOBOT SOIL DIAGNOSTIC 2: mean-training-curve (image-free) baseline ===")
t0 = time.time()


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
SAMPLE_SUB_PATH = locate_path([COMP_ROOT / "sample_submission.csv"], "sample_submission.csv")
TRAIN_LABELS_PATH = locate_path(
    [
        COMP_ROOT / "Training_labels_updated.csv",
        COMP_ROOT / "Training_labels_without_H374.csv",
    ],
    "training labels csv",
)

train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub_df = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLUMNS = [c for c in sample_sub_df.columns if c != "sample_id"]

SUPPORT_DIAMETERS = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
log_diameters = np.log10(SUPPORT_DIAMETERS)
interval_widths = np.diff(log_diameters)


def compute_log_weighted_emd(y_true, y_pred, widths=interval_widths):
    absolute_errors = np.abs(y_true - y_pred)
    interval_errors = (absolute_errors[:, :-1] + absolute_errors[:, 1:]) / 2.0
    weighted_intervals = interval_errors * widths
    return np.sum(weighted_intervals, axis=1)


Y = train_labels_df[TARGET_COLUMNS].values.astype(float)
n = Y.shape[0]
print(f"Training labels: {n} samples x {Y.shape[1]} columns")

# LOO mean-curve baseline: for each held-out row, predict the mean of the
# OTHER n-1 rows (no image features used at all).
oof_pred = np.zeros_like(Y)
for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    oof_pred[i] = Y[mask].mean(axis=0)

meancurve_emd = compute_log_weighted_emd(Y, oof_pred)
print(f"OOF EMD - LOO mean-training-curve baseline (NO images/features used): {np.mean(meancurve_emd):.4f}")
print(f"  min={meancurve_emd.min():.4f}  max={meancurve_emd.max():.4f}  std={meancurve_emd.std():.4f}")

# Also report the full-24-sample mean curve (what would actually be
# submitted for every test row under this baseline) and the training
# curves' spread at each diameter, for context.
full_mean_curve = Y.mean(axis=0)
full_std_curve = Y.std(axis=0)
print("\nFull-sample (n=24) mean curve (this is what a mean-curve submission would predict for every test row):")
for col, m, s in zip(TARGET_COLUMNS, full_mean_curve, full_std_curve):
    print(f"  {col:>8} mm: mean={m:6.2f}  std={s:6.2f}")

# Trivial equal-fraction baseline for reference (organizer-given, never submit).
TRIVIAL = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.00])
trivial_emd = compute_log_weighted_emd(Y, np.tile(TRIVIAL, (n, 1)))
print(f"\nTrivial equal-fraction baseline EMD on training labels: {np.mean(trivial_emd):.4f}")

print(f"\nTotal runtime: {time.time() - t0:.1f}s")
print("=== DIAGNOSTIC COMPLETE ===")
