# ==============================================================================
# Autobot Soil Grain Size — Diagnostic 1: Test image / sample_id match audit
# Competition: soil-grain-size-from-photos (EU project GRID)
#
# NOT a modeling experiment. exp2 (PCA+PLS+LOO blend) validated at OOF EMD
# 46.94 but scored 80.96 on the real public leaderboard -- worse than every
# individual model's OOF number (Ridge 53.96, SVR 56.48, PLS 53.96), not
# just the blend. See THINKING_AND_DECISIONS.md Section 8 ("exp2 Real
# Leaderboard Result") for 4 hypotheses. This script targets hypothesis (3),
# the most actionable to check first: exp1/exp2's fuzzy "all tokens but
# one" filename->sample_id fallback was validated against *training*
# filenames only (0 fallback rows were needed there). If it silently
# mismaps a *test* image to the wrong sample_id (or the wrong camera/PPM),
# OOF would never catch it -- OOF never touches test files at all.
#
# This script reuses exp2's exact matching/camera-detection logic
# UNCHANGED (copy-pasted, not re-implemented) and adds instrumentation:
# for every train and test image, log which match pass succeeded
# (exact-substring / exact-cleaned / fuzzy) and the matched sample_id +
# camera. Then prints:
#   - full per-test-image match audit table (filename, sample_id, pass,
#     camera, ppm)
#   - per-test-sample_id image counts and pass-type breakdown
#   - per-camera image counts, train vs test (distribution-shift check,
#     hypothesis 4 / Section 9's already-flagged item)
# No model is fit, no submission is written. CPU only, no GPU needed.
# ==============================================================================
import os
import re
import glob
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

print("=== AUTOBOT SOIL DIAGNOSTIC 1: test image / sample_id match audit ===")


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

train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
ppm_df = pd.read_csv(PPM_PATH)
sample_sub_df = pd.read_csv(SAMPLE_SUB_PATH)

train_sample_list = train_labels_df["sample_id"].unique().tolist()
test_sample_list = sample_sub_df["sample_id"].unique().tolist()
print(f"Train sample_ids ({len(train_sample_list)}): {train_sample_list}")
print(f"Test sample_ids ({len(test_sample_list)}): {test_sample_list}")

# ------------------------------------------------------------------------------
# EXACT COPY of exp2's matching logic, instrumented to report which pass hit.
# ------------------------------------------------------------------------------
def _tokenize(s):
    return [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t]


def fuzzy_match_sample_id(filename, label_sample_ids):
    fn_clean = re.sub(r"[^a-z0-9]+", "", filename.lower())
    best_id, best_matched, best_total = None, -1, 0
    for sample_id in label_sample_ids:
        tokens = _tokenize(sample_id)
        if len(tokens) < 3:
            continue
        matched = sum(1 for t in tokens if t in fn_clean)
        if matched >= len(tokens) - 1 and matched > best_matched:
            best_matched, best_total, best_id = matched, len(tokens), sample_id
    return best_id, best_matched, best_total


def match_sample_id_instrumented(filename, label_sample_ids):
    fn_lower = filename.lower()
    for sample_id in label_sample_ids:
        if str(sample_id).lower() in fn_lower:
            return sample_id, "exact-substring", None, None
    cleaned_filename = fn_lower.replace(" ", "")
    for sample_id in label_sample_ids:
        cleaned_sid = str(sample_id).lower().replace(" ", "")
        if cleaned_sid in cleaned_filename or cleaned_filename in cleaned_sid:
            return sample_id, "exact-cleaned", None, None
    best_id, matched, total = fuzzy_match_sample_id(filename, label_sample_ids)
    if best_id is None:
        return None, "no-match", None, None
    return best_id, "fuzzy", matched, total


def detect_camera(filename, ppm_df):
    fn_lower = filename.lower().replace(" ", "")
    if "iphone14" in fn_lower:
        return "iPhone 14"
    elif "iphone16" in fn_lower:
        return "iPhone 16"
    elif "60fusion" in fn_lower or ("60" in fn_lower and "fusion" in fn_lower):
        return "Motorola Edge 60 fusion"
    elif "motorola" in fn_lower or "edge" in fn_lower:
        return "motorola edge 20"
    elif "samsung" in fn_lower or "a52" in fn_lower or "sm-a525f" in fn_lower:
        return "SM-A525F"
    else:
        for _, row in ppm_df.iterrows():
            cam_spec = str(row["camera"]).lower()
            phone_spec = str(row["phone"]).lower()
            if cam_spec in filename.lower() or phone_spec in filename.lower():
                return row["camera"]
        return "unknown"


def audit_directory(directory_path, label_sample_ids, split_name):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    all_files = []
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(str(directory_path), ext)))
    all_files.sort()

    records = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        sid, match_pass, matched_tok, total_tok = match_sample_id_instrumented(filename, label_sample_ids)
        camera = detect_camera(filename, ppm_df)
        ppm_row = ppm_df[ppm_df["camera"] == camera]
        if len(ppm_row) == 0:
            ppm_row = ppm_df[ppm_df["phone"] == camera]
        ppm_val = ppm_row["ppm"].values[0] if len(ppm_row) > 0 else ppm_df["ppm"].mean()
        records.append(
            {
                "split": split_name,
                "filename": filename,
                "sample_id": sid,
                "match_pass": match_pass,
                "fuzzy_matched_tok": matched_tok,
                "fuzzy_total_tok": total_tok,
                "camera": camera,
                "ppm": ppm_val,
            }
        )
    return pd.DataFrame(records)


print("\n--- Auditing TRAIN images ---")
train_audit = audit_directory(TRAIN_DIR, train_sample_list, "train")
print(f"Total train images found: {len(train_audit)}")
print("Train match-pass breakdown:")
print(train_audit["match_pass"].value_counts().to_string())
n_train_unmatched = (train_audit["sample_id"].isna()).sum()
print(f"Train images with NO sample_id match: {n_train_unmatched}")

print("\n--- Auditing TEST images ---")
test_audit = audit_directory(TEST_DIR, test_sample_list, "test")
print(f"Total test images found: {len(test_audit)}")
print("Test match-pass breakdown:")
print(test_audit["match_pass"].value_counts().to_string())
n_test_unmatched = (test_audit["sample_id"].isna()).sum()
print(f"Test images with NO sample_id match: {n_test_unmatched}")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 220)

print("\n--- FULL TEST IMAGE AUDIT TABLE (every test image, every match) ---")
print(test_audit[["filename", "sample_id", "match_pass", "fuzzy_matched_tok", "fuzzy_total_tok", "camera", "ppm"]].to_string())

print("\n--- Any FUZZY-matched test images (highest risk for a wrong mapping) ---")
fuzzy_rows = test_audit[test_audit["match_pass"] == "fuzzy"]
if len(fuzzy_rows) == 0:
    print("(none -- all test images matched via exact-substring or exact-cleaned)")
else:
    print(fuzzy_rows[["filename", "sample_id", "fuzzy_matched_tok", "fuzzy_total_tok", "camera"]].to_string())

print("\n--- Per test sample_id: image count + match-pass types ---")
for sid in test_sample_list:
    sub = test_audit[test_audit["sample_id"] == sid]
    passes = Counter(sub["match_pass"])
    cams = Counter(sub["camera"])
    print(f"  {sid!r}: n_images={len(sub)}, passes={dict(passes)}, cameras={dict(cams)}")

print("\n--- Any TEST sample_id with ZERO matched images (would hit the equal-fraction fallback row) ---")
matched_test_sids = set(test_audit["sample_id"].dropna())
missing_test_sids = [s for s in test_sample_list if s not in matched_test_sids]
print(f"Missing: {missing_test_sids if missing_test_sids else '(none)'}")

print("\n--- Camera/PPM distribution: TRAIN vs TEST (raw image counts) ---")
train_cam_counts = train_audit["camera"].value_counts()
test_cam_counts = test_audit["camera"].value_counts()
all_cams = sorted(set(train_cam_counts.index) | set(test_cam_counts.index))
print(f"{'camera':<28}{'train_n':>10}{'train_%':>10}{'test_n':>10}{'test_%':>10}")
for cam in all_cams:
    tr_n = int(train_cam_counts.get(cam, 0))
    te_n = int(test_cam_counts.get(cam, 0))
    tr_pct = 100.0 * tr_n / len(train_audit) if len(train_audit) else 0.0
    te_pct = 100.0 * te_n / len(test_audit) if len(test_audit) else 0.0
    print(f"{cam:<28}{tr_n:>10}{tr_pct:>9.1f}%{te_n:>10}{te_pct:>9.1f}%")

print("\n--- Camera/PPM distribution: TRAIN vs TEST (per unique SAMPLE, not per image) ---")
train_sample_cam = train_audit.dropna(subset=["sample_id"]).groupby("sample_id")["camera"].agg(lambda x: x.mode().iloc[0])
test_sample_cam = test_audit.dropna(subset=["sample_id"]).groupby("sample_id")["camera"].agg(lambda x: x.mode().iloc[0])
print("Train (per-sample dominant camera):")
print(train_sample_cam.value_counts().to_string())
print("Test (per-sample dominant camera):")
print(test_sample_cam.value_counts().to_string())

print("\n=== DIAGNOSTIC COMPLETE ===")
