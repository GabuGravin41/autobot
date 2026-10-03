# Soil Grain Size from Photos (EU project GRID)

Kaggle ref: `soil-grain-size-from-photos` -- "Predicting Soil Grain Size Distributions from Images"

## Task
Predict, from photos of a soil sample's surface, the **cumulative grain size mass
fraction (%)** at 11 fixed diameters (mm), per DIN EN ISO 14688-1:

```
0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20, 63, 200
```

## Metric
Log-weighted Earth Mover's Distance (Wasserstein-1) between predicted and true
cumulative curves on a log-diameter scale, averaged across samples. Lower is
better, range `[0, 500]`. Final leaderboard = `0.30 * public + 0.70 * private`.

Implementation used throughout this project's `main.py` scripts:
```python
log_d = np.log10(SUPPORT_DIAMETERS)          # 11 points
w = np.diff(log_d)                            # 10 interval widths
err = np.abs(y_true - y_pred)
interval_err = (err[:, :-1] + err[:, 1:]) / 2  # trapezoid avg per interval
emd = (interval_err * w).sum(axis=1)          # per-sample score
```

## Trivial baseline (organizer-given, for calibration only -- never submit)
Equal ~9.09% per bin:
```
9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.00
```

## Submission format
CSV, header = `sample_id` + the 11 diameter columns **exactly as spelled in the
competition's own `sample_submission.csv`** (never hand-typed -- see Decision 2
in THINKING_AND_DECISIONS.md for why this matters). Structural rules enforced
by the organizers (auto-reject on failure):
- all 11 values in `[0, 100]`
- non-decreasing across the 11 columns
- last column (200 mm) exactly `100.00`
- every test `sample_id` present exactly once, no extras/duplicates

## Data (lives only on Kaggle -- never pulled to this machine)
- `Training-All_Photos_without_H374/` -- training photos (~25 samples, multiple
  images/angles each)
- `Test_All_Photos/` -- test photos
- `Training_labels_without_H374.csv` -- ground-truth cumulative curves per
  training `sample_id`
- `ppm.csv` -- pixels-per-millimeter calibration per camera/phone model (lets
  features be computed in real physical units instead of raw pixels)
- `sample_submission.csv` -- exact submission shape/column names/id list

Small training pool (~25 labeled samples) is the central modeling constraint:
this is a small-sample regime, not a big-data CV problem. Heavy unconstrained
deep learning will overfit immediately; regularized models and physically
grounded, low-dimensional features matter more than raw model capacity.

## Experiment log
See `THINKING_AND_DECISIONS.md` for the numbered decision/experiment log.

## Workspace layout
- `exp1_public_baseline/` -- adapted from the best-scoring public kernel found
  in Phase 0 discovery (attribution inside `main.py`). Validated OOF EMD
  58.29 (see THINKING_AND_DECISIONS.md Section 6).
- `exp2_pca_pls_loo_blend/` -- PCA-compressed deep features + PLS + LOO CV +
  OOF-searched 3-way blend. Validated OOF EMD 46.94; real public LB score
  80.96495 -- a severe OOF/LB gap, root-caused in Section 11.
- `diag1_test_image_audit/` -- diagnostic-only kernel (no model fit) that
  audits every train/test image's filename->sample_id match and detected
  camera; found 100% train/test camera-model disjointness (Section 11).
- `diag2_meancurve_baseline/` -- diagnostic-only kernel (no images used)
  checking a pure LOO mean-training-curve baseline, to rule out
  "photo features are net-harmful" as an explanation (Section 12).
- `exp3_camera_robust_simple/` -- drops the ImageNet ResNet-34 block
  entirely, fixes a confirmed camera-detection bug, uses a fixed
  (not OOF-searched) 50/50 Ridge+PLS blend on the 24 deterministic CV
  features only. Validated OOF EMD 42.72; real public LB score 80.65489
  -- **current best**, though only marginally ahead of exp2 (Section 12
  explains why the fix's real-world impact was smaller than expected).
- `public_nb/` -- the original public notebook, archived unmodified
- `output_exp1/`, `output_exp2/`, `output_exp3/`, `output_diag1/`,
  `output_diag2/` -- pulled kernel logs + validated `submission.csv` per
  experiment
- Later iterations (`exp4_...` etc.) and open ideas: see
  THINKING_AND_DECISIONS.md Section 12's "Ideas not tried in exp3".
