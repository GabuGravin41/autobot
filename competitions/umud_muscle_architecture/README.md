# UMUD Challenge: Muscle Architecture in Ultrasound Data (University of Basel)

Kaggle ref: `umud-challenge-muscle-architecture-in-ultrasound-data`.

From a single B-mode ultrasound image of skeletal muscle, predict 3 values:
- **PA** — Pennation Angle (degrees)
- **FL** — Fascicle Length (mm)
- **MT** — Muscle Thickness (mm)

The test set includes some 5-frame video sequences (temporal stability angle);
exploiting that is optional, not required.

**Status**: opportunistic / lower priority. Work here only happens with spare
GPU/CPU capacity, per the user's instruction — see `THINKING_AND_DECISIONS.md`
entry 1.

## Metric
"UMUD Score" — normalized MAE across PA/FL/MT (each divided by a predefined
per-target tolerance so the three different units/scales contribute
comparably). **Lower is better.**

## Data layout (do not download locally — see entry 1)
| path (under the Kaggle dataset mount) | contents |
|---|---|
| `test_images_v2/test_set_v2/IMG_*.tif` / `.png` | 309 test images (the only thing exp1 uses) |
| `apo_imgs_v1/`, `apo_masks_v1/` | ~1048 aponeurosis image/mask pairs (train) |
| `fasc_imgs_v1/`, `fasc_masks_v1/` | 2761/2761 fascicle image/mask pairs (train) — **contains 670 exact duplicate pairs, see entry 2** |
| `sample_submission.csv` | exact schema/order — **semicolon-delimited**, not comma (verified by download, see entry 3) |

There is **no `train.csv` of PA/FL/MT ground truth** — only segmentation
masks. Any supervised approach needs to first derive PA/FL/MT labels from the
masks geometrically (aponeurosis centerline separation -> MT, fascicle mask
line angle relative to the aponeuroses -> PA, etc.) before it has something
to regress against.

## Submission format
CSV, semicolon-delimited, columns exactly `image_id;pa_deg;fl_mm;mt_mm` (per
the real `sample_submission.csv` — copy this at runtime, don't hand-type it;
some public kernels assume comma and may be wrong for the actual grader).
Every one of the 309 test `image_id`s exactly once.

## Known data-quality issue: duplicate training images
Confirmed by the organizers on the competition Discussion (topic 740356,
"670 exact duplicate image-mask pairs in the fascicle training set"):
- 2,761 fascicle image-mask records total; 670 exact duplicate groups (1,340
  records affected); 2,091 unique pairs after de-duplication.
- Duplicates are contiguous block remaps: `image_0371-0670 == image_0797-1096`
  (300 pairs), `image_0000-0230 == image_1097-1327` (231 pairs),
  `image_0232-0370 == image_1328-1466` (139 pairs).
- Organizers confirmed the **hidden test set has no overlap** with training
  data — this only inflates internal/local CV scores if duplicate pairs land
  on both sides of a split, not leaderboard scores.
- Recommendation from the organizers: drop one copy per duplicate group
  before building CV splits.

## Experiments
| exp | approach | hardware | status |
|---|---|---|---|
| `exp1_geometric_baseline` | no-train classical CV: ruler/tick-mark calibration -> aponeurosis depth profile -> block-matched fascicle angle -> PA/FL/MT | CPU | dispatched, see log |

See `THINKING_AND_DECISIONS.md` for the full numbered decision log, including
why exp1 deliberately avoids copying the public kernels with the best listed
leaderboard scores.
