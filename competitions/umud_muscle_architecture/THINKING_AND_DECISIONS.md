# UMUD Challenge: Engineering Decisions, Analysis, & Experiment Log

---

## 1. Capacity check & task framing (2026-09-24)
This is an explicitly lower-priority/opportunistic task: work it only with
spare GPU/CPU capacity, don't contend with other agents' higher-priority
Kaggle work. At dispatch time:
- `kaggle kernels status daltongabrielomondi/autobot-gemma4-exp1-baseline-zeroshot`
  → `RUNNING` (1 GPU slot).
- `~/.autobot/kaggle_jobs.json` listed `autobot-biohub-exp2-geometric-leaf-prune`
  as `status: "running", hardware: "gpu"` — but a live `kaggle kernels status`
  check on that kernel returned `COMPLETE`. The ledger's cached status was
  stale (this kernel had finished since the last poll). Conversely, the
  ledger had `autobot-gemma4-exp1-baseline-zeroshot` at `status: "error"`
  (a terminal status, excluded from `active_hardware_count`), while the live
  API said it was actually `RUNNING`.
  - **Net effect / ledger staleness gotcha worth feeding back into AutoBot**:
    `active_hardware_count()` in `kaggle_watchdog.py` only counts jobs whose
    *cached* status is non-terminal. Two independent staleness errors
    happened to cancel out here (one job wrongly counted as active, another
    wrongly excluded, real total unchanged at 1/2 GPU slots), but that's
    luck, not a property of the mechanism — a ledger that's stale in only
    one direction would either falsely block a legitimate dispatch or, worse,
    silently let an agent oversubscribe real Kaggle capacity because
    `check_capacity()` never makes a live API call, only reads the ledger.
    `poll_pending()` exists to refresh this but apparently isn't being
    called often/reliably enough across concurrent agents. Consider: either
    have `push_kernel`'s capacity check do a live status refresh for
    ledger entries older than some TTL, or make `poll_pending` a
    pre-condition of `check_capacity` rather than a separate scheduled
    sweep.
- Given this task doesn't need GPU at all (see entry 4 — exp1 is classical
  CPU image processing, no training), the GPU contention question turned
  out to be moot for exp1. Decision: proceed CPU-only, which was uncontended
  (traffic exp4/5/6 CPU jobs were all already `complete` in the ledger).
- Disk: this machine has ~1.5GB free (confirmed via `shutil.disk_usage`).
  The competition's raw data is thousands of large multi-hundred-KB TIFFs
  (7,930 files across apo/fasc image+mask sets plus the 309-image test set —
  confirmed via `competition_list_files` pagination). **No bulk download to
  this machine** — all image access happens inside the Kaggle kernel. Only
  tiny metadata (`sample_submission.csv`, 104 bytes) and public kernel
  `.ipynb` source files were pulled locally for research.

## 2. Phase 0 SOTA discovery — and a red flag (2026-09-24)
`kaggle kernels list --competition ... --sort-by voteCount` /
`--sort-by scoreDescending`, then pulled the top candidates with
`kaggle kernels pull <ref> -p <dir> -m`.

**Important finding: the best-scoring public kernels are not reproducible
methodology — they're frozen, hand-tuned leaderboard-probed answer sheets.**
- `phuongncn/lb-0-76704-no-train-anatomy-calibrated-dltrack` (LB 0.76704):
  embeds a gzip+base64-encoded "C130 anchor" prediction table and applies a
  documented sequence of scalar calibration constants (`Q_MEAN`, `Q_ROBUST`,
  `LOW_TAIL_GATE_MM`, etc., 15+ decimal digits each) whose own changelog
  ("C130 → C137") shows they were derived through a multi-submission
  LB-probing search, not measured from images. The notebook is explicit that
  it embeds "only a scored prediction anchor... not competition images or
  labels" and fails closed unless the SHA-256 of its frozen anchor matches —
  i.e. it's designed to deterministically reproduce one specific scored
  submission, not to run a measurement pipeline on new images.
- `dreaddevelopment/vera-seg-centerline-mt-correction-lb-0-45134` (LB
  0.45134 — a *better* score, since lower is better here): the entire
  "submission" is a **literal hardcoded CSV string** pasted into the
  notebook (309 rows, `image_id,pa_deg,fl_mm,mt_mm`), not a segmentation
  model that runs on the mounted test images at all despite the title
  ("Seg-Centerline MT Correction"). This lines up with competition
  Discussion topic 739660, "Reproducibility and Reusing Old Submissions in
  Final Ensemble" — the 309-image test set is public and fixed for the
  whole competition, so repeated leaderboard probing over months has let
  some participants converge on (and now publicly share) values that are
  effectively memorized answers for this specific test set, not a
  transferable method.
- **Decision: do not fork, extend, or otherwise build on either of these.**
  Copying a hardcoded/leaked answer table would produce a great local number
  with zero methodological content, doesn't generalize (a private
  re-scoring or any test-set change breaks it completely), and isn't in the
  spirit of the exercise. Also relevant: Discussion topic 690868 ("Test data
  leakage | manual labeling") — organizers acknowledge participants could
  manually label the 309 public test images and explicitly require declaring
  that as external-data usage; the frozen-anchor kernel's provenance is
  consistent with exactly this kind of test-set-specific tuning.
- The genuinely reusable public baseline is
  `ambrosm/umud-quick-and-dirty` (36 votes, no training, no masks, no
  leaderboard-tuned constants) — an honest classical-CV pipeline: detect
  each test image's on-image ruler scale from its tick marks (equipment
  formats vary — the notebook branches on exact image shape/pixel patterns),
  find the two aponeurosis bands as the brightest bands in the vertical
  brightness profile (-> MT), and block-match tissue strips across the
  muscle belly to get the fascicle slope (-> PA), then FL = MT / sin(PA).
  Its own docstring explicitly invites forking/building on it ("whatever
  other algorithm you're implementing, you'll need px_per_cm... take these
  values from here"). This is what exp1 adapts. Its reported LB (1.33851,
  mentioned in the frozen-anchor kernel's own comparison table) is far worse
  in absolute terms than the leaked-anchor scores above, but it's the only
  top-listed kernel that's actually measuring the images.

## 3. Submission schema gotcha: semicolon, not comma (2026-09-24)
Downloaded the real `sample_submission.csv` directly (104 bytes — cheap,
worth doing rather than trusting a hand-typed guess):
```
image_id;pa_deg;fl_mm;mt_mm
IMG_00001.tif;17.334;79.423;21.778
IMG_00002.tif;12.876;69.424;15.478
```
It's **semicolon-delimited with a UTF-8 BOM**, not the comma-delimited
format most public kernels (including AmbrosM's, via `pandas.to_csv`
defaults) assume. Whether Kaggle's grader is lenient about this is unclear
without a real submission — the honest reproduction of the frozen-anchor
kernel (entry 2) explicitly auto-detects the delimiter by counting `;` vs
`,` in the sample file's first line rather than assuming either, which is a
good defensive pattern and the one exp1 follows. exp1's `main.py` reads the
real `sample_submission.csv` from the mounted dataset at runtime and copies
its exact column names, delimiter, and row order — it never hand-types the
schema.

Also confirmed via `competition_list_files` pagination (7,930 files total):
`test_images_v2/test_set_v2/` has exactly 309 images, and **not all are
`.tif`** — the last ~58 (`IMG_00252.png` onward, per the hardcoded-anchor
kernel's embedded IDs) are `.png`, which is exactly why AmbrosM's calibration
table has a separate PNG branch with different tick-mark geometry than the
TIFF branches.

## 4. Duplicate training data (2026-09-24)
Per the task brief's instruction to check before training on it: confirmed
via `api.competition_list_topics(...)` / `competition_list_topic_messages(...)`
(topic 740356). Full detail captured in `README.md`'s "Known data-quality
issue" section — short version: 670 exact duplicate image-mask pairs in the
2,761-image fascicle training set (organizer-confirmed, hidden test set
unaffected, drop one copy per group before any CV split). **Not actually
relevant to exp1**, since exp1 is a zero-training geometric pipeline that
never touches the train/mask folders at all — flagging this for whichever
future experiment first trains on `fasc_imgs_v1`/`fasc_masks_v1`.

## 5. Experiment 1: geometric calibrated baseline (2026-09-24)
- **Kernel**: `daltongabrielomondi/autobot-umud-exp1-geometric-baseline`
- **Path**: `competitions/umud_muscle_architecture/exp1_geometric_baseline/`
- **Hardware**: CPU only (`enable_gpu: false`) — appropriate since the
  method is pure NumPy/SciPy/PIL image processing, runtime is reported as
  "seconds, not minutes" by other kernels doing similar per-image work over
  the same 309 images.
- **Approach**: adapted from `ambrosm/umud-quick-and-dirty` (see entry 2 for
  attribution and why this baseline was chosen over the higher-scoring but
  non-reproducible public kernels). Reimplemented (not copy-pasted) with two
  deliberate changes:
  1. **Defensive per-image fallback.** The original notebook `assert`s on
     every calibration branch and crashes the whole run on the first
     unrecognized image. exp1 wraps each image's calibration+measurement in
     `try/except` and falls back to a fixed population-level prior
     (PA≈16°, FL≈85mm, MT≈21mm — rough physiological defaults, not fitted to
     any leaderboard feedback) so one odd image can't zero out the whole
     submission. It logs a `[warn]` line and a final success-rate count so a
     high fallback rate would be visible as a signal to fix the calibration
     table rather than silently degrading.
  2. **Schema fidelity.** Reads the real `sample_submission.csv` at runtime
     for exact column names, delimiter (`;`), and `image_id` row order,
     instead of assuming a comma-separated schema (see entry 3).
- **Validation performed**: `py_compile` clean locally (no local image data
  available to test against — see entry 1 on disk space — so the geometric
  logic itself was unverified against real pixels until the kernel actually
  ran on Kaggle).

### Run 1 (version 1 push) — two bugs found by actually running it
Dispatched via `Kaggle.push_kernel(...)`. `push_kernel`'s own liveness check
reported `LIVENESS CHECK FAILED at t+30s: status=error` (no error log) —
this turned out to be a transient Kaggle-infra blip: polling
`kaggle kernels status` directly a minute later showed `RUNNING`, then
`COMPLETE` a few minutes after that. **Worth feeding back**: the same
running→error→running flicker shows up in this account's job history for
*other* kernels too (see entry 1's ledger dump), so this looks like a
recurring, not one-off, Kaggle status-API quirk — `push_kernel`'s liveness
check treating one transient `error` read as fatal (vs. re-polling past the
grace window) would cause false-negative aborts on perfectly fine runs if a
caller trusted it literally instead of following up with a manual status
check, which is what happened here.

Pulled output and validated: **only 2 rows**, not 309. Root causes (found by
reading the kernel's own log output, not guesswork):
1. **Critical bug**: `main.py` was pulling the row manifest from
   `sample_submission.csv`, but the real one Kaggle serves for this
   competition is a **2-row format example** (`IMG_00001`, `IMG_00002`),
   not the 309-row full manifest (confirmed directly — see entry 3). Taking
   it as authoritative silently truncated the submission to 2 rows. Fixed:
   now compares its row count against the actual on-disk test-image glob
   count and only trusts the sample file's ID list when they match; falls
   back to the glob-derived list (with a logged `[info]`) otherwise. This is
   a **repo-wide risk**, not specific to this competition — any exp that
   reads `sample_submission.csv` as its row manifest should validate its row
   count against the real test set size rather than assuming it's complete.
2. 58/309 images (`IMG_00252.png` onward) fell back to the fixed prior with
   `unrecognized 800x1200 tick layout`. Root cause: this file's `_calibrate`
   picked the PNG-specific calibration branch using a **shape heuristic**
   (`shape[1] < 400`, assuming PNGs are narrow, per the source notebook's
   comment "images are 2.85 cm wide"). This dataset's actual PNG test
   images have a full raw array shape of **(800, 1200, 3)** — the same as
   one of the TIFF branches — so the heuristic misrouted every PNG into the
   TIFF branch, which correctly rejected them (they don't match either known
   TIFF tick pattern). Fixed: branch on the real file extension instead
   (matching the source notebook's actual `if filetype == 'png'` logic,
   which this port had incorrectly translated into a shape guess).

### Run 2 (version 2 push) — clean
Both fixes applied, re-pushed, `RUNNING` → `COMPLETE` in ~75s of kernel
time. Pulled `submission.csv` and validated locally:
- 309 data rows + 1 header, all 309 `image_id` values unique.
- Columns exactly `['image_id', 'pa_deg', 'fl_mm', 'mt_mm']`, `;`-delimited,
  matching the real `sample_submission.csv` schema.
- No NaN/non-finite/non-positive values in any of the three target columns.
- Ranges: `pa_deg` 5.7-27.5°, `fl_mm` 30-200mm (the pipeline's own physiological
  clip bounds), `mt_mm` 9.9-48.3mm — all physiologically plausible for
  skeletal muscle architecture.
- Kernel log: **"Geometric pipeline succeeded on 309/309 images (0 used the
  population-median fallback)"** — every single test image, including all
  58 PNGs, got a real geometric measurement once the routing bug was fixed.
- Output copied to `output_exp1/submission.csv` +
  `output_exp1/autobot-umud-exp1-geometric-baseline.log` for the record.

**exp1 validated and ready for the user's own review/submit decision** (per
task instructions, this agent does not call `kaggle competitions submit`).
No expected-LB-score claim is made — this baseline was deliberately built
without any leaderboard feedback loop (see entry 2), so its real score is
unknown until submitted.

### Next steps (not done here — opportunistic task, stopping at a validated exp1)
- exp2: derive real PA/FL/MT training labels from `apo_masks_v1`/
  `fasc_masks_v1` (aponeurosis-centerline separation, fascicle-mask line
  angle) and train a learned regressor (e.g. LightGBM on handcrafted
  features) as a second, independent signal to blend with/replace this
  geometric baseline — remembering to drop one copy per duplicate group
  (entry 4) before any CV split.
- Cross-check this baseline's PA/FL/MT distributions against the public
  `stpeteishii/umud-challenge-image-display` / `*-apo-mask-unet` /
  `*-fasc-mask-unet` kernels' visualizations as a sanity check, since no
  local image inspection was possible (entry 1, disk space).
