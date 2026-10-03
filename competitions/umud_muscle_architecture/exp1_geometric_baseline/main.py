"""
UMUD Challenge — Exp1: no-train, CPU-only geometric baseline.

Predicts pennation angle (PA, deg), fascicle length (FL, mm) and muscle
thickness (MT, mm) directly from each test ultrasound image using classical
image processing — no training data, no GPU, no learned model:

  1. Detect the on-image ruler/tick-mark scale (pixels-per-cm) and the real
     image bounding box (cropping out the equipment's UI chrome). The test
     set mixes several ultrasound-machine output formats (different image
     shapes, tick-mark styles/positions), so this is format-specific.
  2. Locate the two aponeurosis bands as the two brightest bands in the
     vertical (depth-axis) brightness profile -> muscle thickness (MT) is
     the distance between them.
  3. Estimate fascicle pennation angle by block-matching short vertical
     strips of tissue between two x-positions inside the muscle belly and
     finding the vertical offset (dy) that maximizes correlation -> that
     offset's angle is PA.
  4. Fascicle length (FL) = MT / sin(PA) (a straight-line chord across the
     muscle belly at the fascicle's angle).

Attribution: the calibration table (which image shape/tick-pattern maps to
which pixels-per-cm and crop box) and the overall measurement approach are
adapted from the public baseline notebook `ambrosm/umud-quick-and-dirty`
(Kaggle, MIT-style "fork and build on this" baseline, 36 votes at time of
writing) — reimplemented here with defensive per-image error handling (the
original raises `AssertionError` and dies on any image it doesn't recognize;
this version instead falls back to a population-median prediction so one
odd image can't sink the whole submission) and explicit sample_submission
delimiter/column matching. No training-set images, masks or leaderboard
feedback were used to derive or tune these constants — see
THINKING_AND_DECISIONS.md for why: the UMUD test set is public and fixed,
and several of the top-voted/top-scoring public kernels turned out to embed
literal frozen/hand-tuned prediction anchors (LB-probed answer sheets, not
methodology) rather than a reproducible measurement pipeline. This baseline
deliberately avoids that path.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Locate the competition data directory (mount path varies across Kaggle
# environments / dataset versions).
# ---------------------------------------------------------------------------

COMPETITION_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"


def _find_competition_dir() -> str:
    candidates = [
        f"/kaggle/input/{COMPETITION_SLUG}",
        f"/kaggle/input/competitions/{COMPETITION_SLUG}",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # Fall back to searching for the known test-set folder anywhere under
    # /kaggle/input.
    hits = glob.glob("/kaggle/input/**/test_images_v2", recursive=True)
    if hits:
        return str(Path(hits[0]).parent)
    raise FileNotFoundError(
        f"Could not locate competition data dir for {COMPETITION_SLUG} under "
        f"/kaggle/input. Tried: {candidates + ['<recursive glob>']}"
    )


COMPETITION_DIR = _find_competition_dir()
TEST_GLOB = f"{COMPETITION_DIR}/test_images_v2/test_set_v2/IMG_*.*"

# Reasonable population-level fallback priors (used only when the per-image
# geometric pipeline fails outright — e.g. an image shape we don't
# recognize). These are rough physiological defaults, not fitted to any
# leaderboard feedback.
FALLBACK_PA_DEG = 16.0
FALLBACK_FL_MM = 85.0
FALLBACK_MT_MM = 21.0


def _sample_submission_path() -> str | None:
    hits = glob.glob(f"{COMPETITION_DIR}/**/sample_submission.csv", recursive=True)
    return hits[0] if hits else None


def _read_sample_submission() -> tuple[list[str], str, list[str]]:
    """Returns (column_names, delimiter, image_id order) exactly as the
    organizers published them — never hand-typed, per project convention."""
    path = _sample_submission_path()
    if path is None:
        # No sample available in this environment; fall back to the schema
        # documented in the competition overview (semicolon-delimited, as
        # observed in the real sample_submission.csv at research time).
        return ["image_id", "pa_deg", "fl_mm", "mt_mm"], ";", []
    text = Path(path).read_text(encoding="utf-8-sig")
    first_line = text.splitlines()[0]
    delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    columns = list(reader.fieldnames or [])
    ids = [row[columns[0]] for row in reader]
    return columns, delimiter, ids


# ---------------------------------------------------------------------------
# Step 1: per-image calibration (pixels-per-cm + real-image bounding box).
# ---------------------------------------------------------------------------


def _calibrate(a: np.ndarray, filetype: str) -> tuple[float, int, int, int, int]:
    """Returns (px_per_cm, left, top, right, bottom). Raises on an
    unrecognized image format so the caller can fall back gracefully.

    Branch selection is keyed off the file extension (`filetype`), not the
    pixel shape — the original ambrosm notebook this is adapted from does
    the same (`if filetype == 'png': ...`). An earlier version of this file
    used a shape heuristic (`shape[1] < 400`) instead, which was wrong: this
    dataset's `.png` test images (IMG_00252.png onward) have a full-canvas
    array shape of (800, 1200, 3) — the same raw shape as one of the TIFF
    branches — so the heuristic misrouted every PNG into the TIFF branch,
    which rejected them (58/309 fallbacks on the first real run). Keying off
    the actual file extension routes them into the PNG branch instead, whose
    tick-search logic doesn't assume a narrow raw array — it finds the real
    ~2.85cm-wide content region by searching inward from the horizontal
    center — so it calibrates correctly even on this wide canvas: second run
    was 309/309 with 0 fallbacks. See THINKING_AND_DECISIONS.md entry 5.
    """
    height, width = a.shape[:2]

    # PNG test images: variable scale (3-6cm high), 2.85cm wide, square px.
    if filetype == "png":
        col = a[:, 6].mean(axis=-1)
        first_tick = int(np.argmax(col > 50))
        second_minor_tick = 150 + int(np.argmax(a[150:, 6].mean(axis=-1) > 50))
        second_major_tick = 150 + int(np.argmax(a[150:, 9].mean(axis=-1) > 50))
        last_tick = len(a) - 1 - int(np.argmax(col[::-1] > 50))
        if (second_major_tick - first_tick) < 3 * (second_minor_tick - first_tick):
            px_per_cm = second_major_tick - first_tick
        else:
            px_per_cm = second_minor_tick - first_tick
        hw = a.shape[1] // 2
        w2 = int(np.argmin(a[:, hw:].sum(axis=(0, 2))))
        l, t, r, b = hw - w2, first_tick, hw + w2, last_tick
        if not (px_per_cm > 0 and 2.6 < (r - l) / px_per_cm < 3.1):
            raise ValueError("png calibration out of expected range")
        return float(px_per_cm), l, t, r, b

    if a.shape == (800, 1200, 3):
        if (a[87, 1147:1157] == 175).all():
            first_tick = int(np.argmax(a[:, 1150].mean(axis=-1) > 50))
            second_major_tick = first_tick + 20 + int(
                np.argmax(a[first_tick + 20:, 1150].mean(axis=-1) > 50)
            )
            last_tick = len(a) - 1 - int(np.argmax(a[:, 1150].mean(axis=-1)[::-1] > 50))
            n_ticks = round((last_tick - first_tick) / max(second_major_tick - first_tick, 1))
            crop_by_n_ticks = {
                7: (142, 91, 1058), 8: (163, 91, 1037), 9: (211, 91, 989),
                10: (249, 91, 951), 11: (282, 91, 918), 12: (308, 91, 892),
                13: (331, 91, 869), 14: (349, 91, 851),
            }
            if n_ticks <= 14 and n_ticks in crop_by_n_ticks:
                px_per_cm = (last_tick - first_tick) / n_ticks * 2
                l, t, r = crop_by_n_ticks[n_ticks]
                b = last_tick
            elif n_ticks == 15:
                px_per_cm = (last_tick - first_tick) / 3
                l, t, r, b = 142, 91, 1058, last_tick
            else:
                raise ValueError(f"unexpected tick count {n_ticks}")
            if not (px_per_cm > 0 and 5.6 < (r - l) / px_per_cm < 5.9):
                raise ValueError("tif(800x1200) right-tick calibration out of range")
            return float(px_per_cm), l, t, r, b
        elif (a[42, 67:74, 0] > 115).all():
            px_per_cm = (783 - 42) / 5
            return float(px_per_cm), 171, 42, 1029, 798
        raise ValueError("unrecognized 800x1200 tick layout")

    if a.shape == (644, 1088, 3):
        px_per_cm = 630.5 / 5
        return float(px_per_cm), 140, 0, 947, 643

    if a.shape[0] in (512, 513):
        for x0, x1 in ((49, 438), (52, 441), (53, 442)):
            if a[-5, x0, 0] == 168 and a[-5, x1, 0] == 168:
                px_per_cm = (442 - 53) / 5
                return float(px_per_cm), 0, 0, width, height - 10
        raise ValueError("unrecognized 512/513-row tick layout")

    if a.shape[0] == 853 and a.ndim == 2:
        for x0, x1 in ((100, 934), (44, 879)):
            if a[-5, x0] == 170 and a[-5, x1] == 170:
                px_per_cm = (x1 - x0) / 5
                return float(px_per_cm), 0, 0, width, height
        raise ValueError("unrecognized 853-row tick layout")

    raise ValueError(f"unknown image format, shape={a.shape}")


# ---------------------------------------------------------------------------
# Step 2 + 3: aponeurosis depths (-> MT) and fascicle angle (-> PA), then FL.
# ---------------------------------------------------------------------------


def _measure(a_full: np.ndarray, l: int, t: int, r: int, b: int, px_per_cm: float):
    b_img = a_full[t:b, l:r]
    if b_img.ndim == 3:
        b_img = b_img.mean(axis=-1)
    mm_per_px = 10.0 / px_per_cm

    # Vertical brightness profile -> two brightest bands = aponeuroses.
    profile = savgol_filter(b_img.mean(axis=1), 31, 3)
    hi = min(int(2.0 * px_per_cm), len(profile) - int(px_per_cm))
    if hi <= 0:
        raise ValueError("image too shallow for calibrated window")
    superficial_px = int(np.argmax(profile[:hi]))
    min_deep = superficial_px + int(px_per_cm)
    max_deep = superficial_px + 5 * int(px_per_cm)
    deep_px = min_deep + int(np.argmax(profile[min_deep:max_deep]))
    mt_px = deep_px - superficial_px
    if mt_px <= 0:
        raise ValueError("non-positive muscle thickness in pixels")

    # Block-match strips across x to find the fascicle slope.
    middle_px = (superficial_px + deep_px) // 2
    dx = 25
    half_len = min(50, mt_px // 2 - dx)
    if half_len < 5:
        raise ValueError("muscle belly too thin for angle block-matching")

    best_dys: list[float] = []
    for x in range(0, b_img.shape[1] - dx, 25):
        strip_left = b_img[middle_px - half_len: middle_px + half_len, x]
        if len(strip_left) < 2 or strip_left.var() == 0:
            continue
        best_corr, best_dy = -np.inf, None
        for dy in range(-dx, dx):
            strip_right = b_img[middle_px - half_len + dy: middle_px + half_len + dy, x + dx]
            if len(strip_right) != len(strip_left):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr = np.corrcoef(strip_left, strip_right)[1, 0]
            if np.isfinite(corr) and corr > best_corr:
                best_corr, best_dy = corr, dy
        if best_dy is not None:
            best_dys.append(best_dy)

    if len(best_dys) < 5:
        raise ValueError("too few reliable block-matches for angle estimation")

    best_dy = float(np.median(best_dys))
    pa_rad = abs(math.atan(best_dy / dx))
    pa_deg = min(pa_rad / math.pi * 180.0, 45.0)
    if pa_deg >= 5:
        fl_px = mt_px / math.sin(pa_rad)
    else:
        pa_deg = 15.0
        fl_px = mt_px / 0.34

    mt_mm = mt_px * mm_per_px
    fl_mm = float(np.clip(fl_px * mm_per_px, 30, 200))
    return pa_deg, fl_mm, mt_mm


def predict_one(path: str) -> tuple[float, float, float]:
    filetype = path.rsplit(".", 1)[-1].lower()
    with Image.open(path) as img:
        a = np.asarray(img)
    px_per_cm, l, t, r, b = _calibrate(a, filetype)
    return _measure(a, l, t, r, b, px_per_cm)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    columns, delimiter, sample_ids = _read_sample_submission()
    test_paths = sorted(glob.glob(TEST_GLOB))
    if not test_paths:
        raise FileNotFoundError(f"No test images found at {TEST_GLOB}")

    results: dict[str, tuple[float, float, float]] = {}
    n_fallback = 0
    for path in test_paths:
        image_id = os.path.basename(path)
        try:
            results[image_id] = predict_one(path)
        except Exception as exc:  # noqa: BLE001 - one bad image must not sink the run
            print(f"[warn] {image_id}: geometric pipeline failed ({exc}); using fallback prior")
            results[image_id] = (FALLBACK_PA_DEG, FALLBACK_FL_MM, FALLBACK_MT_MM)
            n_fallback += 1

    # Prefer the sample submission's own row order/id set, but only when it
    # actually enumerates every test image: Kaggle's real
    # sample_submission.csv for this competition is a 2-row *format example*
    # (IMG_00001/IMG_00002), not the full 309-row manifest — using it
    # unconditionally silently truncated the first real run of this kernel
    # to 2 output rows (caught by post-run validation; see
    # THINKING_AND_DECISIONS.md entry 5). Trust it only when its row count
    # matches the number of test images actually found.
    glob_ids = [os.path.basename(p) for p in test_paths]
    ordered_ids = sample_ids if len(sample_ids) == len(glob_ids) else glob_ids
    if sample_ids and len(sample_ids) != len(glob_ids):
        print(f"[info] sample_submission.csv has {len(sample_ids)} rows but "
              f"{len(glob_ids)} test images were found on disk; using the "
              f"on-disk image list as the row manifest instead (the sample "
              f"file is a format example, not the full manifest).")
    missing = [i for i in ordered_ids if i not in results]
    for image_id in missing:
        print(f"[warn] {image_id}: listed in sample_submission but no test image found; using fallback prior")
        results[image_id] = (FALLBACK_PA_DEG, FALLBACK_FL_MM, FALLBACK_MT_MM)
        n_fallback += 1

    out_path = Path("/kaggle/working/submission.csv")
    if not out_path.parent.exists():
        out_path = Path("submission.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter, lineterminator="\n")
        writer.writerow(columns)
        for image_id in ordered_ids:
            pa_deg, fl_mm, mt_mm = results[image_id]
            writer.writerow([image_id, f"{pa_deg:.3f}", f"{fl_mm:.3f}", f"{mt_mm:.3f}"])

    print(f"Wrote {len(ordered_ids)} rows to {out_path} (delimiter={delimiter!r}, columns={columns})")
    print(f"Geometric pipeline succeeded on {len(ordered_ids) - n_fallback}/{len(ordered_ids)} images "
          f"({n_fallback} used the population-median fallback).")


if __name__ == "__main__":
    main()
