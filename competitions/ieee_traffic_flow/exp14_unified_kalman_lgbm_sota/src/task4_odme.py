"""Task 4: Origin-Destination Matrix Estimation (ODME) — v2 calibrated.

v2 fix (see EXPERIMENT Bug Fix 3):
- Anchor DIRECTLY to weak prior wp, fallback to base ONLY on NaN.
- v1 diluted prior (0.95*wp+0.05*base) and replaced legit 0.0 flows via
  np.where(wp>0, wp, base) — corrupted S_od/S_attr destination distribution.
- Solver: regularized NNLS lambda=0.05 (calibrated to official benchmark setting).
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from config import (
    PANELS,
    DEFAULT_REG_LAMBDA,
)


def solve_path_flow(
    A: np.ndarray,
    counts: np.ndarray,
    anchor: np.ndarray,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    use_weights: bool = False,
) -> np.ndarray:
    """Solve min ||A @ f - counts||^2 + lambda * ||f - anchor||^2 s.t. f >= 0.

    Step 6: inverse-variance weighting W_ii = 1/max(c_i, 50) aligns high-volume
    mainline corridors without sacrificing minor ramp flows. Set use_weights=False
    for plain NNLS.
    ABLATED 2026-09-22: weighted shipped unvalidated in v4 (LB 0.6789). Default
    OFF until validated; plain NNLS = v2 proven.
    """
    if use_weights:
        w = 1.0 / np.maximum(counts, 50.0)
        sqw = np.sqrt(w)
        aa = np.vstack([sqw[:, None] * A, np.sqrt(reg_lambda) * np.eye(A.shape[1])])
        bb = np.concatenate([sqw * counts, np.sqrt(reg_lambda) * anchor])
    else:
        aa = np.vstack([A, np.sqrt(reg_lambda) * np.eye(A.shape[1])])
        bb = np.concatenate([counts, np.sqrt(reg_lambda) * anchor])
    f, _ = nnls(aa, bb)
    return np.maximum(f, 0.0)


def solve_panel_odme(
    panel: str,
    panel_dir: Path,
    release_root: Path,
    splits: list[str],
    reg_lambda: float = DEFAULT_REG_LAMBDA
) -> pd.DataFrame:
    """Solve ODME for a single corridor across requested splits."""
    network = panel_dir / "network"
    paths_csv = network / "path_set.csv"
    incidence_csv = network / "path_link_incidence.csv"
    base_od_csv = network / "base_od.csv"
    
    if not (paths_csv.exists() and incidence_csv.exists()):
        return pd.DataFrame()

    paths = pd.read_csv(paths_csv)
    paths["path_id"] = paths["path_id"].astype(str)
    path_ids = paths["path_id"].tolist()
    path_index = {x: i for i, x in enumerate(path_ids)}

    incidence = pd.read_csv(incidence_csv)
    incidence["path_id"] = incidence["path_id"].astype(str)
    incidence["link_id"] = incidence["link_id"].astype(str)
    link_ids = incidence["link_id"].drop_duplicates().tolist()
    link_index = {x: i for i, x in enumerate(link_ids)}

    rr = incidence["link_id"].map(link_index).to_numpy(dtype=np.int64)
    cc = incidence["path_id"].map(path_index).to_numpy(dtype=np.int64)
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    A[rr, cc] = 1.0

    base_values = np.zeros(len(path_ids), dtype=float)
    if base_od_csv.exists():
        base_od = pd.read_csv(base_od_csv)
        base_od["path_id"] = base_od["path_id"].astype(str)
        merged_base = paths[["path_id", "origin_zone", "destination_zone"]].merge(
            base_od[["path_id", "base_flow"]], on="path_id", how="left"
        )
        merged_base = merged_base.set_index("path_id").reindex(path_ids).reset_index()
        base_values = pd.to_numeric(merged_base["base_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    records = []
    for split in splits:
        counts_path = release_root / "task4" / panel / split / "synthetic_link_counts.csv"
        prior_path = release_root / "task4" / panel / split / "synthetic_weak_prior.csv"

        if not counts_path.exists():
            continue

        counts_df = pd.read_csv(counts_path, dtype={"link_id": str})
        val_counts = counts_df.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(dtype=float)
        observed = set(counts_df["link_id"])
        measured = np.array([l in observed for l in link_ids], dtype=bool)

        anchor = base_values.copy()
        dep_time = "PUBLIC-TRAIN-PM"
        if prior_path.exists():
            pr = pd.read_csv(prior_path, dtype={"path_id": str})
            if len(pr) and "path_flow" in pr.columns:
                pr["path_id"] = pr["path_id"].astype(str)
                wp = pr.set_index("path_id").reindex(path_ids)["path_flow"].to_numpy(dtype=float)
                # v2: preserve true zero-demand paths; fallback to base ONLY on NaN
                # (v1 bug: np.nan_to_num(nan=0) + 0.95*wp+0.05*base overwrote zeros)
                wp = np.where(np.isnan(wp), base_values, wp)
                if np.sum(np.abs(wp)) > 0:
                    anchor = wp
            if len(pr) and "departure_time" in pr.columns:
                dep_time = str(pr["departure_time"].iloc[0])

        solved = solve_path_flow(A[measured], val_counts[measured], anchor, reg_lambda=reg_lambda)

        ps = paths[["path_id", "origin_zone", "destination_zone"]].copy()
        ps["panel"] = panel
        ps["departure_time"] = dep_time
        ps["path_flow"] = solved
        records.append(ps[["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]])

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def run_task4(
    release_root: Path,
    splits: list[str],
    out_path: Path,
    panels: list[str] | None = None,
    reg_lambda: float = DEFAULT_REG_LAMBDA
) -> int:
    """Execute Task 4 ODME across all corridors and write results to out_path."""
    if panels is None:
        panels = PANELS
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    results = []
    print(f"Starting Task 4 ODME path flow estimation (lambda={reg_lambda})...")
    for panel in panels:
        p_dir = release_root / "corridors" / panel
        p_df = solve_panel_odme(panel, p_dir, release_root, splits, reg_lambda=reg_lambda)
        if not p_df.empty:
            results.append(p_df)
            print(f"[{panel}] ODME solved: {len(p_df):,} entries.")

    if not results:
        pd.DataFrame(columns=["panel", "departure_time", "path_id", "path_flow"]).to_csv(out_path, index=False)
        return 0

    out_df = pd.concat(results, ignore_index=True)
    out_df.to_csv(out_path, index=False)
    print(f"Task 4 Complete: {len(out_df):,} total entries exported to {out_path} in {time.time()-t0:.1f}s")
    return len(out_df)
