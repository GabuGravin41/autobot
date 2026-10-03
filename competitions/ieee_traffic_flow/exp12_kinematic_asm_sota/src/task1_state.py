"""Task 1: Traffic State Reconstruction & Task 3 Physics Consistency (Exp 12).

Upgraded with the Treiber-Helbing Adaptive Smoothing Method (ASM):
1. Spatiotemporal wave propagation replaces naive symmetric interpolation:
   - Free-flow perturbations propagate downstream at c_free ~ +75 km/h.
   - Congested shockwaves propagate upstream at c_cong ~ -18 km/h.
   - Hyperbolic tangent nonlinear gate blends free and congested states based on local congestion level.
2. Physical floor guards:
   - Enforces 50 vph floor guard to prevent zero-flow disqualification penalties in Task 3 S_FD.
   - Restricts speed to [5.0, 1.05 * v_f].
3. Iterative conservation balance along corridor link topology for Task 3 S_LWR.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    PANELS,
    INTERVAL_MINUTES,
    SLOTS_PER_DAY,
    SLOTS_PER_WEEK,
)
from network_utils import (
    load_lane_counts,
    load_fd_params,
    load_ordered_links,
    get_lane_vector,
)

# Physics corrections: soft FD projection + iterative conservation balance across links
USE_PHYSICS_CORRECTIONS = True
try:
    from physics_corrections import (
        apply_physics_corrections,
        FD_ALPHA,
        CONSERVATION_ALPHA,
        CONSERVATION_PASSES,
    )
except ImportError:
    apply_physics_corrections = None  # type: ignore
    FD_ALPHA = 0.15
    CONSERVATION_ALPHA = 0.40
    CONSERVATION_PASSES = 5


def compute_time_slots(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute (weekday, time_of_day_slot) from timestamp column."""
    dt = pd.to_datetime(df["timestamp"], utc=True)
    weekday = dt.dt.weekday.to_numpy(dtype=np.int64)
    tod = (dt.dt.hour * 12 + dt.dt.minute // 5).to_numpy(dtype=np.int64)
    return weekday, tod


def learn_historical_profiles(panel_dir: Path) -> tuple[dict, dict]:
    """Learn historical (link x weekday x time_of_day) profiles from train unmasked states."""
    train_files = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
    if not train_files:
        pfiles = sorted(panel_dir.glob("**/mainline_states/**/*.parquet"))
        train_files = [f for f in pfiles if "train" in str(f) and "masked" not in str(f)]
        
    if not train_files:
        raise FileNotFoundError(f"No unmasked train states found under {panel_dir}")

    links_df = pd.read_csv(panel_dir / "network" / "links.csv", dtype={"link_id": str})
    link_ids = sorted(links_df["link_id"].unique())
    link_idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_ids)

    speed_sum = np.zeros((n_links, SLOTS_PER_WEEK), dtype=float)
    speed_cnt = np.zeros((n_links, SLOTS_PER_WEEK), dtype=int)
    flow_sum = np.zeros((n_links, SLOTS_PER_WEEK), dtype=float)
    flow_cnt = np.zeros((n_links, SLOTS_PER_WEEK), dtype=int)

    for path in train_files:
        df = pd.read_parquet(path)
        df["link_id"] = df["link_id"].astype(str)
        idx = df["link_id"].map(link_idx).fillna(-1).to_numpy(dtype=np.int64)
        valid_mask = idx >= 0
        if not np.any(valid_mask):
            continue

        weekday, tod = compute_time_slots(df)
        slot = weekday * SLOTS_PER_DAY + tod

        eligible = (
            pd.to_numeric(df["pct_observed"], errors="coerce").ge(75)
            & df["speed_kmh"].notna()
            & df["flow_vph"].notna()
        ).to_numpy()

        sp = pd.to_numeric(df["speed_kmh"], errors="coerce").to_numpy(dtype=float)
        fl = pd.to_numeric(df["flow_vph"], errors="coerce").to_numpy(dtype=float)

        v_sp = eligible & np.isfinite(sp)
        v_fl = eligible & np.isfinite(fl)

        np.add.at(speed_sum, (idx[v_sp], slot[v_sp]), sp[v_sp])
        np.add.at(speed_cnt, (idx[v_sp], slot[v_sp]), 1)
        np.add.at(flow_sum, (idx[v_fl], slot[v_fl]), fl[v_fl])
        np.add.at(flow_cnt, (idx[v_fl], slot[v_fl]), 1)

    speed_mean = np.zeros_like(speed_sum)
    flow_mean = np.zeros_like(flow_sum)
    np.divide(speed_sum, np.maximum(speed_cnt, 1), out=speed_mean, where=speed_cnt > 0)
    np.divide(flow_sum, np.maximum(flow_cnt, 1), out=flow_mean, where=flow_cnt > 0)
    speed_fb = np.divide(speed_sum.sum(axis=1), np.maximum(speed_cnt.sum(axis=1), 1))
    flow_fb = np.divide(flow_sum.sum(axis=1), np.maximum(flow_cnt.sum(axis=1), 1))

    return (
        {"link_ids": link_ids, "link_idx": link_idx, "mean": speed_mean, "fallback": speed_fb, "cnt": speed_cnt},
        {"link_ids": link_ids, "link_idx": link_idx, "mean": flow_mean, "fallback": flow_fb, "cnt": flow_cnt},
    )


def run_asm_grid(
    sp_grid: np.ndarray,
    fl_grid: np.ndarray,
    x_coords: np.ndarray,
    c_free: float = 75.0,
    c_cong: float = -18.0,
    sigma: float = 2.0,
    tau: float = 0.15,
    v_c: float = 55.0,
    delta_v: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute Treiber-Helbing Adaptive Smoothing Method on a single day's corridor grid."""
    n_links, n_slots = sp_grid.shape
    t_coords = np.arange(n_slots) * (5.0 / 60.0)  # Time in hours
    
    valid_sp = ~np.isnan(sp_grid)
    valid_fl = ~np.isnan(fl_grid)
    
    sp_out = np.copy(sp_grid)
    fl_out = np.copy(fl_grid)
    
    missing_r, missing_c = np.where(~valid_sp)
    
    for r, c in zip(missing_r, missing_c):
        x_tgt, t_tgt = x_coords[r], t_coords[c]
        
        # Local space-time neighborhood
        r_min, r_max = max(0, r - 6), min(n_links, r + 7)
        c_min, c_max = max(0, c - 9), min(n_slots, c + 10)
        
        sub_valid = valid_sp[r_min:r_max, c_min:c_max]
        if not np.any(sub_valid):
            continue
            
        dx = x_tgt - x_coords[r_min:r_max, None]
        dt = t_tgt - t_coords[None, c_min:c_max]
        
        # Downstream free wave kernel: information travels downstream along characteristic rays
        arg_free = -np.abs(dx) / sigma - np.abs(dt - dx / c_free) / tau
        w_free = np.exp(arg_free) * sub_valid
        
        # Upstream congested shockwave kernel: shockwaves propagate backward against traffic
        arg_cong = -np.abs(dx) / sigma - np.abs(dt - dx / c_cong) / tau
        w_cong = np.exp(arg_cong) * sub_valid
        
        sum_w_free = float(np.sum(w_free))
        sum_w_cong = float(np.sum(w_cong))
        
        if sum_w_free < 1e-5 and sum_w_cong < 1e-5:
            continue
            
        sub_sp = np.nan_to_num(sp_grid[r_min:r_max, c_min:c_max], nan=0.0)
        sub_fl = np.nan_to_num(fl_grid[r_min:r_max, c_min:c_max], nan=0.0)
        
        v_free = float(np.sum(w_free * sub_sp) / max(sum_w_free, 1e-6))
        v_cong = float(np.sum(w_cong * sub_sp) / max(sum_w_cong, 1e-6))
        
        q_free = float(np.sum(w_free * sub_fl) / max(sum_w_free, 1e-6))
        q_cong = float(np.sum(w_cong * sub_fl) / max(sum_w_cong, 1e-6))
        
        # Nonlinear hyperbolic tangent gate based on congested speed estimate
        w_blend = 0.5 * (1.0 + np.tanh((v_c - v_cong) / delta_v))
        
        sp_out[r, c] = (1.0 - w_blend) * v_free + w_blend * v_cong
        fl_out[r, c] = (1.0 - w_blend) * q_free + w_blend * q_cong
        
    return sp_out, fl_out


def reconstruct_panel_split(
    panel: str,
    panel_dir: Path,
    split: str,
    sp_prof: dict,
    fl_prof: dict,
    lanes_map: dict[str, float],
    fd: dict[str, dict[str, float]],
    ordered_links: list[str],
) -> pd.DataFrame:
    """Reconstruct masked states for a single corridor panel and split using ASM."""
    link_order_map = {lid: i for i, lid in enumerate(ordered_links)}
    link_idx = sp_prof["link_idx"]
    pfiles = sorted((panel_dir / split / "mainline_states_masked").glob("**/*.parquet"))
    
    # Cumulative link distances for ASM
    link_lens = np.array([fd.get(str(l), {}).get("length", 1.0) for l in ordered_links], dtype=float)
    cum_dist = np.concatenate([[0.0], np.cumsum(link_lens)[:-1]]) if len(link_lens) else np.arange(len(ordered_links), dtype=float)
    n_ordered = len(ordered_links)
    
    batches = []
    for path in pfiles:
        df = pd.read_parquet(path)
        df["station_id"] = df["station_id"].astype(str)
        df["link_id"] = df["link_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        weekday, tod = compute_time_slots(df)
        slot = weekday * SLOTS_PER_DAY + tod
        idx = df["link_id"].map(link_idx).fillna(-1).to_numpy(dtype=np.int64)
        valid_idx = np.where(idx >= 0, idx, 0)

        # Historical baseline priors
        hist_sp = sp_prof["mean"][valid_idx, slot]
        hist_fl = fl_prof["mean"][valid_idx, slot]
        hist_sp = np.where(sp_prof["cnt"][valid_idx, slot] == 0, sp_prof["fallback"][valid_idx], hist_sp)
        hist_fl = np.where(fl_prof["cnt"][valid_idx, slot] == 0, fl_prof["fallback"][valid_idx], hist_fl)

        lanes_arr = get_lane_vector(df["station_id"], df["link_id"], lanes_map)
        df["flow_per_lane"] = df["flow_vph"] / np.maximum(lanes_arr, 1.0)
        hist_fl_pl = hist_fl / np.maximum(lanes_arr, 1.0)

        # Build 2D Spatiotemporal Matrix for this day (ordered_links x 288 slots)
        tod_arr = tod
        order_idx_arr = df["link_id"].map(link_order_map).fillna(-1).to_numpy(dtype=np.int64)
        
        sp_grid = np.full((n_ordered, SLOTS_PER_DAY), np.nan, dtype=float)
        fl_grid = np.full((n_ordered, SLOTS_PER_DAY), np.nan, dtype=float)
        
        # Populate observed cells in grid
        obs_mask = df["speed_kmh"].notna() & df["flow_per_lane"].notna()
        obs_orders = order_idx_arr[obs_mask]
        obs_slots = tod_arr[obs_mask]
        valid_grid_pts = (obs_orders >= 0) & (obs_orders < n_ordered)
        
        if np.any(valid_grid_pts):
            sp_grid[obs_orders[valid_grid_pts], obs_slots[valid_grid_pts]] = df["speed_kmh"].to_numpy(dtype=float)[obs_mask][valid_grid_pts]
            fl_grid[obs_orders[valid_grid_pts], obs_slots[valid_grid_pts]] = df["flow_per_lane"].to_numpy(dtype=float)[obs_mask][valid_grid_pts]
            
        # Run Treiber-Helbing Adaptive Smoothing Method
        sp_asm, fl_asm = run_asm_grid(sp_grid, fl_grid, cum_dist)
        
        # Map ASM results back to dataframe rows
        row_orders = np.clip(order_idx_arr, 0, n_ordered - 1)
        row_slots = np.clip(tod_arr, 0, SLOTS_PER_DAY - 1)
        
        asm_sp_vals = sp_asm[row_orders, row_slots]
        asm_fl_vals = fl_asm[row_orders, row_slots]
        
        # Blend: ASM wave propagation where valid, historical mean where unsupported
        pred_speed = np.where(
            np.isfinite(asm_sp_vals) & (order_idx_arr >= 0),
            asm_sp_vals,
            hist_sp
        )
        pred_fl_pl = np.where(
            np.isfinite(asm_fl_vals) & (order_idx_arr >= 0),
            asm_fl_vals,
            hist_fl_pl
        )
        
        # Gentle regularizing blend with historical profile (5% prior)
        pred_speed = 0.95 * pred_speed + 0.05 * hist_sp
        pred_fl_pl = 0.95 * pred_fl_pl + 0.05 * hist_fl_pl
        
        pred_flow_raw = pred_fl_pl * lanes_arr

        # Physical bounds clipping
        vf_arr = np.array([fd.get(str(l), {}).get("vf", 105.0) for l in df["link_id"]])
        cap_arr = np.array([fd.get(str(l), {}).get("cap", 7200.0) for l in df["link_id"]])

        pred_speed_clipped = np.clip(pred_speed, 5.0, vf_arr * 1.05)
        
        # Enforce physical 50 vph floor guard (prevents zero-flow penalty in S_FD)
        pred_flow_clean = np.maximum(pred_flow_raw, 50.0)

        df["pred_speed_kmh"] = pred_speed_clipped
        df["pred_flow_vph"] = pred_flow_clean
        
        # Only keep target rows that need submission
        batches.append(df[["timestamp", "station_id", "link_id", "pred_speed_kmh", "pred_flow_vph"]])

    if not batches:
        return pd.DataFrame(columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"])

    out_df = pd.concat(batches, ignore_index=True)
    out_df.rename(columns={"pred_speed_kmh": "speed_kmh", "pred_flow_vph": "flow_vph"}, inplace=True)

    # Apply physical conservation balancing along corridor links if enabled
    if USE_PHYSICS_CORRECTIONS and apply_physics_corrections is not None:
        try:
            sp_corr, fl_corr = apply_physics_corrections(
                pred_speed=out_df["speed_kmh"].to_numpy(dtype=float),
                pred_flow=out_df["flow_vph"].to_numpy(dtype=float),
                link_ids=out_df["link_id"].to_numpy(dtype=str),
                timestamps=out_df["timestamp"].to_numpy(dtype=str),
                ordered_links=ordered_links,
                fd=fd,
            )
            out_df["speed_kmh"] = sp_corr
            out_df["flow_vph"] = fl_corr
        except Exception as e:
            print(f"Warning: Physics corrections skipped for {panel} ({e})")

    # Final safe floor enforcement
    out_df["speed_kmh"] = np.clip(pd.to_numeric(out_df["speed_kmh"], errors="coerce").fillna(65.0), 5.0, 130.0)
    out_df["flow_vph"] = np.clip(pd.to_numeric(out_df["flow_vph"], errors="coerce").fillna(1200.0), 50.0, 16000.0)
    
    return out_df


def run_task1(
    release_root: Path,
    splits: list[str],
    out_path: Path,
) -> int:
    """Execute Task 1 traffic state reconstruction across all panels and splits."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    results = []
    for panel in PANELS:
        panel_dir = release_root / "corridors" / panel
        if not panel_dir.exists():
            panel_dir = release_root / panel
        if not panel_dir.exists():
            continue
            
        lanes_map = load_lane_counts(panel_dir)
        fd = load_fd_params(panel_dir)
        ordered_links = load_ordered_links(panel_dir)
        sp_prof, fl_prof = learn_historical_profiles(panel_dir)
        
        for split in splits:
            tmpl_file = release_root / "task1" / panel / split / "sample_submission_state.csv"
            if not tmpl_file.exists():
                tmpl_file = release_root / "task1" / split / panel / "sample_submission_state.csv"
            if not tmpl_file.exists():
                candidates = list(release_root.glob(f"task1/**/{panel}/**/{split}/**/sample_submission_state.csv"))
                if candidates:
                    tmpl_file = candidates[0]
                    
            if not tmpl_file.exists():
                continue
                
            tmpl = pd.read_csv(tmpl_file)
            tmpl["station_id"] = tmpl["station_id"].astype(str)
            tmpl["link_id"] = tmpl["link_id"].astype(str)
            tmpl["timestamp"] = pd.to_datetime(tmpl["timestamp"], utc=True)
            
            recon = reconstruct_panel_split(
                panel=panel,
                panel_dir=panel_dir,
                split=split,
                sp_prof=sp_prof,
                fl_prof=fl_prof,
                lanes_map=lanes_map,
                fd=fd,
                ordered_links=ordered_links,
            )
            
            # Align exact rows to template
            merged = tmpl[["timestamp", "station_id", "link_id"]].merge(
                recon,
                on=["timestamp", "station_id", "link_id"],
                how="left"
            )
            
            # Impute any missing lookup fallback
            merged["speed_kmh"] = merged["speed_kmh"].fillna(65.0)
            merged["flow_vph"] = merged["flow_vph"].fillna(1200.0)
            
            # Preserve original template columns
            for col in tmpl.columns:
                if col not in merged.columns:
                    merged[col] = tmpl[col]
                    
            results.append(merged[tmpl.columns])
            
    if not results:
        print("Warning: No Task 1 outputs generated.")
        pd.DataFrame(columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"]).to_csv(out_path, index=False)
        return 0
        
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(out_path, index=False)
    
    print(f"[task1] Done in {time.time()-t0:.1f}s. Saved {len(final_df):,} rows to {out_path}.")
    return len(final_df)
