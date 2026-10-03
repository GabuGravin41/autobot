"""Task 2: Dynamic Kinematic Shockwave & Saturation Queue Forecasting Engine (Exp 12).

Replaces hardcoded bottleneck dictionaries with physics-grounded dynamics:
1. Ongoing Windows:
   - Identifies active queue boundaries along physical corridor link order.
   - Core bottleneck links persist unless dissipation cues (accelerating speed, falling occupancy) are detected.
   - Backward shockwave expansion: propagates queue tail upstream at physical wave speed w ~ 18 km/h
     conditioned on upstream deceleration and demand pressure.
2. Onset Windows:
   - Computes dynamic Pre-Breakdown Saturation Index (PBSI) across all links from 60-min history:
     combining speed margin to cutoff, short-term deceleration rate, volume/capacity ratio, and occupancy trend.
   - Predicts breakdown step dynamically (earlier for sharp decel/low margin, later for gradual buildup).
   - Once breakdown triggers, expands upstream along corridor topology in subsequent steps.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    TASK2_ACTIVE_PANELS,
    TASK2_HORIZON_STEPS,
)
from network_utils import (
    load_fd_params,
    load_topology,
    load_ordered_links,
)


def _compute_link_history_metrics(
    h_df: pd.DataFrame,
    fd_params: dict[str, dict[str, float]],
    t_last: pd.Timestamp
) -> dict[str, dict[str, float]]:
    """Compute kinematic and state metrics per link from 60-min history."""
    metrics: dict[str, dict[str, float]] = {}
    
    # 15-minute window cutoff
    t_15m = t_last - pd.Timedelta(minutes=15)
    
    for lid, grp in h_df.groupby("link_id"):
        lid = str(lid)
        grp = grp.sort_values("timestamp")
        
        sp = pd.to_numeric(grp["speed_kmh"], errors="coerce").to_numpy(dtype=float)
        sp_valid = sp[np.isfinite(sp)]
        if len(sp_valid) == 0:
            continue
            
        vf = fd_params.get(lid, {}).get("vf", 105.0)
        v_cut = fd_params.get(lid, {}).get("v_cut", vf * 0.60)
        cap = fd_params.get(lid, {}).get("cap", 7200.0)
        
        v_last = float(sp_valid[-1])
        margin = (v_last - v_cut) / max(v_cut, 1.0)
        
        # Recent 15-min speed trend
        grp_15m = grp[grp["timestamp"] >= t_15m]
        sp_15m = pd.to_numeric(grp_15m["speed_kmh"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(sp_15m) >= 2:
            decel_15m = (sp_15m[-1] - sp_15m[0]) / max(len(sp_15m) - 1, 1) / 5.0  # km/h per minute
            v_min15 = float(sp_15m.min())
        else:
            decel_15m = 0.0
            v_min15 = v_last
            
        # Flow ratio
        fl = pd.to_numeric(grp["flow_vph"], errors="coerce").dropna().to_numpy(dtype=float)
        q_last = float(fl[-1]) if len(fl) > 0 else 0.0
        flow_ratio = q_last / max(cap, 1.0)
        
        # Occupancy
        occ = pd.to_numeric(grp["occupancy"], errors="coerce").dropna().to_numpy(dtype=float) if "occupancy" in grp.columns else np.array([])
        occ_last = float(occ[-1]) if len(occ) > 0 else 0.0
        occ_trend = (occ[-1] - occ[0]) / max(len(occ) - 1, 1) if len(occ) >= 2 else 0.0
        
        # Breakdown saturation index (higher = closer to breakdown)
        collapse_index = (
            2.5 * max(0.0, 0.45 - margin)
            + 3.5 * max(0.0, -decel_15m)
            + 1.2 * max(0.0, flow_ratio - 0.70)
            + 1.5 * max(0.0, occ_last - 0.15)
        )
        
        metrics[lid] = {
            "v_last": v_last,
            "v_cut": v_cut,
            "margin": margin,
            "decel_15m": decel_15m,
            "v_min15": v_min15,
            "flow_ratio": flow_ratio,
            "occ_last": occ_last,
            "occ_trend": occ_trend,
            "collapse_index": collapse_index,
            "is_queued_T": float(v_last <= v_cut)
        }
        
    return metrics


def predict_window_queues(
    window_id: str,
    panel: str,
    condition: str,
    h_df: pd.DataFrame,
    target_df: pd.DataFrame,
    fd_params: dict[str, dict[str, float]],
    topology: dict[str, dict[str, list[str]]],
    ordered_links: list[str],
) -> list[int]:
    """Predict binary queue state (0/1) for all (link, timestamp) cells in horizon.
    
    Uses dynamic shockwave propagation and pre-breakdown saturation index.
    """
    if target_df.empty:
        return []
        
    t_last = h_df["timestamp"].max()
    is_ongoing = "ongoing" in condition.lower()
    
    # Corridor topology and link ordering
    link_order_map = {lid: i for i, lid in enumerate(ordered_links)}
    n_links = len(ordered_links)
    
    # Compute metrics from 60-min history
    link_metrics = _compute_link_history_metrics(h_df, fd_params, t_last)
    
    # Active queue at T0
    queued_at_T = {
        lid for lid, m in link_metrics.items()
        if m["is_queued_T"] == 1.0
    }
    
    # Horizon timestamps
    t_horizon = sorted(target_df["timestamp"].unique())
    n_steps = len(t_horizon)  # Expected 6 steps (T+5 to T+30)
    
    # Predicted queue set per step: step_idx (0 to 5) -> set of queued link_ids
    pred_queue_per_step: dict[int, set[str]] = {s: set() for s in range(n_steps)}
    
    if is_ongoing:
        # --- ONGOING QUEUE DYNAMICS ---
        # 1. Baseline: Persistent core queue
        base_queue = set(queued_at_T)
        if not base_queue:
            # Fallback if sensor dropout obscured queue at T: use lowest margin links
            low_m = sorted(link_metrics.keys(), key=lambda l: link_metrics[l]["margin"])
            base_queue = set(low_m[:2]) if low_m else set()
            
        # Get spatial indices of base queue
        base_indices = [link_order_map[lid] for lid in base_queue if lid in link_order_map]
        
        if base_indices:
            idx_tail = min(base_indices)  # Upstream-most queued link
            idx_head = max(base_indices)  # Downstream-most queued link (bottleneck head)
            
            for s in range(n_steps):
                dt_min = 5.0 * (s + 1)
                active_s = set(base_queue)
                
                # Check for queue dissipation (speed recovering rapidly)
                dissipating = False
                if s >= 3:
                    head_lid = ordered_links[idx_head]
                    if head_lid in link_metrics:
                        hm = link_metrics[head_lid]
                        if hm["decel_15m"] > 0.40 and hm["v_last"] > (hm["v_cut"] - 3.0):
                            dissipating = True
                
                if dissipating:
                    # Dissipates from tail to head
                    drop_count = min(s - 2, len(base_indices) - 1)
                    retained_indices = sorted(base_indices)[drop_count:]
                    active_s = {ordered_links[i] for i in retained_indices}
                else:
                    # Backward shockwave expansion:
                    # Wave speed ~ 18 km/h -> expands upstream ~1 link every 5-10 min
                    # Only expands if upstream link is under demand stress
                    max_expansion_hops = min(s + 1, 4)
                    for hop in range(1, max_expansion_hops + 1):
                        up_idx = idx_tail - hop
                        if 0 <= up_idx < n_links:
                            up_lid = ordered_links[up_idx]
                            up_m = link_metrics.get(up_lid)
                            # Expand if upstream link has margin stress or deceleration
                            if up_m is not None and (up_m["margin"] <= 0.35 or up_m["decel_15m"] <= 0.05):
                                active_s.add(up_lid)
                                
                pred_queue_per_step[s] = active_s
        else:
            for s in range(n_steps):
                pred_queue_per_step[s] = set(base_queue)
                
    else:
        # --- ONSET QUEUE DYNAMICS ---
        # No queue at T. Queue must form during horizon.
        # Rank links by Pre-Breakdown Saturation / Collapse Index
        ranked_links = sorted(
            link_metrics.keys(),
            key=lambda l: link_metrics[l]["collapse_index"],
            reverse=True
        )
        
        if ranked_links:
            primary_bn = ranked_links[0]
            secondary_bn = ranked_links[1] if len(ranked_links) > 1 else primary_bn
            
            p_m = link_metrics[primary_bn]
            
            # Estimate breakdown trigger step s_bd (0 to 5)
            # 0: T+5, 1: T+10, 2: T+15, 3: T+20, 4: T+25, 5: T+30
            if p_m["margin"] < 0.15 and p_m["decel_15m"] < -0.20:
                s_bd = 1  # T+10m
            elif p_m["margin"] < 0.25 and p_m["decel_15m"] < -0.10:
                s_bd = 2  # T+15m
            elif p_m["margin"] < 0.35 or p_m["decel_15m"] < -0.05:
                s_bd = 3  # T+20m
            else:
                s_bd = 4  # T+25m (conservative late onset)
                
            p_idx = link_order_map.get(primary_bn, 0)
            
            for s in range(n_steps):
                active_s = set()
                if s >= s_bd:
                    active_s.add(primary_bn)
                    if s >= (s_bd + 1) and secondary_bn != primary_bn:
                        # Secondary bottleneck breakdown
                        if link_metrics[secondary_bn]["collapse_index"] > 0.8:
                            active_s.add(secondary_bn)
                    
                    # Backward shockwave from primary bottleneck
                    hops = s - s_bd
                    for h in range(1, hops + 1):
                        up_idx = p_idx - h
                        if 0 <= up_idx < n_links:
                            active_s.add(ordered_links[up_idx])
                            
                pred_queue_per_step[s] = active_s

    # Generate flat binary predictions aligned with target_df rows
    step_map = {ts: i for i, ts in enumerate(t_horizon)}
    preds: list[int] = []
    
    for row in target_df.itertuples():
        s = step_map.get(row.timestamp, 0)
        lid = str(row.link_id)
        is_q = 1 if lid in pred_queue_per_step.get(s, set()) else 0
        preds.append(is_q)
        
    return preds


def run_task2(
    release_root: Path,
    splits: list[str],
    out_path: Path,
) -> int:
    """Execute Task 2 queue prediction across all evaluation windows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    wi_files = sorted(release_root.glob("task2/**/window_index.csv")) or sorted(release_root.glob("**/window_index.csv"))
    hist_files = sorted(release_root.glob("task2/**/window_history.parquet")) or sorted(release_root.glob("**/window_history.parquet"))
    tmpl_files = sorted(release_root.glob("task2/**/sample_submission_queue.csv")) or sorted(release_root.glob("**/sample_submission_queue.csv"))
    
    if not (wi_files and hist_files and tmpl_files):
        print("Warning: Missing Task 2 input files. Creating empty submission.")
        pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"]).to_csv(out_path, index=False)
        return 0

    wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    tmpl_df = pd.concat([pd.read_csv(f) for f in tmpl_files], ignore_index=True)
    tmpl_df["window_id"] = tmpl_df["window_id"].astype(str)
    tmpl_df["link_id"] = tmpl_df["link_id"].astype(str)
    tmpl_df["timestamp"] = pd.to_datetime(tmpl_df["timestamp"], utc=True)
    
    # Filter splits
    wi_df = wi_df[wi_df["split"].isin(splits)].copy()
    valid_window_ids = set(wi_df["window_id"].astype(str).unique())
    tmpl_df = tmpl_df[tmpl_df["window_id"].isin(valid_window_ids)].copy()
    
    print(f"[task2] Evaluating {len(valid_window_ids)} windows across {splits}...")
    
    # Read history parquet files
    hist_list = []
    cols = ["window_id", "timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"]
    for hf in hist_files:
        try:
            h = pd.read_parquet(hf)
            available_cols = [c for c in cols if c in h.columns]
            h = h[available_cols]
            h["window_id"] = h["window_id"].astype(str)
            h = h[h["window_id"].isin(valid_window_ids)]
            if not h.empty:
                hist_list.append(h)
        except Exception as e:
            print(f"[task2] Error reading {hf}: {e}")
            
    if not hist_list:
        print("Error: No matching history data found.")
        return 0
        
    full_hist = pd.concat(hist_list, ignore_index=True)
    full_hist["link_id"] = full_hist["link_id"].astype(str)
    full_hist["timestamp"] = pd.to_datetime(full_hist["timestamp"], utc=True)
    
    # Pre-cache corridor assets
    fd_params_by_panel: dict[str, dict] = {}
    topo_by_panel: dict[str, dict] = {}
    ordered_links_by_panel: dict[str, list[str]] = {}
    
    for panel in TASK2_ACTIVE_PANELS:
        panel_dir = release_root / "corridors" / panel
        if not panel_dir.exists():
            panel_dir = release_root / panel
        if panel_dir.exists():
            fd_params_by_panel[panel] = load_fd_params(panel_dir)
            topo_by_panel[panel] = load_topology(panel_dir)
            ordered_links_by_panel[panel] = load_ordered_links(panel_dir)

    all_preds_df = []
    
    for _, w_row in tqdm(wi_df.iterrows(), total=len(wi_df), desc="Predicting Task 2 Windows"):
        wid = str(w_row["window_id"])
        panel = str(w_row["panel"])
        condition = str(w_row.get("condition", "queue_onset"))
        
        if panel not in ordered_links_by_panel:
            continue
            
        w_hist = full_hist[full_hist["window_id"] == wid]
        w_tmpl = tmpl_df[tmpl_df["window_id"] == wid].copy()
        
        if w_hist.empty or w_tmpl.empty:
            continue
            
        preds = predict_window_queues(
            window_id=wid,
            panel=panel,
            condition=condition,
            h_df=w_hist,
            target_df=w_tmpl,
            fd_params=fd_params_by_panel[panel],
            topology=topo_by_panel[panel],
            ordered_links=ordered_links_by_panel[panel],
        )
        
        w_tmpl["queue_pred"] = preds
        all_preds_df.append(w_tmpl[["window_id", "timestamp", "link_id", "queue_pred"]])
        
    if not all_preds_df:
        print("Warning: No predictions generated.")
        return 0
        
    final_task2 = pd.concat(all_preds_df, ignore_index=True)
    final_task2.to_csv(out_path, index=False)
    
    pos_count = int(final_task2["queue_pred"].sum())
    total_count = len(final_task2)
    print(f"[task2] Done in {time.time()-t0:.1f}s. Saved {total_count:,} rows to {out_path}.")
    print(f"[task2] Predicted active queue cells: {pos_count:,} / {total_count:,} ({100*pos_count/total_count:.2f}%)")
    return total_count
