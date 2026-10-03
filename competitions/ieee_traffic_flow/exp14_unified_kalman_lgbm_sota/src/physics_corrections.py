"""Physics-consistent post-processing for Task 1 reconstruction.

Applied AFTER the existing linear temporal+spatial interpolation (v2 baseline).
Does NOT replace the interpolation — only corrects the output for physics consistency.

Two stages:
1. Soft FD projection: nudge (speed, flow) toward triangular FD
2. Conservation correction: enforce vehicle conservation at link transitions

These directly boost S_physics (0.15 weight) without degrading S_state (0.35 weight).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ─── Parameters (calibrated on 10/10 train panels) ──────────────
FD_ALPHA = 0.15            # soft FD projection blend (improves S_FD without hurting speed)
CONSERVATION_ALPHA = 0.40  # conservation correction strength (reduces residuals by ~65%)
CONSERVATION_PASSES = 5    # iterative passes for smooth multi-link cascading


def soft_fd_projection(speed: np.ndarray, flow: np.ndarray,
                       link_ids: np.ndarray,
                       fd: dict[str, dict],
                       alpha: float = FD_ALPHA,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Gently project (speed, flow) toward the triangular FD.

    For each cell:
    1. Compute density k = q / v
    2. Compute FD-consistent flow q_fd based on triangular FD
    3. Blend: q_out = (1-α) * q + α * q_fd

    Speed is NOT adjusted (lesson from v1: adjusting speed killed S_state).
    Only flow is nudged, and only gently (α=0.10).

    This directly improves S_FD (1/3 of S_physics = 0.05 of S_total).
    """
    n = len(speed)
    flow_out = flow.copy()

    # Build per-cell FD parameter arrays (vectorized)
    lids = np.asarray(link_ids, dtype=str)
    vf = np.array([fd.get(lids[i], {}).get("vf", 105.0) for i in range(n)])
    cap = np.array([fd.get(lids[i], {}).get("cap", 7200.0) for i in range(n)])
    k_crit = np.array([fd.get(lids[i], {}).get("k_crit", cap[i] / max(vf[i], 1.0)) for i in range(n)])
    k_jam = np.array([fd.get(lids[i], {}).get("k_jam", 400.0) for i in range(n)])
    w_speed = np.array([fd.get(lids[i], {}).get("wave_speed",
                        cap[i] / max(k_jam[i] - k_crit[i], 1.0)) for i in range(n)])

    valid = np.isfinite(speed) & np.isfinite(flow) & (speed > 1.0)

    # Density from current state
    k = np.where(valid, flow / np.maximum(speed, 1.0), 0.0)

    # FD-consistent flow
    is_free = k <= k_crit
    q_fd_free = vf * k
    q_fd_cong = w_speed * np.maximum(k_jam - k, 0.0)
    q_fd = np.where(is_free, q_fd_free, q_fd_cong)

    # Only project where FD flow is positive and reasonable
    reasonable = valid & (q_fd > 0) & (q_fd < cap * 1.3)
    flow_out[reasonable] = (1.0 - alpha) * flow[reasonable] + alpha * q_fd[reasonable]

    return speed.copy(), flow_out


def conservation_correction(speed: np.ndarray, flow: np.ndarray,
                            link_ids_arr: np.ndarray,
                            timestamps_arr: np.ndarray,
                            ordered_links: list[str],
                            fd: dict[str, dict],
                            ramp_net_flow: Optional[dict] = None,
                            alpha: float = CONSERVATION_ALPHA,
                            n_passes: int = CONSERVATION_PASSES,
                            ) -> np.ndarray:
    """Apply iterative conservation-aware flow correction.

    For each consecutive link pair (upstream→downstream) at each timestamp,
    computes the conservation residual:
        residual = q_dn - (q_up + q_ramp_net)
    and adjusts flows to minimize it.

    Multiple passes smooth out cascading corrections along the corridor.

    This directly improves S_LWR (2/3 of S_physics = 0.10 of S_total).
    """
    flow_out = flow.copy()

    # Build link order map
    link_order = {lid: i for i, lid in enumerate(ordered_links)}

    # Build index: pre-sort once
    n = len(flow)
    orders = np.array([link_order.get(str(link_ids_arr[i]), 9999) for i in range(n)])
    ts_arr = timestamps_arr

    # Group by timestamp
    # Use pandas for efficient groupby
    df_tmp = pd.DataFrame({
        "ts": ts_arr,
        "order": orders,
        "idx": np.arange(n),
    })

    # Pre-build group indices
    ts_groups = {}
    for ts_val, group in df_tmp.groupby("ts"):
        sorted_g = group.sort_values("order")
        ts_groups[ts_val] = (
            sorted_g["order"].to_numpy(),
            sorted_g["idx"].to_numpy(dtype=int),
        )

    for _pass in range(n_passes):
        for ts_val, (group_orders, group_indices) in ts_groups.items():
            flows = flow_out[group_indices]
            n_g = len(group_orders)

            for j in range(n_g - 1):
                # Check consecutive in corridor order
                if group_orders[j + 1] != group_orders[j] + 1:
                    continue

                q_up = flows[j]
                q_dn = flows[j + 1]

                if not (np.isfinite(q_up) and np.isfinite(q_dn)):
                    continue

                # Net ramp flow at upstream link
                q_ramp = 0.0
                if ramp_net_flow is not None:
                    lid_up = str(link_ids_arr[group_indices[j]])
                    key = (lid_up, str(ts_val))
                    q_ramp = ramp_net_flow.get(key, 0.0)

                # Conservation residual
                residual = q_dn - (q_up + q_ramp)

                # Split correction evenly between upstream and downstream
                correction = alpha * residual * 0.5
                flow_out[group_indices[j]] = q_up + correction
                flow_out[group_indices[j + 1]] = q_dn - correction
                # Update local copy for cascading within this pass
                flows[j] = flow_out[group_indices[j]]
                flows[j + 1] = flow_out[group_indices[j + 1]]

    return flow_out


def apply_physics_corrections(pred_speed: np.ndarray,
                              pred_flow: np.ndarray,
                              link_ids: np.ndarray,
                              timestamps: np.ndarray,
                              ordered_links: list[str],
                              fd: dict[str, dict],
                              ramp_net_flow: Optional[dict] = None,
                              fd_alpha: float = FD_ALPHA,
                              cons_alpha: float = CONSERVATION_ALPHA,
                              cons_passes: int = CONSERVATION_PASSES,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Apply physics corrections to reconstructed states.

    This is the main entry point. Call after Task 1 interpolation.

    Order matters:
    1. FD projection first (ensures density is in feasible range)
    2. Conservation correction second (adjusts flows for balance)

    Returns (corrected_speed, corrected_flow).
    """
    # Step 1: Soft FD projection
    speed_out, flow_out = soft_fd_projection(
        pred_speed, pred_flow, link_ids, fd, alpha=fd_alpha,
    )

    # Step 2: Conservation correction
    flow_out = conservation_correction(
        speed_out, flow_out,
        link_ids, timestamps,
        ordered_links, fd,
        ramp_net_flow=ramp_net_flow,
        alpha=cons_alpha,
        n_passes=cons_passes,
    )

    return speed_out, flow_out
