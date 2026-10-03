import json
from pathlib import Path

dst_nb = Path('competitions/ieee_traffic_flow/exp9_hydrodynamic_hybrid_sota/notebook.ipynb')

with open(dst_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c16_code = ''.join(nb['cells'][16]['source'])

target_str = "    for row in target_df.itertuples():\n        lid = str(row.link_id)\n        dt_min = 5.0 + max(0.0, (row.timestamp - t_horizon_start).total_seconds() / 60.0)\n        ml_p = ml_probs.get((lid, float(dt_min)), None)\n        if is_ongoing:\n            # Pure persistence — no upstream shockwave expansion (v1 FP bug)\n            if queue_now_map.get(lid, False):\n                preds.append(1)\n            elif ml_p is not None and ml_p >= ONGOING_ML_THRESHOLD:\n                preds.append(1)\n            else:\n                preds.append(0)\n        else:\n            if lid in onset_active_cluster and row.timestamp == t_horizon_end:\n                preds.append(1)\n            elif lid in fast_breakdown and row.timestamp >= t_step5_thresh:\n                preds.append(1)\n            elif ml_p is not None and ml_p >= lgbm_threshold:\n                preds.append(1)\n            else:\n                preds.append(0)"

replacement_str = """    # Dynamic Bottleneck Margin Calculation (Avik Hydrodynamic SOTA)
    at_t0["_margin"] = pd.to_numeric(at_t0["speed_kmh"], errors="coerce") / at_t0["_vcut"]
    dyn_bottlenecks = set(at_t0.sort_values("_margin").head(3)["link_id"].astype(str))

    # Precompute step-by-step hydrodynamic wave propagation for ongoing queues
    future_timestamps = sorted(target_df["timestamp"].unique())
    step_queued_links = {}
    current_queued = {lid for lid, q in queue_now_map.items() if q}
    for step_idx, step_t in enumerate(future_timestamps):
        step_queued_links[step_t] = set(current_queued)
        if step_idx >= 2: # At T+15m onwards, propagate shockwave upstream by 1 hop
            new_q = set(current_queued)
            for ql in current_queued:
                inc_links = topology.get(ql, {}).get("incoming", [])
                for inc_lid in inc_links:
                    inc_s = str(inc_lid)
                    inc_spd = at_t0[at_t0["link_id"].astype(str) == inc_s]["speed_kmh"]
                    if not inc_spd.empty and float(inc_spd.iloc[0]) <= 75.0:
                        new_q.add(inc_s)
            current_queued = new_q

    for row in target_df.itertuples():
        lid = str(row.link_id)
        dt_min = 5.0 + max(0.0, (row.timestamp - t_horizon_start).total_seconds() / 60.0)
        ml_p = ml_probs.get((lid, float(dt_min)), None)
        if is_ongoing:
            # Hydrodynamic shockwave tracking along mainline corridor topology
            if lid in step_queued_links.get(row.timestamp, set()):
                preds.append(1)
            elif ml_p is not None and ml_p >= ONGOING_ML_THRESHOLD:
                preds.append(1)
            else:
                preds.append(0)
        else:
            # Dynamic bottleneck onset (Avik SOTA): activate top-3 lowest margin links at step 2+ (T >= 10m)
            step_num = future_timestamps.index(row.timestamp) if row.timestamp in future_timestamps else 0
            if step_num >= 1 and lid in dyn_bottlenecks:
                preds.append(1)
            elif lid in onset_active_cluster and row.timestamp >= t_step5_thresh:
                preds.append(1)
            elif lid in fast_breakdown and row.timestamp >= t_step5_thresh:
                preds.append(1)
            elif ml_p is not None and ml_p >= lgbm_threshold:
                preds.append(1)
            else:
                preds.append(0)"""

assert target_str in c16_code, "Target string not found in cell 16!"
c16_code_updated = c16_code.replace(target_str, replacement_str)
nb['cells'][16]['source'] = [c16_code_updated]

with open(dst_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("SUCCESS: Successfully upgraded Cell 16 to Hydrodynamic Hybrid SOTA!")
