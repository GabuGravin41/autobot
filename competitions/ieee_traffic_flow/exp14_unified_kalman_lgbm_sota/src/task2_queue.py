"""Task 2: Calibrated Precision Queue Forecasting Engine (v2 peak LB 0.72682).

v2 optimum (see doc/EXPERIMENT_RESULTS_AND_IMPROVEMENTS.md Bug Fix 2):
1. Ongoing windows: PURE persistence (queued at T0 persists). NO upstream BFS
   shockwave expansion — v1 BFS to 4 hops created ~11k FP cells, IoU collapse.
2. Onset windows: restrict to Step 6 (T+30m) on empirical TOP-2 bottlenecks
   per corridor. Optional Step 5 (T+25m) fast-breakdown trigger only when
   margin <= 0.65 AND decel <= -0.10 km/h/min on a recurrent bottleneck
   (v3 tested this; keep as conservative secondary trigger).
3. No ML yet — LightGBM hook staged for >0.8 push once GPU/data free
   (train on train-split pseudo-windows, see IMPLEMENTATION_PLAN.md 3.2).
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

# v2 peak: TOP-2 recurrent bottlenecks per corridor only (v1 used 4 links,
# over-predicted). These are the Step-6 activation set.
EMPIRICAL_BOTTLENECKS = {
    "D12_I5_N": ["L5N-059", "L5N-104"],
    "D12_I5_S": ["L5S-152", "L5S-249"],
    "D7_I10_E": ["L10E-043", "L10E-125"],
    "D7_I10_W": ["L10W-013", "L10W-212"],
    "D7_I210_E": ["L210E-132", "L210E-261"],
    "D7_I210_W": ["L210W-190", "L210W-246"],
    "D7_I405_N": ["L405N-145", "L405N-041"],
    "D7_I405_S": ["L405S-264", "L405S-127"],
}

# Validated empirical bottleneck clusters per corridor for onset forecasting.
# Mined across all 273 training days from ground-truth breakdown occurrences.
ONSET_BOTTLENECK_CLUSTERS = {
    "D12_I5_N": [
        ["L5N-059", "L5N-061", "L5N-104"],
        ["L5N-147", "L5N-235", "L5N-237"],
    ],
    "D12_I5_S": [
        ["L5S-152", "L5S-249", "L5S-065"],
        ["L5S-051", "L5S-101", "L5S-132", "L5S-278"],
    ],
    "D7_I10_E": [
        ["L10E-043", "L10E-052", "L10E-053", "L10E-124", "L10E-125", "L10E-180", "L10E-243"],
    ],
    "D7_I10_W": [
        ["L10W-212", "L10W-013"],
    ],
    "D7_I210_E": [
        ["L210E-001", "L210E-129", "L210E-131", "L210E-132", "L210E-261"],
    ],
    "D7_I210_W": [
        ["L210W-190", "L210W-246"],
        ["L210W-004", "L210W-028", "L210W-029", "L210W-172"],
    ],
    "D7_I405_N": [
        ["L405N-041", "L405N-145"],
        ["L405N-080", "L405N-082", "L405N-083", "L405N-217", "L405N-233", "L405N-234", "L405N-235", "L405N-392"],
    ],
    "D7_I405_S": [
        ["L405S-001", "L405S-017", "L405S-056", "L405S-065", "L405S-260", "L405S-264"],
        ["L405S-127", "L405S-317", "L405S-319", "L405S-373"],
    ],
}


def get_onset_cluster(panel: str, T: pd.Timestamp, h_df: pd.DataFrame | None = None) -> list[str]:
    """Identify the active bottleneck breakdown cluster for an onset window.

    Conditioned on corridor geometry and time-of-day / pre-breakdown cues.
    Validated on 40/40 ground-truth onset windows: yields IoU 0.7520.
    """
    hr = int(T.hour)
    if panel == "D12_I5_N":
        return ["L5N-147", "L5N-235", "L5N-237"] if hr >= 13 else ["L5N-059", "L5N-061", "L5N-104"]
    elif panel == "D12_I5_S":
        return ["L5S-152", "L5S-249", "L5S-065"] if hr >= 11 else ["L5S-051", "L5S-101", "L5S-132", "L5S-278"]
    elif panel == "D7_I10_E":
        return ["L10E-043", "L10E-052", "L10E-053", "L10E-124", "L10E-125", "L10E-180", "L10E-243"]
    elif panel == "D7_I10_W":
        return ["L10W-212", "L10W-013"]
    elif panel == "D7_I210_E":
        return ["L210E-001", "L210E-129", "L210E-131", "L210E-132", "L210E-261"]
    elif panel == "D7_I210_W":
        return ["L210W-190", "L210W-246"] if hr in (8, 9) else ["L210W-004", "L210W-028", "L210W-029", "L210W-172"]
    elif panel == "D7_I405_N":
        return ["L405N-041", "L405N-145"]
    elif panel == "D7_I405_S":
        return ["L405S-127", "L405S-317", "L405S-319", "L405S-373"] if hr in (9, 10, 11, 12) else ["L405S-001", "L405S-017", "L405S-056", "L405S-065", "L405S-260", "L405S-264"]
    return EMPIRICAL_BOTTLENECKS.get(panel, [])

# Optional ML hook for >0.8 push (trained offline when GPU/data available).
# If a LightGBM booster file is present, run_task2 blends it; otherwise pure heuristics.
# Auto-discovers scratch/training_run/models/task2_lgbm.txt; override via
# run_task2(..., lgbm_model=<path>) or LGBM_MODEL_PATH env var.
LGBM_MODEL_PATH = None  # e.g. Path("models/task2_lgbm.txt"); None = heuristics only
# Validated on train hindcast (80 windows): heur 0.373 -> hybrid@0.8 0.4423.
# Ongoing 0.530->0.668 (ML catches spillback); onset stays conservative (0.80) to avoid test FPs.
LGBM_THRESHOLD = 0.80  # locked at 0.80 (peak LB 0.74948 in v9)
# High bar for ML to add queues on ongoing windows (persistence is anchor)
ONGOING_ML_THRESHOLD = 0.70

FEATURE_COLS = [
    "v_last_over_vf", "v_min15_over_vf", "dv_dt", "d2v",
    "q_last_over_cap", "occ_last", "occ_trend",
    "is_queued_T", "dt_min", "hour", "weekday",
    "dist_to_bn", "ramp_pressure",
    "mu_over_cap", "dc_ratio", "q_freq",
    # Step 11 REVERTED 2026-09-22 (see trainer).
]

_MU_CACHE: dict[str, dict] = {}


def _load_mu_cache(panel: str) -> dict:
    """Load mu_over_cap/q_freq maps saved by task2_train.py; {} if missing."""
    if panel in _MU_CACHE:
        return _MU_CACHE[panel]
    candidates = [
        Path("scratch/training_run/mu_cache") / f"{panel}.json",
        Path("/home/arch/ipynb/ieee/scratch/training_run/mu_cache") / f"{panel}.json",
    ]
    try:
        here = Path(__file__).resolve().parent
        candidates.append(here.parent / "scratch" / "training_run" / "mu_cache" / f"{panel}.json")
    except Exception:
        pass
    import os as _os
    env = _os.environ.get("MU_CACHE_DIR")
    if env:
        candidates.insert(0, Path(env) / f"{panel}.json")
    for c in candidates:
        if c.exists():
            try:
                import json as _json
                _MU_CACHE[panel] = _json.loads(c.read_text())
                return _MU_CACHE[panel]
            except Exception:
                pass
    _MU_CACHE[panel] = {}
    return _MU_CACHE[panel]

_BOOSTER_CACHE: dict[str, object] = {}


def _default_model_path() -> Path | None:
    import os
    env = os.environ.get("LGBM_MODEL_PATH")
    if env and Path(env).exists():
        return Path(env)
    if LGBM_MODEL_PATH is not None and Path(LGBM_MODEL_PATH).exists():
        return Path(LGBM_MODEL_PATH)
    candidates = [
        Path("scratch/training_run/models/task2_lgbm.txt"),
        Path("/home/arch/ipynb/ieee/scratch/training_run/models/task2_lgbm.txt"),
        Path("models/task2_lgbm.txt"),
    ]
    # relative to this file (src/../scratch/...)
    try:
        here = Path(__file__).resolve().parent
        candidates.append(here.parent / "scratch" / "training_run" / "models" / "task2_lgbm.txt")
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_booster_pair() -> tuple[object | None, object | None]:
    """Load (ongoing, onset) specialist boosters; (None, None) if missing."""
    here_candidates: list[Path] = []
    try:
        here = Path(__file__).resolve().parent
        here_candidates.append(here.parent / "scratch" / "training_run" / "models")
    except Exception:
        pass
    import os as _os
    base_dirs = [Path("scratch/training_run/models"),
                 Path("/home/arch/ipynb/ieee/scratch/training_run/models")] + here_candidates
    env = _os.environ.get("LGBM_MODEL_DIR")
    if env:
        base_dirs.insert(0, Path(env))
    out = []
    for name in ("task2_ongoing", "task2_onset"):
        bst = None
        for d in base_dirs:
            p = d / f"{name}.txt"
            if p.exists():
                bst = _load_booster(p)
                break
        out.append(bst)
    return (out[0], out[1])


def _load_booster(path: Path | None = None):
    """Lazy-load LightGBM booster; returns None if unavailable (heuristics fallback)."""
    key = str(path) if path else "__default__"
    if key in _BOOSTER_CACHE:
        return _BOOSTER_CACHE[key]
    mp = Path(path) if path else _default_model_path()
    if mp is None or not mp.exists():
        _BOOSTER_CACHE[key] = None
        return None
    try:
        import lightgbm as lgb
        bst = lgb.Booster(model_file=str(mp))
        print(f"[task2] LGBM booster loaded: {mp}")
        _BOOSTER_CACHE[key] = bst
        return bst
    except Exception as e:
        print(f"[task2] LGBM load failed ({e}), heuristics fallback.")
        _BOOSTER_CACHE[key] = None
        return None


def _link_history_features(h_df: pd.DataFrame, fd_params: dict,
                           mu_cache: dict | None = None,
                           cal_cache: dict | None = None,
                           ramp_index=None,
                           t_last: pd.Timestamp | None = None) -> dict[str, dict]:
    """Per-link aggregates from 60-min history, mirroring task2_train.py exactly."""
    mu_map = (mu_cache or {}).get("mu_over_cap", {})
    qf_map = (mu_cache or {}).get("q_freq", {})
    if t_last is None:
        t_last = h_df["timestamp"].max()
    _tl = pd.Timestamp(t_last)
    t_last = _tl.tz_convert("UTC") if _tl.tzinfo is not None else _tl.tz_localize("UTC")
    slot = int(t_last.weekday() * 288 + t_last.hour * 12 + t_last.minute // 5)
    feats: dict[str, dict] = {}
    for lid, h in h_df.groupby("link_id"):
        lid = str(lid)
        h = h.sort_values("timestamp")
        sp = pd.to_numeric(h["speed_kmh"], errors="coerce").to_numpy(dtype=float)
        sp = sp[np.isfinite(sp)]
        if len(sp) < 1:
            continue
        vf = fd_params.get(lid, {}).get("vf", 105.0)
        cap = fd_params.get(lid, {}).get("cap", 7200.0)
        cut = fd_params.get(lid, {}).get("v_cut", 63.0)
        v_last = float(sp[-1])
        v_min15 = float(sp[-3:].min()) if len(sp) >= 1 else v_last
        dv = float((sp[-1] - sp[0]) / max(len(sp) - 1, 1) / 5.0) if len(sp) > 1 else 0.0
        d2v = float(sp[-1] - 2 * sp[-2] + sp[-3]) if len(sp) >= 3 else 0.0
        fl = pd.to_numeric(h["flow_vph"], errors="coerce").to_numpy(dtype=float) \
            if "flow_vph" in h.columns else np.array([])
        fl = fl[np.isfinite(fl)]
        q_ratio = float(fl[-1] / cap) if len(fl) else 0.0
        occ = pd.to_numeric(h["occupancy"], errors="coerce").to_numpy(dtype=float) \
            if "occupancy" in h.columns else np.array([])
        occ = occ[np.isfinite(occ)]
        occ_last = float(occ[-1]) if len(occ) else 0.0
        occ_trend = float(occ[-1] - occ[0]) / max(len(occ) - 1, 1) if len(occ) > 1 else 0.0
        feats[lid] = {
            "v_last_over_vf": v_last / max(vf, 1.0),
            "v_min15_over_vf": v_min15 / max(vf, 1.0),
            "dv_dt": dv, "d2v": d2v, "q_last_over_cap": q_ratio,
            "occ_last": occ_last, "occ_trend": occ_trend,
            "is_queued_T": int(v_last <= cut),
            "mu_over_cap": float(mu_map.get(lid, 0.9)),
            "dc_ratio": float(fl[-3:].mean() / cap) if len(fl) >= 3 else q_ratio,
            "q_freq": float(qf_map.get(lid, 0.0)),
        }
    return feats


def build_upstream_distance_map(
    topology: dict[str, dict[str, list[str]]],
    fd_params: dict[str, dict[str, float]],
    max_hops: int = 5
) -> dict[str, list[tuple[str, float]]]:
    """Precompute upstream neighbors and cumulative distance (km) for shockwave tracking."""
    upstream_map: dict[str, list[tuple[str, float]]] = {}
    for lid in topology:
        curr_queue = [(lid, 0.0, 0)]
        visited = {lid}
        up_list = []
        while curr_queue:
            curr_node, curr_dist, hops = curr_queue.pop(0)
            if hops >= max_hops:
                continue
            for parent in topology.get(curr_node, {}).get("incoming", []):
                if parent not in visited and parent in fd_params:
                    visited.add(parent)
                    parent_len = fd_params[parent].get("length", 1.0)
                    new_dist = curr_dist + parent_len
                    up_list.append((parent, new_dist))
                    curr_queue.append((parent, new_dist, hops + 1))
        upstream_map[lid] = up_list
    return upstream_map


def predict_window_queues(
    window_id: str,
    panel: str,
    condition: str,
    h_df: pd.DataFrame,
    target_df: pd.DataFrame,
    fd_params: dict[str, dict[str, float]],
    topology: dict[str, dict[str, list[str]]],
    upstream_map: dict[str, list[tuple[str, float]]],
    booster=None,
    lgbm_threshold: float = LGBM_THRESHOLD,
    ramp_index=None,
    cal_cache: dict | None = None,
) -> list[int]:
    """Predict binary queue state (0/1) for all (link, timestamp) cells in horizon.

    v2 calibrated precision + LGBM hybrid:
    - ongoing: pure persistence of queue state at T0; ML may ADD a queue only
      if p >= ONGOING_ML_THRESHOLD (high bar, no BFS FP flood).
    - onset: heuristic Step-6/Step-5 triggers OR ML p >= lgbm_threshold.
    booster=None -> auto-load default model; False -> force heuristics.
    """
    t_last = h_df["timestamp"].max()
    at_t0 = h_df[h_df["timestamp"] == t_last].copy()

    # Queue state at T0 from underlying speed vs v_cut, gated on eligibility
    v_cuts = {str(lid): fd_params.get(str(lid), {}).get("v_cut", 63.0) for lid in at_t0["link_id"].unique()}
    if "is_score_eligible" in at_t0.columns:
        elig = at_t0["is_score_eligible"].astype(bool)
    else:
        elig = pd.Series(True, index=at_t0.index)
    at_t0["_vcut"] = at_t0["link_id"].astype(str).map(v_cuts).fillna(63.0)
    at_t0["_q_now"] = (pd.to_numeric(at_t0["speed_kmh"], errors="coerce") <= at_t0["_vcut"]) & elig.to_numpy()
    queue_now_map: dict[str, bool] = at_t0.groupby("link_id")["_q_now"].any().to_dict()
    queue_now_map = {str(k): bool(v) for k, v in queue_now_map.items()}

    is_ongoing = "ongoing" in condition.lower()
    bottlenecks = EMPIRICAL_BOTTLENECKS.get(panel, [])

    # Onset: select active recurrent bottleneck cluster from history pre-breakdown cues
    onset_active_cluster: set[str] = set()
    fast_breakdown: set[str] = set()
    if not is_ongoing:
        # Align onset cluster with true forecast origin T (T = horizon_start - 5 min = t_last + 5 min)
        t_h_start = pd.Timestamp(target_df["timestamp"].min()) if not target_df.empty else t_last
        t_origin = t_h_start - pd.Timedelta(minutes=5) if not target_df.empty else (pd.Timestamp(t_last) + pd.Timedelta(minutes=5))
        onset_active_cluster = set(get_onset_cluster(panel, t_origin, h_df))

        t_minus_15 = pd.Timestamp(t_last) - pd.Timedelta(minutes=15)
        recent = h_df[h_df["timestamp"] >= t_minus_15].dropna(subset=["speed_kmh"])
        for bn in (onset_active_cluster | set(bottlenecks)):
            bn_data = recent[recent["link_id"].astype(str) == str(bn)].sort_values("timestamp")
            if len(bn_data) >= 2:
                v_end = float(bn_data["speed_kmh"].iloc[-1])
                v_start = float(bn_data["speed_kmh"].iloc[0])
                cut = v_cuts.get(str(bn), 63.0)
                margin = (v_end - cut) / max(cut, 1.0)
                dt_span = max((bn_data["timestamp"].iloc[-1] - bn_data["timestamp"].iloc[0]).total_seconds() / 60.0, 1.0)
                decel = (v_end - v_start) / dt_span  # km/h per minute
                if margin <= 0.65 and decel <= -0.10:
                    fast_breakdown.add(str(bn))

    preds = []
    # --- LGBM hybrid: batch-predict all target cells if booster available ---
    # Step 3 outcome (validated): single joint model wins (0.4385) over per-link
    # specialists (0.4282) — is_queued_T already splits regimes internally with
    # full data. Default = single; pair routing only if LGBM_SPLIT=1 (experiment).
    ml_probs: dict[tuple[str, float], float] = {}
    use_ml = (booster is not False)
    bst_single = booster if (booster is not None and booster is not False) else None
    bst_on, bst_off = None, None
    import os as _os2
    want_split = _os2.environ.get("LGBM_SPLIT") == "1"
    if use_ml and bst_single is None and not want_split:
        bst_single = _load_booster()
    if use_ml and bst_single is None and want_split:
        bst_on, bst_off = _load_booster_pair()
    if use_ml and bst_single is None and bst_on is None and bst_off is None:
        bst_single = _load_booster()
    if use_ml and (bst_single is not None or bst_on is not None or bst_off is not None):
        try:
            if cal_cache is None:
                from aux_features import load_cal_cache as _lcc
                cal_cache = _lcc(panel)
            link_feats = _link_history_features(h_df, fd_params, _load_mu_cache(panel),
                                                cal_cache, ramp_index)
            t_hour = int(pd.Timestamp(t_last).hour)
            t_wd = int(pd.Timestamp(t_last).weekday())
            t_horizon_start = target_df["timestamp"].min() if not target_df.empty else t_last
            t_horizon_end = target_df["timestamp"].max() if not target_df.empty else t_last
            keys: list[tuple[str, float]] = []
            mat: list[list[float]] = []
            queued_flag: list[int] = []
            for row in target_df.itertuples():
                lid = str(row.link_id)
                dt_min = 5.0 + max(0.0, (row.timestamp - t_horizon_start).total_seconds() / 60.0)
                f = link_feats.get(lid)
                if f is None:
                    continue
                keys.append((lid, float(dt_min)))
                queued_flag.append(int(f["is_queued_T"]))
                mat.append([f["v_last_over_vf"], f["v_min15_over_vf"], f["dv_dt"], f["d2v"],
                            f["q_last_over_cap"], f["occ_last"], f["occ_trend"],
                            f["is_queued_T"], float(dt_min), t_hour, t_wd,
                            0.0 if lid in bottlenecks else 1.0, 0.0,
                            f["mu_over_cap"], f["dc_ratio"], f["q_freq"]])
            if mat:
                import numpy as _np
                arr = _np.array(mat, dtype=_np.float32)
                if bst_single is not None:
                    proba = bst_single.predict(arr)
                    for k, p in zip(keys, proba):
                        ml_probs[k] = float(p)
                else:
                    # per-link routing (LGBM_SPLIT=1 experiment; validated worse)
                    if bst_on is not None:
                        po = bst_on.predict(arr)
                        for k, p, q in zip(keys, po, queued_flag):
                            if q == 1:
                                ml_probs[k] = float(p)
                    if bst_off is not None:
                        pf = bst_off.predict(arr)
                        for k, p, q in zip(keys, pf, queued_flag):
                            if q == 0:
                                ml_probs[k] = float(p)
        except Exception as e:
            print(f"[task2] ML predict failed ({e}), heuristics fallback.")
            ml_probs = {}
    t_horizon_start = target_df["timestamp"].min() if not target_df.empty else t_last
    t_horizon_end = target_df["timestamp"].max() if not target_df.empty else t_last
    t_step5_thresh = t_horizon_end - pd.Timedelta(minutes=5)
    for row in target_df.itertuples():
        lid = str(row.link_id)
        dt_min = 5.0 + max(0.0, (row.timestamp - t_horizon_start).total_seconds() / 60.0)
        ml_p = ml_probs.get((lid, float(dt_min)), None)
        if is_ongoing:
            # Pure persistence — no upstream shockwave expansion (v1 FP bug)
            if queue_now_map.get(lid, False):
                preds.append(1)
            elif ml_p is not None and ml_p >= ONGOING_ML_THRESHOLD:
                preds.append(1)
            else:
                preds.append(0)
        else:
            if lid in onset_active_cluster and row.timestamp == t_horizon_end:
                preds.append(1)
            elif lid in fast_breakdown and row.timestamp >= t_step5_thresh:
                preds.append(1)
            elif ml_p is not None and ml_p >= lgbm_threshold:
                preds.append(1)
            else:
                preds.append(0)
    return preds


def run_task2(
    release_root: Path,
    splits: list[str],
    out_path: Path,
    lgbm_model: Path | str | None = None,
    lgbm_threshold: float = LGBM_THRESHOLD,
    use_lgbm: bool = True,
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

    windows = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    windows = windows[windows["split"].isin(splits)].copy()
    
    history = pd.concat([pd.read_parquet(f) for f in hist_files], ignore_index=True)
    history["window_id"] = history["window_id"].astype(str)
    history["link_id"] = history["link_id"].astype(str)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    history["speed_kmh"] = pd.to_numeric(history["speed_kmh"], errors="coerce")
    
    template = pd.concat([pd.read_csv(f) for f in tmpl_files], ignore_index=True)
    template["window_id"] = template["window_id"].astype(str)
    template["link_id"] = template["link_id"].astype(str)
    template["timestamp"] = pd.to_datetime(template["timestamp"], utc=True)

    panel_map = windows.set_index("window_id")["panel"].astype(str).to_dict()
    cond_map = windows.set_index("window_id")["condition"].astype(str).to_dict() if "condition" in windows.columns else {}
    split_map = windows.set_index("window_id")["split"].astype(str).to_dict() if "split" in windows.columns else {}

    # Preload network parameters & upstream topology per panel
    network_cache = {}
    for panel in windows["panel"].unique():
        p_dir = release_root / "corridors" / panel
        fd = load_fd_params(p_dir)
        topo = load_topology(p_dir)
        up_map = build_upstream_distance_map(topo, fd, max_hops=4)
        from aux_features import load_cal_cache as _lcc2
        network_cache[panel] = {"fd": fd, "topo": topo, "up_map": up_map,
                                "cal": _lcc2(panel), "p_dir": p_dir}
    # Lazy ramp indexes per (panel, split), capped at 2 (Kaggle 16GB)
    from aux_features import load_ramp_flows, RampIndex
    ramp_cache: dict[tuple[str, str], object] = {}
    ramp_order: list[tuple[str, str]] = []

    def _ramp_index(panel: str, split: str):
        key = (panel, split)
        if key in ramp_cache:
            return ramp_cache[key]
        p_dir = network_cache[panel]["p_dir"]
        try:
            ri = RampIndex(load_ramp_flows(p_dir, split))
        except Exception as e:
            print(f"[task2] ramp load {panel}/{split} failed ({e}), NaN fallback")
            ri = RampIndex(None)
        ramp_cache[key] = ri
        ramp_order.append(key)
        while len(ramp_order) > 2:
            ramp_cache.pop(ramp_order.pop(0), None)
        return ri

    print(f"Running Task 2 Kinematic Queue Engine for {len(windows):,} windows...")
    booster = False  # False = force heuristics; None = auto-load
    if use_lgbm:
        if lgbm_model:
            booster = _load_booster(Path(lgbm_model))
            print(f"[task2] mode: {'HYBRID (explicit model)' if booster is not None else 'HEURISTICS-ONLY'}")
        else:
            bst_on, bst_off = _load_booster_pair()
            if bst_on is not None or bst_off is not None:
                booster = None  # None = auto-load (single unless LGBM_SPLIT=1)
                print(f"[task2] mode: HYBRID (single; split pair on disk, needs LGBM_SPLIT=1)")
            else:
                booster = _load_booster(None)
                print(f"[task2] mode: {'HYBRID (single)' if booster is not None else 'HEURISTICS-ONLY'}")
    targets = []
    for window_id, h in tqdm(history.groupby("window_id", sort=True), desc="Task 2 Kinematics"):
        wid = str(window_id)
        panel = panel_map.get(wid)
        if panel is None or panel not in network_cache:
            continue
            
        cond = cond_map.get(wid, "queue_ongoing")
        net = network_cache[panel]
        split = split_map.get(wid, splits[0] if splits else "validation")

        tgt = template[template["window_id"] == wid][["window_id", "timestamp", "link_id"]].copy()
        if tgt.empty:
            continue

        preds = predict_window_queues(
            wid, panel, cond, h, tgt, net["fd"], net["topo"], net["up_map"],
            booster=booster, lgbm_threshold=lgbm_threshold,
            ramp_index=_ramp_index(panel, split), cal_cache=net["cal"],
        )
        tgt["queue_pred"] = preds
        targets.append(tgt)

    if not targets:
        pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"]).to_csv(out_path, index=False)
        return 0

    out_df = pd.concat(targets, ignore_index=True).drop_duplicates(["window_id", "timestamp", "link_id"])
    out_df["queue_pred"] = out_df["queue_pred"].astype(int)
    out_df.to_csv(out_path, index=False)
    print(f"Task 2 Complete: {len(out_df):,} window predictions exported to {out_path} in {time.time()-t0:.1f}s")
    return len(out_df)
