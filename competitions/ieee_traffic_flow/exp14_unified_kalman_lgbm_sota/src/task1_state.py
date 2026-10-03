"""Task 1: Traffic State Reconstruction & Task 3 Physics Consistency.

v2 Calibrated Precision (LB 0.72682 peak):
1. Physical corridor topological link ordering
2. Temporal interpolation (limit=12) + spatial kernel (limit=8) with 85/15 blend
3. NO congested FD projection (v1 bug: displaced flow RMSE, hurt S_state)
4. Rolling density smoothing (window=3, 85/15 blend) for LWR without flow distortion
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

try:
    from scipy.interpolate import PchipInterpolator
    _HAS_PCHIP = True
except Exception:
    PchipInterpolator = None  # type: ignore
    _HAS_PCHIP = False

# Step 12 hunt knobs (defaults = v2 baseline, LB-proven 0.72682).
# LWR-safe only: monotone/gentle ops. Every change RMSE-scored on train first.
T1_DENOISE = False    # rolling-median-3 on observed per-link values pre-interp
T1_HIST_W = 0.0       # historical-prior blend weight (0.05 = regularizer)
T1_TEMP_LIMIT = 12    # temporal interpolation limit (steps of 5 min)
# Steps 4-5 ABLATED 2026-09-22 (PCHIP hurt 0.8987->0.8723; Kriging no gain).
# Kept as code paths for experiments; defaults OFF = v2 proven baseline.
USE_PCHIP = False
USE_KRIGING = False

# Physics corrections: soft FD projection + iterative conservation balance across links
# Validated across 10/10 train panels: +0.0062 S_state gain, -65% conservation residual
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


def _denoise_link(s: pd.Series) -> pd.Series:
    """Median-3 over observed points only; NaN structure untouched."""
    m = s.notna()
    if m.sum() < 3:
        return s
    v = s[m].rolling(3, center=True, min_periods=1).median()
    out = s.copy()
    out[m] = v.to_numpy()
    return out
# Step 5: Kriging-like distance-weighted spatial fill along corridor order
# (Gaussian kernel sigma=2km, nearest k=8). Falls back to linear limit=8.
KRIGING_SIGMA_KM = 2.0
KRIGING_K = 8


def _pchip_fill(s: pd.Series, ts: pd.Series, limit: int = 12) -> pd.Series:
    """Monotonic PCHIP fill for one link's series; only within `limit` steps of data."""
    y = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    try:
        x = ts.astype("int64").to_numpy() // 1_000_000_000
    except Exception:
        return s.interpolate(method="linear", limit_direction="both", limit=limit)
    m = np.isfinite(y)
    if m.sum() < 3 or not _HAS_PCHIP:
        return s.interpolate(method="linear", limit_direction="both", limit=limit)
    try:
        ser = pd.Series(y[m], index=x[m]).groupby(level=0).mean().sort_index()
        xs = ser.index.to_numpy(dtype=float)
        ys = ser.to_numpy(dtype=float)
        if len(xs) < 3:
            return s.interpolate(method="linear", limit_direction="both", limit=limit)
        f = PchipInterpolator(xs, ys, extrapolate=False)
    except Exception:
        return s.interpolate(method="linear", limit_direction="both", limit=limit)
    out = y.copy()
    na = ~m
    if na.any():
        idx = np.arange(len(y))
        nearest = np.abs(idx[na, None] - idx[m][None, :]).min(axis=1)
        fill = np.zeros(len(y), dtype=bool)
        fill[np.where(na)[0][nearest <= limit]] = True
        with np.errstate(all="ignore"):
            vals = f(x[fill])
        out[fill] = np.where(np.isfinite(vals), vals, out[fill])
    res = pd.Series(out, index=s.index)
    # boundary NaN beyond PCHIP range -> linear fallback within limit
    return res.interpolate(method="linear", limit_direction="both", limit=limit)


def _kriging_fill(vals: np.ndarray, dists: np.ndarray,
                  sigma: float = KRIGING_SIGMA_KM, k: int = KRIGING_K) -> np.ndarray:
    """Distance-weighted fill along corridor order for one timestamp slice.

    vals: temporal values (NaN where missing); dists: cumulative km per cell
    (-1 = unknown link, left for fallback). Weight w=exp(-|d-dk|/sigma) over
    nearest k known neighbours.
    """
    out = vals.copy()
    known = np.isfinite(vals) & (dists >= 0)
    if known.sum() == 0:
        return out
    kd = dists[known]
    kv = vals[known]
    for j in np.where(~np.isfinite(vals))[0]:
        if dists[j] < 0:
            continue
        d = np.abs(kd - dists[j])
        take = np.argsort(d, kind="stable")[:k]
        w = np.exp(-d[take] / sigma)
        ws = w.sum()
        if ws > 0:
            out[j] = float((w * kv[take]).sum() / ws)
    return out


def compute_time_slots(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute 0-indexed weekday (0..6) and 5-min time-of-day slot (0..287)."""
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    weekday = ts.dt.weekday.to_numpy(dtype=np.int16)
    tod = (ts.dt.hour * 12 + ts.dt.minute // int(INTERVAL_MINUTES)).to_numpy(dtype=np.int16)
    return weekday, tod


def build_historical_profiles(panel_dir: Path) -> tuple[dict, dict]:
    """Build fine-grained (link x slot) historical profiles from training mainline states."""
    train_files = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
    if not train_files:
        train_files = sorted((panel_dir / "train" / "mainline_states_masked").glob("**/*.parquet"))
        
    if not train_files:
        raise FileNotFoundError(f"No training parquet files found for corridor {panel_dir.name}")

    first_df = pd.read_parquet(train_files[0], columns=["link_id"])
    link_ids = sorted(first_df["link_id"].astype(str).unique())
    link_idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_ids)
    n_slots = SLOTS_PER_WEEK

    speed_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    speed_cnt = np.zeros((n_links, n_slots), dtype=np.int64)
    flow_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    flow_cnt = np.zeros((n_links, n_slots), dtype=np.int64)

    for path in train_files:
        df = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"])
        df["link_id"] = df["link_id"].astype(str)
        idx = df["link_id"].map(link_idx).to_numpy(dtype=np.int64)
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
    """Reconstruct masked states for a single corridor panel and split."""
    link_order_map = {lid: i for i, lid in enumerate(ordered_links)}
    link_idx = sp_prof["link_idx"]
    pfiles = sorted((panel_dir / split / "mainline_states_masked").glob("**/*.parquet"))
    
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

        hist_sp = sp_prof["mean"][valid_idx, slot]
        hist_fl = fl_prof["mean"][valid_idx, slot]
        hist_sp = np.where(sp_prof["cnt"][valid_idx, slot] == 0, sp_prof["fallback"][valid_idx], hist_sp)
        hist_fl = np.where(fl_prof["cnt"][valid_idx, slot] == 0, fl_prof["fallback"][valid_idx], hist_fl)

        lanes_arr = get_lane_vector(df["station_id"], df["link_id"], lanes_map)
        df["flow_per_lane"] = df["flow_vph"] / np.maximum(lanes_arr, 1.0)
        hist_fl_pl = hist_fl / np.maximum(lanes_arr, 1.0)

        # 1. High-fidelity temporal continuity per link (limit=12 steps = 60 min)
        # Step 4: PCHIP (monotonic) per link; linear fallback. Same 12-step window.
        df_sorted_t = df.sort_values(["link_id", "timestamp"]).reset_index()
        if T1_DENOISE:
            # LWR-safe: denoise OBSERVED inputs only; gaps untouched, no smearing
            df_sorted_t["speed_kmh"] = df_sorted_t.groupby("link_id")["speed_kmh"].transform(_denoise_link)
            df_sorted_t["flow_per_lane"] = df_sorted_t.groupby("link_id")["flow_per_lane"].transform(_denoise_link)
        if USE_PCHIP and _HAS_PCHIP:
            t_sp = np.full(len(df_sorted_t), np.nan)
            t_fl = np.full(len(df_sorted_t), np.nan)
            for _lid, _sel in df_sorted_t.groupby("link_id").indices.items():
                _sel = np.sort(np.asarray(_sel))
                _ts = df_sorted_t["timestamp"].iloc[_sel]
                t_sp[_sel] = _pchip_fill(
                    df_sorted_t["speed_kmh"].iloc[_sel], _ts, T1_TEMP_LIMIT).to_numpy()
                t_fl[_sel] = _pchip_fill(
                    df_sorted_t["flow_per_lane"].iloc[_sel], _ts, T1_TEMP_LIMIT).to_numpy()
            df_sorted_t["t_sp"] = t_sp
            df_sorted_t["t_fl"] = t_fl
        else:
            t_sp = df_sorted_t.groupby("link_id")["speed_kmh"].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both", limit=T1_TEMP_LIMIT)
            )
            t_fl = df_sorted_t.groupby("link_id")["flow_per_lane"].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both", limit=T1_TEMP_LIMIT)
            )
            df_sorted_t["t_sp"] = t_sp
            df_sorted_t["t_fl"] = t_fl
        df = df_sorted_t.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

        # 2. Topological spatial continuity along physical corridor link sequence
        # Step 5: Kriging-like Gaussian distance weighting (sigma=2km, k=8)
        # along cumulative link lengths; v1/v2 used linear limit=8 on rank order.
        df["_spatial_rank"] = df["link_id"].map(link_order_map).fillna(9999).astype(int)
        df_sorted_s = df.sort_values(["timestamp", "_spatial_rank"]).reset_index()
        if USE_KRIGING:
            _lens = np.array(
                [fd.get(str(l), {}).get("length", 1.0) for l in ordered_links],
                dtype=float)
            _cum = np.concatenate([[0.0], np.cumsum(_lens)[:-1]]) if len(_lens) else np.array([])
            _pos2dist = {str(lid): float(_cum[i]) for i, lid in enumerate(ordered_links)}
            _d = df_sorted_s["link_id"].astype(str).map(_pos2dist).fillna(-1.0).to_numpy(dtype=float)
            _tsp = df_sorted_s["t_sp"].to_numpy(dtype=float)
            _tfl = df_sorted_s["t_fl"].to_numpy(dtype=float)
            _ssp = np.full_like(_tsp, np.nan)
            _sfl = np.full_like(_tfl, np.nan)
            for _ts, _sel in df_sorted_s.groupby("timestamp").indices.items():
                _sel = np.sort(np.asarray(_sel))
                _ssp[_sel] = _kriging_fill(_tsp[_sel], _d[_sel])
                _sfl[_sel] = _kriging_fill(_tfl[_sel], _d[_sel])
            df_sorted_s["s_sp"] = _ssp
            df_sorted_s["s_fl"] = _sfl
        else:
            s_sp = df_sorted_s.groupby("timestamp")["t_sp"].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both", limit=8)
            )
            s_fl = df_sorted_s.groupby("timestamp")["t_fl"].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both", limit=8)
            )
            df_sorted_s["s_sp"] = s_sp
            df_sorted_s["s_fl"] = s_fl
        df = df_sorted_s.sort_values("index").drop(columns=["index", "_spatial_rank"]).reset_index(drop=True)

        t_sp_v = df["t_sp"].to_numpy(dtype=float)
        s_sp_v = df["s_sp"].to_numpy(dtype=float)
        t_fl_v = df["t_fl"].to_numpy(dtype=float)
        s_fl_v = df["s_fl"].to_numpy(dtype=float)

        # 3. Calibrated blend surface (85% temporal + 15% spatial, v2 optimum).
        # v1 used 82/18 which over-weighted noisy spatial neighbours.
        pred_speed = np.where(
            np.isfinite(t_sp_v),
            0.85 * t_sp_v + 0.15 * np.where(np.isfinite(s_sp_v), s_sp_v, t_sp_v),
            np.where(np.isfinite(s_sp_v), 0.90 * s_sp_v + 0.10 * hist_sp, hist_sp),
        )

        pred_fl_pl = np.where(
            np.isfinite(t_fl_v),
            0.85 * t_fl_v + 0.15 * np.where(np.isfinite(s_fl_v), s_fl_v, t_fl_v),
            np.where(np.isfinite(s_fl_v), 0.90 * s_fl_v + 0.10 * hist_fl_pl, hist_fl_pl),
        )

        pred_flow_raw = pred_fl_pl * lanes_arr

        # Step 12: historical-prior regularization (gentle, LWR-safe)
        if T1_HIST_W > 0:
            pred_speed = (1.0 - T1_HIST_W) * pred_speed + T1_HIST_W * hist_sp
            pred_fl_pl = (1.0 - T1_HIST_W) * pred_fl_pl + T1_HIST_W * hist_fl_pl
            pred_flow_raw = pred_fl_pl * lanes_arr

        # 4. NO congested FD projection (v1 bug fix).
        # v1 forced q toward q_cong_fd = w*v*k_jam/(v+w) with 12% blend,
        # inflating flow RMSE vs stochastic sim truth. v2 uses raw flux directly.
        # FD params only used for clipping bounds below.
        vf_arr = np.array([fd.get(str(l), {}).get("vf", 105.0) for l in df["link_id"]])
        cap_arr = np.array([fd.get(str(l), {}).get("cap", 7200.0) for l in df["link_id"]])

        pred_speed_clipped = np.clip(pred_speed, 5.0, vf_arr * 1.05)
        pred_flow_fd = pred_flow_raw

        # 5. Continuous vehicle density accumulation smoothing (LWR mass balance)
        df["pred_sp_tmp"] = pred_speed_clipped
        df["pred_fl_tmp"] = np.maximum(pred_flow_fd, 50.0)

        df_sorted_link = df.sort_values(["link_id", "timestamp"]).reset_index()
        df_sorted_link["density_raw"] = df_sorted_link["pred_fl_tmp"] / np.maximum(df_sorted_link["pred_sp_tmp"], 1.0)
        df_sorted_link["density_smooth"] = df_sorted_link.groupby("link_id")["density_raw"].transform(
            lambda s: s.rolling(3, center=True, min_periods=1).mean()
        )
        smooth_fl = df_sorted_link["density_smooth"] * df_sorted_link["pred_sp_tmp"]

        # Gentle mass conservation blend (85% raw + 15% smooth, v2 calibrated)
        df_sorted_link["final_flow"] = 0.85 * df_sorted_link["pred_fl_tmp"] + 0.15 * smooth_fl
        df_back = df_sorted_link.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

        final_speed = np.clip(df_back["pred_sp_tmp"].to_numpy(dtype=float), 5.0, vf_arr * 1.05)
        final_flow = np.clip(df_back["final_flow"].to_numpy(dtype=float), 50.0, cap_arr * 1.15)

        # 6. Physics corrections: soft FD projection + iterative link-transition conservation
        if USE_PHYSICS_CORRECTIONS and apply_physics_corrections is not None:
            final_speed, final_flow = apply_physics_corrections(
                final_speed,
                final_flow,
                df["link_id"].astype(str).to_numpy(),
                df["timestamp"].to_numpy(),
                ordered_links,
                fd,
                fd_alpha=FD_ALPHA,
                cons_alpha=CONSERVATION_ALPHA,
                cons_passes=CONSERVATION_PASSES,
            )
            final_speed = np.clip(final_speed, 5.0, vf_arr * 1.05)
            final_flow = np.clip(final_flow, 50.0, cap_arr * 1.15)

        # Filter only eligible blanked targets
        eligible = df["is_score_eligible"].astype(bool).to_numpy()
        blanked = eligible & df["speed_kmh"].isna().to_numpy() & df["flow_vph"].isna().to_numpy()

        for regime in pd.unique(df["mask_regime"].astype(str)):
            target = blanked & (df["mask_regime"].astype(str) == regime).to_numpy()
            mask = target & np.isfinite(final_speed) & np.isfinite(final_flow)
            if not mask.any():
                continue
            batch = pd.DataFrame({
                "panel": panel,
                "timestamp": df.loc[mask, "timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
                "station_id": df.loc[mask, "station_id"].to_numpy(),
                "link_id": df.loc[mask, "link_id"].to_numpy(),
                "mask_regime": regime,
                "speed_kmh": final_speed[mask],
                "flow_vph": final_flow[mask],
            })
            batches.append(batch)

    if batches:
        return pd.concat(batches, ignore_index=True)
    return pd.DataFrame(columns=["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"])


def run_task1(
    release_root: Path,
    splits: list[str],
    out_path: Path,
    panels: list[str] | None = None
) -> int:
    """Execute complete Task 1 state reconstruction across corridors and export CSV."""
    if panels is None:
        panels = PANELS
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    write_header = True
    t0 = time.time()
    
    print(f"Starting Task 1 state reconstruction for splits: {splits}")
    for panel in panels:
        tp = time.time()
        panel_dir = release_root / "corridors" / panel
        lanes_map = load_lane_counts(panel_dir)
        fd = load_fd_params(panel_dir)
        ordered_links = load_ordered_links(panel_dir)
        sp_prof, fl_prof = build_historical_profiles(panel_dir)
        
        panel_rows = 0
        for split in splits:
            reconstructed_df = reconstruct_panel_split(
                panel, panel_dir, split, sp_prof, fl_prof, lanes_map, fd, ordered_links
            )
            if not reconstructed_df.empty:
                reconstructed_df.to_csv(out_path, mode="w" if write_header else "a", header=write_header, index=False)
                write_header = False
                n = len(reconstructed_df)
                panel_rows += n
                total_rows += n
                
        print(f"[{panel}] Reconstructed {panel_rows:,} cells in {time.time()-tp:.1f}s")
        
    print(f"Task 1 Complete: {total_rows:,} total rows exported to {out_path} in {time.time()-t0:.1f}s")
    return total_rows
