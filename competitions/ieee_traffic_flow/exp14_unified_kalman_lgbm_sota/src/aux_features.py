"""Shared aux features: EB climatology prior + ramp merge pressure.

cal_prior: p_EB(link, slot) = (hits + M*p_bar_corridor(slot)) / (N + M),
  slot = weekday*288 + tod (2016 slots), M=20. Built from train mainline_states.
ramp: on-ramp flows attached via network/ramp_attachment_map.csv
  (ramp_link_id -> nearest_mainline_link_id). Missing = NaN (never zero).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

N_SLOTS = 7 * 288
SHRINK_M = 20.0


def _slot_of(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts, utc=True)
    return (ts.dt.weekday * 288 + ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(dtype=np.int32)


def build_cal_cache(panel: str, release_root: Path, out_path: Path,
                    vcut_map: dict | None = None) -> dict:
    """Count (link, slot) queue hits over train mainline_states; save JSON cache."""
    from network_utils import load_fd_params
    pdir = release_root / "corridors" / panel
    fd = load_fd_params(pdir)
    vcut = vcut_map or {lid: fd.get(lid, {}).get("v_cut", 63.0) for lid in fd}
    files = sorted((pdir / "train" / "mainline_states").glob("**/*.parquet"))
    hits: dict[tuple[str, int], int] = {}
    tot_link_slot: dict[tuple[str, int], int] = {}
    hits_slot = np.zeros(N_SLOTS, dtype=np.float64)
    tot_slot = np.zeros(N_SLOTS, dtype=np.float64)
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh"])
        except Exception:
            continue
        df["link_id"] = df["link_id"].astype(str)
        sp = pd.to_numeric(df["speed_kmh"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(sp)
        if not ok.any():
            continue
        slot = _slot_of(df["timestamp"])
        cut = df["link_id"].map(vcut).fillna(63.0).to_numpy(dtype=float)
        q = (ok & (sp <= cut)).astype(np.int32)
        for lid, s, qq, oo in zip(df["link_id"].to_numpy(), slot, q, ok.astype(np.int32)):
            k = (str(lid), int(s))
            hits[k] = hits.get(k, 0) + int(qq)
            tot_link_slot[k] = tot_link_slot.get(k, 0) + int(oo)
        np.add.at(hits_slot, slot[ok], q[ok])
        np.add.at(tot_slot, slot[ok], 1)
    p_bar = hits_slot / np.maximum(tot_slot, 1.0)
    cal = {"p_bar": [float(x) for x in p_bar], "links": {}}
    by_link: dict[str, dict[str, list]] = {}
    for (lid, s), h in hits.items():
        n = tot_link_slot.get((lid, s), 0)
        if n == 0:
            continue
        p_eb = (h + SHRINK_M * float(p_bar[s])) / (n + SHRINK_M)
        by_link.setdefault(lid, []).append([s, round(p_eb, 4)])
    cal["links"] = by_link
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(cal))
    os.replace(tmp, out_path)
    return cal


def load_cal_cache(panel: str) -> dict:
    cands = [Path("scratch/training_run/cal_cache") / f"{panel}.json",
             Path("/home/arch/ipynb/ieee/scratch/training_run/cal_cache") / f"{panel}.json"]
    try:
        here = Path(__file__).resolve().parent
        cands.append(here.parent / "scratch" / "training_run" / "cal_cache" / f"{panel}.json")
    except Exception:
        pass
    env = os.environ.get("CAL_CACHE_DIR")
    if env:
        cands.insert(0, Path(env) / f"{panel}.json")
    for c in cands:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                pass
    return {}


def cal_prior_for(link: str, slot: int, cache: dict) -> float:
    if not cache:
        return 0.0
    for s, p in cache.get("links", {}).get(str(link), []):
        if s == int(slot):
            return float(p)
    pb = cache.get("p_bar", [])
    if pb and 0 <= int(slot) < len(pb):
        return float(pb[int(slot)])
    return 0.0


def load_attach(panel_dir: Path) -> pd.DataFrame:
    for name in ("ramp_attachment_map.csv", "synthetic_ramp_attachment_map.csv"):
        p = panel_dir / "network" / name
        if p.exists():
            df = pd.read_csv(p, dtype=str)
            if {"ramp_link_id", "nearest_mainline_link_id"}.issubset(df.columns):
                if "ramp_type" not in df.columns:
                    df["ramp_type"] = "OR"
                return df[["ramp_link_id", "ramp_type", "nearest_mainline_link_id"]].copy()
    return pd.DataFrame(columns=["ramp_link_id", "ramp_type", "nearest_mainline_link_id"])


def load_ramp_flows(panel_dir: Path, split: str,
                    dates: set | None = None) -> pd.DataFrame:
    """Long frame: timestamp, ml_link, ramp_flow (NaN if missing/ineligible).

    dates: optional set of datetime.date to restrict (trainer sampled days).
    """
    base = panel_dir / split / "ramp_states" if split != "train" else panel_dir / "train" / "ramp_states"
    if split == "train":
        base = panel_dir / "train" / "ramp_states"
    files = sorted(base.glob("**/*.parquet"))
    if dates:
        files = [f for f in files if any(d.isoformat().replace("-", "_") in f.name or
                                         d.strftime("%Y_%m_%d") in f.name for d in dates)]
        if not files:  # fall back: filter after load by date below
            files = sorted(base.glob("**/*.parquet"))
    attach = load_attach(panel_dir)
    on = attach[attach["ramp_type"] == "OR"][["ramp_link_id", "nearest_mainline_link_id"]].copy()
    if on.empty:
        return pd.DataFrame(columns=["timestamp", "ml_link", "ramp_flow", "ramp_ok"])
    rmap = dict(zip(on["ramp_link_id"].astype(str), on["nearest_mainline_link_id"].astype(str)))
    parts = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "flow_vph",
                                             "pct_observed", "is_score_eligible"])
        except Exception:
            continue
        df["ramp_link_id"] = df["ramp_link_id"].astype(str)
        df = df[df["ramp_link_id"].isin(rmap)].copy()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if dates:
            df = df[df["timestamp"].dt.date.isin(dates)]
            if df.empty:
                continue
        elig = (pd.to_numeric(df["pct_observed"], errors="coerce").ge(75)
                & df["flow_vph"].notna())
        if "is_score_eligible" in df.columns:
            elig = elig & df["is_score_eligible"].astype(bool)
        df["ml_link"] = df["ramp_link_id"].map(rmap)
        df["ramp_flow"] = np.where(elig.to_numpy(),
                                   pd.to_numeric(df["flow_vph"], errors="coerce").to_numpy(dtype=float),
                                   np.nan)
        df["ramp_ok"] = elig.to_numpy().astype(np.int8)
        parts.append(df[["timestamp", "ml_link", "ramp_flow", "ramp_ok"]])
    if not parts:
        return pd.DataFrame(columns=["timestamp", "ml_link", "ramp_flow", "ramp_ok"])
    out = pd.concat(parts, ignore_index=True)
    g = out.groupby(["ml_link", "timestamp"], as_index=False).agg(
        ramp_flow=("ramp_flow", "sum"), ramp_ok=("ramp_ok", "max"))
    # sum of NaNs -> 0 falsely; fix: NaN where no valid ramp
    g["ramp_flow"] = np.where(g["ramp_ok"] > 0, g["ramp_flow"], np.nan)
    return g.sort_values(["ml_link", "timestamp"]).reset_index(drop=True)


class RampIndex:
    """Fast per-(link, T) ramp lookup over sorted int64 timestamps."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.d: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if df is not None and len(df):
            df = df.sort_values(["ml_link", "timestamp"])
            ts = pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy()
            fl = df["ramp_flow"].to_numpy(dtype=float)
            ok = df["ramp_ok"].to_numpy(dtype=np.int8)
            for lid, idx in df.groupby("ml_link").indices.items():
                sel = np.sort(np.asarray(idx))
                self.d[str(lid)] = (ts[sel], fl[sel], ok[sel])

    def query(self, ml_link: str, T: pd.Timestamp, cap: float,
              main_flow_last: float) -> tuple[float, float, float, float]:
        nan = (np.nan, np.nan, np.nan, 1.0)
        e = self.d.get(str(ml_link))
        if e is None:
            return nan
        ts, fl, ok = e
        Tt = pd.Timestamp(T)
        if Tt.tzinfo is None:
            Tt = Tt.tz_localize("UTC")
        else:
            Tt = Tt.tz_convert("UTC")
        t = Tt.value
        lo = np.searchsorted(ts, t - 30 * 60 * 10**9)
        hi = np.searchsorted(ts, t)
        if hi <= lo:
            return nan
        c0 = np.searchsorted(ts, t - 15 * 60 * 10**9)
        cur_ok = ok[c0:hi] > 0
        if cur_ok.sum() == 0:
            return nan
        cur_fl = fl[c0:hi][cur_ok]
        cur_fl = cur_fl[np.isfinite(cur_fl)]
        if not len(cur_fl):
            return nan
        f15 = float(cur_fl.mean())
        prv_fl = fl[lo:c0][ok[lo:c0] > 0]
        prv_fl = prv_fl[np.isfinite(prv_fl)]
        surge = f15 - float(prv_fl.mean()) if len(prv_fl) else 0.0
        dem = ((main_flow_last if np.isfinite(main_flow_last) else 0.0) + f15) / max(cap, 1.0)
        return f15, float(surge), float(dem), 0.0


def ramp_features_at(ml_link: str, T: pd.Timestamp, ramp_df: pd.DataFrame,
                     cap: float, main_flow_last: float) -> tuple[float, float, float, float]:
    """(ramp_flow_15, ramp_surge, demand_ratio, ramp_missing) at forecast origin T."""
    return RampIndex(ramp_df).query(ml_link, T, cap, main_flow_last)
