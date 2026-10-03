"""Network, topology, and Fundamental Diagram parameters loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def load_lane_counts(panel_dir: Path) -> dict[str, float]:
    """Extract lane counts keyed by link_id and station_id."""
    lanes: dict[str, float] = {}
    links_path = panel_dir / "network" / "links.csv"
    fd_path = panel_dir / "network" / "fd_parameters.csv"
    
    if links_path.exists():
        df = pd.read_csv(links_path, dtype={"link_id": str})
        if "lanes" in df.columns:
            lns = pd.to_numeric(df["lanes"], errors="coerce")
            valid = lns.notna() & (lns >= 1) & (lns <= 16)
            for lid, l in zip(df["link_id"][valid], lns[valid]):
                lanes[str(lid)] = float(l)
                
    if fd_path.exists():
        df = pd.read_csv(fd_path, dtype={"station_id": str, "link_id": str})
        if "lanes" in df.columns:
            lns = pd.to_numeric(df["lanes"], errors="coerce")
            valid = lns.notna() & (lns >= 1) & (lns <= 16)
            for _, row in df[valid].iterrows():
                if pd.notna(row.get("station_id")):
                    lanes[str(row["station_id"])] = float(row["lanes"])
                if pd.notna(row.get("link_id")):
                    lanes[str(row["link_id"])] = float(row["lanes"])
    return lanes


def load_fd_params(panel_dir: Path) -> dict[str, dict[str, float]]:
    """Load per-link fundamental diagram parameters with validated triangular FD properties.
    
    Computes:
    - vf: free-flow speed (km/h)
    - cap: capacity (veh/h total)
    - lanes: number of lanes
    - length: length of link (km)
    - k_crit: critical density (veh/km total) = cap / vf
    - k_jam: jam density (veh/km total)
    - wave_speed: backward shockwave speed w = cap / (k_jam - k_crit) (km/h)
    - v_cut: queue cutoff speed = 0.60 * vf (km/h)
    """
    links_path = panel_dir / "network" / "links.csv"
    fd_path = panel_dir / "network" / "fd_parameters.csv"
    params: dict[str, dict[str, Any]] = {}
    
    if links_path.exists():
        df = pd.read_csv(links_path, dtype={"link_id": str})
        for _, r in df.iterrows():
            lid = str(r["link_id"])
            vf = float(pd.to_numeric(r.get("free_speed_kmh"), errors="coerce") or 105.0)
            cap = float(pd.to_numeric(r.get("capacity_vph"), errors="coerce") or 0)
            ln = float(pd.to_numeric(r.get("lanes"), errors="coerce") or 4.0)
            ln = max(ln, 1.0)
            length = float(pd.to_numeric(r.get("length_km"), errors="coerce") or 1.0)
            if cap <= 0:
                cap = 1800.0 * ln
            params[lid] = {"vf": vf, "cap": cap, "lanes": ln, "length": length}

    if fd_path.exists():
        fd = pd.read_csv(fd_path, dtype={"link_id": str, "station_id": str})
        for lid, g in fd.groupby("link_id"):
            lid = str(lid)
            if lid not in params:
                params[lid] = {"vf": 105.0, "cap": 7200.0, "lanes": 4.0, "length": 1.0}
            if "free_speed_kmh" in g.columns:
                v = g["free_speed_kmh"].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]["vf"] = float(v)
            if "capacity_vph" in g.columns:
                v = g["capacity_vph"].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]["cap"] = float(v)
            if "lanes" in g.columns:
                v = g["lanes"].dropna().mean()
                if pd.notna(v) and v >= 1:
                    params[lid]["lanes"] = float(v)
            if "k_jam" in g.columns:
                v = g["k_jam"].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]["k_jam_raw"] = float(v)
            if "v_cut" in g.columns:
                v = g["v_cut"].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]["v_cut_raw"] = float(v)

    for lid, p in params.items():
        vf = p["vf"]
        cap = p["cap"]
        ln = p["lanes"]
        p["k_crit"] = cap / max(vf, 1.0)
        p["k_jam"] = p.get("k_jam_raw", 100.0 * max(ln, 1.0))
        p["k_jam"] = max(p["k_jam"], p["k_crit"] * 1.05)
        p["wave_speed"] = cap / max(p["k_jam"] - p["k_crit"], 1e-9)
        p["v_cut"] = p.get("v_cut_raw", vf * 0.60)
        
    return params


def load_topology(panel_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Parse lwr_mainline_topology.csv to extract upstream and downstream link relations."""
    topo_path = panel_dir / "network" / "lwr_mainline_topology.csv"
    topology: dict[str, dict[str, list[str]]] = {}
    if not topo_path.exists():
        return topology
        
    df = pd.read_csv(topo_path, dtype=str)
    for _, row in df.iterrows():
        lid = str(row["link_id"])
        incoming = [x.strip() for x in str(row.get("incoming_link_ids", "")).split(";") if x.strip() and x != "nan"]
        outgoing = [x.strip() for x in str(row.get("outgoing_link_ids", "")).split(";") if x.strip() and x != "nan"]
        on_ramps = [x.strip() for x in str(row.get("on_ramp_link_ids", "")).split(";") if x.strip() and x != "nan"]
        off_ramps = [x.strip() for x in str(row.get("off_ramp_link_ids", "")).split(";") if x.strip() and x != "nan"]
        topology[lid] = {
            "incoming": incoming,
            "outgoing": outgoing,
            "on_ramps": on_ramps,
            "off_ramps": off_ramps
        }
    return topology


def load_ordered_links(panel_dir: Path) -> list[str]:
    """Determine physical upstream-to-downstream link sequence along the corridor.
    
    Uses milepost from links.csv or train mainline states if available,
    falling back to topological sorting on the directed corridor graph.
    """
    links_path = panel_dir / "network" / "links.csv"
    if links_path.exists():
        df = pd.read_csv(links_path, dtype={"link_id": str})
        if "milepost" in df.columns:
            mp = pd.to_numeric(df["milepost"], errors="coerce")
            if mp.notna().sum() > len(df) * 0.5:
                df["mp"] = mp.fillna(9999.0)
                ordered = df.sort_values(["mp", "link_id"])["link_id"].unique().tolist()
                return [str(x) for x in ordered]
                
    # Fallback to topology traversal
    topo = load_topology(panel_dir)
    if not topo:
        if links_path.exists():
            return [str(x) for x in pd.read_csv(links_path, dtype={"link_id": str})["link_id"].unique()]
        return []
        
    # Find upstream root nodes (in-degree == 0)
    in_degrees = {k: len(v["incoming"]) for k, v in topo.items()}
    visited: list[str] = []
    queue = [k for k, deg in in_degrees.items() if deg == 0]
    if not queue:
        queue = list(topo.keys())[:1]
        
    seen = set()
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        for succ in topo.get(node, {}).get("outgoing", []):
            if succ not in seen and succ in topo:
                queue.append(succ)
                
    for k in topo:
        if k not in seen:
            visited.append(k)
    return visited


def get_lane_vector(stations: pd.Series | np.ndarray, links: pd.Series | np.ndarray, lanes_map: dict[str, float]) -> np.ndarray:
    """Vectorized retrieval of lane counts for given stations and links."""
    st_arr = np.asarray(stations, dtype=str)
    lk_arr = np.asarray(links, dtype=str)
    n = len(st_arr)
    result = np.ones(n, dtype=float)
    for i in range(n):
        s = st_arr[i]
        l = lk_arr[i]
        if s in lanes_map:
            result[i] = lanes_map[s]
        elif l in lanes_map:
            result[i] = lanes_map[l]
        else:
            result[i] = 4.0
    return np.maximum(result, 1.0)
