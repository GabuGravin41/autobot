"""Configuration constants and path discovery for TrafficFlowBench 2026."""
from __future__ import annotations

import glob
from pathlib import Path

# Competition Corridor Panels (10 directional panels across 5 freeway corridor families)
PANELS = [
    "D7_I10_E", "D7_I10_W",
    "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S",
    "D12_I5_N", "D12_I5_S",
    "D12_I405_N", "D12_I405_S"
]

CORRIDOR_FAMILIES = {
    "D7_I10_E": "D7_I10",
    "D7_I10_W": "D7_I10",
    "D7_I210_E": "D7_I210",
    "D7_I210_W": "D7_I210",
    "D7_I405_N": "D7_I405",
    "D7_I405_S": "D7_I405",
    "D12_I5_N": "D12_I5",
    "D12_I5_S": "D12_I5",
    "D12_I405_N": "D12_I405",
    "D12_I405_S": "D12_I405"
}

# Corridors excluded from Task 2 (Queue Forecasting) by competition specification
TASK2_EXCLUDED_PANELS = {"D12_I405_N", "D12_I405_S"}
TASK2_ACTIVE_PANELS = [p for p in PANELS if p not in TASK2_EXCLUDED_PANELS]

# Task Weights in Final Composite Leaderboard Score
# S_total = 0.35 * S_state + 0.30 * S_queue + 0.15 * S_physics + 0.20 * S_ODME
WEIGHT_TASK1_STATE = 0.35
WEIGHT_TASK2_QUEUE = 0.30
WEIGHT_TASK3_PHYSICS = 0.15
WEIGHT_TASK4_ODME = 0.20

# Task 1 Scoring Parameters
SPEED_NORMALIZER = 25.0       # km/h
FLOW_NORMALIZER = 600.0       # veh/h/lane
SPEED_WEIGHT = 0.54
FLOW_WEIGHT = 0.46
REGIMES = ("R1", "R2", "R3")

# Time Resolution Constants
INTERVAL_MINUTES = 5.0
SLOTS_PER_DAY = 288           # 24 hours * 12 slots/hour
SLOTS_PER_WEEK = 7 * 288      # 2,016 slots/week

# Task 2 Horizon Constants
TASK2_HISTORY_STEPS = 12      # 60 minutes history
TASK2_HORIZON_STEPS = 6       # 30 minutes horizon (T+5, T+10, T+15, T+20, T+25, T+30)

# Task 4 Default Regularization Parameter (calibrated to official benchmark setting: S_link = 0.9999)
DEFAULT_REG_LAMBDA = 0.05


def locate_release_root(custom_path: str | Path | None = None) -> Path:
    """Locate the competition dataset release directory.
    
    Checks standard Kaggle paths, local paths, and environment overrides.
    """
    if custom_path:
        p = Path(custom_path).resolve()
        if (p / "config" / "corridors.json").exists() or (p / "corridors").exists():
            return p
        if p.name == "corridors" and p.parent.exists():
            return p.parent

    candidates = [
        "/kaggle/input/2026-ieee-big-data-traffic-flow-bench/kaggle_public",
        "/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public",
        "/kaggle/input/2026-ieee-big-data-traffic-flow-bench",
        "/home/arch/ipynb/ieee/data/kaggle_public",
        "/home/arch/ipynb/ieee/data",
        "/home/arch/ipynb/ieee/kaggle_public",
        "kaggle_public",
        "data/kaggle_public",
        "."
    ]
    # Also look inside /kaggle/input if running on Kaggle
    candidates += glob.glob("/kaggle/input/**/kaggle_public", recursive=True)
    candidates += glob.glob("/kaggle/input/**/corridors", recursive=True)

    for c in candidates:
        p = Path(c)
        if (p / "config" / "corridors.json").exists() or (p / "corridors").exists():
            return p
        if p.name == "corridors" and p.parent.exists():
            return p.parent

    for p in Path(".").rglob("submission_key.csv"):
        return p.parent

    raise FileNotFoundError(
        "Could not find competition release root (kaggle_public). "
        "Please provide --release-root or place data under data/kaggle_public."
    )
