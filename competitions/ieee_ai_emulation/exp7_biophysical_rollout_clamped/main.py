"""
Autobot AI Emulation Exp 6: STEP=5 (8 Hops) Log-Residual Delta Rollout SOTA
Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)

ARCHITECTURAL INNOVATION
---------------------------------------------------------------------------
In Exp 4 (STEP=20, 2 hops) score was 0.270.
In Exp 5 (STEP=10, 4 hops) score improved to 0.222 (Rank #76).
The primary failure mode diagnosed by Claude Opus 5.5 was tree-model extrapolation
error when predicting absolute states:
1. Tree models cannot extrapolate beyond training extremes; predicting absolute
   next state pulls predictions toward the training mean, compounding error
   exponentially over multi-hop rollouts.
2. Exp 6 implements:
   a. Log-residual delta target: \\Delta = log1p(s_{t+5}) - log1p(s_t).
      State update: s_{t+5} = expm1(log1p(s_t) + \\Delta).
      This guarantees states remain strictly non-negative by definition!
   b. STEP=5 (8 hops: 0->5->10->15->20->25->30->35->40) for high dynamical fidelity.
   c. Per-hop intermediate biophysical clipping (preventing cascading runaway drift).
   d. Initial stand state s_0 conditioned as persistent anchor features.
"""

from __future__ import annotations

import gc
import os
import random
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=== AUTOBOT AI EMULATION EXP 6: STEP=5 (8 HOPS) LOG-RESIDUAL ROLLOUT ===")
print(f"Python Runtime: {sys.version.split()[0]}")
t0_global = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Path Resolution
# ------------------------------------------------------------------------------
def find_file(name: str, search_dirs: tuple[str, ...] = ("/kaggle/input", ".")) -> str:
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            if name in files:
                return os.path.join(root, name)
    raise FileNotFoundError(f"Cannot find {name}")

glob_x_path = find_file("glob_X_fea.npy")
BASE = os.path.dirname(os.path.dirname(glob_x_path))
print("Resolved CarbonGlobe BASE:", BASE)

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
AGE_FILES = {age: f"glob_Y{age:03d}.npy" for age in SEED_AGES}
ALL_TARGET_NAMES = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
SCALE = {"height": 10.90645, "agb": 3.39567}

STATE_VARS = ["height", "agb", "soil", "lai"]
STATE_IDX = [ALL_TARGET_NAMES.index(v) for v in STATE_VARS]

STEP = 5
STEP_STARTS = list(range(0, 40, STEP))  # [0, 5, 10, 15, 20, 25, 30, 35] (8 hops)
N_HOPS = len(STEP_STARTS)
print(f"STEP={STEP}, STEP_STARTS={STEP_STARTS}, N_HOPS={N_HOPS} (Exp 5 used STEP=10, N_HOPS=4)")

SITE_SUBSAMPLE_FRAC = 0.12  # Bounds memory to ~1.6 GB and keeps CPU runtime ~50 min
NUM_FOLDS = 3
EPS = 1e-6

X = np.load(f"{BASE}/data_global/glob_X_fea.npy", mmap_mode="r")  # (54152, 40, 12, 136)
print("X shape:", X.shape)

train_sites = pd.read_csv(f"{BASE}/train.csv", header=None).iloc[:, 0].astype(int).values
val_sites = pd.read_csv(f"{BASE}/val.csv", header=None).iloc[:, 0].astype(int).values
test_sites_bench = pd.read_csv(f"{BASE}/test.csv", header=None).iloc[:, 0].astype(int).values
all_sites = np.unique(np.concatenate([train_sites, val_sites, test_sites_bench]))

rng = np.random.RandomState(RANDOM_SEED)
n_keep = int(len(all_sites) * SITE_SUBSAMPLE_FRAC)
all_sites = rng.choice(all_sites, size=n_keep, replace=False)
print(f"Subsampled site pool: {len(all_sites)} sites ({len(all_sites)/54152:.1%})")

stats = np.load(f"{BASE}/data_stats/data_stats.npz")
x_mean, x_std = stats["x_mean"], stats["x_std"]

# ------------------------------------------------------------------------------
# 2. Feature Engineering
# ------------------------------------------------------------------------------
def climate_window_feats(raw_window: np.ndarray) -> np.ndarray:
    """raw_window: (n, STEP, 12, 136). Returns (n, 4*136)."""
    raw_annual_mean = raw_window.mean(axis=2)  # (n, STEP, 136)
    Xn = (raw_annual_mean - x_mean) / (x_std + 1e-8)

    feat_mean = Xn.mean(axis=1)
    feat_std = Xn.std(axis=1)

    years = np.arange(raw_window.shape[1])
    years_c = years - years.mean()
    denom = max((years_c ** 2).sum(), 1e-6)
    feat_trend = (Xn * years_c[None, :, None]).sum(axis=1) / denom

    rel_change = np.clip((raw_annual_mean[:, -1, :] - raw_annual_mean[:, 0, :]) / (np.abs(raw_annual_mean[:, 0, :]) + EPS), -5, 5)

    return np.concatenate([feat_mean, feat_std, feat_trend, rel_change], axis=1).astype(np.float32)

def build_feature_names() -> list[str]:
    names = []
    for tag in ["mean", "std", "trend", "relchange"]:
        names += [f"clim_{tag}_{i}" for i in range(136)]
    names += [f"state_{v}" for v in STATE_VARS]
    names += [f"init_{v}" for v in STATE_VARS]
    names += ["stand_age"]
    return names

FEATURE_NAMES = build_feature_names()
N_FEAT = len(FEATURE_NAMES)
print(f"Features per step-transition row: {N_FEAT}")

# ------------------------------------------------------------------------------
# 3. Assemble Training Dataset
# ------------------------------------------------------------------------------
print(f"\nExtracting climate blocks for {len(all_sites)} sites...")
Xf_full = np.array(X[all_sites], dtype=np.float32)

window_feats_cache = {}
for t0 in STEP_STARTS:
    window_feats_cache[t0] = climate_window_feats(Xf_full[:, t0 : t0 + STEP, :, :])
del Xf_full
gc.collect()

n_sites = len(all_sites)
n_rows = n_sites * len(SEED_AGES) * N_HOPS
X_all = np.empty((n_rows, N_FEAT), dtype=np.float32)
y_all = np.empty((n_rows, len(STATE_VARS)), dtype=np.float32)
sid_all = np.empty(n_rows, dtype=np.int64)

state_max_bounds = {v: 0.0 for v in STATE_VARS}

print(f"Assembling {n_rows:,} step-transition rows...")
row = 0
for age in SEED_AGES:
    Y = np.load(f"{BASE}/data_global/{AGE_FILES[age]}", mmap_mode="r")
    Yf = np.array(Y[all_sites], dtype=np.float32)
    s0 = Yf[:, 0, -1, STATE_IDX]  # Initial state at t=0

    for vi, v in enumerate(STATE_VARS):
        state_max_bounds[v] = max(state_max_bounds[v], float(Yf[:, :, -1, STATE_IDX[vi]].max()))

    for t0 in STEP_STARTS:
        t1 = t0 + STEP
        st0 = Yf[:, t0, -1, STATE_IDX]
        st1 = Yf[:, t1, -1, STATE_IDX]

        # Log-residual delta target: log1p(st1) - log1p(st0)
        delta_target = np.log1p(np.maximum(st1, 0.0)) - np.log1p(np.maximum(st0, 0.0))
        stand_age = np.full((n_sites, 1), age + t0, dtype=np.float32)

        feats = np.concatenate([window_feats_cache[t0], st0, s0, stand_age], axis=1).astype(np.float32)

        lo, hi = row, row + n_sites
        X_all[lo:hi] = feats
        y_all[lo:hi] = delta_target
        sid_all[lo:hi] = all_sites
        row = hi
    del Yf
    gc.collect()

del window_feats_cache
gc.collect()
print(f"X_all shape: {X_all.shape} ({X_all.nbytes / (1024**3):.2f} GB)")
print("Physical state max bounds:", {k: round(v, 2) for k, v in state_max_bounds.items()})

# ------------------------------------------------------------------------------
# 4. Train LightGBM Models (GroupKFold by site_id)
# ------------------------------------------------------------------------------
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=50,
    lambda_l1=0.2,
    lambda_l2=0.5,
    feature_fraction=0.75,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
    n_jobs=-1,
)
NUM_BOOST_ROUND = 1500
EARLY_STOP = 60

gkf = GroupKFold(n_splits=NUM_FOLDS)
fold_indices = list(gkf.split(X_all, groups=sid_all))

models = {v: [] for v in STATE_VARS}
print(f"\n=== Training {len(STATE_VARS)} Delta-Transition Models x {NUM_FOLDS} Folds ===")

for vi, var in enumerate(STATE_VARS):
    y_target = y_all[:, vi]
    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
        dtrain = lgb.Dataset(X_all[tr_idx], label=y_target[tr_idx], feature_name=FEATURE_NAMES)
        dval = lgb.Dataset(X_all[va_idx], label=y_target[va_idx], feature_name=FEATURE_NAMES, reference=dtrain)

        m = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        models[var].append(m)
        val_pred = m.predict(X_all[va_idx])
        val_rmse = np.sqrt(mean_squared_error(y_target[va_idx], val_pred))
        print(f"  [{var} | Fold {fold_i+1}/{NUM_FOLDS}] best_iter={m.best_iteration} val delta RMSE={val_rmse:.5f}")

del X_all, y_all
gc.collect()

# ------------------------------------------------------------------------------
# 5. Future SSP Scenario Rollout Inference
# ------------------------------------------------------------------------------
def find_test_base():
    p = find_file("sites_ssp.csv")
    return os.path.dirname(p)

TEST_BASE = find_test_base()
print(f"\nResolved TEST_BASE: {TEST_BASE}")
df_sites = pd.read_csv(f"{TEST_BASE}/sites_ssp.csv")
if "site" in df_sites.columns:
    test_sites = df_sites["site"].astype(int).values
else:
    test_sites = pd.to_numeric(df_sites.iloc[:, 0], errors="coerce").dropna().astype(int).values
n_test = len(test_sites)
print(f"Loaded {n_test} test sites from sites_ssp.csv")

def rollout_scenario(scenario_name: str) -> pd.DataFrame:
    print(f"\n--- Commencing Rollout for Scenario: {scenario_name} ---")
    data = np.load(f"{TEST_BASE}/test_{scenario_name}.npz")
    X_test_clim = data["test_x"] if "test_x" in data else data["X"]  # (n_test, 40, 12, 136)
    Y_test_init = data["test_y0"] if "test_y0" in data else data["Y"]  # (n_test, 15, 7)

    # Precompute 5-year climate window features for test sites
    test_window_cache = {}
    for t0 in STEP_STARTS:
        test_window_cache[t0] = climate_window_feats(X_test_clim[:, t0 : t0 + STEP, :, :])

    dfs = []
    for age_idx, seed_age in enumerate(SEED_AGES):
        # Initial state at t=0
        current_state = np.array(Y_test_init[:, age_idx, STATE_IDX], dtype=np.float32)
        initial_anchor = current_state.copy()

        # Execute 8-hop rollout
        for t0 in STEP_STARTS:
            clim_f = test_window_cache[t0]
            stand_age = np.full((n_test, 1), seed_age + t0, dtype=np.float32)
            f_in = np.concatenate([clim_f, current_state, initial_anchor, stand_age], axis=1).astype(np.float32)

            next_state = np.empty_like(current_state)
            for vi, var in enumerate(STATE_VARS):
                # Ensemble average across folds
                delta_pred = np.mean([m.predict(f_in) for m in models[var]], axis=0)
                # Log-space reconstruction: expm1(log1p(current_state) + delta)
                reconstructed = np.expm1(np.log1p(np.maximum(current_state[:, vi], 0.0)) + delta_pred)
                # Intermediate biophysical clipping: bounded at strictly positive & 1.1x max historical
                max_bound = state_max_bounds[var] * 1.10
                next_state[:, vi] = np.clip(reconstructed, 0.05, max_bound)

            current_state = next_state

        pred_height = np.clip(current_state[:, 0] / SCALE["height"], 0.05, 45.0)
        pred_agb = np.clip(current_state[:, 1] / SCALE["agb"], 0.01, 50.0)

        df_block = pd.DataFrame({
            "id": [f"{s}_{seed_age:03d}_{scenario_name}" for s in test_sites],
            "height": pred_height,
            "agb": pred_agb,
        })
        dfs.append(df_block)

    return pd.concat(dfs, ignore_index=True)

sub_126 = rollout_scenario("ssp126")
sub_585 = rollout_scenario("ssp585")

submission = pd.concat([sub_126, sub_585], ignore_index=True)

# ------------------------------------------------------------------------------
# 6. Integrity Verification & Submission Export
# ------------------------------------------------------------------------------
print("\n=== Submission Integrity Verification ===")
expected_rows = 2 * len(test_sites) * len(SEED_AGES)
print(f"Total Rows: {len(submission)} (expected {expected_rows})")
assert len(submission) == expected_rows, f"Row count mismatch! {len(submission)} vs {expected_rows}"
assert list(submission.columns) == ["id", "height", "agb"], "Column header mismatch!"
assert submission.isna().sum().sum() == 0, "Null values found!"
assert (submission["height"] > 0).all(), "Zero or negative height values found!"
assert (submission["agb"] > 0).all(), "Zero or negative agb values found!"

print("Height Stats:")
print(submission["height"].describe())
print("AGB Stats:")
print(submission["agb"].describe())

out_file = "/kaggle/working/submission.csv"
submission.to_csv(out_file, index=False)
print(f"\nSUCCESS: Exported {out_file} ({os.path.getsize(out_file):,} bytes)")
print(f"Total Wall Clock: {time.time() - t0_global:.1f}s")
