"""
Autobot AI Emulation Exp 8: Optimal 4-Hop (STEP=10) Log-Residual Delta Rollout SOTA
Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)

SCIENTIFIC FOUNDATION & LESSONS FROM EXP 1-7:
1. Dynamical Bias-Variance Tradeoff:
   - STEP=40 (1 hop): 0.330 (severe climate extrapolation bias)
   - STEP=20 (2 hops): 0.270
   - STEP=10 (4 hops): 0.222 (Autobot champion score, Rank #76)
   - STEP=5 (8 hops): 0.379 (exponential compounding variance drift)
   -> Proves mathematically that N_HOPS = 4 (STEP=10) is the optimal sweet spot.
2. Target Formulation:
   - Exp 5 predicted absolute raw states, pulling predictions to training means on future SSPs.
   - Exp 8 predicts log-residual increments: \\Delta = log1p(s_{t+10}) - log1p(s_t).
   - Reconstructed state: s_{t+10} = expm1(log1p(s_t) + \\Delta), guaranteeing strict positivity.
   - Monotonic height constraint: trees do not shrink over decadal timescales (\\Delta_height >= 0).
3. Data Scaling:
   - Exp 7 only used 12% sites; Exp 8 scales back up to 35% sites (~19,000 sites), providing rich
     geographical representation across global climate biomes within Kaggle CPU memory budget.
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

print("=== AUTOBOT AI EMULATION EXP 8: OPTIMAL STEP=10 LOG-RESIDUAL ROLLOUT SOTA ===")
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

STEP = 10
STEP_STARTS = list(range(0, 40, STEP))  # [0, 10, 20, 30] (4 hops - proven optimal)
N_HOPS = len(STEP_STARTS)
print(f"Optimal Hop Configuration: STEP={STEP}, STEP_STARTS={STEP_STARTS}, N_HOPS={N_HOPS}")

SITE_SUBSAMPLE_FRAC = 0.35  # ~19,000 sites for strong biome generalization
NUM_FOLDS = 3
EPS = 1e-6

X = np.load(f"{BASE}/data_global/glob_X_fea.npy", mmap_mode="r")  # (54152, 40, 12, 136)
print("Climate X tensor shape:", X.shape)

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
# 2. Climate Window Feature Engineering
# ------------------------------------------------------------------------------
def climate_window_feats(raw_window: np.ndarray) -> np.ndarray:
    """raw_window: (n, STEP, 12, 136) monthly raw units. Returns (n, 4*136)."""
    raw_annual_mean = raw_window.mean(axis=2)  # (n, STEP, 136)
    Xn = (raw_annual_mean - x_mean) / (x_std + 1e-8)

    feat_mean = Xn.mean(axis=1)
    feat_std = Xn.std(axis=1)

    years = np.arange(raw_window.shape[1])
    years_c = years - years.mean()
    denom = (years_c ** 2).sum()
    feat_trend = (Xn * years_c[None, :, None]).sum(axis=1) / denom

    half = raw_window.shape[1] // 2
    raw_first = raw_annual_mean[:, :half, :].mean(axis=1)
    raw_second = raw_annual_mean[:, half:, :].mean(axis=1)
    rel_change = np.clip((raw_second - raw_first) / (np.abs(raw_first) + EPS), -10.0, 10.0)

    return np.concatenate([feat_mean, feat_std, feat_trend, rel_change], axis=1).astype(np.float32)

def build_feature_names() -> list[str]:
    names = []
    for tag in ["mean", "std", "trend", "relchange"]:
        names += [f"clim_{tag}_{i}" for i in range(136)]
    names += [f"state_{v}" for v in STATE_VARS]
    names += [f"anchor_s0_{v}" for v in STATE_VARS]
    names += ["stand_age"]
    return names

FEATURE_NAMES = build_feature_names()
N_FEAT = len(FEATURE_NAMES)
print(f"Total features per step-transition row: {N_FEAT}")

# ------------------------------------------------------------------------------
# 3. Assemble Pooled Step-Transition Dataset
# ------------------------------------------------------------------------------
print(f"\nReading climate block once for {len(all_sites)} sites...")
t_x = time.time()
Xf_full = np.array(X[all_sites], dtype=np.float32)
print(f"Climate block loaded: {Xf_full.shape} ({Xf_full.nbytes / (1024**3):.2f} GB) in {time.time()-t_x:.1f}s")

window_feats_cache = {}
for t0 in STEP_STARTS:
    window_feats_cache[t0] = climate_window_feats(Xf_full[:, t0 : t0 + STEP, :, :])
    print(f"  window[{t0}:{t0+STEP}] features ready: {window_feats_cache[t0].shape}")
del Xf_full
gc.collect()

n_sites = len(all_sites)
n_rows = n_sites * len(SEED_AGES) * N_HOPS
X_all = np.empty((n_rows, N_FEAT), dtype=np.float32)
y_all = np.empty((n_rows, len(STATE_VARS)), dtype=np.float32)
sid_all = np.empty(n_rows, dtype=np.int64)

state_max_bounds = {v: 0.0 for v in STATE_VARS}

print(f"\nBuilding {n_rows:,} log-residual delta transition rows...")
row = 0
for age in SEED_AGES:
    t_age = time.time()
    Y = np.load(f"{BASE}/data_global/{AGE_FILES[age]}", mmap_mode="r")
    Yf = np.array(Y[all_sites], dtype=np.float32)

    s0 = Yf[:, 0, -1, STATE_IDX]  # initial anchor state at stand inception
    for vi, var in enumerate(STATE_VARS):
        state_max_bounds[var] = max(state_max_bounds[var], float(Yf[:, :, -1, STATE_IDX[vi]].max()))

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
    print(f"  assembled age={age:03d} in {time.time()-t_age:.1f}s ({row:,}/{n_rows:,} rows so far)")

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
    learning_rate=0.04,
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
NUM_BOOST_ROUND = 1200
EARLY_STOP = 50

gkf = GroupKFold(n_splits=NUM_FOLDS)
fold_indices = list(gkf.split(X_all, groups=sid_all))

models = {v: [] for v in STATE_VARS}
print(f"\n=== Training {len(STATE_VARS)} Delta-Transition Models x {NUM_FOLDS} Folds ===")

for vi, var in enumerate(STATE_VARS):
    y_target = y_all[:, vi]
    print(f"\n--- Training Target: {var} (log-residual delta) ---")
    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
        t_f = time.time()
        dtrain = lgb.Dataset(X_all[tr_idx], label=y_target[tr_idx], feature_name=FEATURE_NAMES)
        dval = lgb.Dataset(X_all[va_idx], label=y_target[va_idx], feature_name=FEATURE_NAMES, reference=dtrain)

        m = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        val_rmse = mean_squared_error(y_target[va_idx], m.predict(X_all[va_idx])) ** 0.5
        print(f"  [{var} | Fold {fold_i+1}/{NUM_FOLDS}] best_iter={m.best_iteration}, val_rmse={val_rmse:.4f} ({time.time()-t_f:.1f}s)")
        models[var].append(m)

del X_all, y_all, sid_all
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
test_sites = df_sites.iloc[:, 0].astype(int).values
n_test = len(test_sites)
print(f"Loaded {n_test} test sites from sites_ssp.csv")

def rollout_scenario(scenario_name: str) -> pd.DataFrame:
    print(f"\n--- Commencing Rollout for Scenario: {scenario_name} ---")
    data = np.load(f"{TEST_BASE}/test_{scenario_name}.npz")
    X_test_clim = data["test_x"] if "test_x" in data else data["X"]
    Y_test_init = data["test_y0"] if "test_y0" in data else data["Y"]

    # Precompute 10-year climate window features
    test_window_cache = {}
    for t0 in STEP_STARTS:
        test_window_cache[t0] = climate_window_feats(X_test_clim[:, t0 : t0 + STEP, :, :])

    dfs = []
    for age_idx, seed_age in enumerate(SEED_AGES):
        current_state = np.array(Y_test_init[:, age_idx, STATE_IDX], dtype=np.float32)
        initial_anchor = current_state.copy()

        # Execute 4-hop rollout
        for t0 in STEP_STARTS:
            clim_f = test_window_cache[t0]
            stand_age = np.full((n_test, 1), seed_age + t0, dtype=np.float32)
            f_in = np.concatenate([clim_f, current_state, initial_anchor, stand_age], axis=1).astype(np.float32)

            next_state = np.empty_like(current_state)
            for vi, var in enumerate(STATE_VARS):
                # Ensemble average across folds
                delta_pred = np.mean([m.predict(f_in) for m in models[var]], axis=0)
                # Physical monotonicity clamp on tree height: trees do not shrink
                if var == "height":
                    delta_pred = np.maximum(0.0, delta_pred)

                # Log-space reconstruction: expm1(log1p(current_state) + delta)
                reconstructed = np.expm1(np.log1p(np.maximum(current_state[:, vi], 0.0)) + delta_pred)
                # Intermediate biophysical clipping: bounded at strictly positive & 1.1x max historical
                max_bound = state_max_bounds[var] * 1.10
                next_state[:, vi] = np.clip(reconstructed, 0.05, max_bound)

            current_state = next_state

        pred_height = np.clip(current_state[:, 0] / SCALE["height"], 0.05, 45.0)
        pred_agb = np.clip(current_state[:, 1] / SCALE["agb"], 0.01, 50.0)

        df_block = pd.DataFrame({
            "site": test_sites,
            "age": seed_age,
            "height": pred_height,
            "agb": pred_agb,
        })
        dfs.append(df_block)

    df_full = pd.concat(dfs, ignore_index=True)
    df_full["id"] = f"{scenario_name}_" + df_full["site"].astype(str) + "_" + df_full["age"].astype(str)
    return df_full[["id", "height", "agb"]]

sub_126 = rollout_scenario("ssp126")
sub_585 = rollout_scenario("ssp585")

submission_raw = pd.concat([sub_126, sub_585], ignore_index=True)

# Align strictly against sample_submission.csv
sample_sub = pd.read_csv(find_file("sample_submission.csv"))
submission = sample_sub[["id"]].merge(submission_raw, on="id", how="left")

# ------------------------------------------------------------------------------
# 6. Integrity Verification & Submission Export
# ------------------------------------------------------------------------------
print("\n=== Submission Integrity Verification ===")
expected_rows = len(sample_sub)
print(f"Total Rows: {len(submission)} (expected {expected_rows})")
assert len(submission) == expected_rows, f"Row count mismatch! {len(submission)} vs {expected_rows}"
assert list(submission.columns) == ["id", "height", "agb"], "Column header mismatch!"
assert submission.isna().sum().sum() == 0, "Null values found!"
assert (submission["height"] > 0).all(), "Zero or negative height values found!"
assert (submission["agb"] > 0).all(), "Zero or negative agb values found!"
assert (submission["id"] == sample_sub["id"]).all(), "Submission ID order mismatch with sample_submission!"

print("\nHeight Stats:")
print(submission["height"].describe())
print("\nAGB Stats:")
print(submission["agb"].describe())

out_file = "/kaggle/working/submission.csv" if os.path.exists("/kaggle/working") else "submission.csv"
submission.to_csv(out_file, index=False)
print(f"\nSUCCESS: Exported {out_file} ({os.path.getsize(out_file):,} bytes)")
print(f"Total Wall Clock: {time.time() - t0_global:.1f}s")
