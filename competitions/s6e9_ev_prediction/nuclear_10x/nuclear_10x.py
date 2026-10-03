"""
Autobot S6E9 Nuclear 10x Final Attack
Target: Beat 0.94705 (Chris Deotte / real ceiling). Our best: 0.94656. Gap: +0.00049.

10 Experiments:
  A: LGBM 10-seed
  B: LGBM 20-seed (max variance reduction)
  C: LGBM depth=4/leaves=16
  D: LGBM max_bin=4096 (high-bin)
  E: LGBM + interaction features
  F: XGBoost GPU 5-seed
  G: CatBoost GPU 5000 trees
  H: 60% B + 40% F rank blend
  I: 50% B + 30% F + 20% G three-way blend
  J: Grand ensemble (B+D+E+F+G)

All saved to /kaggle/working/expX_name.csv
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
warnings.filterwarnings("ignore")

t_start = time.time()
OUT = Path('/kaggle/working')

# ── Data discovery (walk /kaggle/input) ─────────────────────────────────────
def find_file(keyword, exclude=('sample',)):
    for root, _, files in os.walk('/kaggle/input'):
        for f in files:
            fl = f.lower()
            if keyword.lower() in fl and fl.endswith('.csv'):
                if not any(e in fl for e in exclude):
                    return os.path.join(root, f)
    return None

train_path = find_file('train.csv')
test_path  = find_file('test.csv')
sub_path   = find_file('sample_submission')
orig_path  = find_file('ev_adoption') or find_file('anxiety') or find_file('adoption')

print(f'train: {train_path}')
print(f'test:  {test_path}')
print(f'orig:  {orig_path}')

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

TARGET = 'Will_Buy_EV'
y = (train[TARGET].map({'No': 0, 'Yes': 1})
     if not pd.api.types.is_numeric_dtype(train[TARGET])
     else train[TARGET]).to_numpy(np.uint8)

print(f'Train: {len(train):,}  Test: {len(test):,}  Positive rate: {y.mean():.4f}')
print(f'Cols: {list(train.columns)}')
TEST_IDS = test['id'].values

# ── Feature Engineering ──────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    'Gender', 'City_Type', 'Current_Car_Type',
    'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level'
]

def build_row_local(frame):
    out = frame.copy()
    income      = out['Annual_Income_USD'].to_numpy(dtype=np.int64)
    commute_x10 = np.round(out['Daily_Commute_km'].to_numpy(float) * 10).astype(np.int64)

    out['inc_d1']     = (income % 10).astype('int8')
    out['inc_d2']     = (income // 10 % 10).astype('int8')
    out['inc_d3']     = (income // 100 % 10).astype('int8')
    out['inc_mod100'] = (income % 100).astype('int16')
    out['inc_mod1000']= (income % 1000).astype('int16')
    out['km_d1']      = (commute_x10 % 10).astype('int8')
    out['km_mod100']  = (commute_x10 % 100).astype('int8')

    for d in (50, 100, 250, 500, 1000, 2500, 5000):
        out[f'inc_q{d}'] = (income // d).astype('int32')
    for d in (5, 10, 25, 50):
        out[f'km_q{d}']  = (commute_x10 // d).astype('int32')

    out['is_30k_spike']         = (income == 30000).astype('int8')
    out['is_millionaire_cliff'] = (income >= 170537).astype('int8')
    out['is_dead_zone']         = ((income >= 38000) & (income <= 42000)).astype('int8')
    out['is_env_hater']         = (out['Environmental_Concern_Level'] == 1).astype('int8')

    keys = pd.DataFrame(index=out.index)
    keys['k_inc_exact'] = income.astype(str)
    keys['k_inc100']    = (income // 100).astype(str)
    keys['k_inc1000']   = (income // 1000).astype(str)
    keys['k_km_int']    = (commute_x10 // 10).astype(str)
    for col in CATEGORICAL_FEATURES + [
        'Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
        'Charging_Stations_Near_Work', 'Environmental_Concern_Level'
    ]:
        keys[f'k_{col}'] = out[col].astype(str).to_numpy()
    return out.reset_index(drop=True), keys.reset_index(drop=True)

def map_zero(values, mapping):
    return values.map(mapping).fillna(0.0).astype(np.float32).to_numpy()

def bin_statistics(codes, y_fit, n_bins, prior, smooth):
    sums   = np.bincount(codes, weights=y_fit, minlength=n_bins).astype(np.float64)
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    central = (sums + smooth * prior) / (counts + smooth)
    ls, lc  = np.r_[0., sums[:-1]], np.r_[0., counts[:-1]]
    rs, rc  = np.r_[sums[1:], 0.], np.r_[counts[1:], 0.]
    left  = (ls + smooth * prior) / (lc + smooth)
    right = (rs + smooth * prior) / (rc + smooth)
    kern  = np.exp(-0.5 * (np.arange(-1, 2) / 0.8) ** 2)
    ns = np.convolve(sums, kern, mode='same')
    nc = np.convolve(counts, kern, mode='same')
    sym   = (ns + smooth * kern.sum() * prior) / (nc + smooth * kern.sum())
    slope = right - left
    curv  = central - 0.5 * (left + right)
    return np.column_stack([central, sym, left, right, slope, curv, np.log1p(counts)]).astype(np.float32)

def income_asymmetric_full(train_inc, y_arr, test_inc, q, smooth=10.0):
    train_inc = np.asarray(train_inc, np.float64)
    test_inc  = np.asarray(test_inc,  np.float64)
    y_arr     = np.asarray(y_arr, np.uint8)
    prior     = float(y_arr.mean())
    edges     = np.linspace(train_inc.min(), train_inc.max(), q + 1)
    tr_codes  = np.searchsorted(edges[1:-1], train_inc)
    te_codes  = np.searchsorted(edges[1:-1], test_inc)
    n_bins    = len(edges)
    ps        = max(n_bins - 2, 1)
    tr_feats  = np.zeros((len(train_inc), 8), dtype=np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=17)
    for fi, vi in inner.split(tr_codes, y_arr):
        stats = bin_statistics(tr_codes[fi], y_arr[fi], n_bins, float(y_arr[fi].mean()), smooth)
        tr_feats[vi, 0] = tr_codes[vi] / ps
        tr_feats[vi, 1:] = stats[tr_codes[vi]]
    stats = bin_statistics(tr_codes, y_arr, n_bins, prior, smooth)
    te_feats = np.column_stack([te_codes / ps, stats[te_codes]]).astype(np.float32)
    return tr_feats, te_feats

def build_full_features(X_tr, tr_keys, X_te, te_keys, y_arr, orig_df=None, extra=False):
    X_tr = X_tr.reset_index(drop=True).copy()
    X_te = X_te.reset_index(drop=True).copy()
    tr_keys = tr_keys.reset_index(drop=True).copy()
    te_keys = te_keys.reset_index(drop=True).copy()

    inc_tr  = pd.Series(X_tr['Annual_Income_USD'].to_numpy(np.int64))
    inc_te  = pd.Series(X_te['Annual_Income_USD'].to_numpy(np.int64))
    inc_cnt = inc_tr.value_counts(dropna=False)
    X_tr['fq_inc'] = map_zero(inc_tr, inc_cnt)
    X_te['fq_inc'] = map_zero(inc_te, inc_cnt)

    km_tr  = pd.Series(np.round(X_tr['Daily_Commute_km'].to_numpy(float) * 10).astype(np.int64))
    km_te  = pd.Series(np.round(X_te['Daily_Commute_km'].to_numpy(float) * 10).astype(np.int64))
    km_cnt = km_tr.value_counts(dropna=False)
    X_tr['fq_km'] = map_zero(km_tr, km_cnt)
    X_te['fq_km'] = map_zero(km_te, km_cnt)

    for key in tr_keys.columns:
        freq = tr_keys[key].value_counts(normalize=True, dropna=False)
        X_tr[f'{key}_fe'] = map_zero(tr_keys[key], freq)
        X_te[f'{key}_fe'] = map_zero(te_keys[key], freq)

    for col in CATEGORICAL_FEATURES:
        cats = sorted(X_tr[col].astype(str).unique().tolist())
        dt   = pd.CategoricalDtype(categories=cats, ordered=False)
        X_tr[col] = X_tr[col].astype(str).astype(dt)
        X_te[col] = X_te[col].astype(str).astype(dt)

    for smooth, tag in (('auto', 'auto'), (10.0, '10'), (100.0, '100')):
        enc    = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=42)
        enc_tr = enc.fit_transform(tr_keys, y_arr)
        enc_te = enc.transform(te_keys)
        for i, key in enumerate(tr_keys.columns):
            X_tr[f'{key}_te{tag}'] = enc_tr[:, i].astype(np.float32)
            X_te[f'{key}_te{tag}'] = enc_te[:, i].astype(np.float32)

    # Per-exact-income prior (key trick: maps income value → P(buy) in training)
    inc_prior = pd.Series(y_arr.astype(float)).groupby(inc_tr.values).mean()
    X_tr['inc_exact_prior'] = inc_tr.map(inc_prior).fillna(y_arr.mean()).astype(np.float32).values
    X_te['inc_exact_prior'] = inc_te.map(inc_prior).fillna(y_arr.mean()).astype(np.float32).values

    # Original dataset prior mapping (if available)
    if orig_df is not None:
        try:
            orig_target_col = [c for c in orig_df.columns if 'buy' in c.lower() or 'ev' in c.lower()][-1]
            orig_y = (orig_df[orig_target_col].map({'No': 0, 'Yes': 1})
                     if not pd.api.types.is_numeric_dtype(orig_df[orig_target_col])
                     else orig_df[orig_target_col]).astype(float)
            if 'Annual_Income_USD' in orig_df.columns:
                orig_inc = orig_df['Annual_Income_USD'].astype(np.int64)
                orig_prior = orig_y.groupby(orig_inc).mean()
                X_tr['orig_inc_prior'] = inc_tr.map(orig_prior).fillna(orig_y.mean()).astype(np.float32).values
                X_te['orig_inc_prior'] = inc_te.map(orig_prior).fillna(orig_y.mean()).astype(np.float32).values
                print(f'  Added orig_inc_prior from original dataset ({len(orig_df):,} rows)')
        except Exception as e:
            print(f'  orig_df mapping failed: {e}')

    if extra:
        for ca, cb in [('k_inc_exact', 'k_Subsidy_Available'),
                       ('k_inc_exact', 'k_Environmental_Concern_Level'),
                       ('k_inc_exact', 'k_Range_Anxiety_Level'),
                       ('k_inc1000',   'k_City_Type'),
                       ('k_inc1000',   'k_Current_Car_Type')]:
            combo_tr = tr_keys[ca] + '_' + tr_keys[cb]
            combo_te = te_keys[ca] + '_' + te_keys[cb]
            freq_c   = combo_tr.value_counts(normalize=True, dropna=False)
            X_tr[f'ix_{ca}_{cb}_fe'] = map_zero(combo_tr, freq_c)
            X_te[f'ix_{ca}_{cb}_fe'] = map_zero(combo_te, freq_c)
            enc_c = TargetEncoder(shuffle=True, cv=5, smooth=10.0, random_state=42)
            enc_c.fit(combo_tr.values.reshape(-1, 1), y_arr)
            X_tr[f'ix_{ca}_{cb}_te'] = enc_c.transform(combo_tr.values.reshape(-1, 1))[:, 0].astype(np.float32)
            X_te[f'ix_{ca}_{cb}_te'] = enc_c.transform(combo_te.values.reshape(-1, 1))[:, 0].astype(np.float32)

    for res in (8192, 16384):
        tr_blk, te_blk = income_asymmetric_full(
            X_tr['Annual_Income_USD'].values, y_arr,
            X_te['Annual_Income_USD'].values, res)
        for j in range(1 if res == 16384 else 0, tr_blk.shape[1]):
            X_tr[f'own_income_{res}_{j}'] = tr_blk[:, j]
            X_te[f'own_income_{res}_{j}'] = te_blk[:, j]

    return X_tr, X_te

# Load original dataset if available
orig_df = None
if orig_path:
    try:
        orig_df = pd.read_csv(orig_path)
        print(f'Original dataset loaded: {orig_df.shape}  cols={list(orig_df.columns)}')
    except Exception as e:
        print(f'Could not load original dataset: {e}')

print('\nBuilding base feature matrix...')
X_raw, tr_keys = build_row_local(train.drop(columns=['id', TARGET]))
X_tst, te_keys = build_row_local(test.drop(columns='id'))
X_tr_base, X_te_base = build_full_features(X_raw, tr_keys, X_tst, te_keys, y, orig_df=orig_df)
print(f'Base features: {X_tr_base.shape[1]}')

print('\nBuilding extra-interaction feature matrix...')
X_raw2, tr_keys2 = build_row_local(train.drop(columns=['id', TARGET]))
X_tst2, te_keys2 = build_row_local(test.drop(columns='id'))
X_tr_ext, X_te_ext = build_full_features(X_raw2, tr_keys2, X_tst2, te_keys2, y, orig_df=orig_df, extra=True)
print(f'Extended features: {X_tr_ext.shape[1]}')

# ── Utilities ────────────────────────────────────────────────────────────────
def rank_pct(arr):
    return pd.Series(arr).rank(pct=True).values

def save_sub(preds, name):
    path = OUT / f'{name}.csv'
    pd.DataFrame({'id': TEST_IDS, 'Will_Buy_EV': preds}).to_csv(path, index=False)
    print(f'  Saved: {name}.csv  ({len(np.unique(preds))} unique values)')
    return path

ALL = {}  # name -> (oof_rank, te_rank, auc)

LGBM_BASE = dict(
    n_estimators=2000, learning_rate=0.025,
    max_depth=3, num_leaves=8,
    min_child_samples=15, subsample=0.85, subsample_freq=1,
    colsample_bytree=0.35, reg_alpha=0.08, reg_lambda=2.5,
    max_bin=255, feature_pre_filter=False,
    n_jobs=-1, verbose=-1, metric='auc'
)

def run_lgbm(name, X_tr, X_te, seeds, params=None, n_folds=5):
    p = {**LGBM_BASE, **(params or {})}
    print(f'\n[{name}] LGBM: {len(seeds)}seeds x {n_folds}folds  '
          f'depth={p["max_depth"]} leaves={p["num_leaves"]} bin={p["max_bin"]}')
    t0 = time.time()
    oof_rk = np.zeros(len(y), np.float64)
    te_rk  = np.zeros(len(X_te), np.float64)
    for seed in seeds:
        oof = np.zeros(len(y), np.float64)
        tst = np.zeros(len(X_te), np.float64)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tri, vali) in enumerate(skf.split(X_tr, y)):
            m = LGBMClassifier(**{**p, 'random_state': seed * 100 + fold})
            m.fit(X_tr.iloc[tri], y[tri],
                  eval_set=[(X_tr.iloc[vali], y[vali])],
                  callbacks=[log_evaluation(-1)])
            oof[vali] = m.predict_proba(X_tr.iloc[vali])[:, 1]
            tst += m.predict_proba(X_te)[:, 1] / n_folds
        auc = roc_auc_score(y, oof)
        print(f'  seed={seed} AUC={auc:.6f}')
        oof_rk += rank_pct(oof)
        te_rk  += rank_pct(tst)
    oof_rk /= len(seeds); te_rk /= len(seeds)
    final_auc = roc_auc_score(y, oof_rk)
    print(f'  => {name} OOF AUC: {final_auc:.6f}  {time.time()-t0:.0f}s')
    save_sub(rank_pct(te_rk), name)
    ALL[name] = (oof_rk, te_rk, final_auc)
    return oof_rk, te_rk

# GPU availability
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except Exception:
    HAS_GPU = False
DEVICE = 'cuda' if HAS_GPU else 'cpu'
print(f'\nGPU available: {HAS_GPU}  device: {DEVICE}')

def run_xgb(name, X_tr, X_te, seeds, n_folds=5):
    print(f'\n[{name}] XGBoost: {len(seeds)}seeds x {n_folds}folds  device={DEVICE}')
    t0 = time.time()
    # Encode categoricals
    X_trn = X_tr.copy(); X_ten = X_te.copy()
    for col in X_trn.select_dtypes('category').columns:
        X_trn[col] = X_trn[col].cat.codes
        X_ten[col] = X_ten[col].cat.codes
    X_trn = X_trn.astype(np.float32); X_ten = X_ten.astype(np.float32)
    oof_rk = np.zeros(len(y), np.float64)
    te_rk  = np.zeros(len(X_te), np.float64)
    for seed in seeds:
        oof = np.zeros(len(y), np.float64)
        tst = np.zeros(len(X_te), np.float64)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tri, vali) in enumerate(skf.split(X_trn, y)):
            m = XGBClassifier(
                n_estimators=3000, learning_rate=0.02,
                max_depth=6, subsample=0.8, colsample_bytree=0.4,
                tree_method='hist', device=DEVICE,
                eval_metric='auc', random_state=seed * 100 + fold,
                verbosity=0
            )
            m.fit(X_trn.iloc[tri], y[tri],
                  eval_set=[(X_trn.iloc[vali], y[vali])],
                  verbose=False)
            oof[vali] = m.predict_proba(X_trn.iloc[vali])[:, 1]
            tst += m.predict_proba(X_ten)[:, 1] / n_folds
        auc = roc_auc_score(y, oof)
        print(f'  seed={seed} AUC={auc:.6f}')
        oof_rk += rank_pct(oof)
        te_rk  += rank_pct(tst)
    oof_rk /= len(seeds); te_rk /= len(seeds)
    final_auc = roc_auc_score(y, oof_rk)
    print(f'  => {name} OOF AUC: {final_auc:.6f}  {time.time()-t0:.0f}s')
    save_sub(rank_pct(te_rk), name)
    ALL[name] = (oof_rk, te_rk, final_auc)
    return oof_rk, te_rk

def run_catboost(name, X_tr, X_te, n_folds=5):
    print(f'\n[{name}] CatBoost: {n_folds}folds  task={"GPU" if HAS_GPU else "CPU"}')
    t0 = time.time()
    X_trc = X_tr.copy(); X_tec = X_te.copy()
    cat_idx = [X_trc.columns.get_loc(c)
               for c in X_trc.select_dtypes('category').columns]
    for col in X_trc.select_dtypes('category').columns:
        X_trc[col] = X_trc[col].cat.codes.astype(str)
        X_tec[col] = X_tec[col].cat.codes.astype(str)
    oof = np.zeros(len(y), np.float64)
    tst = np.zeros(len(X_te), np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (tri, vali) in enumerate(skf.split(X_trc, y)):
        m = CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=6,
            eval_metric='AUC',
            task_type='GPU' if HAS_GPU else 'CPU',
            random_seed=42 + fold, verbose=0,
            cat_features=cat_idx
        )
        m.fit(X_trc.iloc[tri], y[tri],
              eval_set=(X_trc.iloc[vali], y[vali]))
        oof[vali] = m.predict_proba(X_trc.iloc[vali])[:, 1]
        tst += m.predict_proba(X_tec)[:, 1] / n_folds
        print(f'  fold {fold+1}/{n_folds} AUC={roc_auc_score(y[vali], oof[vali]):.6f}')
    final_auc = roc_auc_score(y, oof)
    print(f'  => {name} OOF AUC: {final_auc:.6f}  {time.time()-t0:.0f}s')
    save_sub(rank_pct(tst), name)
    ALL[name] = (rank_pct(oof), rank_pct(tst), final_auc)
    return rank_pct(oof), rank_pct(tst)

def blend(name, oof_list, te_list, weights):
    w = np.array(weights, np.float64); w /= w.sum()
    oof_b = sum(wi * rank_pct(o) for wi, o in zip(w, oof_list))
    te_b  = sum(wi * rank_pct(t) for wi, t in zip(w, te_list))
    auc   = roc_auc_score(y, oof_b)
    print(f'\n[{name}] Blend OOF AUC: {auc:.6f}')
    save_sub(rank_pct(te_b), name)
    ALL[name] = (oof_b, te_b, auc)
    return oof_b, te_b

# ════════════════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

# A: 10-seed LGBM (2x our Exp10A seeds)
oof_A, te_A = run_lgbm('expA_lgbm_10seed', X_tr_base, X_te_base, seeds=list(range(10)))

# B: 20-seed LGBM (maximum variance reduction)
oof_B, te_B = run_lgbm('expB_lgbm_20seed', X_tr_base, X_te_base, seeds=list(range(20)))

# C: Deeper trees (depth=4, leaves=16)
oof_C, te_C = run_lgbm('expC_lgbm_depth4', X_tr_base, X_te_base, seeds=list(range(5)),
                        params={'max_depth': 4, 'num_leaves': 16, 'learning_rate': 0.02})

# D: High-bin (max_bin=4096)
oof_D, te_D = run_lgbm('expD_lgbm_highbin', X_tr_base, X_te_base, seeds=list(range(5)),
                        params={'max_bin': 4096, 'learning_rate': 0.02})

# E: Interaction features + 5 seeds
oof_E, te_E = run_lgbm('expE_lgbm_interact', X_tr_ext, X_te_ext, seeds=list(range(5)))

# F: XGBoost GPU 5-seed
oof_F, te_F = run_xgb('expF_xgb_5seed', X_tr_base, X_te_base, seeds=list(range(5)))

# G: CatBoost GPU 5000 trees
oof_G, te_G = run_catboost('expG_catboost5k', X_tr_base, X_te_base)

# H: 20-seed LGBM + XGB blend (60/40)
oof_H, te_H = blend('expH_lgbm20_xgb', [oof_B, oof_F], [te_B, te_F], [0.6, 0.4])

# I: LGBM + XGB + CAT three-way (50/30/20)
oof_I, te_I = blend('expI_3way', [oof_B, oof_F, oof_G], [te_B, te_F, te_G], [0.5, 0.3, 0.2])

# J: Grand ensemble (B+D+E+F+G)
oof_J, te_J = blend('expJ_grand', [oof_B, oof_D, oof_E, oof_F, oof_G],
                    [te_B, te_D, te_E, te_F, te_G], [0.35, 0.20, 0.15, 0.20, 0.10])

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('FINAL SUMMARY — SUBMIT IN THIS ORDER (highest OOF AUC first)')
print('=' * 70)
ranked = sorted(ALL.items(), key=lambda kv: kv[1][2], reverse=True)
for rank, (name, (_, _, auc)) in enumerate(ranked, 1):
    delta = auc - 0.94656
    print(f'  #{rank:2d}  {name:<30}  OOF AUC={auc:.6f}  ({delta:+.5f} vs our best)')

print(f'\nTotal time: {(time.time() - t_start)/60:.1f} minutes')
print(f'\nFiles in /kaggle/working/:')
for f in sorted(OUT.glob('exp*.csv')):
    print(f'  {f.name}  ({f.stat().st_size:,} bytes)')
print('\nDOWNLOAD AND SUBMIT ALL 10 VIA: kaggle kernels output <kernel-slug>')
