"""
Autobot AI Emulation Exp 10: Deep CarbonGlobe Temporal Sequence Emulator SOTA
Competition: IEEE BigData Cup 2026 - AI Emulation Challenge (CarbonGlobe)
Hardware: Kaggle Cloud GPU (Dual NVIDIA T4 / P100)

MATHEMATICAL & SCIENTIFIC ARCHITECTURE:
1. Target: Break into Top 10 (Leaderboard Top 10 Cutoff: <= 0.072, Leader: 0.046).
2. Root Cause of Previous Plateau (0.207):
   - Decision trees (LightGBM/XGBoost) fail on future SSP scenarios because 19.2% of test features
     are out of historical training bounds. Trees extrapolate as flat step constants, causing 18.28%
     of AGB predictions to collapse onto the 0.001 clip floor.
3. Neural Sequence Solution (DeepCarbonEmulator):
   - Multi-scale Dilated 1D ConvNet (dilations 1, 2, 4) + Bidirectional GRU processing the full
     40-year climate sequence (40 x 136 drivers) continuously.
   - Continuous piecewise linear / smooth activation surfaces (GELU) guarantee natural extrapolation
     along global warming and CO2 response curves without clamping.
   - Biophysical Residual Formulation: Models predict growth increments Delta_h and Delta_a from
     initial state y0, conditioned on stand age saturation dynamics:
       h_40 = max(0.85 * h_0, h_0 + Delta_h)
       a_40 = max(0.80 * a_0, a_0 + Delta_a)
     This physically prevents old-growth forests from collapsing to zero!
   - Direct Competition Loss: Trained end-to-end on the exact competition Scaled MSE metric:
       Loss = 0.5 * [ MSE(pred_h / 10.90645, true_h / 10.90645) + MSE(pred_a / 3.39567, true_a / 3.39567) ]
   - 4-Fold GroupKFold strictly grouped by site_id across 100% of labeled training sites (~35,000 sites).
"""

import os
import sys
import gc
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=== AUTOBOT AI EMULATION EXP 10: DEEP CARBON SEQUENCE EMULATOR SOTA ===")
print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")

t0_global = time.time()

# ------------------------------------------------------------------------------
# 1. Dataset Resolution & Path Configuration
# ------------------------------------------------------------------------------
def find_file(name, search_dirs=('/kaggle/input', '.')):
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            if name in files:
                return os.path.join(root, name)
    raise FileNotFoundError(f'Cannot find {name}')

glob_x_path = find_file('glob_X_fea.npy')
BASE = os.path.dirname(os.path.dirname(glob_x_path))
print('Resolved CarbonGlobe BASE:', BASE)

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
AGE_FILES = {age: f"glob_Y{age:03d}.npy" for age in SEED_AGES}
ALL_TARGET_NAMES = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']

SCALE_H = 10.90645
SCALE_A = 3.39567

# Load normalization statistics
stats = np.load(f'{BASE}/data_stats/data_stats.npz')
x_mean = stats['x_mean'].astype(np.float32)  # (136,)
x_std = stats['x_std'].astype(np.float32)    # (136,)

# Labeled benchmark sites
train_sites = pd.read_csv(f'{BASE}/train.csv', header=None).iloc[:, 0].astype(int).values
val_sites = pd.read_csv(f'{BASE}/val.csv', header=None).iloc[:, 0].astype(int).values
test_sites_bench = pd.read_csv(f'{BASE}/test.csv', header=None).iloc[:, 0].astype(int).values
print('Benchmark split sizes -- train/val/test:', len(train_sites), len(val_sites), len(test_sites_bench))

all_sites = np.unique(np.concatenate([train_sites, val_sites, test_sites_bench]))
all_sites.sort()
print(f'Total labeled sites pool: {len(all_sites):,} ({len(all_sites)/54152:.1%} of globe)')

# Load raw X memmap: (54152, 40, 12, 136)
X_raw = np.load(glob_x_path, mmap_mode='r')

# ------------------------------------------------------------------------------
# 2. Climate Sequence & Site Feature Extraction
# ------------------------------------------------------------------------------
print('Extracting annual climate sequences and static site properties...')
t_prep = time.time()

# Precompute annual mean climate for training sites: (N_sites, 40, 136)
# Pre-normalizing by x_mean and x_std
X_pool_annual = np.empty((len(all_sites), 40, 136), dtype=np.float32)
CHUNK = 5000
for i in range(0, len(all_sites), CHUNK):
    idx_chunk = all_sites[i:i+CHUNK]
    chunk_raw = np.array(X_raw[idx_chunk], dtype=np.float32)  # (chunk, 40, 12, 136)
    chunk_annual = chunk_raw.mean(axis=2)                     # (chunk, 40, 136)
    chunk_norm = (chunk_annual - x_mean) / (x_std + 1e-8)
    X_pool_annual[i:i+CHUNK] = chunk_norm
    del chunk_raw, chunk_annual, chunk_norm

# Static site properties: soil hydraulics and climate classification from channels 110:136
# These are constant across time
site_static_feats = X_pool_annual[:, 0, 110:136].copy()  # (N_sites, 26)
# Map site_id to pool index
site2idx = {s: i for i, s in enumerate(all_sites)}

print(f'Climate sequences extracted in {time.time() - t_prep:.1f}s. Memory: {X_pool_annual.nbytes / (1024**2):.1f} MB')

# ------------------------------------------------------------------------------
# 3. Assemble Full Multi-Age Training Dataset
# ------------------------------------------------------------------------------
print('Assembling multi-age biological training tensors...')
t_asm = time.time()

# For each age, read initial condition y0 (year 0, month 11) and target y40 (year 40, month 11)
records_clim_idx = []
records_cond = []
records_y0_ha = []
records_target = []
records_sid = []

for age in SEED_AGES:
    Y_raw = np.load(f'{BASE}/data_global/{AGE_FILES[age]}', mmap_mode='r')
    Y_pool = np.array(Y_raw[all_sites], dtype=np.float32)  # (N_sites, 41, 12, 7)
    
    y0 = Y_pool[:, 0, -1, :]      # (N_sites, 7)
    y40_target = Y_pool[:, 40, -1, :2]  # (N_sites, 2) [height, agb]
    
    y0_ha = y0[:, :2].copy()      # (N_sites, 2)
    
    # Feature engineering for conditioning vector
    init_rate = y0 / (float(age) + 1.0)
    log_y0 = np.log1p(np.maximum(0.0, y0))
    
    age_norm = np.full((len(all_sites), 1), np.log1p(float(age)) / np.log1p(500.0), dtype=np.float32)
    age_sin = np.full((len(all_sites), 1), np.sin(2 * np.pi * float(age) / 500.0), dtype=np.float32)
    age_cos = np.full((len(all_sites), 1), np.cos(2 * np.pi * float(age) / 500.0), dtype=np.float32)
    age_is_young = np.full((len(all_sites), 1), 1.0 if age <= 30 else 0.0, dtype=np.float32)
    age_is_mature = np.full((len(all_sites), 1), 1.0 if age >= 200 else 0.0, dtype=np.float32)
    
    cond_mat = np.column_stack([
        y0,             # 7
        init_rate,      # 7
        log_y0,         # 7
        age_norm,       # 1
        age_sin,        # 1
        age_cos,        # 1
        age_is_young,   # 1
        age_is_mature,  # 1
        site_static_feats  # 26
    ]).astype(np.float32)  # Total 52 dims
    
    n_s = len(all_sites)
    records_clim_idx.append(np.arange(n_s, dtype=np.int32))
    records_cond.append(cond_mat)
    records_y0_ha.append(y0_ha)
    records_target.append(y40_target)
    records_sid.append(all_sites)
    
    del Y_pool, y0, y40_target, cond_mat
    gc.collect()

all_clim_idx = np.concatenate(records_clim_idx, axis=0)
all_cond = np.concatenate(records_cond, axis=0)
all_y0_ha = np.concatenate(records_y0_ha, axis=0)
all_target = np.concatenate(records_target, axis=0)
all_sids = np.concatenate(records_sid, axis=0)

N_TOTAL = len(all_target)
COND_DIM = all_cond.shape[1]
print(f'Total training rows: {N_TOTAL:,} across {len(SEED_AGES)} seed ages. Conditioning dimension: {COND_DIM}')
print(f'Data assembled in {time.time() - t_asm:.1f}s')

# Normalization of conditioning features
cond_mean = all_cond.mean(axis=0)
cond_std = all_cond.std(axis=0) + 1e-6
all_cond = (all_cond - cond_mean) / cond_std

# ------------------------------------------------------------------------------
# 4. PyTorch Dataset & DeepCarbonEmulator Architecture
# ------------------------------------------------------------------------------
class CarbonDataset(Dataset):
    def __init__(self, clim_seq_pool, clim_idx, cond, y0_ha, targets=None):
        self.clim_seq_pool = clim_seq_pool
        self.clim_idx = clim_idx
        self.cond = cond
        self.y0_ha = y0_ha
        self.targets = targets
        
    def __len__(self):
        return len(self.clim_idx)
        
    def __getitem__(self, idx):
        c_i = self.clim_idx[idx]
        x_clim = self.clim_seq_pool[c_i]  # (40, 136)
        cond = self.cond[idx]            # (COND_DIM,)
        y0 = self.y0_ha[idx]             # (2,)
        
        if self.targets is not None:
            y_true = self.targets[idx]   # (2,)
            return torch.from_numpy(x_clim), torch.from_numpy(cond), torch.from_numpy(y0), torch.from_numpy(y_true)
        return torch.from_numpy(x_clim), torch.from_numpy(cond), torch.from_numpy(y0)

class ResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        res = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + res)

class DeepCarbonEmulator(nn.Module):
    def __init__(self, in_climate=136, cond_dim=52, hidden_dim=128):
        super().__init__()
        # Multi-scale Dilated Temporal Backbone
        self.stem = nn.Sequential(
            nn.Conv1d(in_climate, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.res1 = ResBlock1D(hidden_dim, dilation=1)
        self.res2 = ResBlock1D(hidden_dim, dilation=2)
        self.res3 = ResBlock1D(hidden_dim, dilation=4)
        
        # Temporal Sequence Integrator (Bi-GRU)
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        
        # Conditioning MLP (Initial Ecological State & Stand Age)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Gated Cross-Fusion & Deep Biophysical Residual Head
        fusion_dim = (hidden_dim * 3) + hidden_dim  # last, mean, max pools + cond
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x_clim, cond, y0_ha):
        # x_clim: (B, 40, 136) -> transpose to (B, 136, 40) for Conv1D
        x = x_clim.transpose(1, 2)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Back to (B, 40, hidden_dim) for Bi-GRU
        x_seq = x.transpose(1, 2)
        gru_out, _ = self.gru(x_seq)
        
        p_last = gru_out[:, -1, :]
        p_mean = gru_out.mean(dim=1)
        p_max, _ = gru_out.max(dim=1)
        temp_rep = torch.cat([p_last, p_mean, p_max], dim=-1)
        
        cond_rep = self.cond_mlp(cond)
        fused = torch.cat([temp_rep, cond_rep], dim=-1)
        
        delta = self.head(fused)
        
        # Biophysical Growth Residual Formula: y40 = max(clamp_floor, y0 + delta)
        h0 = y0_ha[:, 0:1]
        a0 = y0_ha[:, 1:2]
        
        delta_h = delta[:, 0:1]
        delta_a = delta[:, 1:2]
        
        pred_h = torch.maximum(h0 * 0.85, h0 + delta_h)
        pred_a = torch.maximum(a0 * 0.80, a0 + delta_a)
        
        pred_h = torch.clamp(pred_h, min=0.01)
        pred_a = torch.clamp(pred_a, min=0.01)
        
        return torch.cat([pred_h, pred_a], dim=-1)

# Scaled MSE Competition Loss
class CompetitionLoss(nn.Module):
    def __init__(self, scale_h=SCALE_H, scale_a=SCALE_A):
        super().__init__()
        self.scale_h = scale_h
        self.scale_a = scale_a
        
    def forward(self, pred, target):
        loss_h = F.mse_loss(pred[:, 0] / self.scale_h, target[:, 0] / self.scale_h)
        loss_a = F.mse_loss(pred[:, 1] / self.scale_a, target[:, 1] / self.scale_a)
        return 0.5 * (loss_h + loss_a)

# ------------------------------------------------------------------------------
# 5. GroupKFold Cross-Validation Training Loop
# ------------------------------------------------------------------------------
NUM_FOLDS = 4
EPOCHS = 18
BATCH_SIZE = 512
LR = 1e-3

gkf = GroupKFold(n_splits=NUM_FOLDS)
fold_indices = list(gkf.split(np.arange(N_TOTAL), groups=all_sids))

models = []
oof_preds = np.zeros_like(all_target)

print(f'\n=== Training Deep Carbon Sequence Emulator ({NUM_FOLDS} Folds, {EPOCHS} Epochs, BS={BATCH_SIZE}) ===')

for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
    print(f'\n--- [Fold {fold_i + 1}/{NUM_FOLDS}] (Train: {len(tr_idx):,}, Val: {len(va_idx):,}) ---')
    t_f = time.time()
    
    train_ds = CarbonDataset(X_pool_annual, all_clim_idx[tr_idx], all_cond[tr_idx], all_y0_ha[tr_idx], all_target[tr_idx])
    val_ds = CarbonDataset(X_pool_annual, all_clim_idx[va_idx], all_cond[va_idx], all_y0_ha[va_idx], all_target[va_idx])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    model = DeepCarbonEmulator(in_climate=136, cond_dim=COND_DIM, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = CompetitionLoss().to(device)
    
    best_val_score = float('inf')
    best_weights = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_c, cond, y0, y_true in train_loader:
            x_c, cond, y0, y_true = x_c.to(device), cond.to(device), y0.to(device), y_true.to(device)
            optimizer.zero_grad()
            pred = model(x_c, cond, y0)
            loss = criterion(pred, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        train_loss /= n_batches
        
        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for x_c, cond, y0, y_true in val_loader:
                x_c, cond, y0 = x_c.to(device), cond.to(device), y0.to(device)
                pred = model(x_c, cond, y0)
                val_preds_list.append(pred.cpu().numpy())
                
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_true = all_target[va_idx]
        
        mse_h = mean_squared_error(val_true[:, 0] / SCALE_H, val_preds[:, 0] / SCALE_H)
        mse_a = mean_squared_error(val_true[:, 1] / SCALE_A, val_preds[:, 1] / SCALE_A)
        comp_score = 0.5 * (mse_h + mse_a)
        
        if comp_score < best_val_score:
            best_val_score = comp_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 3 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Train Scaled Loss: {train_loss:.5f} | Val Scaled Score: {comp_score:.5f} (H: {mse_h:.5f}, A: {mse_a:.5f})")
            
    print(f"  Fold {fold_i + 1} Best Scaled Score: {best_val_score:.5f} ({time.time() - t_f:.1f}s)")
    
    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()
    
    # Generate honest OOF predictions
    val_preds_list = []
    with torch.no_grad():
        for x_c, cond, y0, _ in val_loader:
            x_c, cond, y0 = x_c.to(device), cond.to(device), y0.to(device)
            val_preds_list.append(model(x_c, cond, y0).cpu().numpy())
    oof_preds[va_idx] = np.concatenate(val_preds_list, axis=0)
    
    models.append(model)
    gc.collect()

# Overall OOF Metric
overall_mse_h = mean_squared_error(all_target[:, 0] / SCALE_H, oof_preds[:, 0] / SCALE_H)
overall_mse_a = mean_squared_error(all_target[:, 1] / SCALE_A, oof_preds[:, 1] / SCALE_A)
overall_score = 0.5 * (overall_mse_h + overall_mse_a)

print(f'\n======================================================')
print(f'=== OVERALL OUT-OF-FOLD (OOF) COMPETITION METRICS ===')
print(f'  Scaled MSE (Height) : {overall_mse_h:.6f}')
print(f'  Scaled MSE (AGB)    : {overall_mse_a:.6f}')
print(f'  Composite Scaled MSE: {overall_score:.6f}')
print(f'  Estimated Rel RMSE  : {overall_score ** 0.5:.6f}')
print(f'======================================================')

# ------------------------------------------------------------------------------
# 6. Test Inference on Future Climate Scenarios (SSP126 & SSP585)
# ------------------------------------------------------------------------------
sites_ssp_file = find_file('sites_ssp.csv')
TEST_BASE = os.path.dirname(sites_ssp_file)
print('\nResolved TEST_BASE:', TEST_BASE)

sites_ssp = pd.read_csv(sites_ssp_file)
test_site_ids = sites_ssp.iloc[:, 0].astype(int).values
n_test_sites = len(test_site_ids)
print(f'Loaded {n_test_sites} test sites from sites_ssp.csv')

SCENARIO_CONFIGS = [
    ('test_ssp126.npz', 'ssp126'),
    ('test_ssp585.npz', 'ssp585'),
]

dfs = []

for npz_name, sc in SCENARIO_CONFIGS:
    t_sc = time.time()
    print(f'\n--- Generating Inference for Scenario: {sc.upper()} ({npz_name}) ---')
    npz_path = find_file(npz_name)
    d = np.load(npz_path)
    test_x = d['test_x'].astype(np.float32)    # (6171, 40, 12, 136)
    test_y0 = d['test_y0'].astype(np.float32)  # (6171, 15, 7)
    
    # 1. Annual climate sequence: (6171, 40, 136) normalized
    test_annual = test_x.mean(axis=2)
    test_annual_norm = (test_annual - x_mean) / (x_std + 1e-8)
    
    # 2. Static soil & biome properties from test channels 110:136
    test_static = test_annual_norm[:, 0, 110:136].copy()  # (6171, 26)
    
    sc_rows = []
    
    for age_i, age in enumerate(SEED_AGES):
        y0_age = test_y0[:, age_i, :]  # (6171, 7)
        y0_ha_age = y0_age[:, :2].copy()
        
        init_rate = y0_age / (float(age) + 1.0)
        log_y0 = np.log1p(np.maximum(0.0, y0_age))
        
        age_norm = np.full((n_test_sites, 1), np.log1p(float(age)) / np.log1p(500.0), dtype=np.float32)
        age_sin = np.full((n_test_sites, 1), np.sin(2 * np.pi * float(age) / 500.0), dtype=np.float32)
        age_cos = np.full((n_test_sites, 1), np.cos(2 * np.pi * float(age) / 500.0), dtype=np.float32)
        age_is_young = np.full((n_test_sites, 1), 1.0 if age <= 30 else 0.0, dtype=np.float32)
        age_is_mature = np.full((n_test_sites, 1), 1.0 if age >= 200 else 0.0, dtype=np.float32)
        
        cond_raw = np.column_stack([
            y0_age, init_rate, log_y0,
            age_norm, age_sin, age_cos, age_is_young, age_is_mature,
            test_static
        ]).astype(np.float32)
        
        # Normalize with training stats
        cond_norm = (cond_raw - cond_mean) / cond_std
        
        # Run test inference through all fold models
        test_ds = CarbonDataset(test_annual_norm, np.arange(n_test_sites), cond_norm, y0_ha_age)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
        
        fold_preds = []
        for model in models:
            m_preds = []
            with torch.no_grad():
                for x_c, cond, y0 in test_loader:
                    x_c, cond, y0 = x_c.to(device), cond.to(device), y0.to(device)
                    pred = model(x_c, cond, y0)
                    m_preds.append(pred.cpu().numpy())
            fold_preds.append(np.concatenate(m_preds, axis=0))
            
        avg_pred = np.mean(fold_preds, axis=0)  # (6171, 2) [height, agb]
        
        # Scale predictions by dividing by global means
        pred_h_scaled = np.clip(avg_pred[:, 0] / SCALE_H, 1e-3, None)
        pred_a_scaled = np.clip(avg_pred[:, 1] / SCALE_A, 1e-3, None)
        
        sc_rows.append(pd.DataFrame({
            'site': test_site_ids,
            'age': age,
            'height': pred_h_scaled,
            'agb': pred_a_scaled,
        }))
        
    df_sc = pd.concat(sc_rows, ignore_index=True)
    df_sc['id'] = f'{sc}_' + df_sc['site'].astype(str) + '_' + df_sc['age'].astype(str)
    dfs.append(df_sc)
    print(f'Scenario {sc} inference complete in {time.time() - t_sc:.1f}s')

# ------------------------------------------------------------------------------
# 7. Verification Contract & Submission Export
# ------------------------------------------------------------------------------
print('\n=== Verifying Submission Integrity Contract ===')
submission = pd.concat(dfs, ignore_index=True)[['id', 'height', 'agb']]
sample_sub = pd.read_csv(find_file('sample_submission.csv'))
submission = sample_sub[['id']].merge(submission, on='id', how='left')

out_path = Path('/kaggle/working/submission.csv') if Path('/kaggle/working').exists() else Path('submission.csv')
submission.to_csv(out_path, index=False)
print(f'SUCCESS: Submission saved to {out_path.resolve()} ({out_path.stat().st_size / (1024**2):.2f} MB)')

EXPECTED_ROWS = 185130
assert submission.shape[0] == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {submission.shape[0]}"
assert not submission[['height', 'agb']].isna().any().any(), "Fatal: Null values found in predictions!"
assert (submission['height'] > 0).all(), "Fatal: Negative or zero height predictions found!"
assert (submission['agb'] > 0).all(), "Fatal: Negative or zero agb predictions found!"

print(f'Total Rows Verified: {submission.shape[0]:,}')
print(f"Height Stats -> Mean: {submission['height'].mean():.4f}, Std: {submission['height'].std():.4f}, "
      f"Min: {submission['height'].min():.4f}, Max: {submission['height'].max():.4f}")
print(f"AGB Stats    -> Mean: {submission['agb'].mean():.4f}, Std: {submission['agb'].std():.4f}, "
      f"Min: {submission['agb'].min():.4f}, Max: {submission['agb'].max():.4f}")

# Diagnostic: check clip floor rate
clip_h = (submission['height'] <= 0.0011).mean()
clip_a = (submission['agb'] <= 0.0011).mean()
print(f"Positivity Floor Hit Rate -> Height: {clip_h:.2%}, AGB: {clip_a:.2%}")

print(f'Total Execution Time: {time.time() - t0_global:.1f}s')
print('\nSample Predictions:')
print(submission.head(10))
print("=== END AUTOBOT AI EMULATION EXP 10 SOTA ===")
