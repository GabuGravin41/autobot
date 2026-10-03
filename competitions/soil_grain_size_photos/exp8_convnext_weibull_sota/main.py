"""
Autobot Soil Exp 8: ConvNeXt-Tiny + Physical PPM Scaling + Analytical Weibull CDF SOTA
Competition: Predicting Soil Grain Size Distributions from Images

HYPOTHESIS & SCIENTIFIC FOUNDATION:
1. In Exp 7, frozen DINOv2 patch features + Ridge regression scored 59.99 EMD due to camera-sensor
   domain gap across 24 train images.
2. Grandmaster AmbrosM demonstrated that direct parametric Weibull CDF modeling with physical
   pixel-per-millimeter (ppm) scaling achieves single-digit EMD (~8.0) locally.
3. The closed-form Weibull CDF: F(d) = 100 * (1 - exp(-(d/b)^c)) enforces strict monotonicity,
   eliminates unphysical mass leakage, and optimizes the exact Earth Mover's Distance loss:
   L_EMD = (n_sizes / 2) * MAE(y_pred, y_true).
4. Physical 40x40 mm crops scale dynamically with image ppm: crop_size = int(40 * ppm).
5. Robust path discovery and image-sample matching from diag1 ensures 100% data discovery on Kaggle.
"""

import os
import sys
import gc
import re
import unicodedata
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import GroupKFold
import timm

print("=== AUTOBOT SOIL EXP 8: CONVNEXT + WEIBULL CDF SOTA ===")
print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
class CFG:
    model_name = "convnext_tiny.fb_in22k_ft_in1k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 320 # pixels
    batch_size = 16
    T_0 = 20 # epochs for cosine annealing
    epochs = T_0
    lr = 1e-5 # initial learning rate
    sizes = ['0.002', '0.0063', '0.02', '0.063', '0.2', '0.63', '2', '6.3', '20', '63']
    n_sizes = len(sizes)

# ------------------------------------------------------------------------------
# 2. Robust Path Discovery & Diagnostics
# ------------------------------------------------------------------------------
def locate_path(candidates, desc="path"):
    for c in candidates:
        p = Path(c)
        if p.exists():
            print(f"Found {desc}: {p}")
            return p
    for root in ["/kaggle/input", "."]:
        if os.path.exists(root):
            for r, dirs, files in os.walk(root):
                for c in candidates:
                    target = Path(c).name
                    if target in files or target in dirs:
                        found = Path(r) / target
                        print(f"Discovered {desc} via walk: {found}")
                        return found
    raise FileNotFoundError(f"Could not find {desc} among candidates: {candidates}")

COMP_ROOT = None
for candidate in [
    Path("/kaggle/input/competitions/soil-grain-size-from-photos"),
    Path("/kaggle/input/soil-grain-size-from-photos"),
    Path("."),
]:
    if candidate.exists():
        COMP_ROOT = candidate
        break

if COMP_ROOT is None:
    COMP_ROOT = Path("/kaggle/input/competitions/soil-grain-size-from-photos")

print(f"Using COMP_ROOT: {COMP_ROOT}")

TRAIN_LABELS_PATH = locate_path([
    COMP_ROOT / "Training_labels_updated.csv",
    COMP_ROOT / "Training_labels_without_H374.csv",
    COMP_ROOT / "Training_labels.csv",
], "training labels csv")

PPM_PATH = locate_path([
    COMP_ROOT / "ppm_updated.csv",
    COMP_ROOT / "ppm.csv",
], "ppm csv")

SAMPLE_SUB_PATH = locate_path([
    COMP_ROOT / "sample_submission.csv",
], "sample submission csv")

TRAIN_DIR = locate_path([
    COMP_ROOT / "Training-All_Photos_updated" / "Training-All_Photos_updated",
    COMP_ROOT / "Training-All_Photos_updated",
    COMP_ROOT / "Training-All_Photos_without_H374" / "Training-All_Photos_without_H374",
    COMP_ROOT / "Training-All_Photos_without_H374",
    COMP_ROOT / "Training-All_Photos" / "Training-All_Photos",
    COMP_ROOT / "Training-All_Photos",
], "training photos directory")

TEST_DIR = locate_path([
    COMP_ROOT / "Test_All_Photos" / "Test_All_Photos",
    COMP_ROOT / "Test_All_Photos",
], "test photos directory")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
ppm_df = pd.read_csv(PPM_PATH)
submission_sample = pd.read_csv(SAMPLE_SUB_PATH)

train_labels['sample_id'] = train_labels['sample_id'].astype(str)
submission_sample['sample_id'] = submission_sample['sample_id'].astype(str)

print(f"Loaded {len(train_labels)} train label rows, {len(submission_sample)} submission rows, {len(ppm_df)} ppm rows.")

# ------------------------------------------------------------------------------
# 3. Robust Filename Parsing & Metadata Linking (Proven by diag1)
# ------------------------------------------------------------------------------
def _tokenize(s):
    return [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t]

def fuzzy_match_sample_id(filename, label_sample_ids):
    fn_clean = re.sub(r"[^a-z0-9]+", "", filename.lower())
    best_id, best_matched, best_total = None, -1, 0
    for sample_id in label_sample_ids:
        tokens = _tokenize(sample_id)
        if len(tokens) < 3:
            continue
        matched = sum(1 for t in tokens if t in fn_clean)
        if matched >= len(tokens) - 1 and matched > best_matched:
            best_matched, best_total, best_id = matched, len(tokens), sample_id
    return best_id

def match_sample_id(filename, label_sample_ids):
    fn_lower = filename.lower()
    for sample_id in label_sample_ids:
        if str(sample_id).lower() in fn_lower:
            return sample_id
    cleaned_filename = fn_lower.replace(" ", "")
    for sample_id in label_sample_ids:
        cleaned_sid = str(sample_id).lower().replace(" ", "")
        if cleaned_sid in cleaned_filename or cleaned_filename in cleaned_sid:
            return sample_id
    return fuzzy_match_sample_id(filename, label_sample_ids)

def detect_camera(filename, ppm_df):
    fn_lower = filename.lower().replace(" ", "")
    if "iphone14" in fn_lower:
        return "iPhone 14"
    elif "iphone16" in fn_lower:
        return "iPhone 16"
    elif "60fusion" in fn_lower or ("60" in fn_lower and "fusion" in fn_lower):
        return "Motorola Edge 60 fusion"
    elif "motorola" in fn_lower or "edge" in fn_lower:
        return "motorola edge 20"
    elif "samsung" in fn_lower or "a52" in fn_lower or "sm-a525f" in fn_lower:
        return "SM-A525F"
    else:
        for _, row in ppm_df.iterrows():
            cam_spec = str(row["camera"]).lower()
            phone_spec = str(row["phone"]).lower()
            if cam_spec in filename.lower() or phone_spec in filename.lower():
                return row["camera"]
        return "unknown"

camera_to_ppm = {}
for _, row in ppm_df.iterrows():
    camera_to_ppm[str(row["phone"])] = float(row["ppm"])
    camera_to_ppm[str(row["camera"])] = float(row["ppm"])
default_ppm = float(ppm_df["ppm"].median())

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".mpo", ".JPG", ".JPEG", ".PNG"}
raw_train_paths = sorted([str(p) for p in TRAIN_DIR.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS])
raw_test_paths = sorted([str(p) for p in TEST_DIR.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS])

train_sample_list = train_labels['sample_id'].unique().tolist()
test_sample_list = submission_sample['sample_id'].unique().tolist()

train_records = []
for p in raw_train_paths:
    fn = os.path.basename(p)
    sid = match_sample_id(fn, train_sample_list)
    cam = detect_camera(fn, ppm_df)
    ppm_val = camera_to_ppm.get(cam, default_ppm)
    if sid is not None:
        train_records.append({'path': p, 'filename': fn, 'sample_id': sid, 'camera': cam, 'ppm': ppm_val})

test_records = []
for p in raw_test_paths:
    fn = os.path.basename(p)
    sid = match_sample_id(fn, test_sample_list)
    cam = detect_camera(fn, ppm_df)
    ppm_val = camera_to_ppm.get(cam, default_ppm)
    if sid is not None:
        test_records.append({'path': p, 'filename': fn, 'sample_id': sid, 'camera': cam, 'ppm': ppm_val})

train = pd.DataFrame(train_records)
test = pd.DataFrame(test_records)

print(f"Matched {len(train)} train images across {train['sample_id'].nunique()} samples.")
print(f"Matched {len(test)} test images across {test['sample_id'].nunique()} samples.")
assert len(train) > 0, "No train images matched!"
assert len(test) > 0, "No test images matched!"
assert set(test['sample_id']).issubset(set(submission_sample['sample_id'])), "Test sample_id mismatch!"

# Join labels to train dataframe
train = train.join(train_labels.set_index('sample_id')[CFG.sizes], on='sample_id')
print(f"Train DF shape: {train.shape}, Test DF shape: {test.shape}")

# ------------------------------------------------------------------------------
# 4. Physical Dataset & Transforms
# ------------------------------------------------------------------------------
class PSDDataset(Dataset):
    def __init__(self, ds_type, df, input_size=CFG.input_size):
        self.ds_type = ds_type
        self.df = df.reset_index(drop=True)
        self.return_label = ds_type != 'test'
        self.input_size = input_size
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.tfm0 = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=3),
            v2.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"])
        ppm = float(r['ppm'])
        w, h = img.size
        # Margin crop (5%)
        img = img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
        w_c, h_c = img.size
        crop_size = min(max(16, int(40 * ppm)), min(w_c, h_c))

        if self.ds_type == 'train':
            img = v2.RandomCrop(crop_size)(img)
            img = img.resize((self.input_size, self.input_size))
            img = v2.RandomHorizontalFlip()(img)
            img = v2.RandomVerticalFlip()(img)
            img = v2.RandomApply([v2.RandomRotation(degrees=(90, 90))])(img)
            img = v2.ColorJitter(brightness=0.1, contrast=0.1)(img)
        else:
            img = v2.CenterCrop(crop_size)(img)
            img = img.resize((self.input_size, self.input_size))

        ppm_scaled = ppm / crop_size * self.input_size
        label = torch.tensor([float(r[s]) for s in CFG.sizes], dtype=torch.float32) if self.return_label else str(r['sample_id'])
        return self.tfm0(img), torch.tensor(ppm_scaled, dtype=torch.float32), label

# ------------------------------------------------------------------------------
# 5. Parametric Weibull CDF Model
# ------------------------------------------------------------------------------
class PSDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model(
            model_name=CFG.model_name,
            pretrained=True,
            features_only=True
        )
        self.linear = nn.Linear(768, 2)
        with torch.no_grad():
            self.linear.bias[0] = float(np.log(20.0))
            self.linear.bias[1] = float(np.log(0.7))
        self.register_buffer("size_tensor", torch.tensor([[float(s) for s in CFG.sizes]], dtype=torch.float32))

    def forward(self, img_batch, ppm_batch):
        features = self.base_model(img_batch)[3]
        features = nn.functional.avg_pool2d(features, features.size()[2:]).flatten(1)
        weibull = torch.exp(self.linear(features))
        b = torch.clamp(weibull[:, :1] / ppm_batch.view(-1, 1), min=1e-4, max=1e4)
        c = torch.clamp(weibull[:, 1:], min=1e-4, max=10.0)
        ratio = torch.clamp(self.size_tensor / b, min=1e-6, max=1e6)
        exponent = torch.clamp(c * torch.log(ratio), min=-20.0, max=20.0)
        y_pred = 1.0 - torch.exp(- torch.exp(exponent))
        return torch.clamp(y_pred * 100.0, min=0.0, max=100.0)

loss_l1 = nn.L1Loss()
def loss_fn(y_pred, y_true):
    return loss_l1(y_pred, y_true) * CFG.n_sizes / 2.0

# ------------------------------------------------------------------------------
# 6. Training & Validation Loops
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total = 0.0, 0
    for img_batch, ppm_batch, label_batch in loader:
        img_batch = img_batch.to(CFG.device)
        ppm_batch = ppm_batch.to(CFG.device)
        label_batch = label_batch.to(CFG.device)
        optimizer.zero_grad()
        preds = model(img_batch, ppm_batch)
        loss = loss_fn(preds, label_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img_batch.size(0)
        total += label_batch.size(0)
    return total_loss / max(1, total)

def validate(model, loader):
    model.eval()
    total_loss, total = 0.0, 0
    preds_fold = []
    with torch.no_grad():
        for img_batch, ppm_batch, label_batch in loader:
            img_batch = img_batch.to(CFG.device)
            ppm_batch = ppm_batch.to(CFG.device)
            label_batch = label_batch.to(CFG.device)
            preds = model(img_batch, ppm_batch)
            loss = loss_fn(preds, label_batch)
            total_loss += loss.item() * img_batch.size(0)
            preds_fold.append(preds.cpu().numpy())
            total += label_batch.size(0)
    return total_loss / max(1, total), np.vstack(preds_fold)

def predict(model, loader):
    model.eval()
    ids, preds = [], []
    with torch.no_grad():
        for img_batch, ppm_batch, id_batch in loader:
            img_batch = img_batch.to(CFG.device)
            ppm_batch = ppm_batch.to(CFG.device)
            logits = model(img_batch, ppm_batch)
            ids.extend(id_batch)
            preds.append(logits.cpu().numpy())
    return ids, np.vstack(preds)

# ------------------------------------------------------------------------------
# 7. Group-KFold Cross-Validation on sample_id
# ------------------------------------------------------------------------------
print("\n=== STARTING 5-FOLD GROUP-KFOLD CV ON SAMPLE_ID ===")
gkf = GroupKFold(n_splits=5)
test_preds = []
oof_df = train[['sample_id', 'camera']].copy()
for s in CFG.sizes:
    oof_df[s] = 0.0

test_loader = DataLoader(
    PSDDataset('test', test),
    batch_size=CFG.batch_size,
    shuffle=False
)

for fold, (train_idx, val_idx) in enumerate(gkf.split(train, groups=train['sample_id'])):
    print(f"\n--- FOLD {fold+1}/5 ---")
    df_tr = train.iloc[train_idx]
    df_va = train.iloc[val_idx]
    print(f"Train samples: {df_tr['sample_id'].nunique()}, Val samples: {df_va['sample_id'].nunique()}")

    model = PSDModel().to(CFG.device)
    train_loader = DataLoader(PSDDataset('train', df_tr), batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(PSDDataset('val', df_va), batch_size=CFG.batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)

    best_val_loss = float('inf')
    best_preds = None

    for epoch in range(CFG.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_preds = validate(model, val_loader)
        scheduler.step()
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_preds = va_preds
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")

    print(f"Fold {fold+1} Best Val EMD Loss: {best_val_loss:.4f}")
    oof_df.iloc[val_idx, 2:] = best_preds

    model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
    _, fold_test_preds = predict(model, test_loader)
    test_preds.append(fold_test_preds)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

# Overall OOF Evaluation
oof_agg = oof_df.groupby('sample_id')[CFG.sizes].mean().reset_index()
train_labels_matched = train_labels.set_index('sample_id').loc[oof_agg['sample_id']].reset_index()
oof_score = np.mean(np.abs(train_labels_matched[CFG.sizes].values - oof_agg[CFG.sizes].values)) * CFG.n_sizes / 2.0
print(f"\n=======================================================")
print(f"OVERALL OOF EMD LOSS AFTER ENSEMBLING: {oof_score:.4f}")
print(f"=======================================================")

# ------------------------------------------------------------------------------
# 8. Test Submission Generation & Contract Verification
# ------------------------------------------------------------------------------
mean_test_preds = np.mean(test_preds, axis=0) # average across folds
test_sub_raw = pd.DataFrame(mean_test_preds, columns=CFG.sizes)
test_sub_raw['sample_id'] = test['sample_id']

# Group by sample_id to average predictions across multiple images of same test sample
submission = test_sub_raw.groupby('sample_id')[CFG.sizes].mean().reset_index()
submission['200'] = 100.0 # Upper bound constant particle size

# Align with exact sample_submission.csv
submission = submission_sample[['sample_id']].merge(submission, on='sample_id', how='left')

# Contract Invariants:
assert submission.shape == submission_sample.shape, f"Shape mismatch: {submission.shape} vs {submission_sample.shape}"
assert (submission['sample_id'] == submission_sample['sample_id']).all(), "Sample ID mismatch!"
assert submission.isnull().sum().sum() == 0, "Contains NaNs!"
assert (np.diff(submission.iloc[:, 1:].values, axis=1) >= -1e-5).all(), "Non-monotonic cumulative curve detected!"

sub_out_path = "submission.csv"
submission.to_csv(sub_out_path, index=False)
print(f"Saved submission to {sub_out_path} ({len(submission)} rows).")
print(submission.head(10))
