import os, sys

class SuppressStderr:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.stderr

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
cv2.setNumThreads(0)

import re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG

class CFG:
    IMG_SIZE = 256
    BATCH = 16
    EPOCHS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_PATH = "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"

APO_IMG_DIR = os.path.join(BASE_PATH, "apo_imgs_v1/apo_images_new_model_v1")
APO_MASK_DIR = os.path.join(BASE_PATH, "apo_masks_v1/apo_masks_new_model_v1")
TEST_DIR = os.path.join(BASE_PATH, "test_images_v2/test_set_v2")

# HELPERS

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.uint8)
    return img

def is_image(f):
    return f.endswith((".png",".tif",".jpg"))

def get_id(name):
    nums = re.findall(r'\d+', name)
    return nums[0] if nums else name

# LOAD TRAIN

img_dict, mask_dict = {}, {}

for f in os.listdir(APO_IMG_DIR):
    if is_image(f):
        img_dict[get_id(f)] = os.path.join(APO_IMG_DIR, f)

for f in os.listdir(APO_MASK_DIR):
    if is_image(f):
        mask_dict[get_id(f)] = os.path.join(APO_MASK_DIR, f)

common_ids = list(set(img_dict.keys()) & set(mask_dict.keys()))

apo_imgs = [img_dict[i] for i in common_ids]
apo_masks = [mask_dict[i] for i in common_ids]

print("TRAIN SAMPLES:", len(apo_imgs))

# AUGMENTATION

train_tfms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

# DATASET

class SegDataset(Dataset):
    def __init__(self, imgs, masks):
        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = read_img(self.imgs[i])
        mask = read_img(self.masks[i])

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        img = np.stack([img]*3, -1)
        mask = (mask > 0).astype(np.float32)

        data = train_tfms(image=img, mask=mask)
        return data["image"], data["mask"].unsqueeze(0)

# MODEL

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3,16,3,1,1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16,32,3,2,1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64,32,2,2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(32,16,2,2), nn.ReLU())
        self.out = nn.Conv2d(16,1,1)

    def forward(self,x):
        return self.out(self.dec2(self.dec1(self.enc3(self.enc2(self.enc1(x))))))

# TRAIN

train_ds = SegDataset(apo_imgs, apo_masks)
loader = DataLoader(train_ds, batch_size=CFG.BATCH, shuffle=True)

model = UNet().to(CFG.DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

print("Training...")

with SuppressStderr():
    model.train()
    for epoch in range(CFG.EPOCHS):
        for imgs, masks in loader:
            imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

print("Training complete")

# LOAD TEST

sample_sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"), sep=";")
id_col = [c for c in sample_sub.columns if "id" in c.lower()][0]

test_dict = {}
for f in os.listdir(TEST_DIR):
    if is_image(f):
        test_dict[get_id(f)] = os.path.join(TEST_DIR, f)

def get_path(img_id):
    return test_dict.get(get_id(str(img_id)), list(test_dict.values())[0])

sample_sub["path"] = sample_sub[id_col].apply(get_path)

# GEOMETRY

def compute_mt(mask):
    ys = np.where(mask>0)[0]
    if len(ys)<20: return 25
    return np.percentile(ys,95) - np.percentile(ys,5)

def compute_fl(mask):
    ys,xs = np.where(mask>0)
    if len(xs)<20: return 100
    return np.sqrt((xs.max()-xs.min())**2+(ys.max()-ys.min())**2)

def compute_pa(mask):
    ys,xs = np.where(mask>0)
    if len(xs)<30: return 20
    pts = np.column_stack([xs,ys]).astype(np.float32)
    [vx,vy,_,_] = cv2.fitLine(pts, cv2.DIST_L2,0,0.01,0.01)
    angle = abs(np.degrees(np.arctan2(vy,vx)))
    if angle > 90:
        angle = 180 - angle
    return angle

# MASK PREDICTION

def predict_mask(img):
    img = cv2.resize(img,(CFG.IMG_SIZE,CFG.IMG_SIZE))

    def infer(x):
        x = x.copy()
        x = torch.tensor(x).float().unsqueeze(0).unsqueeze(0)/255
        x = x.repeat(1,3,1,1).to(CFG.DEVICE)
        with torch.no_grad():
            return torch.sigmoid(model(x)).cpu().numpy()[0,0]

    p1 = infer(img)
    p2 = np.fliplr(infer(np.fliplr(img).copy()))

    pred = (p1 + p2) / 2

    thresh = np.percentile(pred, 75)

    mask = (pred > thresh).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)

    return mask

# INFERENCE

pa_list, fl_list, mt_list = [], [], []

with SuppressStderr():
    for path in tqdm(sample_sub["path"]):
        img = read_img(path)
        mask = predict_mask(img)

        pa_list.append(compute_pa(mask))
        fl_list.append(compute_fl(mask))
        mt_list.append(compute_mt(mask))

# SAVE

sample_sub["pa_deg"] = np.clip(pa_list,5,45)
sample_sub["fl_mm"] = np.clip(fl_list,30,200)
sample_sub["mt_mm"] = np.clip(mt_list,10,50)

sample_sub.to_csv("submission.csv", index=False)

print("'submission.csv' created.")