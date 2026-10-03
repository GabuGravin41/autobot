import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, GroupKFold
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import timm
import pickle
from glob import glob
from colorama import Fore, Style
import gc

is_interactive = os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'
tqdm_if_int = tqdm if is_interactive else lambda x, **kwargs: x

class CFG:
    model_name = "convnext_tiny.fb_in22k_ft_in1k" # 114MByte download, 28.6M parameters
    pretrained_cfg_overlay = {'file': '/kaggle/input/models/ambrosm/convnext-tiny-fb-in22k-ft-in1k/pytorch/default/1/model.safetensors'}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 320 # pixels
    batch_size = 32
    T_0 = 20 # epochs for cyclic learning rate schedule
    epochs = T_0
    lr = 1e-5 # initial learning rate
    sizes = ['0.002', '0.0063', '0.02', '0.063', '0.2', '0.63', '2', '6.3', '20', '63'] # particle sizes except the last one (200)
    n_sizes = len(sizes) # 10
    

train_labels = pd.read_csv('/kaggle/input/competitions/soil-grain-size-from-photos/Training_labels.csv')
submission = pd.read_csv('/kaggle/input/competitions/soil-grain-size-from-photos/sample_submission.csv')
# display(train_labels.head(2))

ppm = pd.read_csv('/kaggle/input/competitions/soil-grain-size-from-photos/ppm.csv')
ppm

# Create a train dataframe with one row per image
train = pd.DataFrame({
    'path': sorted(glob("/kaggle/input/competitions/soil-grain-size-from-photos/Training-All_Photos/Training-All_Photos/*.jpg"))
})
temp = train.path.apply(lambda path: path.split('/')[-1].split('_'))
train['sample_id'] = temp.apply(lambda t: t[2])
train['phone'] = temp.apply(lambda t: ' '.join(t[:2]))
train = train.join(ppm.set_index('phone')['ppm'], on='phone')
train = train.join(train_labels.set_index('sample_id').iloc[:,:-1], on='sample_id') # drop the constant eleventh column
train.head()


# Create a test dataframe with one row per image
test = pd.DataFrame({
    'path': sorted(glob("/kaggle/input/competitions/soil-grain-size-from-photos/Test_All_Photos/Test_All_Photos/*.JPG"))
})
temp = test.path.apply(lambda path: path.split('/')[-1])
test['sample_id'] = temp.apply(lambda t: t[t.index('HPC'):-4])
test['sample_id'] = test['sample_id'].str.replace(r' \(\d\)', '', regex=True).str.replace('HPC_Münster_BS6_9,0-10m', 'HPC_Muenster_BS6_9_0-10m')
assert test.sample_id.isin(submission.sample_id).all()
test['phone'] = temp.apply(lambda t: t[:t.index('1') + 2].replace('_', '').replace('iPhone', 'iPhone '))
test = test.join(ppm.set_index('phone')['ppm'], on='phone')
test.head(2)

class PSDDataset(Dataset):
    
    def __init__(self, ds_type, df, model, input_size=CFG.input_size):
        """Image dataset.

        Returns square images of the correct resolution for model.

        Parameters
        ds_type: 'train', 'val', or 'test'
        df: either a subset of train or test
        model: a timm model, which defines the normalization parameters
        """
        assert ds_type in ['train', 'val', 'test']
        self.ds_type = ds_type
        self.df = df
        self.return_label = ds_type != 'test'
        self.shapes = []
        self.model = model
        self.input_size = input_size
        self.tfm0 = v2.Compose([ # applied to all images after augmentation transforms
            v2.ToImage(), # converts PIL to tensor
            v2.ToDtype(torch.float32, scale=True), # converts uint8 to float and divides by 255
            v2.Grayscale(num_output_channels=3),
            v2.Normalize(mean=model.pretrained_cfg['mean'], std=model.pretrained_cfg['std'])
        ])

    def __len__(self):
        """Length of the dataset"""
        return len(self.df)

    def __getitem__(self, i):
        """Return a 320x320px image covering 40x40mm, its ppm and its labels or its sample_id."""
        r = self.df.iloc[i]
        img = Image.open(r["path"])#.convert("RGB")
        crop_size = int(40 * r['ppm'])
        img = img.crop((int(img.width * 0.05), int(img.height * 0.05), int(img.width * 0.95), int(img.height * 0.95))) # left, top, right, bottom
        if self.ds_type == 'train':
            img = v2.RandomCrop(crop_size)(img)
        else:
            img = v2.CenterCrop(crop_size)(img)
        img = img.resize((self.input_size, self.input_size))
        ppm = r['ppm'] / crop_size * self.input_size
        if self.ds_type == 'train':
            img = v2.RandomHorizontalFlip()(img)
            img = v2.RandomVerticalFlip()(img)
            img = v2.RandomApply([v2.RandomRotation(degrees=(90,90))])(img)
            img = v2.ColorJitter(brightness=0.1, contrast=0.1)(img)
        return (self.tfm0(img), 
                torch.tensor(ppm),
                torch.tensor(list(r.loc[CFG.sizes])) if self.return_label else r.sample_id
               )
        

class PSDModel(nn.Module):
    """Model for predicting particle size distribution.

    The model is based on a Convnet. It predicts the two parameters of a Weibull distribution and then
    computes the cumulative density function for the ten defined particle sizes.

    Input: batch of images (n, 3, h, w)

    Output: batch of n*n_sizes cdf values in percent, evaluated at CFG.sizes
    """
    def __init__(self):
        super().__init__()
        # pretrained_cfg_overlay loads the weights from the Kaggle model, which is more reliable than the download from Huggingface.
        # The weights are the same as on Huggingface.
        self.base_model = timm.create_model(
            model_name=CFG.model_name,
            pretrained_cfg_overlay=CFG.pretrained_cfg_overlay,
            pretrained=True,
            features_only=True
        )
        
        # Add a regression head
        self.linear = nn.Linear(768, 2)
        with torch.no_grad():
            self.linear.bias[0] = np.log(20) # depends on ppm and image scaling
            self.linear.bias[1] = np.log(0.7)
            # torch.nn.init.zeros_(self.linear.weight)
        self.size_tensor = torch.tensor([[float(s) for s in CFG.sizes]]).to(CFG.device)

    def forward(self, img_batch, ppm_batch):
        features = self.base_model(img_batch)
        # print(features[0].shape) # [n, 96, h // 4, w // 4]
        # print(features[1].shape) # [n, 192, h // 8, w // 8]
        # print(features[2].shape) # [n, 384, h // 16, w // 16]
        # print(features[3].shape) # [n, 768, h // 32, w // 32]
        features = features[3]
        features = nn.functional.avg_pool2d(features, features.size()[2:]).flatten(1) # [n, c]
        weibull = torch.exp(self.linear(features)) # [n, 2], parameters b and c of the Weibull distribution
        b, c = weibull[:, :1] / ppm_batch.view(-1, 1), weibull[:, 1:]
        y_pred = 1 - torch.exp(- torch.exp(c * torch.log(self.size_tensor / b)))
        y_pred *= 100 # convert predicted cdf to percent
        return y_pred

# PSDModel().to(CFG.device)(torch.randn(2, 3, 320, 320).to(CFG.device), torch.tensor([8, 8]).to(CFG.device))

loss0 = nn.L1Loss()

def loss_fn(y_pred, y_true):
    """Earth Mover's Distance (EMD) or Wasserstein-1 distance.
    
    Scalar MAE loss, summed over the n_sizes elements of every row, multiplied with distance, mean over batch"""
    return loss0(y_pred, y_true) * CFG.n_sizes / 2

def train_one_epoch(model, loader, optimizer, scaler):
    """Train the model for one epoch.
    
    Return cross-entropy loss and accuracy.
    """
    model.train()
    
    total_loss = 0
    total = 0

    pbar = loader

    for i, (img_batch, ppm_batch, label_batch) in enumerate(pbar):
        img_batch = img_batch.to(CFG.device, memory_format=torch.channels_last)
        ppm_batch = ppm_batch.to(CFG.device)
        label_batch = label_batch.to(CFG.device)

        optimizer.zero_grad()
        with torch.amp.autocast(CFG.device, enabled=False and CFG.device=='cuda'):
            preds = model(img_batch, ppm_batch)
            loss = loss_fn(preds, label_batch)
            # print('loss', loss)
        
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        total_loss += loss.item() * img_batch.size(0)

        total += label_batch.size(0)

    return total_loss / total

def validate(model, loader):
    """Validate the model.

    Return cross-entropy loss, accuracy and logits.
    """
    model.eval()

    total_loss = 0
    total = 0
    preds_fold = []

    pbar = loader

    with torch.no_grad():
        for img_batch, ppm_batch, label_batch in pbar:
            img_batch = img_batch.to(CFG.device, memory_format=torch.channels_last)
            ppm_batch = ppm_batch.to(CFG.device)
            label_batch = label_batch.to(CFG.device)

            preds_batch = model(img_batch, ppm_batch)
            loss = loss_fn(preds_batch, label_batch)

            total_loss += loss.item() * img_batch.size(0)

            preds_fold.append(preds_batch.cpu().numpy())
            total += label_batch.size(0)

    return total_loss / total, np.vstack(preds_fold)

def predict(model, loader):
    """Predict the particle size distributions of the images using model.
    
    Return a list of ids and a prediction array of shape (n_samples, n_sizes).
    """
    model.eval()
    ids, logits = [], []
    with torch.no_grad():
        for img_batch, ppm_batch, id_batch in loader: # tqdm_if_int(loader, desc="Test"):
            img_batch = img_batch.to(CFG.device, memory_format=torch.channels_last)
            ppm_batch = ppm_batch.to(CFG.device)
            logits_batch = model(img_batch, ppm_batch)
            ids.extend(id_batch)
            logits.append(logits_batch.cpu().numpy())
    return ids, np.vstack(logits)

%%time
print(f"Device: {CFG.device}   Model: {CFG.model_name}")
kf = GroupKFold(n_splits=25, shuffle=True, random_state=1)

def run_all_folds():
    if torch.cuda.is_available():
        torch.cuda.memory.empty_cache()
    preds_te = []
    preds_oof = np.full((len(train), CFG.n_sizes), np.nan)
    
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, groups=train.sample_id)):
        df_tr = train.iloc[idx_tr]
        df_va = train.iloc[idx_va]

        print()
        # dummy_score = np.abs(df_tr[CFG.sizes].median().values.reshape(1, -1) - df_va[CFG.sizes].values).mean() * CFG.n_sizes / 2
        # print(f'Fold {fold:2} dummy loss: {dummy_score:7.3f}')
    
        model = PSDModel()
        model = model.to(CFG.device)
        model = model.to(memory_format=torch.channels_last)
        
        train_loader = DataLoader(
            PSDDataset('train', df_tr, model=model.base_model),
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 1,
            pin_memory=CFG.device=='cuda'
        )
        
        val_loader = DataLoader(
            PSDDataset('val', df_va, model=model.base_model),
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 1,
            pin_memory=CFG.device=='cuda'
        )
        
        test_loader = DataLoader(
            PSDDataset('test', test, model=model.base_model),
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 1,
            pin_memory=CFG.device=='cuda'
        )
    
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=CFG.T_0,
            eta_min=1e-6
        )
        scaler = torch.amp.GradScaler(CFG.device, enabled=CFG.device=='cuda')
        
        # Train
        best_loss = np.inf
        for epoch in range(CFG.epochs):
            # Train one epoch
            print(f"Fold {fold:2} epoch {epoch+1:2}/{CFG.epochs:2}: lr={scheduler.get_last_lr()[0]:.1e}", end='   ')
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
            print(f"train_loss={train_loss:7.3f}", end='   ')
            scheduler.step()
    
            # Validate after every epoch
            val_loss, preds_fold = validate(model, val_loader)
            preds_oof[idx_va] = preds_fold
            c1 = Fore.GREEN if epoch == CFG.epochs - 1 else ''
            c2 = f"{Style.RESET_ALL} ***" if epoch == CFG.epochs - 1 else ''
            print(f"{c1}val_loss={val_loss:7.3f}{c2}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"best_model{fold}.pth")

        del optimizer, scaler, scheduler
        gc.collect()
        
        # Predict
        model.load_state_dict(torch.load(f"best_model{fold}.pth"))
        ids, preds = predict(model, test_loader)
        assert (test["sample_id"] == ids).all()
        preds_te.append(preds)

        del model

    print(f"Overall loss = {np.abs(train[CFG.sizes].values - preds_oof).mean() * CFG.n_sizes / 2:.5f}")
    return preds_oof, preds_te

preds_oof, preds_te = run_all_folds()

oof = pd.concat([
    train.sample_id,
    pd.DataFrame(preds_oof, columns=CFG.sizes),
], axis=1)
oof['phone'] = train['phone']

# # Plot particle size distributions per sample_id
# for sample_id, cdf in oof.groupby('sample_id'):
#     print(cdf.phone.iloc[0])
#     plt.title(f"oof {sample_id}")
#     plt.plot([float(s) for s in CFG.sizes], cdf[CFG.sizes].T, color='m' if cdf.phone.iloc[0].startswith('M') else 'c')
#     plt.xscale('log')
#     # plt.show()
    
for _, row in oof.iterrows():
    plt.plot([float(s) for s in CFG.sizes], row[CFG.sizes], color='m' if row.phone.startswith('M') else 'c')
plt.xscale('log')
plt.title('OOF particle size distributions')
plt.xlabel('particle size')
plt.ylabel('cdf')
plt.show()

# Ensemble the oof predictions
oof = oof.groupby('sample_id')[CFG.sizes].mean()
oof.reset_index(inplace=True)
display(oof)

# Score oof values after tta
assert (oof.sample_id == train_labels.sample_id).all()
score = np.mean(np.abs(train_labels[CFG.sizes] - oof[CFG.sizes])) * CFG.n_sizes / 2
print(f"OOF loss after ensembling: {score:7.3f}")


submission = pd.concat([
    test.sample_id,
    pd.DataFrame(np.mean(preds_te, axis=0), columns=CFG.sizes),
], axis=1)
submission['phone'] = test['phone']

# # Plot particle size distributions per sample_id
# for sample_id, cdf in submission.groupby('sample_id')[CFG.sizes]:
#     plt.title(f"test {sample_id}")
#     plt.plot([float(s) for s in CFG.sizes], cdf.T)
#     plt.xscale('log')
#     plt.show()

# Plot all particle size distribution, colored by camera type
for _, row in submission.iterrows():
    plt.plot([float(s) for s in CFG.sizes], row[CFG.sizes], color='m' if '4' in row.phone else 'c')
plt.xscale('log')
plt.title('Test particle size distributions')
plt.xlabel('particle size')
plt.ylabel('cdf')
plt.show()

# Ensemble the test predictions
submission = submission.groupby('sample_id')[CFG.sizes].mean()
submission.reset_index(inplace=True)
submission['200'] = 100.0
display(submission)

submission.to_csv('submission.csv', index=False)
!head submission.csv



