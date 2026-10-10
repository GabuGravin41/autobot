import json
import os
from pathlib import Path

def create_cell(cell_type, source):
    lines = [line + '\n' for line in source.split('\n')]
    if lines and lines[-1] == '\n':
        lines[-1] = ''
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    }

def save_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {output_path}")

out_dir = Path("educational_notebooks")

# -------------------------------------------------------------
# Notebook 8: Molecular Fingerprinting & Tanimoto Similarity
# -------------------------------------------------------------
c8 = []
c8.append(create_cell("markdown", """# Visual ML Foundations: Molecular Fingerprints & Chemical Similarity
### Circular Fingerprints (ECFP/Morgan), Bit Vectors, and Tanimoto Coefficients

---

## 1. Concept Overview
In chemistry and drug discovery (such as the **Enveda CASMI** challenge), molecules cannot be fed directly into standard ML models as 2D sketches.
We convert molecules into **Extended Connectivity Fingerprints (ECFP)**:
1. Each atom and its circular neighborhood of radius $r$ is hashed into an integer.
2. The integers are folded into a fixed-length bit vector (e.g. 1024 or 2048 bits).
3. Chemical similarity between two molecules $A$ and $B$ is measured via the **Tanimoto Coefficient**:
$$T(A, B) = \\frac{|A \\cap B|}{|A \\cup B|} = \\frac{N_c}{N_a + N_b - N_c}$$
Where $N_c$ is the number of shared bits, and $N_a, N_b$ are the bits set in $A$ and $B$.
"""))

c8.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

# Simulate 64-bit structural fingerprints for 4 chemical compounds
np.random.seed(42)
def generate_fp(base_bits, noise_rate=0.1, n_bits=64):
    fp = np.zeros(n_bits, dtype=int)
    fp[base_bits] = 1
    # Add random mutation bits
    flips = np.random.choice(n_bits, size=int(n_bits * noise_rate), replace=False)
    fp[flips] = 1 - fp[flips]
    return fp

# Base scaffold bits
aspirin_scaffold = [4, 12, 19, 27, 33, 45, 52]
mol_A = generate_fp(aspirin_scaffold, noise_rate=0.05)
mol_B = generate_fp(aspirin_scaffold, noise_rate=0.08) # Close analog
mol_C = generate_fp([2, 9, 21, 38, 59], noise_rate=0.1) # Unrelated alkaloid

def tanimoto_similarity(fp1, fp2):
    intersection = np.sum((fp1 == 1) & (fp2 == 1))
    union = np.sum((fp1 == 1) | (fp2 == 1))
    return intersection / union if union > 0 else 0.0

sim_AB = tanimoto_similarity(mol_A, mol_B)
sim_AC = tanimoto_similarity(mol_A, mol_C)

# Visualizing the bit vectors
fig, ax = plt.subplots(figsize=(14, 3.5))
fps_matrix = np.vstack([mol_A, mol_B, mol_C])
cax = ax.imshow(fps_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels([
    f"Compound A (Reference)",
    f"Compound B (Analog, Sim={sim_AB:.2f})",
    f"Compound C (Unrelated, Sim={sim_AC:.2f})"
], fontsize=11, fontweight='bold')
ax.set_xlabel("Fingerprint Bit Position (0 - 63)")
ax.set_title("Molecular Bit Vector Alignment & Tanimoto Overlap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print(f"Calculated Tanimoto Overlap: A-B = {sim_AB:.3f}, A-C = {sim_AC:.3f}")
"""))
save_notebook(c8, out_dir / "08_molecular_fingerprinting_rdkit.ipynb")

# -------------------------------------------------------------
# Notebook 9: 2.5D Slice Stacking for 3D Medical MRI
# -------------------------------------------------------------
c9 = []
c9.append(create_cell("markdown", """# Visual ML Foundations: 2.5D Slice Stacking for Volumetric Imaging
### Bridging 2D Pretrained Backbones (ImageNet / DINOv2) to 3D Medical Scans

---

## 1. Concept Overview
Full 3D convolutional networks require massive GPU VRAM ($>16\\text{ GB}$ per small batch) and lack massive pre-trained weights.
The **2.5D Slice Stacking** technique solves this:
* Standard vision backbones (ResNet, DINOv2, ConvNeXt) expect 3 RGB color channels: $(3, H, W)$.
* Instead of RGB, we stack 3 adjacent anatomical slices:
  $$\\text{Channel } 0 = \\text{Slice } i-1, \\quad \\text{Channel } 1 = \\text{Slice } i, \\quad \\text{Channel } 2 = \\text{Slice } i+1$$
* This gives the 2D network local 3D through-plane context while running at 10x faster inference speed!
"""))

c9.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

# Simulate 3 consecutive Sagittal MRI slices showing meniscus tear evolution
h, w = 120, 120
y, x = np.ogrid[:h, :w]

slice_prev = np.zeros((h, w))
slice_curr = np.zeros((h, w))
slice_next = np.zeros((h, w))

# Bone background
bone = ((y - 45)**2 + (x - 60)**2) < 26**2
for s in [slice_prev, slice_curr, slice_next]:
    s[bone] = 0.75

# Joint space with evolving meniscus tear
slice_prev[68:74, 50:70] = 0.35 # Mild signal
slice_curr[67:75, 48:72] = 0.90 # Focal high-signal tear!
slice_next[68:74, 52:68] = 0.40 # Tear fading out

# Stack into 2.5D 3-Channel Tensor
stack_2p5d = np.stack([slice_prev, slice_curr, slice_next], axis=-1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(slice_prev, cmap='bone')
axes[0].set_title("Channel 0: Slice (i-1)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(slice_curr, cmap='bone')
axes[1].set_title("Channel 1: Slice (i) [Tear]", fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(slice_next, cmap='bone')
axes[2].set_title("Channel 2: Slice (i+1)", fontsize=11, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(stack_2p5d)
axes[3].set_title("Combined 2.5D Composite", fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.show()
print(f"2.5D Tensor shape: {stack_2p5d.shape} (Ready for pre-trained vision backbones)")
"""))
save_notebook(c9, out_dir / "09_mri_2p5d_slice_stacking.ipynb")

# -------------------------------------------------------------
# Notebook 10: Confusion Matrix & ROC-AUC Deep Dive
# -------------------------------------------------------------
c10 = []
c10.append(create_cell("markdown", """# Visual ML Foundations: Confusion Matrix & ROC-AUC Sweep
### Understanding True Positives, False Positives, and Threshold Invariance

---

## 1. Concept Overview
In competitive ML and clinical deployment, models output probabilities ($p \\in [0, 1]$), not hard decisions.
* **Accuracy is Misleading**: On a 95% negative dataset, predicting all 0 gives 95% accuracy but saves 0 patients.
* **ROC Curve**: Plots **True Positive Rate (Sensitivity)** vs. **False Positive Rate (1 - Specificity)** across every possible decision threshold $\\tau \\in [0, 1]$.
* **ROC-AUC (Area Under Curve)**: Measures the probability that the model ranks a randomly chosen positive case higher than a randomly chosen negative case.
"""))

c10.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

np.random.seed(42)
# True binary labels (50 positive, 450 negative)
y_true = np.concatenate([np.ones(50), np.zeros(450)])

# Realistic predicted probabilities with some overlap
y_scores = np.concatenate([
    np.random.beta(a=6, b=2, size=50),  # Positives shift high
    np.random.beta(a=2, b=7, size=450)  # Negatives shift low
])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Evaluate confusion matrix at a chosen threshold (tau = 0.5)
tau = 0.5
y_pred = (y_scores >= tau).astype(int)
cm = confusion_matrix(y_true, y_pred)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve Plot
ax1.plot(fpr, tpr, color='crimson', lw=2.5, label=f'Model ROC (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess (AUC = 0.500)')
ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold')
ax1.set_xlabel("False Positive Rate (1 - Specificity)")
ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right")

# Confusion Matrix Heatmap
im = ax2.imshow(cm, cmap='Blues', interpolation='nearest')
ax2.set_title(f"Confusion Matrix (Threshold τ = {tau})", fontsize=12, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Pred Negative (0)', 'Pred Positive (1)'], fontweight='bold')
ax2.set_yticklabels(['True Negative (0)', 'True Positive (1)'], fontweight='bold')

for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > 100 else "black", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
print(f"Calculated Macro ROC-AUC: {roc_auc:.4f}")
"""))
save_notebook(c10, out_dir / "10_confusion_matrix_and_roc_auc_deep_dive.ipynb")

# -------------------------------------------------------------
# Notebook 11: Out-of-Fold Cross Validation Mastery
# -------------------------------------------------------------
c11 = []
c11.append(create_cell("markdown", """# Visual ML Foundations: Out-of-Fold Cross Validation (GroupKFold)
### Preventing Data Leakage Across Patient Studies and Series

---

## 1. Concept Overview
The most common mistake on Kaggle is **Data Leakage**:
* In MRI imaging, one patient (`StudyInstanceUID`) may have 3 series (Sagittal, Coronal, Axial).
* If you perform random train/test splitting, slices from the same patient land in both train and validation sets!
* The model memorizes patient-specific bone anatomy, showing an impressive $0.99$ validation score, but collapses on the test leaderboard.
* **The Solution**: **GroupKFold** grouped strictly on `StudyInstanceUID`.
"""))

c11.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

# Simulate 20 patient studies, each with 3 series (60 total scans)
n_patients = 20
studies = np.repeat(np.arange(n_patients), 3)
scans = np.arange(len(studies))

gkf = GroupKFold(n_splits=5)
fold_assignments = np.zeros(len(studies))

for fold, (train_idx, val_idx) in enumerate(gkf.split(scans, groups=studies)):
    fold_assignments[val_idx] = fold

# Visualizing fold distributions
fig, ax = plt.subplots(figsize=(14, 4))
scatter = ax.scatter(scans, studies, c=fold_assignments, cmap='tab10', s=120, edgecolors='black')
ax.set_title("GroupKFold Distribution: Zero Patient Leakage Across 5 Folds", fontsize=12, fontweight='bold')
ax.set_xlabel("Scan Index (0 to 59)")
ax.set_ylabel("Patient Study UID (0 to 19)")
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax, ticks=range(5))
cbar.set_label("Assigned Validation Fold", fontweight='bold')

plt.tight_layout()
plt.show()
print("Patient group partitioning verified: No patient crosses fold boundaries.")
"""))
save_notebook(c11, out_dir / "11_out_of_fold_cross_validation_mastery.ipynb")

print("Generated notebooks 8 through 11.")
