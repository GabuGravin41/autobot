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
# Notebook 4: DICOM Windowing and Hounsfield Unit Scaling
# -------------------------------------------------------------
c4 = []
c4.append(create_cell("markdown", """# Visual ML Foundations: Medical Image Windowing & Intensity Scaling
### Dynamic Range Compression: Bone vs. Soft Tissue Windows

---

## 1. Concept Overview
Medical scans (CT and MRI) produce 12-bit to 16-bit grayscale values (ranging from -1000 to +3000 Hounsfield Units, or raw MRI proton density values). Computer monitors and vision models expect 8-bit values ($[0, 255]$).

If you normalize globally with `img / max`, subtle soft tissue contrasts (ligament tears, cartilage wear) disappear into black or gray mush.
We use **Window Center (Level)** and **Window Width**:
$$\\text{Val}_{\\text{min}} = \\text{Center} - \\frac{\\text{Width}}{2}, \\quad \\text{Val}_{\\text{max}} = \\text{Center} + \\frac{\\text{Width}}{2}$$
Values outside this window are clamped, and the remaining band is mapped linearly to $[0, 1]$.
"""))

c4.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

def apply_windowing(image, window_center, window_width):
    \"\"\"Apply radiological window level and width.\"\"\"
    img_min = window_center - window_width / 2.0
    img_max = window_center + window_width / 2.0
    windowed = np.clip(image, img_min, img_max)
    return (windowed - img_min) / (img_max - img_min)

# Simulate wide dynamic-range radiologic scan (-500 to +1500 intensity)
np.random.seed(42)
y, x = np.ogrid[:200, :200]
base_scan = np.full((200, 200), -200.0) # Background fat/air

# Soft tissue muscle/capsule (-50 to +150)
soft_tissue = ((y - 100)**2 + (x - 100)**2) < 70**2
base_scan[soft_tissue] = np.random.normal(50, 25, size=np.sum(soft_tissue))

# Dense cortical bone (800 to 1400)
bone_ring = (((y - 100)**2 + (x - 100)**2) < 35**2) & (((y - 100)**2 + (x - 100)**2) > 20**2)
base_scan[bone_ring] = np.random.normal(1100, 50, size=np.sum(bone_ring))

# Compare Default Global Scaling vs. Specialized Windows
global_scaled = (base_scan - base_scan.min()) / (base_scan.max() - base_scan.min())
soft_tissue_win = apply_windowing(base_scan, window_center=50, window_width=250)
bone_win = apply_windowing(base_scan, window_center=1000, window_width=800)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(global_scaled, cmap='bone')
axes[0].set_title("Global Min-Max (Poor Contrast)", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(soft_tissue_win, cmap='bone')
axes[1].set_title("Soft Tissue Window (C=50, W=250)", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(bone_win, cmap='bone')
axes[2].set_title("Bone Window (C=1000, W=800)", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
print("Radiological windowing comparison rendered successfully.")
"""))
save_notebook(c4, out_dir / "04_dicom_windowing_and_hu_scaling.ipynb")

# -------------------------------------------------------------
# Notebook 5: Medical Image Augmentations
# -------------------------------------------------------------
c5 = []
c5.append(create_cell("markdown", """# Visual ML Foundations: Medical Image Augmentations
### Safe vs. Dangerous Augmentations for Radiologic Diagnostic Models

---

## 1. Concept Overview
In natural image classification (ImageNet), arbitrary rotations and vertical flips are common. In clinical radiology:
* **Vertical Flip**: Often anatomically impossible (knees do not invert along the proximal-distal axis in a scanner).
* **Safe Augmentations**: Slight horizontal flip (with careful class re-mapping e.g., Medial vs Lateral!), mild elastic deformation (simulating patient positioning variance), subtle affine rotations ($\\pm 10^\\circ$), and Gaussian noise/blur.
"""))

c5.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter

# Generate baseline knee joint slice
y, x = np.ogrid[:150, :150]
knee = np.zeros((150, 150))
knee[((y - 50)**2 + (x - 75)**2) < 30**2] = 0.8 # Femur
knee[((y - 110)**2 + (x - 75)**2) < 35**2] = 0.7 # Tibia

# 1. Subtle Affine Rotation (-8 deg)
rotated = rotate(knee, angle=-8, reshape=False, mode='nearest')

# 2. Gaussian Noise (Low SNR simulation)
noisy = np.clip(knee + np.random.normal(0, 0.08, knee.shape), 0, 1)

# 3. Elastic / Blur Deformation (Soft tissue deformation)
blurred = gaussian_filter(knee, sigma=1.2)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ["Original Knee", "Affine Rotation (-8°)", "Gaussian Sensor Noise", "Gaussian Smoothing"]
imgs = [knee, rotated, noisy, blurred]

for ax, im, t in zip(axes, imgs, titles):
    ax.imshow(im, cmap='bone')
    ax.set_title(t, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()
print("Clinical image augmentation suite visualized.")
"""))
save_notebook(c5, out_dir / "05_medical_image_augmentations.ipynb")

# -------------------------------------------------------------
# Notebook 6: Multilabel Class Imbalance Strategies
# -------------------------------------------------------------
c6 = []
c6.append(create_cell("markdown", """# Visual ML Foundations: Handling Extreme Multilabel Class Imbalance
### Positive Weighting, Asymmetric Loss, and Focal Loss Mechanics

---

## 1. Concept Overview
In clinical datasets like RSNA Knee, common findings (like Joint Effusion) appear in 60% of patients, whereas rare findings (like MCL tears or Fractures) appear in only 3% to 5% of cases.
If trained with standard Binary Cross Entropy (BCE):
$$\\mathcal{L} = - [y \\log p + (1-y) \\log(1-p)]$$
The model quickly discovers that predicting $0$ for rare classes yields $97\\%$ accuracy, completely failing to learn rare abnormalities.
Solutions:
1. **Positive Weight ($w_\\text{pos}$)**: Multiplies loss on positive errors by $\\frac{N_\\text{neg}}{N_\\text{pos}}$.
2. **Focal Loss**: Adds a $(1-p)^\\gamma$ focusing parameter that down-weights easy negatives.
"""))

c6.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 500)

# True positive case (y=1)
bce_pos = -np.log(p)
focal_gamma1 = -(1 - p)**1 * np.log(p)
focal_gamma2 = -(1 - p)**2 * np.log(p)

# True negative case (y=0)
bce_neg = -np.log(1 - p)
focal_neg_gamma2 = -p**2 * np.log(1 - p)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Positive Loss Curves
ax1.plot(p, bce_pos, label='Standard BCE', color='black', lw=2)
ax1.plot(p, focal_gamma1, label='Focal Loss (γ=1)', color='orange', lw=2)
ax1.plot(p, focal_gamma2, label='Focal Loss (γ=2)', color='crimson', lw=2)
ax1.set_title("Loss on Positive Cases (y=1)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicted Probability p")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot Negative Loss Curves (Down-weighting easy negatives)
ax2.plot(p, bce_neg, label='Standard BCE Negative', color='black', lw=2)
ax2.plot(p, focal_neg_gamma2, label='Focal Loss Negative (γ=2)', color='navy', lw=2)
ax2.set_title("Loss on Negative Cases (y=0)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Predicted Probability p")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
print("Loss curve dynamics under class imbalance demonstrated.")
"""))
save_notebook(c6, out_dir / "06_multilabel_class_imbalance_strategies.ipynb")

# -------------------------------------------------------------
# Notebook 7: Attention Map & Grad-CAM Visualizer
# -------------------------------------------------------------
c7 = []
c7.append(create_cell("markdown", """# Visual ML Foundations: Attention Map & Grad-CAM Visualizer
### Visualizing What Vision Transformers and CNNs Look at in Knee MRIs

---

## 1. Concept Overview
Black-box predictions are dangerous in medicine. When a model predicts a $92\\%$ probability of an ACL tear, a radiologist must know: *Did the model look at the anterior cruciate ligament, or did it cheat by looking at an image timestamp or bone drill artifact?*
**Grad-CAM (Gradient-weighted Class Activation Mapping)** uses the gradients flowing into the final feature maps to produce a heatmap highlighting the exact discriminative regions.
"""))

c7.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Simulate MRI knee slice
np.random.seed(42)
img = np.zeros((128, 128), dtype=np.float32)
y, x = np.ogrid[:128, :128]
img[((y - 45)**2 + (x - 64)**2) < 25**2] = 0.8
img[((y - 90)**2 + (x - 64)**2) < 28**2] = 0.7

# Simulate deep feature map (8x8 grid) with high activation at the ACL joint gap
feature_map = np.zeros((8, 8), dtype=np.float32)
feature_map[4, 4] = 1.0  # Center joint space activation
feature_map[4, 3] = 0.7
feature_map[3, 4] = 0.5

# Upsample feature activation to image dimensions
cam_heatmap = zoom(feature_map, 128 / 8.0, order=1)
cam_heatmap = (cam_heatmap - cam_heatmap.min()) / (cam_heatmap.max() - cam_heatmap.min() + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='bone')
axes[0].set_title("Input Knee MRI Slice", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cam_heatmap, cmap='jet')
axes[1].set_title("Grad-CAM Feature Activation", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(img, cmap='bone')
axes[2].imshow(cam_heatmap, cmap='jet', alpha=0.45)
axes[2].set_title("Diagnostic Visual Overlay", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
print("Grad-CAM radiological visualizer rendered successfully.")
"""))
save_notebook(c7, out_dir / "07_attention_map_visualizer.ipynb")

print("Generated notebooks 4 through 7.")
