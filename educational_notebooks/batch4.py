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
# Notebook 12: Feature Importance and SHAP Values
# -------------------------------------------------------------
c12 = []
c12.append(create_cell("markdown", """# Visual ML Foundations: Feature Importance & SHAP Values
### Interpreting Model Decisions in Tabular and Spectroscopy Pipelines

---

## 1. Concept Overview
How do you explain *why* a gradient boosted tree or neural network flagged a patient for surgery or identified a chemical compound?
**SHAP (SHapley Additive exPlanations)** borrows from cooperative game theory:
* It computes the marginal contribution of each feature across all possible subsets of features.
* Positive SHAP values push the model toward a positive prediction; negative values push toward negative.
"""))

c12.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

# Simulate 8 clinical / spectral features and their calculated mean absolute SHAP values
features = [
    'Precursor m/z Exact Mass', 'Retention Time (min)', 'Effusion Volume (mL)',
    'Cartilage Thickness (mm)', 'Patient Age', 'Joint Space Width',
    'ACL Signal Intensity', 'Bone Marrow Edema Area'
]
shap_importance = np.array([0.42, 0.35, 0.28, 0.24, 0.19, 0.15, 0.11, 0.08])

# Sort ascending for horizontal bar chart
idx = np.argsort(shap_importance)
sorted_feats = [features[i] for i in idx]
sorted_vals = shap_importance[idx]

plt.figure(figsize=(10, 5.5))
bars = plt.barh(sorted_feats, sorted_vals, color='steelblue', edgecolor='black', height=0.65)
plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight='bold')
plt.xlabel("Average Impact on Model Output Magnitude", fontsize=11)
plt.grid(axis='x', alpha=0.3)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va='center', fontweight='bold')

plt.tight_layout()
plt.show()
print("Feature importance visualization complete.")
"""))
save_notebook(c12, out_dir / "12_feature_importance_and_shap_analysis.ipynb")

# -------------------------------------------------------------
# Notebook 13: Image Patching & Vision Transformer Basics
# -------------------------------------------------------------
c13 = []
c13.append(create_cell("markdown", """# Visual ML Foundations: Vision Transformers & Patch Tokenization
### How DINOv2 and ViT Break 2D Medical Slices into Token Sequences

---

## 1. Concept Overview
Vision Transformers (ViT, DINOv2) do not use convolutional sliding kernels. Instead:
1. An input image $X \\in \\mathbb{R}^{H \\times W}$ is divided into non-overlapping grid patches of size $P \\times P$ (e.g. $16 \\times 16$).
2. Each patch is flattened and linearly projected into an embedding vector of dimension $D$.
3. A learnable **`[CLS]` token** and **positional embeddings** are added.
4. Self-attention layers compute all-to-all relationships across all patches.
"""))

c13.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

# Create sample MRI image (64x64)
y, x = np.ogrid[:64, :64]
img = np.zeros((64, 64))
img[((y - 25)**2 + (x - 32)**2) < 18**2] = 0.8
img[((y - 45)**2 + (x - 32)**2) < 16**2] = 0.6

# Decompose into 16x16 patches (4x4 = 16 total patches)
patch_size = 16
patches = []
for r in range(0, 64, patch_size):
    for c in range(0, 64, patch_size):
        patches.append(img[r:r+patch_size, c:c+patch_size])

fig = plt.figure(figsize=(12, 6))

# Show original image with patch grid lines
ax_orig = plt.subplot(1, 2, 1)
ax_orig.imshow(img, cmap='bone')
for p in range(0, 64, patch_size):
    ax_orig.axhline(p - 0.5, color='yellow', lw=1.5)
    ax_orig.axvline(p - 0.5, color='yellow', lw=1.5)
ax_orig.set_title("Input Image with 16x16 Grid", fontsize=12, fontweight='bold')
ax_orig.axis('off')

# Display tokenized patch sequence (4x4 layout)
for i in range(16):
    ax_p = fig.add_subplot(4, 8, (i // 4) * 8 + (i % 4) + 5)
    ax_p.imshow(patches[i], cmap='bone')
    ax_p.axis('off')
    ax_p.set_title(f"T{i+1}", fontsize=9)

plt.suptitle("Vision Transformer (ViT) Patch Tokenization Sequence", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print(f"Divided image into {len(patches)} sequential patch tokens.")
"""))
save_notebook(c13, out_dir / "13_image_patching_and_vision_transformer_basics.ipynb")

# -------------------------------------------------------------
# Notebook 14: Clinical NegEx Scope Parser
# -------------------------------------------------------------
c14 = []
c14.append(create_cell("markdown", """# Visual ML Foundations: Clinical Report NegEx & Scope Parsing
### Handling Medical Negations in Multilingual Radiology Reports

---

## 1. Concept Overview
When processing clinical text (Spanish, Dutch, English):
* *"No se observa rotura del tendón rotuliano"* $\\rightarrow$ **Negative (0)**.
* *"Rotura completa de ligamento cruzado"* $\\rightarrow$ **Positive (1)**.
A naive keyword search for *"rotura"* produces disastrous false positives.
**NegEx** uses rule-based windowing to flag whether a target pathology term falls inside the syntactic scope of a negation phrase.
"""))

c14.append(create_cell("code", """import re

NEGATION_TRIGGERS = [
    r'sin\\s+(?:signos\\s+de\\s+|evidencia\\s+de\\s+)?',
    r'no\\s+(?:se\\s+observa|hay|presenta|aprecia)\\s+',
    r'conservad[oa]s?',
    r'dentro\\s+de\\s+l[ií]mites\\s+normales',
    r'intact[oa]s?'
]
NEG_REGEX = re.compile('|'.join(NEGATION_TRIGGERS), re.IGNORECASE)

SAMPLE_SENTENCES = [
    "No se observa rotura del ligamento cruzado anterior.",
    "Rotura completa del cuerno posterior del menisco interno.",
    "Ligamentos colaterales conservados sin alteraciones.",
    "Derrame articular moderado con sinovitis reactiva."
]

print("=== Clinical NegEx Scope Parsing Demo ===\\n")
for sent in SAMPLE_SENTENCES:
    is_negated = bool(NEG_REGEX.search(sent))
    has_rotura = "rotura" in sent.lower()
    has_derrame = "derrame" in sent.lower()
    
    status = "NEGATIVE / RULED OUT" if is_negated else "POSITIVE FINDING"
    print(f"Sentence: '{sent}'")
    print(f"  Negation detected: {is_negated}")
    print(f"  Clinical Status:   {status}\\n")
"""))
save_notebook(c14, out_dir / "14_text_report_clinical_negex_parser.ipynb")

# -------------------------------------------------------------
# Notebook 15: Test-Time Augmentation (TTA) Visualizer
# -------------------------------------------------------------
c15 = []
c15.append(create_cell("markdown", """# Visual ML Foundations: Test-Time Augmentation (TTA) Mechanics
### Squeezing $0.005$ to $0.015$ Metric Gains on the Leaderboard

---

## 1. Concept Overview
During inference on test images, single predictions are vulnerable to slight pixel noise.
**Test-Time Augmentation (TTA)**:
1. Feeds the original image plus subtle geometric variants (Horizontal Flip, Slight Scale).
2. Computes model predictions on each variant.
3. Inverts the transformation (if spatial) and averages the predicted probability distributions:
   $$\\bar{p} = \\frac{1}{K} \\sum_{k=1}^K p(T_k(X))$$
This dramatically stabilizes model predictions on difficult borderline cases!
"""))

c15.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

# Simulate knee image
np.random.seed(42)
img = np.zeros((100, 100))
y, x = np.ogrid[:100, :100]
img[((y - 40)**2 + (x - 50)**2) < 22**2] = 0.8
img[((y - 75)**2 + (x - 50)**2) < 20**2] = 0.7

# Create 4 TTA variants
orig = img
flip_h = np.fliplr(img)
shift_pos = np.roll(img, shift=4, axis=1)
contrast_boost = np.clip(img * 1.25, 0, 1)

tta_variants = [orig, flip_h, shift_pos, contrast_boost]
titles = ["Original", "Horizontal Flip", "Horizontal Shift (+4px)", "Contrast Shift (1.25x)"]
simulated_preds = [0.912, 0.884, 0.905, 0.931]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, im, t, p in zip(axes, tta_variants, titles, simulated_preds):
    ax.imshow(im, cmap='bone')
    ax.set_title(f"{t}\\nPred: {p:.3f}", fontsize=11, fontweight='bold')
    ax.axis('off')

mean_p = np.mean(simulated_preds)
plt.suptitle(f"Test-Time Augmentation (TTA) Ensemble | Calibrated Prediction: {mean_p:.4f}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print(f"TTA Ensembled Probability: {mean_p:.4f} (Variance reduced across transforms)")
"""))
save_notebook(c15, out_dir / "15_test_time_augmentation_tta_visualizer.ipynb")

print("Generated notebooks 12 through 15.")
