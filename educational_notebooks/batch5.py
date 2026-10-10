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
# Notebook 16: Hierarchical Clustering of Chemical Spectra
# -------------------------------------------------------------
c16 = []
c16.append(create_cell("markdown", """# Visual ML Foundations: Hierarchical Clustering of Spectral Profiles
### Dendrograms, Molecular Cosine Distances, and Chemical Family Trees

---

## 1. Concept Overview
In untargeted metabolomics (like **Enveda CASMI**), we often discover unknown molecules that have never been seen before.
By computing the **pairwise cosine distance** across mass spectra, we can perform **Hierarchical Agglomerative Clustering (HAC)**:
* Molecules with matching fragmentation patterns cluster together into chemical families (flavonoids, lipids, alkaloids).
* A **Dendrogram** visualizes the taxonomy of structural relationships.
"""))

c16.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Simulate spectral cosine similarity profiles for 8 compounds
np.random.seed(42)
compound_names = [
    'Flavonoid A', 'Flavonoid B', 'Alkaloid X', 'Alkaloid Y',
    'Lipid Core 1', 'Lipid Core 2', 'Peptide Frag 1', 'Peptide Frag 2'
]

# Generate synthetic 10-dimensional spectral feature vectors
features = np.array([
    [0.9, 0.8, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.1], # Flavonoid A
    [0.85, 0.82, 0.15, 0.0, 0.05, 0.0, 0.0, 0.1, 0.18, 0.12], # Flavonoid B
    [0.0, 0.1, 0.9, 0.85, 0.1, 0.2, 0.0, 0.0, 0.0, 0.1], # Alkaloid X
    [0.05, 0.08, 0.88, 0.91, 0.05, 0.15, 0.0, 0.0, 0.0, 0.08], # Alkaloid Y
    [0.1, 0.0, 0.1, 0.0, 0.95, 0.89, 0.1, 0.0, 0.1, 0.0], # Lipid 1
    [0.12, 0.05, 0.08, 0.0, 0.91, 0.94, 0.08, 0.0, 0.08, 0.0], # Lipid 2
    [0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.92, 0.88, 0.15, 0.2], # Peptide 1
    [0.05, 0.0, 0.0, 0.08, 0.0, 0.08, 0.89, 0.93, 0.12, 0.18] # Peptide 2
])

# Hierarchical linkage (Ward distance)
Z = linkage(features, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z, labels=compound_names, leaf_rotation=45, leaf_font_size=10)
plt.title("Hierarchical Chemical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight='bold')
plt.ylabel("Cluster Distance")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("Dendrogram clusters correctly group chemical families.")
"""))
save_notebook(c16, out_dir / "16_hierarchical_clustering_chemical_spectra.ipynb")

# -------------------------------------------------------------
# Notebook 17: Contrastive Learning (SimCLR) Concepts
# -------------------------------------------------------------
c17 = []
c17.append(create_cell("markdown", """# Visual ML Foundations: Self-Supervised Contrastive Learning (SimCLR)
### Training Encoders on Millions of Unlabeled Medical Images Without Labels

---

## 1. Concept Overview
Labeling medical images is prohibitively expensive ($50+ per scan for radiologist annotation).
**Self-Supervised Contrastive Learning (SimCLR / MoCo / DINO)** trains models on unannotated images:
1. Take an unlabeled MRI slice $x$.
2. Apply two different random stochastic augmentations to generate views $x_i$ and $x_j$ (positive pair).
3. The model minimizes **NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)**:
   $$\\mathcal{L}_{i,j} = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j) / \\tau)}{\\sum_{k} \\exp(\\text{sim}(z_i, z_k) / \\tau)}$$
4. Pulls representations of the same image together, while pushing all other images in the batch apart!
"""))

c17.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter

# Original unannotated knee MRI
np.random.seed(42)
y, x = np.ogrid[:100, :100]
x_orig = np.zeros((100, 100))
x_orig[((y - 40)**2 + (x - 50)**2) < 22**2] = 0.8
x_orig[((y - 75)**2 + (x - 50)**2) < 20**2] = 0.7

# Generate Positive Pair (Augmented Views of SAME image)
view_i = rotate(x_orig, angle=6, reshape=False) + np.random.normal(0, 0.05, (100, 100))
view_j = gaussian_filter(x_orig, sigma=1.0) * 1.2

# Generate Negative Sample (DIFFERENT image)
x_neg = np.zeros((100, 100))
x_neg[((y - 50)**2 + (x - 50)**2) < 35**2] = 0.65

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(x_orig, cmap='bone')
axes[0].set_title("Source Unlabeled Image x", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(view_i, cmap='bone')
axes[1].set_title("Positive View x_i (PULL)", fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(view_j, cmap='bone')
axes[2].set_title("Positive View x_j (PULL)", fontsize=11, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(x_neg, cmap='bone')
axes[3].set_title("Negative Sample x_neg (PUSH)", fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.suptitle("Self-Supervised Contrastive Representation Learning Dynamics", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("Contrastive positive/negative pair dynamics visualized.")
"""))
save_notebook(c17, out_dir / "17_contrastive_learning_simclr_concepts.ipynb")

# -------------------------------------------------------------
# Notebook 18: Metric Learning & t-SNE / UMAP Feature Projections
# -------------------------------------------------------------
c18 = []
c18.append(create_cell("markdown", """# Visual ML Foundations: Latent Space Visualization with t-SNE
### Inspecting High-Dimensional Embeddings in 2D Latent Space

---

## 1. Concept Overview
High-capacity models (ViT, ResNet) project complex inputs into 768-dimensional or 1024-dimensional feature embeddings.
To inspect if our model has learned meaningful clinical features:
* We project high-dimensional representations into 2D using **t-SNE (t-Distributed Stochastic Neighbor Embedding)**.
* Well-trained models form distinct, well-separated visual clusters for each abnormality!
"""))

c18.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Simulate 300 high-dimensional feature vectors (dim=64) from 3 diagnostic classes
np.random.seed(42)
n_samples = 100
dim = 64

# Cluster 1: Normal Knees
c1 = np.random.normal(0.0, 1.0, size=(n_samples, dim))
# Cluster 2: ACL Tears
c2 = np.random.normal(4.0, 1.0, size=(n_samples, dim))
# Cluster 3: Severe Osteoarthritis
c3 = np.random.normal(-4.0, 1.2, size=(n_samples, dim))

X = np.vstack([c1, c2, c3])
y = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

# Run 2D t-SNE projection
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(9, 6))
labels = ['Normal Anatomy', 'ACL Tear', 'Severe OA']
colors = ['teal', 'crimson', 'darkorange']

for cl in range(3):
    idx = (y == cl)
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=labels[cl], color=colors[cl], alpha=0.8, edgecolors='black', s=60)

plt.title("2D t-SNE Latent Space Embedding Projection", fontsize=13, fontweight='bold')
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
print("t-SNE diagnostic latent clustering visualized.")
"""))
save_notebook(c18, out_dir / "18_metric_learning_and_embeddings_tsne.ipynb")

# -------------------------------------------------------------
# Notebook 19: Learning Rate Schedulers in Practice
# -------------------------------------------------------------
c19 = []
c19.append(create_cell("markdown", """# Visual ML Foundations: Learning Rate Schedulers in Deep Learning
### Cosine Annealing vs. OneCycleLR vs. StepLR Dynamics

---

## 1. Concept Overview
The choice of learning rate scheduler often makes the difference between a model getting stuck in a local minimum vs. achieving state-of-the-art generalization:
* **StepLR**: Drops LR by $\\gamma$ at fixed epoch intervals (coarse, can shock gradients).
* **Cosine Annealing**: Smoothly decreases LR along a cosine curve toward a minimum $\\eta_{\\text{min}}$.
* **OneCycleLR**: Warms up rapidly in early iterations, then smoothly anneals down to near zero (super-convergence).
"""))

c19.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

steps = np.arange(100)
base_lr = 1e-3

# 1. StepLR (decay by 0.5 every 25 steps)
step_lr = base_lr * (0.5 ** (steps // 25))

# 2. Cosine Annealing
cosine_lr = 1e-5 + 0.5 * (base_lr - 1e-5) * (1 + np.cos(np.pi * steps / 100))

# 3. OneCycleLR (Warmup 30%, Cosine Anneal 70%)
warmup_steps = 30
onecycle_lr = np.zeros(100)
onecycle_lr[:warmup_steps] = np.linspace(1e-4, 3e-3, warmup_steps)
onecycle_lr[warmup_steps:] = 1e-5 + 0.5 * (3e-3 - 1e-5) * (1 + np.cos(np.pi * (steps[warmup_steps:] - warmup_steps) / (100 - warmup_steps)))

plt.figure(figsize=(11, 5))
plt.plot(steps, step_lr, label='StepLR (gamma=0.5, step=25)', color='gray', linestyle='--', lw=2)
plt.plot(steps, cosine_lr, label='Cosine Annealing', color='navy', lw=2.5)
plt.plot(steps, onecycle_lr, label='OneCycleLR (Super-Convergence)', color='crimson', lw=2.5)

plt.title("Learning Rate Trajectory Comparison Across 100 Optimization Steps", fontsize=13, fontweight='bold')
plt.xlabel("Training Step / Epoch")
plt.ylabel("Learning Rate (Log Scale)")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
print("LR dynamics visualizer complete.")
"""))
save_notebook(c19, out_dir / "19_learning_rate_schedulers_in_practice.ipynb")

# -------------------------------------------------------------
# Notebook 20: Ensembling & Rank Averaging Mastery
# -------------------------------------------------------------
c20 = []
c20.append(create_cell("markdown", """# Visual ML Foundations: Model Ensembling & Rank Averaging
### How Competition Grandmasters Blend Diverse Architectures (ViT + CNN)

---

## 1. Concept Overview
No single model wins a competitive ML competition.
* **Why Ensembles Work**: Individual models make uncorrelated errors. Averaging their predictions cancels out individual model noise.
* **Probability Mean vs. Rank Averaging**:
  * **Simple Mean**: $\\bar{p} = \\frac{1}{M} \\sum_{m=1}^M p_m$. Susceptible if one model is poorly calibrated (overconfident).
  * **Rank Averaging**: Converts probabilities into uniform percentile ranks $[0, 1]$ before averaging. Highly robust against calibration shifts and metric outliers!
"""))

c20.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Simulate 5 test predictions from 3 different architectures
np.random.seed(42)
y_true = np.array([0, 1, 0, 1, 1])

# Model 1: DINOv2 (well calibrated)
m1 = np.array([0.15, 0.85, 0.25, 0.70, 0.90])
# Model 2: CoAtNet (slightly underconfident)
m2 = np.array([0.30, 0.65, 0.35, 0.58, 0.72])
# Model 3: EfficientNet (overconfident)
m3 = np.array([0.05, 0.98, 0.12, 0.94, 0.99])

# Simple Arithmetic Mean
simple_mean = (m1 + m2 + m3) / 3.0

# Percentile Rank Averaging
r1 = rankdata(m1) / len(m1)
r2 = rankdata(m2) / len(m2)
r3 = rankdata(m3) / len(m3)
rank_avg = (r1 + r2 + r3) / 3.0

x = np.arange(len(y_true))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - 2*width, m1, width, label='DINOv2', color='steelblue')
ax.bar(x - width, m2, width, label='CoAtNet', color='darkorange')
ax.bar(x, m3, width, label='EfficientNet', color='forestgreen')
ax.bar(x + width, simple_mean, width, label='Simple Mean Ensemble', color='purple')
ax.bar(x + 2*width, rank_avg, width, label='Rank Average Ensemble', color='crimson')

ax.set_xticks(x)
ax.set_xticklabels([f"Sample {i+1} (True={y_true[i]})" for i in range(len(y_true))], fontweight='bold')
ax.set_title("Model Ensembling: Individual Backbones vs. Blended Predictions", fontsize=13, fontweight='bold')
ax.set_ylabel("Predicted Score / Rank")
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
print("Ensemble comparison rendered successfully.")
"""))
save_notebook(c20, out_dir / "20_model_ensembling_and_rank_averaging.ipynb")

print("Generated notebooks 16 through 20. Full 20 notebook series complete!")
