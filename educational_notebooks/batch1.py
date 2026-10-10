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
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Notebook 1: 3-Plane MRI Visualizer
# -------------------------------------------------------------
c1 = []
c1.append(create_cell("markdown", """# Visual ML Foundations: 3-Plane MRI Volume Reconstruction
### Understanding Sagittal, Coronal, and Axial Orthogonal Perspectives in Knee Imaging

---

## 1. Concept Overview
Magnetic Resonance Imaging (MRI) produces three-dimensional volumetric scans of human anatomy. To analyze a joint, radiologic scanners acquire slices across three orthogonal planes:
* **Sagittal Plane**: Slices from lateral to medial (side view of the knee). Essential for viewing the Anterior Cruciate Ligament (**ACL**), Posterior Cruciate Ligament (**PCL**), and meniscal horns.
* **Coronal Plane**: Slices from anterior to posterior (front-facing view). Best for viewing the Medial Collateral Ligament (**MCL**), Lateral Collateral Ligament (**LCL**), and tibial plateau.
* **Axial Plane**: Slices from superior to inferior (cross-sectional top-down view). Essential for the patellofemoral joint, cartilage thickness, and joint effusions.

In this notebook, we generate an interactive, visual representation of 3-plane MRI volumes with side-by-side rendering.
"""))

c1.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_knee_volume(size=(64, 128, 128)):
    \"\"\"Simulate a 3D MRI volume containing femur, tibia, and joint space.\"\"\"
    depth, height, width = size
    volume = np.zeros(size, dtype=np.float32)
    
    # Coordinates grid
    z, y, x = np.ogrid[:depth, :height, :width]
    
    # Femur bone condyle (upper sphere)
    femur = ((y - 45)**2 + (x - 64)**2 + (z - 32)**2 * 2) < 28**2
    # Tibial plateau (lower cylinder/ellipsoid)
    tibia = ((y - 95)**2 + (x - 64)**2 + (z - 32)**2 * 2) < 32**2
    # Joint capsule / effusion space
    capsule = ((y - 68)**2 / 10**2 + (x - 64)**2 / 40**2 + (z - 32)**2 / 20**2) < 1.0
    
    volume[femur] = 0.85
    volume[tibia] = 0.80
    volume[capsule] = 0.45
    
    # Add tissue texture and scanner noise
    noise = np.random.normal(0, 0.05, size)
    volume = np.clip(volume + noise, 0, 1)
    return volume

volume = generate_synthetic_knee_volume()
print(f"Volume shape (Depth x Height x Width): {volume.shape}")
"""))

c1.append(create_cell("code", """# Extract cross-sectional center slices across the 3 planes
axial_slice = volume[volume.shape[0] // 2, :, :]       # Slice along Z axis (Top-down)
sagittal_slice = volume[:, :, volume.shape[2] // 2]    # Slice along X axis (Side view)
coronal_slice = volume[:, volume.shape[1] // 2, :]     # Slice along Y axis (Front view)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(sagittal_slice, cmap='bone', aspect='auto')
axes[0].set_title("Sagittal View (Side: ACL & Meniscus)", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(coronal_slice, cmap='bone', aspect='auto')
axes[1].set_title("Coronal View (Front: MCL & Collaterals)", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(axial_slice, cmap='bone')
axes[2].set_title("Axial View (Cross-section: Patella & Effusion)", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
print("3-Plane orthogonal views rendered successfully.")
"""))
save_notebook(c1, out_dir / "01_medical_mri_3plane_visualizer.ipynb")

# -------------------------------------------------------------
# Notebook 2: Mass Spectrometry Peak Processing
# -------------------------------------------------------------
c2 = []
c2.append(create_cell("markdown", """# Visual ML Foundations: Mass Spectrometry Peak Processing
### Centroiding, Noise Thresholding, and m/z Spectral Alignment

---

## 1. Concept Overview
In natural product and metabolomic discovery (like the **Enveda CASMI** challenge), molecules are identified by ionizing them and measuring the mass-to-charge ratio ($m/z$) of their fragment ions.
Raw mass spectra contain thousands of raw detector noise spikes alongside true chemical adducts. Preprocessing requires:
1. **Intensity Thresholding**: Eliminating electronic baseline noise.
2. **Peak Centroiding**: Converting continuous mass peaks into discrete $(m/z, \text{intensity})$ tuples.
3. **Square Root / Log Transformation**: Compressing intense precursor peaks so subtle fragment ions remain informative.
"""))

c2.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt

# Generate simulated mass spectrum with precursor and fragment peaks
np.random.seed(42)
mz_axis = np.linspace(50, 600, 2000)
raw_spectrum = np.random.exponential(scale=2.0, size=len(mz_axis))

# Known chemical ion fragments (m/z, intensity)
true_peaks = [(105.04, 85.0), (149.02, 60.0), (279.16, 95.0), (391.28, 100.0), (413.26, 45.0)]
for mz, intensity in true_peaks:
    idx = np.argmin(np.abs(mz_axis - mz))
    raw_spectrum[idx-3:idx+4] += intensity * np.exp(-0.5 * (np.linspace(-2, 2, 7))**2)

# Apply noise thresholding & square root transformation
threshold = 10.0
clean_spectrum = np.where(raw_spectrum > threshold, raw_spectrum, 0.0)
normalized_spectrum = np.sqrt(clean_spectrum)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax1.plot(mz_axis, raw_spectrum, color='crimson', lw=1)
ax1.axhline(threshold, color='black', linestyle='--', label=f'Noise Gate ({threshold})')
ax1.set_title("Raw Mass Spectrum with Baseline Noise", fontsize=12, fontweight='bold')
ax1.set_ylabel("Intensity")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.vlines(mz_axis[clean_spectrum > 0], 0, normalized_spectrum[clean_spectrum > 0], color='navy', lw=1.5)
ax2.set_title("Centroided & Sqrt-Normalized Feature Spectrum", fontsize=12, fontweight='bold')
ax2.set_xlabel("Mass-to-charge ratio (m/z)")
ax2.set_ylabel("Normalized Intensity")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print(f"Extracted {np.sum(clean_spectrum > 0)} significant spectral feature peaks.")
"""))
save_notebook(c2, out_dir / "02_mass_spec_peak_processing.ipynb")

# -------------------------------------------------------------
# Notebook 3: ARC Grid Transformation Visualizer
# -------------------------------------------------------------
c3 = []
c3.append(create_cell("markdown", """# Visual ML Foundations: ARC Grid Transformation & Symmetry Engine
### Visualizing Few-Shot AGI Grid Transformations and Spatial Operations

---

## 1. Concept Overview
The **ARC Prize (Abstraction and Reasoning Corpus)** tests few-shot fluid visual reasoning. Each puzzle consists of small $2D$ grids (values $0-9$ mapped to distinct colors).
Core spatial transformations include:
* **Dihedral Group Operations ($D_4$)**: Rotations ($90^\circ, 180^\circ, 270^\circ$) and reflections (horizontal, vertical, diagonal).
* **Color Permutations**: Color inversion and palette substitution.
* **Connected Component Extraction**: Isolating objects and their bounding boxes.
"""))

c3.append(create_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Standard ARC 10-Color Palette
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
arc_cmap = ListedColormap(ARC_COLORS)

# Create a sample ARC puzzle grid
sample_grid = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 2, 0],
    [0, 1, 0, 0, 2, 0],
    [0, 0, 0, 3, 3, 0],
    [0, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

# Generate spatial transformations
rot_90 = np.rot90(sample_grid, 1)
flip_h = np.fliplr(sample_grid)
flip_v = np.flipud(sample_grid)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ["Original Grid", "Rotated 90°", "Horizontal Flip", "Vertical Flip"]
grids = [sample_grid, rot_90, flip_h, flip_v]

for ax, g, t in zip(axes, grids, titles):
    ax.imshow(g, cmap=arc_cmap, vmin=0, vmax=9)
    ax.set_title(t, fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(-0.5, g.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, g.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

plt.tight_layout()
plt.show()
print("ARC transformation visual suite generated successfully.")
"""))
save_notebook(c3, out_dir / "03_arc_grid_transformations_visualized.ipynb")

print("Generated initial batch of foundational notebooks.")
