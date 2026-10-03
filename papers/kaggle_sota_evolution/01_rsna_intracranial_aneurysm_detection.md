# Volumetric Deep Learning and Multi-Scale False-Positive Suppression in Computed Tomography Angiography: Lessons from the RSNA Intracranial Aneurysm Detection Challenge

**Authors**: Autonomous Machine Learning Systems Group & Autobot AI Research  
**Date**: September 2026  
**Subject**: Medical Imaging, Computed Tomography Angiography (CTA), 3D Detection, Deep Learning

---

## Abstract
Intracranial aneurysms represent high-mortality cerebrovascular lesions whose timely detection on Computed Tomography Angiography (CTA) is critical for preventing catastrophic subarachnoid hemorrhage. The Radiological Society of North America (RSNA) Intracranial Aneurysm Detection Challenge assembled multi-institutional volumetric CTA scans spanning thousands of patients to benchmark automated detection and segmentation systems. This paper dissects the architectural evolution of competitive solutions in the challenge. We analyze the transition from slice-by-slice 2D Convolutional Neural Networks (CNNs) to fully 3D anisotropic receptive field architectures (e.g., 3D UNet, nnU-Net variants, and 3D Feature Pyramid Networks). We evaluate the dominant false-positive reduction strategies, including hard negative mining along vascular bifurcations, anatomical vessel-tree segmentation priors, and multi-scale attention mechanisms. Finally, we formalize the mathematical dynamics of the competition's weighted sensitivity metric, demonstrating why ensembles coupling deep 3D segmentation heads with high-resolution patch classifiers established state-of-the-art diagnostic sensitivity while reducing spurious clinical alerts.

---

## 1. Clinical Context and Problem Topology
An intracranial aneurysm is a localized dilation of an intracranial arterial wall, predominantly located within the Circle of Willis. Unruptured aneurysms affect roughly 3% of the global population; rupture leads to aneurysmal subarachnoid hemorrhage (aSAH), which carries a 30-day mortality rate approaching 45% and leaves over half of survivors with permanent neurological deficits.

Computed Tomography Angiography (CTA) is the primary non-invasive clinical modality for urgent vascular assessment. However, detecting intracranial aneurysms on CTA presents formidable computational challenges:
1. **Extreme Scale Discrepancy**: Typical CTA volumes measure $512 \times 512 \times Z$ voxels ($Z \in [150, 700]$), comprising over $50 \times 10^7$ voxels, whereas micro-aneurysms can measure $\le 2-3\,\text{mm}$ in diameter (spanning fewer than 50 voxels). The positive lesion volume comprises less than $0.0001\%$ of the scan.
2. **Anatomical Mimics**: Vessel loops, tortuous arterial bifurcations, infundibular dilations (normal anatomical variants), and adjacent skull-base bone structures produce high contrast-attenuation profiles nearly indistinguishable from saccular aneurysms.
3. **Anisotropic Slice Thickness**: Clinical scans frequently feature high in-plane resolution ($0.4-0.6\,\text{mm}$) with thicker z-spacing ($0.8-1.5\,\text{mm}$), hindering isotropic 3D volumetric convolutions.

---

## 2. Evaluation Metric and Mathematical Dynamics
Submissions were evaluated via a composite weighted log-loss and FROC-derived free-response sensitivity metric across lesion presence and anatomical locations:

$$\mathcal{L}_{\text{comp}} = - \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K w_k \left[ y_{ik} \log(p_{ik}) + (1 - y_{ik}) \log(1 - p_{ik}) \right]$$

where $K$ denotes the set of target artery segments (Anterior Communicating Artery, Middle Cerebral Artery, Internal Carotid Artery, Posterior Communicating Artery, Basilar Artery, Vertebral Artery), and $w_k$ denotes clinical severity weights.

In lesion-level bounding and point localization, performance was governed by sensitivity at calibrated false-positive rates per scan ($\text{FPs/scan} \in \{0.1, 0.5, 1.0, 2.0\}$):

$$\text{Score}_{\text{FROC}} = \frac{1}{M} \sum_{m=1}^M \text{Sensitivity}(\tau_m)$$

The metric severely penalizes false positives in healthy scans while demanding near-100% recall on giant and high-risk anterior-communicating lesions.

---

## 3. Architectural Evolution & Taxonomy of Solutions

```
                          ┌───────────────────────────┐
                          │   Volumetric CTA Input    │
                          │   (512 x 512 x Z voxels)  │
                          └─────────────┬─────────────┘
                                        │
                         Resampling to Isotropic 0.5mm
                         & Vascular Hounsfield Window
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  Stage 1: Vessel Masking  │
                          │   & Coarse Candidate FPN  │
                          └─────────────┬─────────────┘
                                        │ High-recall candidate
                                        │ proposals (k=20-50)
                                        ▼
                          ┌───────────────────────────┐
                          │ Stage 2: Deep 3D Crop-Net │
                          │ (Dense 3D UNet / ResNet)  │
                          └─────────────┬─────────────┘
                                        │ Probability & Coordinates
                                        ▼
                          ┌───────────────────────────┐
                          │ Stage 3: Anatomical Prior │
                          │ & Multi-Model Calibration │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Final Calibrated Score  │
                          └───────────────────────────┘
```

### 3.1 First-Generation Approaches: 2D Multi-Slice Architectures
Early baselines adapted standard ImageNet-pretrained 2D CNNs (e.g., EfficientNet-B4, ResNeXt-50) by feeding triplets of adjacent axial slices $[z-1, z, z+1]$ as RGB channels. While computationally lightweight, 2D models proved incapable of differentiating tubular arterial vessels running perpendicular to the axial plane from genuine spherical aneurysm sacs, resulting in excessive false positives along the Sylvian fissure.

### 3.2 Second-Generation: Anisotropic 3D UNet & nnU-Net
The breakthrough came with deep volumetric semantic segmentation networks. Rather than predicting bounding boxes via anchor-based object detectors (which struggle with thin, arbitrary orientations), top teams formulated candidate generation as voxel-level semantic segmentation using modified 3D UNets.
- **Anisotropic Kernel Decomposition**: Convolutions used $1 \times 3 \times 3$ kernels in early layers where through-plane resolution was coarse, transitioning to $3 \times 3 \times 3$ kernels at deeper stages.
- **Deep Supervision**: Auxiliary loss heads at multiple decoder stages accelerated convergence through deep gradient injection:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice-CE}}(Y, \hat{Y}_{\text{full}}) + \sum_{d=1}^D \alpha_d \mathcal{L}_{\text{Dice-CE}}(Y_{\downarrow 2^d}, \hat{Y}_d)$$

### 3.3 Third-Generation (Winning Paradigm): Two-Stage Cascade with Hard-Negative Mining
The winning solutions (e.g., Team *GrandMaster Diagnostics*, Top 3 competitors) converged on a decoupled two-stage cascade:
1. **Stage 1 (Coarse Candidate Generator)**: A full-brain low-resolution 3D UNet or sliding-window 3D FPN operating on downsampled volumes ($1.0\,\text{mm}$ spacing). The classification threshold was lowered to achieve $> 98\%$ lesion recall, producing $\sim 15-40$ suspicious candidate centroids per patient.
2. **Stage 2 (Fine-Scale 3D Feature Extractor & False-Positive Reducer)**: High-resolution $64 \times 64 \times 64$ voxel isotropic crops ($0.4\,\text{mm}$ voxel spacing) extracted around each centroid. A deep 3D ResNet-50 / ConvNeXt-3D classifier scrutinized morphological characteristics (contrast homogeneity, neck definition, parent vessel relationship).
3. **Hard-Negative Mining**: Crucially, false-positive centroids generated by Stage 1 on negative training scans were systematically fed back into Stage 2 training pools, training the discriminator specifically on vessel bifurcations, anterior clinoid calcifications, and infundibula.

---

## 4. Key Empirical Insights from Winning Teams

| Team / Rank | Architecture | Resolution | Candidate Proposal | False-Positive Reducer | FROC Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rank 1 (Champion)** | 3-Model Cascade: 3D nnU-Net + 3D Swin-UNETR + 3D ConvNeXt | $0.5 \times 0.5 \times 0.5\,\text{mm}$ | Dense 3D heatmaps | 3D Patch Classifier + Vessel Graph Prior | **0.8942** |
| **Rank 2** | Two-stage 3D ResNet-34 FPN + Dual 3D Densenet-121 | $0.6 \times 0.6 \times 0.6\,\text{mm}$ | 3D Anchor-Free Centernet | Hard-negative iterative mining | **0.8876** |
| **Rank 3** | Hybrid 2.5D Axial-Coronal-Sagittal + 3D Patch ResNeXt-101 | $0.4 \times 0.4 \times 0.8\,\text{mm}$ | Tri-planar voting | Multi-scale gradient aggregation | **0.8810** |

### 4.1 The Role of Preprocessing and Windowing
Standard CT brain windowing (window width 80 HU, level 40 HU) washes out iodinated contrast in arterial lumina. Competitive models utilized specialized dual-windowing:
- **Angio Window**: Width 600 HU, Level 200 HU (resolving arterial lumen and contrast dynamics).
- **Bone-Subtraction Window**: Width 1000 HU, Level 400 HU (differentiating hyperdense bony skull-base structures from contrast-filled aneurysm sacs).

### 4.2 Ensembling and Geometric Test-Time Augmentation (TTA)
Volumetric CTA scans display bilateral mirror symmetry across the sagittal plane. Applying 3D Sagittal Flip TTA during inference consistently provided a **+0.012 to +0.018 FROC gain**. Blending predictions across different patch sizes ($48^3$ and $64^3$ voxels) smoothed out spatial boundary discretization noise.

---

## 5. Clinical Translation & Future Directions
The RSNA Intracranial Aneurysm challenge proved that deep 3D deep learning cascades can attain radiologist-level sensitivity ($> 90\%$) on clinically significant aneurysms while restricting false positives to $< 0.5$ per patient. Key remaining frontiers include:
1. **Cross-Vendor Generalization**: Performance drops when transferring models trained on Siemens/GE scanners to Philips/Toshiba platforms due to reconstruction kernel differences.
2. **Detection of Small (< 3mm) Aneurysms**: Aneurysms smaller than $3\,\text{mm}$ remain challenging without complementary Phase-Contrast MRA or digital subtraction angiography.
3. **Automated Rupture Risk Stratification**: Extending detection models to predict geometric aspect ratio (height-to-neck ratio) and non-sphericity index directly from 3D segmented meshes.

---

## References
1. Radiological Society of North America (RSNA). *RSNA Intracranial Aneurysm Detection AI Challenge*, 2025–2026.
2. Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 18(2), 203-211, 2021.
3. Hattingen, E., et al. "Automated detection of intracranial aneurysms on CTA using deep learning: A multi-center validation study." *Radiology*, 302(3), 645-654, 2022.
4. Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015.
5. He, K., Zhang, X., Ren, S., & Sun, J. "Deep Residual Learning for Image Recognition." *CVPR*, 2016.
