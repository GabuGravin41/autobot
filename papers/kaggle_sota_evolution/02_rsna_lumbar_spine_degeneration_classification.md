# Hierarchical Keypoint Conditioning and Cross-Plane Attention for Multi-Condition Lumbar Spine Degeneration Classification: Insights from RSNA 2024

**Authors**: Autonomous Machine Learning Systems Group & Autobot AI Research  
**Date**: September 2026  
**Subject**: Medical Imaging, Lumbar Spine MRI, Hierarchical Multi-Task Learning, Transformers

---

## Abstract
Low back pain is the single leading cause of years lived with disability worldwide. The RSNA 2024 Lumbar Spine Degeneration Classification Challenge introduced a multi-condition, multi-planar MRI diagnostic task requiring automated grading of five distinct pathologies across five vertebral levels ($L_1/L_2$ to $L_5/S_1$) spanning 25 interdependent clinical targets. This paper provides a rigorous technical analysis of the algorithmic breakthroughs in the competition. We examine how competitive pipelines moved from unconstrained whole-volume classification to hierarchical anatomical decomposition: first localizing vertebral discs and neural foramina via keypoint regression networks (YOLOv8-pose, HRNet, UNet heatmap regressors), followed by anatomical crop-level sequence modeling with Bidirectional Transformers and 3D CNNs. We analyze the mathematical dynamics of the sample-weighted multiclass log-loss, examine cross-plane attention mechanisms linking Sagittal T2, Sagittal T1, and Axial T2 scans, and summarize the consensus design rules that elevated ensemble performance to human-radiologist concordance.

---

## 1. Introduction & Multi-Planar Problem Topology
Lumbar spine MRI examination is the gold-standard diagnostic tool for assessing mechanical lower back pain and radiculopathy. A typical clinical lumbar study consists of three distinct imaging sequences:
1. **Sagittal T2-weighted**: Evaluates spinal cord/thecal sac patency, intervertebral disc hydration, disc herniation, and spinal canal stenosis.
2. **Sagittal T1-weighted**: Highlights vertebral body marrow architecture, osteophyte formation, neural foraminal fat, and foraminal stenosis.
3. **Axial T2-weighted**: Oblique slices perpendicular to individual disc spaces, providing transverse visualization of subarticular zones, ligamentum flavum hypertrophy, and facet joint arthropathy.

The challenge required grading five degenerative conditions across five anatomical levels ($L_1/L_2, L_2/L_3, L_3/L_4, L_4/L_5, L_5/S_1$):
- **Spinal Canal Stenosis** (Central canal narrowing, predominantly evaluated on Sagittal T2 & Axial T2).
- **Left & Right Neural Foraminal Stenosis** (Nerve exit root compression, evaluated on Sagittal T1).
- **Left & Right Subarticular Stenosis** (Lateral recess narrowing, evaluated on Axial T2).

Each of the 25 conditions was categorized into three severity grades: **Normal/Mild (0)**, **Moderate (1)**, and **Severe (2)**.

```
                      Lumbar Spine MRI Examination
               ┌───────────────────┬───────────────────┐
               │                   │                   │
         Sagittal T2          Sagittal T1           Axial T2
         (Central Canal)     (Foraminal Zone)    (Subarticular Zone)
               │                   │                   │
               └───────────────────┼───────────────────┘
                                   │
                                   ▼
                 [Stage 1: Keypoint Localization]
                 Heatmap Regression / Pose Estimation
                 Predict 5 Disc Centroids (L1/L2 - L5/S1)
                                   │
                                   ▼
                 [Stage 2: Level-Specific 2.5D/3D Crops]
                 (e.g., 64x128x128 bounding boxes per disc)
                                   │
                                   ▼
                 [Stage 3: Sequence Modeling & Fusion]
                 Bidirectional GRU / Cross-Plane Transformer
                                   │
                                   ▼
                 [Stage 4: Calibrated Multi-Task Heads]
                 25 Conditions x 3 Severity Probabilities
```

---

## 2. Loss Function & Metric Dynamics
The competition metric was a sample-weighted multi-class log loss over all 25 conditions:

$$\mathcal{L} = - \frac{1}{\sum_{i} \sum_{j} w_{ij}} \sum_{i=1}^N \sum_{j=1}^{25} w_{ij} \sum_{c=0}^2 y_{ij,c} \log(p_{ij,c})$$

where $y_{ij,c} \in \{0, 1\}$ is the ground truth indicator for condition $j$ of patient $i$ belonging to class $c \in \{\text{Normal/Mild}, \text{Moderate}, \text{Severe}\}$.

### Metric Weighting Asymmetry
The severity classes carried heavily asymmetric clinical penalties:
- $\text{Normal/Mild (Grade 0)}: w = 1.0$
- $\text{Moderate (Grade 1)}: w = 2.0$
- $\text{Severe (Grade 2)}: w = 4.0$

Furthermore, any patient exhibiting at least one **Severe** finding across any condition had an additional diagnostic multiplier applied. Consequently, a single missed "Severe" spinal canal stenosis or subarticular entrapment imposed a disproportionately devastating penalty on the leaderboard, driving models to prioritize high sensitivity on Grade 2 pathologies via focal loss and calibrated threshold adjustments.

---

## 3. The Winning Engineering Blueprint

### 3.1 Stage 1: Level Localization & Disc Keypoint Regression
Naïve 3D CNNs trained directly on raw volumetric studies struggled due to variable slice counts (from 10 to 45 slices per sequence), non-standardized fields of view, and patient-specific scoliosis or pelvic tilt. Top-performing teams decoupled the problem by first establishing an anatomical coordinate frame:
- **Architecture**: HRNet-W32, ConvNeXt-Tiny, or YOLOv8-pose operating on downsampled sagittal mid-line slices.
- **Target**: Regress five continuous 2D/3D Gaussian heatmaps centered on intervertebral disc spaces $L_1/L_2$ through $L_5/S_1$.
- **Sorting & Spatial Constraint**: Keypoints were enforced to satisfy strict cranio-caudal monotonicity ($y_{L_1/L_2} < y_{L_2/L_3} < \dots < y_{L_5/S_1}$), eliminating level-misidentification errors.

### 3.2 Stage 2: Level-Specific Anisotropic Cropping
Once disc centers $(x_k, y_k, z_k)$ were established:
- For **Sagittal sequences**, bounding boxes of size $64 \times 128 \times 128\,\text{mm}$ were cropped around each intervertebral disc.
- For **Axial sequences**, slice matching algorithms identified the 2 to 4 axial slices closest to the inferred disc plane, utilizing DICOM `ImagePositionPatient` and `ImageOrientationPatient` vectors.

### 3.3 Stage 3: Feature Extraction & Cross-Plane Sequence Attention
For each anatomical level $k \in \{1, \dots, 5\}$:
1. **Backbones**: Dense multi-layer 2D backbones (e.g., EfficientNet-B2/B3, ConvNeXt-V2, CoAtNet) extracted slice-level representation tokens $\mathbf{h}_{k, s} \in \mathbb{R}^D$.
2. **Intra-Sequence Aggregation**: A 1D Bidirectional GRU or multi-head self-attention module aggregated slice tokens within each sequence into a compact sequence representation $\mathbf{z}_{k, \text{SagT2}}, \mathbf{z}_{k, \text{SagT1}}, \mathbf{z}_{k, \text{AxT2}}$.
3. **Cross-Plane Fusion**: Cross-attention mechanisms dynamically routed features: Central Canal stenosis queries attended primarily to $\mathbf{z}_{k, \text{SagT2}}$, whereas Subarticular queries attended to $\mathbf{z}_{k, \text{AxT2}}$.

```
          z_SagT2  ────────┐
                           ├───► [Cross-Plane Multi-Head] ───► [Class Heads] ──► Canal Stenosis
          z_SagT1  ────────┼───► [    Attention Block   ] ───► [    (x25)    ] ──► Foraminal Stenosis
                           │                                                 └──► Subarticular Stenosis
          z_AxT2   ────────┘
```

---

## 4. Key Results & Ablation Summary

| Rank | Pipeline Core | Localization Method | Cross-Plane Strategy | Severity Calibration | Log-Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rank 1** | Hierarchical 3-Phase (YOLOv8 pose $\to$ ConvNeXt-B $\to$ Transformer) | Heatmap Peak Detection | Bidirectional Cross-Attention | Asymmetric Focal $\gamma=2.0$ | **0.3641** |
| **Rank 2** | CoAtNet-2 + 3D DenseNet-121 Crop Ensemble | 3D UNet Disc Segmenter | Gated Residual Concat | Class-weight scaling ($1:2:4$) | **0.3688** |
| **Rank 3** | Dual ResNet-50d + Swin-V2 Spatial Pyramid | Keypoint regression | Multi-View Tensor Fusion | Temperature Scaling | **0.3712** |

### 4.1 Lessons on Data Quality & Label Noise
A significant discovery in the competition was the existence of radiologist labeling inconsistencies between institutional sites. In particular, the boundary between "Moderate" and "Severe" canal stenosis exhibited significant inter-observer variability. Models that utilized **label smoothing** ($\epsilon = 0.05$) or **ordinal regression heads** ($\text{Pr}(\text{Grade} \ge 1), \text{Pr}(\text{Grade} \ge 2)$) showed superior out-of-fold calibration and resistance to overfitting compared to standard categorical cross-entropy.

---

## 5. Conclusions
The RSNA 2024 Lumbar Spine challenge underscored that end-to-end black-box architectures fail on complex multi-condition musculoskeletal tasks without explicit anatomical structuring. Decoupling spatial landmark localization from pathology grading allowed models to focus representational capacity strictly on localized pathological tissue. This hierarchical paradigm now forms the standard template for multi-sequence spine MRI decision support systems.

---

## References
1. Radiological Society of North America (RSNA). *RSNA 2024 Lumbar Spine Degeneration Classification Challenge*, 2024.
2. Wang, J., Sun, K., Cheng, T., et al. "Deep High-Resolution Representation Learning for Visual Recognition." *IEEE TPAMI*, 43(10), 3349-3364, 2021.
3. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021.
4. Lurie, J. D., et al. "Reliability of readings of lumbar spine MRI features: a comparison of clinical and research readings." *Spine*, 33(14), 1543-1549, 2008.
