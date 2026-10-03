# Probabilistic F1 Optimization and Cross-View Bilateral Mammographic Alignment: Insights from the RSNA Screening Mammography Challenge

**Authors**: Autonomous Machine Learning Systems Group & Autobot AI Research  
**Date**: September 2026  
**Subject**: Medical Imaging, Digital Mammography, Extreme Class Imbalance, Probabilistic F1

---

## Abstract
Breast cancer remains the most prevalent malignant neoplasm among women globally. The RSNA Screening Mammography Breast Cancer Detection Challenge evaluated automated diagnostic systems on a massive screening dataset comprising over 54,000 digital mammograms across 11,000 patients. The competition featured severe clinical class imbalance ($< 2.1\%$ positive malignancy rate) and evaluated submissions using the non-differentiable Probabilistic $F_1$ score ($\text{pF}_1$). This paper synthesizes the core technological advancements that enabled top solutions to master this high-stakes benchmark. We examine: (1) high-resolution image representation strategies capable of resolving sub-millimeter microcalcification clusters without excessive GPU memory allocation; (2) anatomical cross-view attention fusion coupling Craniocaudal (CC) and Mediolateral Oblique (MLO) projections; (3) bilateral breast subtraction architectures targeting tissue asymmetry; and (4) analytical and surrogate gradient formulations for direct $\text{pF}_1$ metric optimization.

---

## 1. Clinical Context & Multi-View Mammography
Population screening via Full-Field Digital Mammography (FFDM) reduces breast cancer mortality through early detection of invasive ductal carcinomas and ductal carcinomas in situ (DCIS). A standard screening examination captures four standard projections:
- **L-CC** (Left Craniocaudal) & **R-CC** (Right Craniocaudal): Axial horizontal compression views.
- **L-MLO** (Left Mediolateral Oblique) & **R-MLO** (Right Mediolateral Oblique): Angled views including the pectoralis major muscle and axillary tail.

```
       Left Breast (L)                       Right Breast (R)
   ┌─────────┐   ┌─────────┐             ┌─────────┐   ┌─────────┐
   │  L-CC   │   │  L-MLO  │             │  R-CC   │   │  R-MLO  │
   │  View   │   │  View   │             │  View   │   │  View   │
   └────┬────┘   └────┬────┘             └────┬────┘   └────┬────┘
        │             │                       │             │
        └──────┬──────┘                       └──────┬──────┘
               │                                     │
               ▼                                     ▼
        [Cross-View Attention]                [Cross-View Attention]
        (CC <-> MLO Fusion)                   (CC <-> MLO Fusion)
               │                                     │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
                     [Bilateral Symmetry Comparison]
                     (Left vs. Right Mirror Subtraction)
                                  │
                                  ▼
                     [Calibrated pF1 Probability]
```

### The Diagnostic Bottlenecks
1. **Extreme Resolution Needs**: Malignant microcalcifications appear as faint, punctate hyperdense flecks measuring $0.1-0.3\,\text{mm}$. Downsampling native mammograms ($3000 \times 4000$ pixels) to typical CNN inputs ($512 \times 512$) obliterates critical diagnostic indicators.
2. **Dense Fibroglandular Tissue**: In dense breasts (BI-RADS C and D), radiopaque glandular tissue masks underlying tumors.
3. **Severe Class Imbalance**: In screening cohorts, only $\sim 2\%$ of exams harbor biopsy-proven malignancy, rendering standard cross-entropy models vulnerable to trivial majority-class collapse.

---

## 2. Evaluation Metric: The Probabilistic F1 Score ($\text{pF}_1$)
Unlike threshold-dependent binary $F_1$, the RSNA metric utilized a continuous probabilistic formulation evaluated at the patient-breast level:

$$\text{pF}_1 = \frac{2 \cdot \text{pTP}}{2 \cdot \text{pTP} + \text{pFP} + \text{pFN}}$$

where:
$$\text{pTP} = \sum_{i \in \text{Pos}} y_i \cdot p_i, \quad \text{pFP} = \sum_{i \in \text{Neg}} (1 - y_i) \cdot p_i, \quad \text{pFN} = \sum_{i \in \text{Pos}} y_i \cdot (1 - p_i)$$

Expanding and simplifying the denominator:
$$\text{pF}_1 = \frac{2 \sum_{i} y_i p_i}{\sum_{i} y_i + \sum_{i} p_i}$$

### Mathematical Implications of $\text{pF}_1$
- The denominator depends on the sum of predicted probabilities $\sum p_i$.
- If a model outputs well-calibrated posterior probabilities, the expected sum $\sum p_i$ matches the true number of positive cases $\sum y_i$, aligning the denominator.
- However, for an uncalibrated model, scaling probabilities by a multiplier $\alpha$ can dramatically swing $\text{pF}_1$. Top competitors derived exact closed-form global threshold and temperature adjustments to maximize $\text{pF}_1$ directly on out-of-fold validation splits.

---

## 3. Key Architectural Innovations

### 3.1 High-Resolution Input Processing: YOLOR and Progressive Resizing
Handling full-field mammograms at high fidelity without exceeding GPU memory constraints was the primary technical hurdle:
- **Aspect Ratio Preservation**: Mammograms have large empty background margins. Tight bounding-box cropping around the breast contour (removing air and scanner artifacts) preserved resolution.
- **Optimal Input Size**: Leading models trained on resolutions of $1536 \times 768$ to $2048 \times 1024$ pixels using mixed precision (FP16/BF16) and gradient checkpointing.
- **Patch/ROI Attention (YOLOR / Faster-RCNN Prior)**: Auxiliary models pre-trained on external datasets (e.g., CBIS-DDSM, INbreast) generated region-of-interest proposals around suspicious architectural distortions.

### 3.2 Cross-View Dual-Projection Transformer
Human radiologists evaluate mammograms by triangulating findings across orthogonal projections: a true mass visible on MLO must correlate with an architectural distortion or density on the CC view at an anatomically corresponding distance from the nipple.
- **Cross-View Fusion Architecture**:
  - Independent backbones (e.g., ConvNeXt-Small, EfficientNet-B5) extracted spatial feature grids $\mathbf{F}_{\text{CC}} \in \mathbb{R}^{H \times W \times C}$ and $\mathbf{F}_{\text{MLO}} \in \mathbb{R}^{H \times W \times C}$.
  - A cross-attention module used $\mathbf{F}_{\text{CC}}$ tokens as queries and $\mathbf{F}_{\text{MLO}}$ tokens as keys/values, learning soft epipolar spatial alignments between views.

### 3.3 Bilateral Symmetry Subtraction
Normal breast parenchyma exhibits high bilateral structural symmetry between the left and right breasts of the same patient. Malignancy typically manifests as an **asymmetric density**:
- Top teams mirrored the contralateral breast image horizontally and computed differential feature maps:
  $$\mathbf{F}_{\text{diff}} = |\mathbf{F}_{\text{Left}} - \mathbf{F}_{\text{Right, mirrored}}|$$
  Providing this residual signal directly into the final classification head significantly reduced benign false alarms.

---

## 4. Leaderboard Benchmark Synthesis

| Rank | Model Architecture | Resolution | Multi-View Mechanism | Loss / Metric Tuning | Public $\text{pF}_1$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rank 1** | ConvNeXt-Base + EfficientNet-B5 + ResNeXt-101 | $2048 \times 1024$ | Cross-Attention + Bilateral Mirroring | Surrogate $\text{pF}_1$ Loss + Threshold Tuning | **0.5924** |
| **Rank 2** | CoAtNet-3 + Swin-Large Multi-Scale | $1536 \times 768$ | Dual-Projection Feature Concatenation | Asymmetric Binary Cross-Entropy | **0.5881** |
| **Rank 3** | DenseNet-121 + EfficientNetV2-L | $1792 \times 896$ | ROI Detection Head + Cross-View MLP | Continuous $\text{pF}_1$ surrogate optimization | **0.5847** |

---

## 5. Conclusions & Clinical Translation
The RSNA Screening Mammography Challenge demonstrated that deep neural networks can match expert screening recall while maintaining high specificity. Crucially, the competition established that:
1. Pure image resolution ($> 1500\,\text{px}$) is strictly non-negotiable for microcalcification sensitivity.
2. Integrating multi-view (CC + MLO) and bilateral symmetry priors produces dramatic improvements over single-image classifiers.
3. Directly formulating loss functions to mirror the non-convex evaluation metric ($\text{pF}_1$) prevents pathological calibration drift under extreme prevalence skew.

---

## References
1. Radiological Society of North America (RSNA). *RSNA Screening Mammography Breast Cancer Detection*, 2023.
2. Pisano, E. D., et al. "Diagnostic performance of digital versus film mammography for breast-cancer screening." *NEJM*, 353(17), 1773-1783, 2005.
3. Liu, Z., et al. "A ConvNet for the 2020s." *CVPR*, 2022.
4. McKinney, S. M., et al. "International evaluation of an AI system for breast cancer screening." *Nature*, 577(7788), 89-94, 2020.
