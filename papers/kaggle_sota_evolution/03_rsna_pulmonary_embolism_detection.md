# Multi-Task Temporal Sequence Convolutions for Volumetric CT Pulmonary Angiography: Insights from the RSNA Pulmonary Embolism Detection Challenge

**Authors**: Autonomous Machine Learning Systems Group & Autobot AI Research  
**Date**: September 2026  
**Subject**: Medical Imaging, CTPA, Sequence Modeling, Multi-Task Learning, Deep Learning

---

## Abstract
Acute pulmonary embolism (PE) is the third most common cause of cardiovascular death worldwide. Rapid and accurate detection on Computed Tomography Pulmonary Angiography (CTPA) is essential for acute therapeutic interventions. The RSNA Pulmonary Embolism Detection Challenge remains one of the largest medical machine learning benchmarks ever mounted, encompassing over 12,000 volumetric CTPA examinations and 1.8 million annotated slices. This paper presents an architectural and clinical review of the state-of-the-art methodologies developed during the benchmark. We analyze the canonical two-stage decomposition: slice-level 2D feature representations followed by volumetric sequence aggregation using Bidirectional LSTMs, GRUs, and 1D Convolutions. We examine how competitive pipelines enforced biophysical consistency constraints across exam-level targets (e.g., right ventricular strain, chronic vs. acute PE, central vs. peripheral localization) and slice-level thrombus indicators. Finally, we analyze the impact of contrast-phase windowing, loss re-weighting, and post-processing calibration on real-world diagnostic triage.

---

## 1. Clinical Challenge and Dataset Complexity
Pulmonary embolism occurs when a deep venous thrombus detaches, travels through the right cardiac chambers, and occludes the pulmonary arterial tree. Diagnostic delays lead to acute right ventricular failure and hemodynamic collapse.

CTPA imaging visualizes intraluminal filling defects surrounded by iodinated contrast. The challenge dataset presented unprecedented scale and structural complexity:
- **12,000+ CTPA scans**: Spanning multiple institutional archives, scanner manufacturers, and acquisition protocols.
- **1.8 Million Slices**: Each study contained between 150 and 900 axial slices.
- **Hierarchical Multi-Task Structure**:
  - **Slice-Level**: Binary indicator for filling defect presence on each individual slice.
  - **Exam-Level (9 Targets)**:
    - Overall presence of PE.
    - Right Ventricular (RV) to Left Ventricular (LV) diameter ratio $\ge 1.0$ (RV/LV ratio $\ge 1.0$, direct marker of acute right heart strain).
    - RV/LV ratio $< 1.0$.
    - Chronic PE.
    - Acute and Chronic PE.
    - Central PE (main pulmonary trunk occlusion).
    - Left main pulmonary artery PE.
    - Right main pulmonary artery PE.
    - Segmental or Subsegmental PE.

```
                          CTPA Axial Series
                         (150 - 900 Slices)
                                 │
                                 ▼
                     [3-Channel Dual Windowing]
                     - PE Window (W:700, L:100)
                     - Lung Window (W:1500, L:-600)
                     - Mediastinal Window (W:400, L:40)
                                 │
                                 ▼
               [Stage 1: 2D Feature Extractor (CNN)]
               Extract Feature Vector per Slice (e.g. 512-dim)
                                 │
                                 ▼
               [Stage 2: 1D Temporal Sequence Model]
               Bidirectional GRU / 1D Dilated ConvNeXt
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
      [Slice-Level Classifier]       [Exam-Level Classifier]
        PE presence per slice           - PE Present / Absent
        (Sequence of Length T)          - RV/LV Ratio >= 1.0
                                        - Central / Lobar / Subseg
```

---

## 2. Mathematical Metric Formulation & Consistency Constraints
The competition metric was an exam-weighted multi-label log-loss over slices and exam-level categories:

$$\mathcal{L}_{\text{comp}} = \frac{1}{\sum_i w_i} \left[ \sum_{s=1}^{S} w_{\text{slice}} \cdot \mathcal{L}_{\text{BCE}}(y_s, p_s) + \sum_{k=1}^K w_k \cdot \mathcal{L}_{\text{BCE}}(y_k, p_k) \right]$$

### The Crucial Consistency Rules
Kaggle's scoring engine enforced rigid logical constraints between slice predictions and exam predictions:
1. **Existential Consistency**: If any slice is positive for PE ($p_s > 0.5$), the exam-level PE prediction $p_{\text{PE}}$ must be high:
   $$p_{\text{PE}} \approx \max_{s=1}^S (p_s)$$
2. **Mutual Exclusivity**:
   - $p_{\text{RV/LV} \ge 1.0}$ and $p_{\text{RV/LV} < 1.0}$ are mutually exclusive and should sum to $p_{\text{PE}}$.
   - If $p_{\text{PE}} = 0$, all subtype probabilities must collapse to zero.

Submissions that violated these joint distribution properties incurred substantial log-loss penalties.

---

## 3. Methodological Taxonomy of Winning Solutions

### 3.1 Domain-Specific Windowing
Raw CT Hounsfield Units (HU) span $[-1024, +3071]$. Slicing into 8-bit image representations requires deliberate windowing. Standard single-window transforms either overexposed the iodinated blood pool or obliterated peripheral pulmonary parenchyma. Winning pipelines mapped each slice into a 3-channel RGB image using three distinct clinical windows:
- **Red Channel (PE Window)**: Center $+100\,\text{HU}$, Width $700\,\text{HU}$ (targets pulmonary artery lumen).
- **Green Channel (Lung Window)**: Center $-600\,\text{HU}$, Width $1500\,\text{HU}$ (targets lung parenchyma to contextualize peripheral wedges and infarcts).
- **Blue Channel (Mediastinum Window)**: Center $+40\,\text{HU}$, Width $400\,\text{HU}$ (targets cardiac chambers, myocardium, and mediastinal lymphadenopathy).

### 3.2 Two-Stage Sequential Decoupling
Directly processing an entire $512 \times 512 \times 400$ volume through a 3D CNN exceeded the GPU memory limits of contemporary accelerators (16GB V100s). The universal architectural consensus was a decoupled two-stage paradigm:
1. **Stage 1 (Feature Extractor)**: A 2D CNN (e.g., EfficientNet-B0 to B4, ResNeXt-50) trained with binary cross-entropy on slice-level labels. Intermediate feature maps before the classification head were pooled into a 512- or 1024-dimensional feature vector $\mathbf{v}_t$ for each slice $t \in [1, T]$.
2. **Stage 2 (Sequence Aggregator)**: The sequence of feature vectors $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_T] \in \mathbb{R}^{T \times D}$ was passed through a sequence model (Bidirectional GRU, 1D WaveNet-style dilated convolutions, or a 1D Transformer encoder).

### 3.3 Biophysical Post-Processing Optimization
To satisfy competition consistency constraints and minimize log-loss, top teams optimized post-processing transforms directly on out-of-fold validation sets:
- **Logit Scaling & Calibration**: Applying isotonic regression or temperature scaling to match positive class base rates.
- **Extreme Value Smoothing**: Clipping extreme probabilities $[\epsilon, 1-\epsilon]$ to prevent catastrophic log-loss outliers from noisy edge slices.
- **Top-$K$ Max-Pooling**: Replacing simple average pooling with learnable attention-weighted top-$K$ pooling:
  $$p_{\text{PE, exam}} = \frac{1}{K} \sum_{j \in \text{Top-}K} p_j$$

---

## 4. Benchmark Results and Leaderboard Progression

| Rank / Team | Stage 1 Backbone | Stage 2 Sequence Model | Consistency Strategy | Log-Loss |
| :--- | :--- | :--- | :--- | :--- |
| **Rank 1** | EfficientNet-B3 + B4 + SEResNeXt50 | 2-Layer BiGRU + 1D Multi-Scale CNN | End-to-end consistency loss penalty | **0.1504** |
| **Rank 2** | ResNet-50d + EfficientNet-B2 | Multi-Head Self-Attention + BiLSTM | Mathematical projection optimization | **0.1512** |
| **Rank 3** | Dual EfficientNet-B0/B3 | 1D Dilated ConvNet (WaveNet style) | Joint multi-task head training | **0.1520** |

---

## 5. Clinical Impact and Architectural Legacy
The RSNA Pulmonary Embolism Challenge demonstrated that 2D-to-1D decoupled sequence modeling provides an extraordinarily efficient and effective surrogate for full 3D volumetric convolutions in long-axis medical CT series. The resulting architectures have since been adapted across whole-body oncology staging, aortic dissection detection, and cervical spine fracture triage.

---

## References
1. Radiological Society of North America (RSNA). *RSNA Pulmonary Embolism Detection Challenge*, 2020.
2. Konstantinides, S. V., et al. "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism." *European Heart Journal*, 41(4), 543-603, 2020.
3. Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*, 2019.
4. Vaswani, A., et al. "Attention Is All You Need." *NeurIPS*, 2017.
