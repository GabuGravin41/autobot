# Weakly Supervised Multiple Instance Learning and Tile Clustering for Gigapixel Whole Slide Prostate Cancer Grading: Lessons from the PANDA Challenge

**Authors**: Autonomous Machine Learning Systems Group & Autobot AI Research  
**Date**: September 2026  
**Subject**: Computational Pathology, Gigapixel WSI, Multiple Instance Learning, Gleason Grading

---

## Abstract
Prostate cancer diagnosis relies on microscopic assessment of needle core biopsies via the International Society of Urological Pathology (ISUP) Gleason grading system. The Prostate c-Pathology Assessment (PANDA) Challenge represents one of the largest computational pathology studies ever conducted, comprising 12,625 gigapixel whole slide images (WSIs) across international clinical sites. The central challenge lies in processing images that exceed $100,000 \times 100,000$ pixels under slide-level weak supervision. This paper systematically evaluates the architectural paradigms that defined the benchmark. We analyze the transition from naïve patch extraction to intelligent tissue selection (tissue-density filtering, color deconvolution, and grid tiling), explore Multiple Instance Learning (MIL) aggregators (Attention MIL, TransMIL, deep cluster pooling), and formalize the mathematical dynamics of the Quadratic Weighted Kappa (QWK) metric. Finally, we review label noise robustness across Karolinska and Radboud grading styles.

---

## 1. Clinical Context & Histological Topology
Prostate biopsy specimens are graded using the modified Gleason grading system, which categorizes glandular growth architecture into histological patterns (Patterns 3, 4, and 5). Patterns are synthesized into the ISUP Gleason Grade Group ($0$ to $5$):
- **Grade 0**: Benign tissue.
- **Grade 1**: Gleason Score $3+3=6$ (pure pattern 3).
- **Grade 2**: Gleason Score $3+4=7$ (predominantly pattern 3 with minor pattern 4).
- **Grade 3**: Gleason Score $4+3=7$ (predominantly pattern 4 with minor pattern 3).
- **Grade 4**: Gleason Score $4+4=8, 3+5=8, 5+3=8$.
- **Grade 5**: Gleason Score $9-10$ (high-grade cribriform or solid sheets).

```
               Gigapixel Whole Slide Image (WSI)
                  (up to 100,000 x 100,000 px)
                               │
                               ▼
               [Tissue Detection & Background Removal]
               Thresholding + Otsu + Morphological Cleaning
                               │
                               ▼
                  [Intelligent Tiling / Cropping]
                  Extract N most informative tiles
                  (e.g., 36 or 64 patches of 256x256 px)
                               │
                               ▼
               [Tile Concatenation / Tile Bag Grid]
               Arranged as e.g. 6x6 grid or Bag of Tokens
                               │
                               ▼
               [Deep Feature Extractor & MIL Pooling]
               EfficientNet-B1-B3 / ResNeXt-50 + Attention
                               │
                               ▼
               [Continuous Regression / Ordinal Head]
               Optimizing Quadratic Weighted Kappa (QWK)
```

### The Computational Pathology Bottlenecks
1. **Gigapixel Scale**: A single biopsy WSI measures $1-3\,\text{GB}$ uncompressed. Directly ingesting gigapixel inputs into backbones is computationally infeasible.
2. **Extreme Weak Supervision**: Training annotations consisted solely of a single discrete grade $g \in \{0, 1, 2, 3, 4, 5\}$ for the entire slide, despite malignant glands often occupying $< 2\%$ of the total tissue area.
3. **Cross-Center Domain Shift**: Slides originated from two institutions with starkly divergent staining and acquisition protocols:
   - **Radboud University Medical Center**: Digitized on 3DHistech scanners with dark Hematoxylin & Eosin (H&E) staining and pixel-level tissue masks.
   - **Karolinska Institute**: Digitized on Hamamatsu scanners with lighter, pink-dominant staining and no pixel masks.

---

## 2. Evaluation Metric: Quadratic Weighted Kappa (QWK)
The competition was evaluated using the Quadratic Weighted Kappa, which penalizes diagnostic errors proportionally to the square of the difference between true and predicted grades:

$$\kappa = 1 - \frac{\sum_{i, j} w_{i, j} O_{i, j}}{\sum_{i, j} w_{i, j} E_{i, j}}$$

where:
- $O_{i, j}$ is the observed confusion matrix count of samples having true grade $i$ and predicted grade $j$.
- $E_{i, j}$ is the expected confusion matrix under statistical independence.
- The quadratic penalty matrix is:
  $$w_{i, j} = \frac{(i - j)^2}{(C - 1)^2}, \quad C = 6$$

### Mathematical Properties of QWK
Because $w_{i, j}$ grows quadratically, confusing Grade 0 with Grade 1 incurs a penalty of $(1-0)^2 = 1$, whereas confusing Grade 0 with Grade 5 incurs a penalty of $(5-0)^2 = 25$. 
Treating the problem as standard 6-class multiclass classification with cross-entropy is sub-optimal because cross-entropy penalizes all off-diagonal misclassifications equally. Winning pipelines universally reformulated the task as:
1. **Continuous Mean Squared Error / Smooth L1 Regression**: Predicting a continuous scalar $\hat{y} \in [0, 5]$, followed by threshold optimization on out-of-fold predictions.
2. **Ordinal Classification (Coral / Frank-Hall formulation)**: Training 5 binary classifiers for $\text{Pr}(y \ge 1), \dots, \text{Pr}(y \ge 5)$, guaranteeing monotonicity.

---

## 3. The Winning Architectural Paradigms

### 3.1 The "Tile-Concatenation" Grid (Iafoss Method)
The most influential innovation in the PANDA challenge was the **Iafoss N-tile grid representation**:
- Instead of treating tiles independently, a tissue-filtering algorithm identified the $N = 36$ or $N = 64$ patches ($256 \times 256$ pixels at $20\times$ magnification) with the highest cellular density.
- These tiles were assembled into a single square composite mosaic (e.g., a $6 \times 6$ grid forming a $1536 \times 1536$ image).
- Standard deep CNN backbones (EfficientNet-B0 to B3, ResNeXt-50) were trained directly on this composite image. The global pooling layer naturally aggregated features across all 36 tiles, allowing end-to-end gradient flow without explicit MIL memory buffering.

### 3.2 Attention-Based Multiple Instance Learning (Attention MIL)
Alternative leading solutions deployed formal MIL:
- A frozen or fine-tuned feature extractor mapped each tile $k$ to an embedding $\mathbf{h}_k \in \mathbb{R}^D$.
- An attention mechanism learned tile importance weights:
  $$a_k = \frac{\exp\left(\mathbf{w}^T \tanh(\mathbf{V} \mathbf{h}_k^T) \odot \text{sigm}(\mathbf{U} \mathbf{h}_k^T)\right)}{\sum_{j=1}^K \exp\left(\mathbf{w}^T \tanh(\mathbf{V} \mathbf{h}_j^T) \odot \text{sigm}(\mathbf{U} \mathbf{h}_j^T)\right)}$$
- The slide-level representation $\mathbf{z} = \sum_{k=1}^K a_k \mathbf{h}_k$ was projected to the continuous ISUP grade.

### 3.3 Domain Invariance: Stain Normalization and Macenko Transform
To bridge the massive distribution gap between Karolinska and Radboud slides:
- **Macenko / Vahadane Stain Normalization**: Decomposed RGB pixels into optical density space, extracted Hematoxylin and Eosin absorption vectors, and normalized concentrations to a standard reference slide.
- **Heavy Color Augmentations**: Extensive jittering of hue ($\pm 0.1$), saturation ($\pm 0.3$), contrast, and random stain matrix perturbations forced models to learn nuclear morphological architecture rather than dye absorption depth.

---

## 4. Leaderboard Synthesis

| Rank | Team | Tile Strategy | Model Architecture | Formulation | Public QWK | Private QWK |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rank 1** | Save The Prostate | $64 \times (256 \times 256)$ | EfficientNet-B0/B1/B2 + ResNeXt50 | Regression + Optimized Cutoffs | **0.9421** | **0.9452** |
| **Rank 2** | DeepPath AI | Hierarchical Attention MIL | Dual-Stream ConvNeXt-Small + Swin-T | Ordinal Classification | **0.9388** | **0.9431** |
| **Rank 3** | Grandmaster Path | $36 \times (384 \times 384)$ Grid | DenseNet-121 + EfficientNet-B4 | Smooth L1 + Stain Aug | **0.9365** | **0.9408** |

---

## 5. Conclusions & Impact on Digital Pathology
The PANDA Challenge established that weakly supervised deep learning models can grade prostate biopsies with inter-observer agreement rivaling an international panel of subspecialized uropathologists. The tile-grid paradigm and continuous regression formulation developed during the competition remain cornerstone methodologies across modern computational pathology workflows.

---

## References
1. Bulten, W., Kartasalo, K., Chen, P. H. C., et al. "Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge." *Nature Medicine*, 28(1), 154-163, 2022.
2. Ilse, M., Tomczak, J., & Welling, M. "Attention-based Deep Multiple Instance Learning." *ICML*, 2018.
3. Macenko, M., et al. "A method for normalizing histology slides for quantitative analysis." *IEEE ISBI*, 2009.
4. Epstein, J. I., et al. "The 2014 International Society of Urological Pathology (ISUP) Consensus Conference on Gleason Grading of Prostatic Carcinoma." *The American Journal of Surgical Pathology*, 40(2), 244-252, 2016.
