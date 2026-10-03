# Extreme Classification and Geometric Verification in Web-Scale Visual Instance Retrieval: Lessons from the Google Landmark Challenges

**Authors**: Autonomous Machine Learning Systems Group & Autobot AI Research  
**Date**: September 2026  
**Subject**: Computer Vision, Instance Retrieval, Metric Learning, Geometric Verification

---

## Abstract
Visual landmark recognition and retrieval evaluate machine learning models on extreme multi-class classification and instance-level matching across millions of images and hundreds of thousands of classes. The Google Landmark Recognition and Retrieval (GLD) challenges assembled the largest worldwide dataset of natural and human-made architectural landmarks, spanning over 5 million images across 200,000 distinct landmark entities. This paper investigates the algorithmic paradigms that achieved state-of-the-art retrieval performance at web scale. We analyze deep metric learning loss functions (ArcFace, CosFace, SubCenter ArcFace), spatial pooling operations (Generalized Mean Pooling - GeM), and the two-stage retrieval cascade: coarse global descriptor indexing followed by local feature extraction (DELG, SuperPoint, LoFTR) and RANSAC-based geometric spatial verification. Finally, we review query expansion techniques (alpha-query expansion, diffusion on k-nearest neighbor graphs) and database re-ranking strategies.

---

## 1. Problem Formulation & Scale Dimensions
Instance retrieval seeks to identify the exact physical landmark entity depicted in a query image, even when photographed under extreme variations in viewpoint, illumination, occlusion, seasonal foliage, and optical focal length.

The Google Landmark Dataset presented extreme computer vision challenges:
- **Number of Classes ($C$)**: Over $200,000$ distinct global landmark categories.
- **Dataset Size ($N$)**: Exceeding $5,000,000$ images.
- **Extreme Long-Tail Distribution**: A tiny fraction of famous global landmarks (e.g., Eiffel Tower, Colosseum) have thousands of images, while tens of thousands of obscure historical monuments have only $2-5$ photos.
- **Dual Evaluation Tracks**:
  1. **Recognition Track**: Classify the query image into one of 200,000 classes or identify it as non-landmark. Evaluated via Global Average Precision (GAP).
  2. **Retrieval Track**: Return a ranked list of index images depicting the exact physical entity. Evaluated via Mean Average Precision at $k=100$ ($\text{mAP}@100$).

```
                      Query Image (q)
                             │
                             ▼
              [Global Feature Extractor (Backbone + GeM)]
              ConvNeXt-Large / Swin-B / ResNeXt-101
                             │
                             ▼
              [L2-Normalized Global Descriptor (512-d)]
                             │
                             ▼
          [Stage 1: Approximate Nearest Neighbor Search]
          Cosine Similarity against Index Database (1M+ images)
          Retrieve Top-K Candidates (e.g., K = 100)
                             │
                             ▼
          [Stage 2: Local Feature Extraction & Matching]
          Extract Keypoints & Descriptors (DELG / LoFTR)
                             │
                             ▼
          [Stage 3: Spatial Geometric Verification (RANSAC)]
          Count Epipolar Inlier Matches -> Re-rank Top-K
                             │
                             ▼
          [Stage 4: Alpha-Query Expansion (alpha-QE)]
          Weighted Feature Aggregation with Top Inlier Neighbors
                             │
                             ▼
                     Final Ranked List
```

---

## 2. Metric Formulations: GAP and mAP@100
The **Global Average Precision (GAP)** metric evaluates recognition predictions ranked across the entire test set:

$$\text{GAP} = \sum_{i=1}^N P(i) \cdot \Delta R(i)$$

where $P(i)$ is precision calculated over all predictions down to rank $i$, and $\Delta R(i)$ is the recall delta when a true positive landmark is encountered. GAP demands both high classification accuracy and near-perfect confidence calibration: assigning a high confidence to a false positive severely penalizes all subsequent correct predictions.

---

## 3. The Modern Deep Metric Learning Paradigm

### 3.1 Angular Margin Losses: ArcFace & SubCenter ArcFace
Standard softmax loss is insufficient for extreme classification with 200k classes because it does not enforce intra-class compactness or inter-class margin separation on the hypersphere.
- **Additive Angular Margin (ArcFace)**:
  $$\mathcal{L}_{\text{ArcFace}} = - \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \ne y_i} e^{s \cdot \cos(\theta_j)}}$$
  where $s$ is the hypersphere radius scale and $m$ is the angular margin penalty ($m \approx 0.3-0.5$).
- **SubCenter ArcFace**: In web-scraped landmark datasets, significant intra-class variance exists (e.g., interior photos vs. exterior aerial photos of Notre-Dame). SubCenter ArcFace assigns $K$ distinct sub-centers per class (e.g., $K=3$). During training, the sample is penalized only relative to its closest sub-center, preventing gradients from destroying multi-modal cluster topology.

### 3.2 Generalized Mean (GeM) Pooling
Traditional spatial pooling operations (average pooling or max pooling) either dilute localized architectural cues or over-emphasize point specular highlights.
- **Generalized Mean Pooling (GeM)**:
  $$\mathbf{f} = \left( \frac{1}{|\Omega|} \sum_{x \in \Omega} x^p \right)^{1/p}$$
  where $p$ is a learnable parameter ($p \to 1$ yields average pooling; $p \to \infty$ yields max pooling). Training with learnable $p \approx 3.0$ focuses network attention on prominent architectural silhouettes while maintaining broad spatial context.

### 3.3 Two-Stage Retrieval with DELG and RANSAC
The ultimate frontier in visual instance retrieval is spatial geometric verification:
1. **DEep Local and Global features (DELG)**: A unified CNN/ViT backbone that simultaneously predicts a 512-dimensional global descriptor and a set of localized patch feature vectors with auto-learned spatial attention masks.
2. **First Stage**: Fast cosine similarity search over global descriptors retrieves the top $k = 100$ candidate images.
3. **Second Stage**: Pairwise local feature matching between query keypoints and retrieved candidate keypoints using mutual nearest neighbors.
4. **Geometric Verification**: A RANSAC estimator fits an affine or homography transformation matrix $\mathbf{H}$ between matching keypoints. Candidates are re-ranked based on the number of verified geometric inliers:
   $$\text{Score}_{\text{final}} = \text{Score}_{\text{global}} + \beta \cdot \log(1 + N_{\text{inliers}})$$

### 3.4 Alpha-Query Expansion ($\alpha\text{-QE}$)
Once initial retrieval ranks are established, the query vector $\mathbf{q}$ is updated by aggregating top-ranked neighbor descriptors:
$$\mathbf{q}_{\text{new}} = \mathbf{q} + \sum_{i=1}^m \left( \mathbf{q}^T \mathbf{d}_i \right)^\alpha \mathbf{d}_i$$
Re-querying the database with the expanded vector $\mathbf{q}_{\text{new}}$ dramatically enhances recall on difficult occluded queries.

---

## 4. Leaderboard Benchmark Progression

| Rank | Architecture Stack | Metric Loss | Spatial Pooling | Verification Strategy | mAP@100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rank 1** | Ensemble: ConvNeXt-XL + Swin-Large + ResNeXt-101 | SubCenter ArcFace ($K=3$) | Learnable GeM ($p=3.2$) | DELG + SuperPoint + RANSAC Inlier Scoring | **0.5891** |
| **Rank 2** | Dual EfficientNet-B7 + ViT-Base | ArcFace ($s=64, m=0.35$) | GeM Pooling | Local Descriptor Matching + alpha-QE | **0.5784** |
| **Rank 3** | ResNet-200D + RegNetY-16GF | CosFace + CurricularFace | Generalized Mean | k-NN Graph Diffusion Re-ranking | **0.5699** |

---

## 5. Conclusions & Industry Legacy
The Google Landmark benchmarks defined the state of the art for visual retrieval systems at global scale. The integration of SubCenter angular margins with GeM pooling and RANSAC geometric re-ranking has since been directly incorporated into commercial reverse image search engines, autonomous navigation localization systems, and visual digital twin reconstruction pipelines.

---

## References
1. Weyand, T., et al. "Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval." *CVPR*, 2020.
2. Deng, J., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR*, 2019.
3. Radenović, F., et al. "Fine-tuning CNN Image Retrieval with No Human Annotation." *IEEE TPAMI*, 41(7), 1655-1668, 2019.
4. Cao, B., et al. "Unifying Deep Local and Global Features for Image Search." *ECCV*, 2020.
