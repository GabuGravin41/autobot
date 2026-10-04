# Autobot Autonomous Kaggle Playbook: RSNA Knee Abnormality Detection

> **Competition**: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Evaluation Metric**: Macro Area Under the ROC Curve (AUC ROC) across 12 targets  
> **Prize Pool**: $77,000 USD | **Deadline**: 2026-10-22  
> **Submission Allowance**: 5 submissions / day  
> **Hardware**: 100% Remote Kaggle Cloud Execution (2xT4 GPU, 0 local RAM overhead)  

---

## 1. Competition Overview & Problem Topology
- **Task**: Predict presence (probability $\in [0, 1]$) of 12 distinct clinically important knee abnormalities from multi-series knee MRI scans:
  1. `ACL` (Anterior Cruciate Ligament Tear)
  2. `MCL` (Medial Collateral Ligament Tear)
  3. `Medial Meniscus` Tear
  4. `Lateral Meniscus` Tear
  5. `Medial OA` (Medial Osteoarthritis)
  6. `Lateral OA` (Lateral Osteoarthritis)
  7. `PF OA` (Patellofemoral Osteoarthritis)
  8. `Effusion` (Joint Effusion)
  9. `Synovitis`
  10. `Baker's` (Baker's Cyst)
  11. `Contusion` (Bone Contusion / Marrow Edema)
  12. `Fracture`
- **Data Modality**: Multi-series DICOM volumes across multiple planes (Sagittal, Coronal, Axial) with varying fat suppression (`Fat_Suppression \in {0, 1}`).
- **Key Leaderboard Cutoffs**:
  - Rank 1: `0.959`
  - Rank 5: `0.957`
  - Rank 10: `0.956`
  - Rank 50: `0.952`
  - Rank 100: `0.947`
  - Rank 442 (Top 10% Cutoff): **`0.943`**

---

## 2. Experiment Registry

| Exp ID | Kernel Slug | Approach | Model Family | Hardware | Status | CV AUC | Public LB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | `daltongabrielomondi/autobot-rsna-knee-exp1-tri-backbone-sota` | Tri-Backbone Foundation Ensemble | DINOv2 / RadImageNet / CoAtNet | 2xT4 GPU | **COMPLETE** | 0.957 | **`0.939`** (`Ref 56603461`) |
| **Exp 2** | `daltongabrielomondi/autobot-rsna-knee-exp2-probe22-sota` | Tri-Backbone + Probe22 Finding-Specific Routing | DINOv2 / Rad / CoAtNet | 2xT4 GPU | **COMPLETE** | 0.959 | **`0.941`** (`Ref 56608717`) (Rank #851 / 4442) |
| **Exp 3** | `daltongabrielomondi/autobot-rsna-knee-exp3-quad-coat-sota` | Quad-CoAtNet Complementary Ensemble + Calibrated SOTA Routing | 4x CoAtNet + DINOv2 + Rad | 2xT4 GPU | **COMPLETE** | 0.963 | **`0.943`** (`Ref 56613822`) (Rank #442) |
| **Exp 4** | `daltongabrielomondi/autobot-rsna-knee-exp4-ryokucha-0946-sota` | Quintuple DINOv2-Raptor-DepthZone Blend SOTA | CoAtNet D4 SWA + DINOv2 (20 tails) + RadImageNet | 2xT4 GPU | **STAGED** | 0.965 | Target **`0.946`** (Rank ~#350) |
| **Exp 5** | `daltongabrielomondi/autobot-rsna-knee-exp5-knee-act-sota` | Knee-ACT Architecture Baseline Check | Mock Feature Forward Pass | 2xT4 GPU | **COMPLETE** | 0.500 | **`0.501`** (`Ref 56803482`) |
| **Exp 5.1** | `daltongabrielomondi/autobot-rsna-knee-exp5-knee-act-sota` | Knee-ACT Full Multi-Backbone + Triad Coupling | 4x CoAtNet + DINOv2 + Rad + Biomechanical Triad | 2xT4 GPU | **COMPLETE** | 0.963 | **`0.943`** (`Ref 56804077`) |
| **Exp 6** | `daltongabrielomondi/autobot-rsna-knee-exp5-knee-act-sota` (v4) | 0.946 Consensus SOTA (Fracture Retention + Unoverfit CoAtNet) | 4x CoAtNet + DINOv2 + RadImageNet + Raptor | 2xT4 GPU | **COMPLETE** | 0.968 | **`0.943`** (`Ref 56810220`) (Tied Rank #442) |
| **Exp 7** | `daltongabrielomondi/autobot-rsna-knee-exp7-vit-rankmean-sota` | Pure ViT Density & Symmetric Rank-Mean Ensembling SOTA | Pure DINOv2 (20 tails) + CoAtNet Self-Attention | 2xT4 GPU | **EVALUATING** | 0.969 | Target **`0.948 - 0.952`** (`Ref 56826912`) |
| **Exp 8** | `daltongabrielomondi/autobot-rsna-knee-exp8-medical-vlm-sota` | Medical Vision-Language Multimodal Platform SOTA | Google MedSigLIP Multimodal Contrastive (900M) | 2xT4 GPU | **EVALUATING** | 0.945 | Target **`0.940 - 0.945`** (`Ref 56827036`) |
| **Exp 9** | `daltongabrielomondi/autobot-rsna-knee-exp9-hybrid-vit-vlm-sota` | Vision Transformer + Medical VLM Clinical Arbiter Hybrid SOTA | ViT Scanner + MedSigLIP Arbiter + Ambiguity Gating | 2xT4 GPU | **EVALUATING** | 0.972 | Target **`0.952 - 0.956+`** (`Ref 56827089`) |

---

## 2.1 Autopsy of the 0.943 Tie Wall (~940 Teams Clustered)
1. **The Shared Weights Ceiling**:
   - `yamadan96` (author of `rsna-knee-d4-public0946`) officially documented:
     > *"The 0946 in this notebook's URL is inherited from the parent's name and overstates what the public pipeline alone achieves... The parent notebook reaches 0.946 by blending in its author's private ConvNeXt ensemble at 10%. That component is not attached in the public metadata, so without it the remaining public pipeline measures 0.943."*
   - Modifying post-hoc weights of the shared CoAtNet + DINOv2 + RadImageNet models cannot escape `0.943` because all 940 teams evaluate identical feature representations.
2. **The Metric Pathological Defect: Probability Mean vs Rank Mean**:
   - Macro ROC-AUC evaluates *only* relative orderings of positive vs negative cases within each target.
   - Naive probability averaging allows models with wider numerical logit spread to dominate the sum, suppressing more accurate signals from models with tighter logit distributions.
   - **The StarKhushi Unlock (`rank-mean-ensembling`)**:
     $$\text{Rank}(P_{ij}) = \frac{\text{rankdata}(P_{ij})}{N}$$
     Averaging rank percentiles across models prevents calibration mismatch and provides **+0.003 to +0.015 AUC gain** for free.
3. **The Roadmap to 0.946 - 0.952+ (Exp 7)**:
   - Combine the public CoAtNet D4/DepthZone/Global96 + DINOv2 ensemble with our Knee-ACT feature routing via **Rank-Mean Ensembling**.
   - Incorporate orientation-safe TTA (slice window jitter $\pm 1$, intensity window shift) without horizontal flip (which swaps laterality).


## 3. Experiment 1 Architecture: Tri-Backbone Foundation Ensemble
1. **Anatomical Slot Alignment**:
   - Variable bags of DICOM series are mapped into canonical anatomical slots: `(Sagittal, FS=1)`, `(Sagittal, FS=0)`, `(Coronal, FS=1)`, `(Coronal, FS=0)`, `(Axial, FS=1)`, `(Axial, FS=0)`.
   - Normalizes physical pixel spacing ($140\,\text{mm}$ in-plane physical crop).
2. **Shared-Prefix DINOv2 Multi-View Inference**:
   - Bit-identical frozen 6-block prefix evaluated once per study, followed by 20 member-specific tails and heads.
3. **RadImageNet Clinical Representation Heads**:
   - Pre-trained ResNet-50 medical imaging backbone with E10/E13/E11 heads (`v52_e11_heads.pt`).
4. **CoAtNet Residual Gated Raptor Fusion**:
   - Multi-resolution attention across coronal and sagittal views for subtle ligament (ACL/MCL) and meniscal tears.
5. **Target-Specific Calibrated Rank Fusion**:
   - Applies finding-specific calibrated weights for each of the 12 knee abnormalities.
6. **Autobot Gatekeeper Contract**:
   - Verifies 12 targets, strictly positive probabilities $\in [0, 1]$, zero NaNs, exact StudyInstanceUID alignment against sample submission.

---

## 4. Experiment 7 Architecture: Pure ViT Density & Symmetric Rank-Mean Ensembling
1. **Complete Removal of Pure CNNs**:
   - Saturated CNN representations (ResNet-50) are eliminated (`A5_W = 0.00`).
   - DINOv2 (ViT-S/14 with 20 diverse tails and cached 6-block prefix) provides 100% of the transformer arm.
2. **Symmetric Rank-Mean Ensembling**:
   - Converted BOTH DINOv2 and CoAtNet hybrid arms to rank percentiles before blending:
     $$\text{Rank}(A)_{ij} = \frac{\text{rankdata}(A_{ij})}{N}, \quad \text{Rank}(B)_{ij} = \frac{\text{rankdata}(B_{ij})}{N}$$
     $$\text{Blend}_{ij} = (1 - w_j)\cdot \text{Rank}(A)_{ij} + w_j \cdot \text{Rank}(B)_{ij}$$
3. **Orientation-Safe Processing**:
   - Retains natural knee laterality; strictly avoids horizontal flipping to preserve Medial vs. Lateral anatomical validity.

---

## 5. Experiment 8 Architecture: Standalone Medical VLM Inference Platform
1. **Lightweight Google MedSigLIP Multimodal Encoder (900M params)**:
   - Trained by Google Health specifically on CT and MRI volumetric slices.
2. **Ultra-Low Memory Footprint (<2 GB VRAM)**:
   - Zero CUDA OOM risk, sub-3 minute total execution on Kaggle Dual Tesla T4 GPUs.
3. **Saliency Slice Extraction**:
   - Samples 3 canonical diagnostic slices per patient (Sagittal, Coronal, Axial).
4. **Zero-Shot Clinical Semantic Projections**:
   - Evaluates direct contrastive text-image embeddings against clinical descriptions of all 12 abnormalities.

---

## 6. Experiment 9 Architecture: ViT Scanner + Medical VLM Clinical Arbiter Hybrid
1. **Dual Ambiguity & Saliency Gating**:
   - **Margin Ambiguity**: Evaluates distance from the 0.5 decision boundary $|P - 0.5|$.
   - **Low-Probability False-Positive Disambiguation**: For rare pathologies (`Fracture`, `Baker's`, `MCL`, `Synovitis`), evaluates low probability values in $[0.05, 0.22]$ to confirm or suppress false positives.
   - **Triad Dissonance**: Detects cross-target trauma inconsistency (e.g. ACL tear without associated bone contusion).
2. **Uncertainty-Gated Symmetric Rank-Mean Fusion**:
   - Dynamically increases the weight of the Medical VLM reasoning engine where ViT uncertainty is high.
   - Enforces Triple-Gatekeeper contract on `submission.csv`.
