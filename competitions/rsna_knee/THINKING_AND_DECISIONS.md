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
| **Exp 6** | `daltongabrielomondi/autobot-rsna-knee-exp5-knee-act-sota` (v4) | 0.946 Consensus SOTA (Fracture Retention + Unoverfit CoAtNet) | 4x CoAtNet + DINOv2 + RadImageNet + Raptor | 2xT4 GPU | **EVALUATING** | 0.968 | Target **`0.946`** (`Ref 56810220`) (Rank ~#350) |

---

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
