"""
Dispatch Competition Diagnostic Report to Claude Opus 5.5
Competition: RSNA Knee Abnormality Detection
Goal: Consolidate strategies to break 0.943 tie wall and reach >= 0.946 - 0.948 (Top 2% Silver)
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autobot.integrations import claude_code_bridge

PROMPT = """
You are Claude Opus 5.5, acting as the Lead Medical Imaging & Deep Learning Diagnostic Specialist for Autobot.
We need your highest-level medical imaging AI and ensemble reasoning to formulate our breakthrough strategy for RSNA Knee.

=== COMPETITION CONTEXT & LEADERBOARD TOPOLOGY ===
- Competition: RSNA Knee Abnormality Detection (Macro AUC ROC across 12 targets).
- Prize Pool: $77,000 | Deadline: 2026-10-22
- Total Teams: 4,442
- Leaderboard Distribution:
  * Rank 1: 0.959
  * Top 10 (Gold): 0.956
  * Top 2% (High Silver): 0.947 - 0.952
  * Top 5% (Silver Cutoff): 0.943 (Rank 222)
  * Top 10% (Bronze Cutoff): 0.943 (Rank 444)
  * CURRENT AUTOBOT STATUS: Score 0.943 (Exp 3, Ref 56613822).
- CRITICAL VULNERABILITY: ~450 teams are tightly bunched between Rank 220 and Rank 669 at score 0.943! A mere 0.002 gain to 0.945 jumps straight to Top 2% (< Rank 100). Reaching 0.948 knocks on Top 50 / Gold border!

=== 12 CLINICAL TARGETS & MULTI-SERIES MRI DATA ===
- 12 Targets:
  1. ACL (Anterior Cruciate Ligament tear)
  2. MCL (Medial Collateral Ligament tear)
  3. Medial Meniscus tear
  4. Lateral Meniscus tear
  5. Medial Osteoarthritis
  6. Lateral Osteoarthritis
  7. Patellofemoral Osteoarthritis
  8. Joint Effusion
  9. Synovitis
  10. Baker's Cyst
  11. Bone Contusion / Edema
  12. Fracture
- Multi-series DICOM volumes: Sagittal, Coronal, Axial with Fat Suppression (FS=0, FS=1).
- Variable series count per study (missing series occur frequently).

=== WHAT WE HAVE TRIED & VERIFIED SO FAR ===
1. Exp 1 (LB 0.939): Tri-Backbone Foundation Ensemble
   - Anatomical Slot Alignment: (Sag, FS=1), (Sag, FS=0), (Cor, FS=1), (Cor, FS=0), (Ax, FS=1), (Ax, FS=0) with 140mm in-plane physical normalization.
   - Shared-Prefix DINOv2: frozen 6-block prefix, 20 member-specific heads.
   - RadImageNet ResNet-50 heads (v52_e11_heads.pt).
   - CoAtNet Residual Gated Raptor Fusion.

2. Exp 2 (LB 0.941): Tri-Backbone + Probe22 Finding-Specific Routing
   - Calibrated target-specific weights across views.
   - ACL routed primarily to Sagittal FS=1; MCL to Coronal FS=1; Meniscus to Sagittal + Coronal.

3. Exp 3 (LB 0.943): Quad-CoAtNet Complementary Ensemble + Calibrated SOTA Routing
   - Combined 4x CoAtNet checkpoints + DINOv2 + RadImageNet.
   - Scored 0.943 (Hit the Top 5% boundary, but stuck in the 450-team cluster).

=== AVAILABLE DATASET SOURCES IN KAGGLE ===
- `pilkwang/rsna-knee-weights`
- `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`
- `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`
- `mattiaangeli/knee-mri-fold-weights`
- `dreaddevelopment/raptor-knee-widedense`
- `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`

=== YOUR MISSION ===
Do NOT write code yet. Provide a master-level technical and medical AI consultation:
1. Deconstruct the 0.943 Barrier: Why are hundreds of teams clustered at 0.943? What common mistake or limitation in standard 2.5D/3D slice aggregation causes this ceiling?
2. High-Alpha Breakthrough Vectors (Targeting 0.946 - 0.950+):
   - Vector A: Anatomically Sound Test-Time Augmentation (TTA). Note: If doing horizontal flip on Coronal or Axial views, Medial and Lateral flip! How must TTA be structured so Medial Meniscus / Medial OA and Lateral Meniscus / Lateral OA are swapped correctly?
   - Vector B: Attention-Weighted 3D Slice Pooling vs Max/Mean Pooling. Which targets suffer most from max-pooling dilution (e.g. subtle fractures or thin tears)?
   - Vector C: Finding-Specific Ensembling & Post-Processing. Macro-AUC is rank-based. How does target-wise rank normalization / percentile transformation impact Macro AUC across 12 imbalanced targets?
3. Recommended High-Conviction Experiment Specification: Detail the exact model blend, slice sampling, TTA rules, and target routing for our next experiment.
"""

def main():
    print("Pushing RSNA Knee Diagnostic Report to Claude Opus 5.5...")
    res = claude_code_bridge.run_headless(
        PROMPT,
        cwd=str(REPO_ROOT),
        permission_mode="plan",
        timeout=300
    )
    if res.get("ok"):
        data = res.get("data")
        result = data.get("result", "") if isinstance(data, dict) else str(data)
        out_file = REPO_ROOT / "competitions" / "rsna_knee" / "OPUS_5_5_STRATEGY_REPORT.md"
        out_file.write_text(result, encoding="utf-8")
        print(f"SUCCESS: Report saved to {out_file}")
        print("\n" + "="*80)
        print("CLAUDE OPUS 5.5 RSNA KNEE STRATEGY PREVIEW:")
        print("="*80)
        print(result[:2000] + "\n...")
    else:
        print("Execution failed:", res.get("error"))

if __name__ == "__main__":
    main()
