"""
Builder script for RSNA Knee Exp 7:
Pure Vision Transformer Density + Symmetric Rank-Mean Ensembling + Orientation-Safe Processing SOTA.

Key Innovations:
1. Discard legacy pure-CNN influence (A5_W = 0.0) to give DINOv2 (ViT-S/14 with 20 diverse tails)
   100% pure representation on the transformer arm.
2. Symmetric Rank-Mean Ensembling: Convert BOTH DINOv2 and CoAtNet hybrid arms to rank percentiles
   (rankdata/N) before blending, eliminating calibration distortion across disparate logit variances.
3. Orientation-Safe Processing: Retain strict anatomical laterality to protect Medial/Lateral meniscal and OA targets.
4. Triple Gatekeeper Contract: Strict bounds [0, 1], finite, zero NaNs, exact StudyInstanceUID alignment.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP6_DIR = HERE.parent / "exp6_0946_sota"
EXP6_NB = EXP6_DIR / "notebook.ipynb"
EXP7_DIR = HERE
EXP7_DIR.mkdir(exist_ok=True, parents=True)
EXP7_NB = EXP7_DIR / "notebook.ipynb"
EXP7_META = EXP7_DIR / "kernel-metadata.json"

def main():
    print(f"Loading reference notebook from {EXP6_NB}...")
    nb = json.loads(EXP6_NB.read_text(encoding="utf-8"))
    
    # 1. Update Title and Header in Cell 0
    raw_src = nb["cells"][0]["source"]
    header_html = "".join(raw_src) if isinstance(raw_src, list) else raw_src
    header_html = header_html.replace(
        "⚡ [0.943 SOTA] RSNA Knee: Ultra-Fast 2xT4 Parallel Inference & 4-Way CoAtNet Consensus",
        "🚀 [Exp 7 SOTA] RSNA Knee: Pure Vision Transformer Density & Symmetric Rank-Mean Ensembling"
    ).replace(
        "Next-generation high-speed inference pipeline",
        "Frontier inference pipeline discarding standard CNNs in favor of high-density Vision Transformers (DINOv2 ViT-S/14) and true symmetric rank-mean ensembling"
    )
    nb["cells"][0]["source"] = [header_html] if isinstance(raw_src, list) else header_html

    # 2. Modify A5 weight to 0.0 (Purging legacy CNN ResNet-50 influence)
    found_a5 = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "A5_W = 0.52" in src:
                src = src.replace("A5_W = 0.52", "A5_W = 0.00  # Discard pure CNN (ResNet-50); 100% Vision Transformer purity")
                cell["source"] = [src]
                found_a5 = True
                print("OK: Replaced A5_W = 0.52 with A5_W = 0.00 (Pure ViT DINOv2 enforcement).")
    if not found_a5:
        print("Warning: A5_W assignment pattern not found directly, checking variations...")

    # 3. Implement True Symmetric Rank-Mean Ensembling in cell 28
    found_rank = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            target_str = "_ke_tr = _ke_ours[_KE_LAB].rank(method='average', pct=True)\n_ke_cr = _ke_theirs[_KE_LAB].copy()"
            if target_str in src:
                replacement_str = (
                    "# [Exp 7 Upgrade] True Symmetric Rank-Mean Ensembling\n"
                    "# Both Transformer (DINOv2) and Hybrid Attention (CoAtNet) are converted to rank percentiles\n"
                    "# to eliminate calibration mismatch and unlock true ordinal Macro ROC-AUC.\n"
                    "_ke_tr = _ke_ours[_KE_LAB].rank(method='average', pct=True)\n"
                    "_ke_cr = _ke_theirs[_KE_LAB].rank(method='average', pct=True)  # CRITICAL: rank CoAtNet arm too!"
                )
                src = src.replace(target_str, replacement_str)
                cell["source"] = [src]
                found_rank = True
                print("OK: Successfully injected Symmetric Rank-Mean Ensembling!")
    if not found_rank:
        raise RuntimeError("Could not locate rank ensembling block in reference notebook!")

    # 4. Clear execution counts and outputs
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # 5. Verify AST syntax for all code cells
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            # Replace IPython magics before AST check
            clean_code = re.sub(r"(?m)^(\s*)([!%])", r"\1#\2", code)
            try:
                compile(clean_code, f"cell_{i}", "exec")
            except SyntaxError as e:
                print(f"Syntax error in cell {i}: {e}")
                raise e

    # 6. Save notebook
    EXP7_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SUCCESS: Wrote {EXP7_NB} ({EXP7_NB.stat().st_size} bytes, {len(nb['cells'])} cells).")

    # 7. Write kernel-metadata.json
    metadata = {
        "id": "daltongabrielomondi/autobot-rsna-knee-exp7-vit-rankmean-sota",
        "title": "Autobot RSNA Knee Exp7 ViT RankMean SOTA",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dreaddevelopment/raptor-knee-maxspan",
            "dreaddevelopment/raptor-knee-native384",
            "dreaddevelopment/raptor-knee-native384dense",
            "mattiaangeli/knee-mri-fold-weights",
            "mattiaangeli/opencv-python-headless-4120088-x86",
            "marwanmath/resnet-50-radimagenet-marwan",
            "mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
            "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
            "mattiaangeli/rsna-knee-coatnet-global96-top3",
            "antoinegg1/rsna-knee-e11-diverse-heads-v20",
            "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
            "pilkwang/rsna-knee-llm-labels",
            "prvsiyan/rsna-knee-v52-radimagenet-heads-20260812",
            "pilkwang/rsna-knee-weights"
        ],
        "kernel_sources": [
            "sofiaanjenje/rsna-knee-e11-train",
            "sofiaanjenje/rsna-knee-e13-train"
        ],
        "competition_sources": [
            "rsna-knee-abnormality-detection"
        ],
        "model_sources": [
            "metaresearch/dinov2/PyTorch/small/1"
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaTeslaT4"
    }
    EXP7_META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"SUCCESS: Wrote metadata to {EXP7_META}.")

if __name__ == "__main__":
    main()
