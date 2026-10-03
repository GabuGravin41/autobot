import json
from pathlib import Path
import ast

src_nb_path = Path("competitions/rsna_knee/exp3_quad_coat_sota/notebook.ipynb")
dst_nb_path = Path("competitions/rsna_knee/exp5_knee_act_sota/notebook.ipynb")

nb = json.loads(src_nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]

# Define the Knee-ACT Biomechanical Triad Coupling Cell
knee_act_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =========================================================================\n",
        "# Knee-ACT: Biomechanical Triad Coupling & Anatomical Calibration\n",
        "# =========================================================================\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "sub_path = Path('/kaggle/working/submission.csv')\n",
        "sub = pd.read_csv(sub_path)\n",
        "targets = [\"ACL\", \"MCL\", \"Medial Meniscus\", \"Lateral Meniscus\", \"Medial OA\",\n",
        "           \"Lateral OA\", \"PF OA\", \"Effusion\", \"Synovitis\", \"Baker's\",\n",
        "           \"Contusion\", \"Fracture\"]\n",
        "\n",
        "# Convert percentile ranks to continuous logit representation\n",
        "ranks = sub[targets].to_numpy(dtype=np.float64)\n",
        "ranks_clipped = np.clip(ranks, 1e-4, 1.0 - 1e-4)\n",
        "logits = np.log(ranks_clipped / (1.0 - ranks_clipped))\n",
        "\n",
        "idx = {t: i for i, t in enumerate(targets)}\n",
        "\n",
        "# 1. Pivot-Shift Marrow Edema Coupling: Contusion reinforces ACL tear\n",
        "contusion_gate = 1.0 / (1.0 + np.exp(-logits[:, idx[\"Contusion\"]]))\n",
        "logits[:, idx[\"ACL\"]] += 0.25 * (contusion_gate - 0.5)\n",
        "\n",
        "# 2. O'Donoghue's Unhappy Triad Coupling: ACL + MCL co-occurrence reinforces Medial Meniscus\n",
        "acl_gate = 1.0 / (1.0 + np.exp(-logits[:, idx[\"ACL\"]]))\n",
        "mcl_gate = 1.0 / (1.0 + np.exp(-logits[:, idx[\"MCL\"]]))\n",
        "med_men_gate = 1.0 / (1.0 + np.exp(-logits[:, idx[\"Medial Meniscus\"]]))\n",
        "triad_co_occurrence = (acl_gate * mcl_gate) ** 0.5\n",
        "logits[:, idx[\"Medial Meniscus\"]] += 0.20 * (triad_co_occurrence - 0.5)\n",
        "\n",
        "# 3. Joint Capsule Reactive Synovitis Coupling: Effusion and Synovitis\n",
        "effusion_gate = 1.0 / (1.0 + np.exp(-logits[:, idx[\"Effusion\"]]))\n",
        "synovitis_gate = 1.0 / (1.0 + np.exp(-logits[:, idx[\"Synovitis\"]]))\n",
        "capsule_coupling = 0.5 * (effusion_gate + synovitis_gate)\n",
        "logits[:, idx[\"Effusion\"]] += 0.15 * (capsule_coupling - 0.5)\n",
        "logits[:, idx[\"Synovitis\"]] += 0.15 * (capsule_coupling - 0.5)\n",
        "\n",
        "# 4. Chronic Joint Space Wear Coupling: Medial Meniscal Wear + Medial OA\n",
        "med_oa_gate = 1.0 / (1.0 + np.exp(-logits[:, idx[\"Medial OA\"]]))\n",
        "wear_coupling = 0.5 * (med_men_gate + med_oa_gate)\n",
        "logits[:, idx[\"Medial OA\"]] += 0.15 * (wear_coupling - 0.5)\n",
        "\n",
        "# Convert back to probabilities and apply final Percentile Rank Normalization\n",
        "adjusted_probs = 1.0 / (1.0 + np.exp(-logits))\n",
        "adjusted_ranks = pd.DataFrame(adjusted_probs, columns=targets).rank(method='average', pct=True).to_numpy()\n",
        "\n",
        "sub[targets] = adjusted_ranks\n",
        "sub.to_csv(sub_path, index=False)\n",
        "print('[Knee-ACT] Biomechanical Triad coupling applied and rank-normalized successfully!')\n"
    ]
}

# Insert before the last cell (the Gatekeeper cell)
new_cells = cells[:-1] + [knee_act_cell] + [cells[-1]]
nb["cells"] = new_cells

# Verify AST syntax
for i, cell in enumerate(new_cells):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        ast.parse(code)

dst_nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"SUCCESS: Assembled full Knee-ACT ensemble notebook with {len(new_cells)} cells ({dst_nb_path.stat().st_size} bytes).")
