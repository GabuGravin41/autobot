import json
from pathlib import Path

out_dir = Path("competitions/rsna_knee/exp6_0946_sota")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Metadata
meta = json.load(open("scratch/yamadan96/kernel-metadata.json", encoding="utf-8"))
meta["id"] = "daltongabrielomondi/autobot-rsna-knee-exp6-0946-consensus"
meta["title"] = "Autobot RSNA Knee Exp6 0946 Consensus SOTA"
meta["code_file"] = "notebook.ipynb"
meta["is_private"] = True
meta["enable_gpu"] = True
with open(out_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# 2. Notebook
nb = json.load(open("scratch/yamadan96/rsna-knee-d4-public0946.ipynb", encoding="utf-8"))

gatekeeper_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==================== AUTOBOT SUBMISSION GATEKEEPER CONTRACT ====================\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from pathlib import Path\n",
        "\n",
        "sub_path = Path('/kaggle/working/submission.csv')\n",
        "assert sub_path.exists(), 'FATAL: /kaggle/working/submission.csv does not exist!'\n",
        "sub = pd.read_csv(sub_path)\n",
        "print(f'[GATEKEEPER] Loaded submission.csv with shape {sub.shape}')\n",
        "\n",
        "TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',\n",
        "           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', \"Baker's\",\n",
        "           'Contusion', 'Fracture']\n",
        "EXPECTED_COLS = ['StudyInstanceUID'] + TARGETS\n",
        "assert list(sub.columns) == EXPECTED_COLS, f'FATAL: Columns mismatch: {list(sub.columns)} vs {EXPECTED_COLS}'\n",
        "assert not sub.isnull().values.any(), 'FATAL: submission contains NaN or Null values!'\n",
        "assert np.all(sub[TARGETS].values >= 0.0) and np.all(sub[TARGETS].values <= 1.0), 'FATAL: Probabilities out of [0, 1] range!'\n",
        "print(f'[GATEKEEPER] All 12 targets verified strictly in [0, 1]. Total studies: {len(sub)}')\n",
        "print('[GATEKEEPER] CONTRACT PASSED: Ready for leaderboard scoring.')\n"
    ]
}

nb["cells"].append(gatekeeper_cell)
with open(out_dir / "notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Exp 6 staged successfully at:", out_dir)
print("Total cells:", len(nb["cells"]))
