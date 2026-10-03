import json
from pathlib import Path

nb_path = Path(r"c:\Users\User 1\OneDrive\Desktop\projects\django projects\personal projects\autobot\competitions\rsna_knee\exp1_tri_backbone_sota\notebook.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

contract_lines = [
    "# ==================== AUTOBOT SUBMISSION GATEKEEPER CONTRACT ====================\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pathlib import Path\n\n",
    "sub_path = Path('/kaggle/working/submission.csv')\n",
    "assert sub_path.exists(), 'FATAL: /kaggle/working/submission.csv does not exist!'\n\n",
    "sub = pd.read_csv(sub_path)\n",
    "print(f'[GATEKEEPER] Loaded submission.csv with shape {sub.shape}')\n\n",
    "TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', \"Baker's\", 'Contusion', 'Fracture']\n",
    "EXPECTED_COLS = ['StudyInstanceUID'] + TARGETS\n",
    "assert list(sub.columns) == EXPECTED_COLS, f'FATAL: Columns mismatch: {list(sub.columns)} vs {EXPECTED_COLS}'\n",
    "assert not sub.isnull().values.any(), 'FATAL: submission contains NaN or Null values!'\n",
    "for col in TARGETS:\n",
    "    vals = sub[col].values\n",
    "    assert np.all((vals >= 0.0) & (vals <= 1.0)), f'FATAL: Column {col} contains values outside [0, 1]!'\n\n",
    "sample_candidates = [Path('/kaggle/input/rsna-knee-abnormality-detection/sample_submission.csv'), Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/sample_submission.csv')]\n",
    "sample_path = next((p for p in sample_candidates if p.exists()), None)\n",
    "if sample_path:\n",
    "    sample = pd.read_csv(sample_path)\n",
    "    assert len(sub) == len(sample), f'FATAL: Row count mismatch: {len(sub)} vs {len(sample)}'\n",
    "    assert (sub['StudyInstanceUID'].values == sample['StudyInstanceUID'].values).all(), 'FATAL: StudyInstanceUID alignment mismatch!'\n",
    "    print('[GATEKEEPER] Row count and StudyInstanceUID alignment verified against sample_submission.csv!')\n\n",
    "print('[GATEKEEPER] ALL 6 SUBMISSION CONTRACT INVARIANTS PASSED SUCCESSFULLY!')\n",
    "print(sub.head())\n"
]

contract_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": contract_lines
}

nb["cells"].append(contract_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("SUCCESS: Added Autobot Submission Gatekeeper contract cell to notebook.ipynb")
