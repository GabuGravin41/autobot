import json

src_path = 'competitions/rsna_knee/research_haideptry/rsna-knee-speedy-raptors-coatnet-d4-0943.ipynb'
dst_path = 'competitions/rsna_knee/exp3_quad_coat_sota/notebook.ipynb'

with open(src_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 27 routing weights
target_src = ''.join(nb['cells'][27]['source'])
old_block = """_coatnet_weight = {label: 0.60 for label in _blend_labels}
_coatnet_weight.update({
    'ACL': 0.75,
    'Medial Meniscus': 0.80,
    'Lateral Meniscus': 1.00,
    'Lateral OA': 0.75,
    'Fracture': 0.75,
})"""

new_block = """_coatnet_weight = {label: 0.60 for label in _blend_labels}
_coatnet_weight.update({
    'ACL': 0.85,
    'MCL': 0.75,
    'Medial Meniscus': 0.85,
    'Lateral Meniscus': 1.00,
    'Medial OA': 0.75,
    'Lateral OA': 0.75,
    'PF OA': 0.65,
    'Contusion': 0.70,
    'Fracture': 0.75,
})"""

assert old_block in target_src, 'Target block not found in cell 27!'
updated_src = target_src.replace(old_block, new_block)
nb['cells'][27]['source'] = [line + '\n' for line in updated_src.splitlines()]

# Add Autobot Gatekeeper cell at the end
gatekeeper_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# ==================== AUTOBOT SUBMISSION GATEKEEPER CONTRACT ====================\n',
        'import pandas as pd\n',
        'import numpy as np\n',
        'from pathlib import Path\n',
        '\n',
        'sub_path = Path("/kaggle/working/submission.csv")\n',
        'assert sub_path.exists(), "FATAL: /kaggle/working/submission.csv does not exist!"\n',
        '\n',
        'sub = pd.read_csv(sub_path)\n',
        'print(f"[GATEKEEPER] Loaded submission.csv with shape {sub.shape}")\n',
        '\n',
        'TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",\n',
        '           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker\'s",\n',
        '           "Contusion", "Fracture"]\n',
        'EXPECTED_COLS = ["StudyInstanceUID"] + TARGETS\n',
        'assert list(sub.columns) == EXPECTED_COLS, f"FATAL: Columns mismatch: {list(sub.columns)} vs {EXPECTED_COLS}"\n',
        'assert not sub.isnull().values.any(), "FATAL: submission contains NaN or Null values!"\n',
        'assert np.all(sub[TARGETS].values >= 0.0) and np.all(sub[TARGETS].values <= 1.0), "FATAL: Probabilities out of [0, 1] range!"\n',
        'print(f"[GATEKEEPER] All 12 targets verified strictly in [0, 1]. Total studies: {len(sub)}")\n',
        'print("[GATEKEEPER] CONTRACT PASSED: Ready for leaderboard scoring.")\n'
    ]
}
nb['cells'].append(gatekeeper_cell)

with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully generated Exp 3 notebook!')
