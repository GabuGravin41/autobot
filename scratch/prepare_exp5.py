import os
import json

exp5_dir = 'competitions/biohub_cell_tracking/exp5_division_precision_sota'
os.makedirs(exp5_dir, exist_ok=True)

with open('competitions/biohub_cell_tracking/exp4_harmonic_subvoxel_sota/harmonic_fusion_sota.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'BIOHUB_SAFE_DIV_MAX_UM' in src:
        print(f'Modifying cell {i}')
        lines = []
        for line in c['source']:
            if 'BIOHUB_SAFE_DIV_MAX_UM' in line and '=' in line and not line.strip().startswith('#'):
                lines.append('os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "10.0"\n')
            elif 'BIOHUB_SAFE_DIV_SISTER_MAX_UM' in line and '=' in line and not line.strip().startswith('#'):
                lines.append('os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "15.0"\n')
            elif 'BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM' in line and '=' in line and not line.strip().startswith('#'):
                lines.append('os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "11.0"\n')
            elif 'BIOHUB_SAFE_DIV_DIVERGE_UM' in line and '=' in line and not line.strip().startswith('#'):
                lines.append('os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "1.75"\n')
            elif 'BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU' in line and '=' in line and not line.strip().startswith('#'):
                lines.append('os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.75"\n')
            elif 'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD' in line and '=' in line and not line.strip().startswith('#'):
                lines.append('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.15"\n')
            elif 'BIOHUB_MOTION_RELINK_TIGHT_UM' in line and '=' in line and not line.strip().startswith('#'):
                lines.append('os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n')
            else:
                lines.append(line)
        c['source'] = lines

with open(os.path.join(exp5_dir, 'harmonic_division_sota.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

metadata = {
  'id': 'daltongabrielomondi/autobot-biohub-exp5-division-precision-sota',
  'title': 'Autobot Biohub Exp5 Division Precision SOTA',
  'code_file': 'harmonic_division_sota.ipynb',
  'language': 'python',
  'kernel_type': 'notebook',
  'is_private': True,
  'enable_gpu': True,
  'enable_tpu': False,
  'enable_internet': False,
  'dataset_sources': [
    'pilkwang/biohub-tracking-support-pack-50ep-v1',
    'pilkwang/biohub-deepcenter-unet3d-center-prior-v1',
    'pilkwang/biohub-temporal-unet3d-seed314159-v1',
    'anvithpothula/biohub-v1284-head-s075'
  ],
  'competition_sources': [
    'biohub-cell-tracking-during-development'
  ]
}

with open(os.path.join(exp5_dir, 'kernel-metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print('Exp 5 prepared successfully')
