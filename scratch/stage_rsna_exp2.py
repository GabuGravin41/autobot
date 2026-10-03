import shutil, json
from pathlib import Path

src_dir = Path('competitions/rsna_knee/exp1_tri_backbone_sota')
dst_dir = Path('competitions/rsna_knee/exp2_probe22_sota')
dst_dir.mkdir(parents=True, exist_ok=True)

with open(src_dir / 'notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2_src = ''.join(nb['cells'][2]['source'])
cell2_mod = cell2_src.replace('PRESET = "parent"', 'PRESET = "probe22"')
nb['cells'][2]['source'] = [cell2_mod]

with open(dst_dir / 'notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

meta = {
  'id': 'daltongabrielomondi/autobot-rsna-knee-exp2-probe22-sota',
  'title': 'Autobot RSNA Knee Exp2 Probe22 SOTA',
  'code_file': 'notebook.ipynb',
  'language': 'python',
  'kernel_type': 'notebook',
  'is_private': 'true',
  'enable_gpu': 'true',
  'enable_tpu': 'false',
  'enable_internet': 'false',
  'competition_sources': ['rsna-knee-abnormality-detection'],
  'dataset_sources': [
    'pilkwang/rsna-knee-weights',
    'antoinegg1/rsna-knee-e9-radimagenet-heads-v15',
    'tonylica/rsna-knee-bend-dinov3-0917-repro-assets',
    'mattiaangeli/knee-mri-fold-weights',
    'dreaddevelopment/raptor-knee-widedense',
    'mattiaangeli/rsna-knee-coat-resgated-ep10-top3'
  ],
  'kernel_sources': [],
  'model_sources': []
}
with open(dst_dir / 'kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print('Exp 2 staged successfully at', dst_dir)
