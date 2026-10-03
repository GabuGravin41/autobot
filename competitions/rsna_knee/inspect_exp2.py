import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('competitions/rsna_knee/exp2_probe22_sota/notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    txt = ''.join(c['source'])
    if any(k in txt.lower() for k in ['probe22', 'weights', 'routing', 'lateral', 'acl', 'blend']):
        print(f"Cell {i} ({c['cell_type']}):")
        for l in txt.splitlines()[:15]:
            print('  ', l[:110])
        print('-'*50)
