import json
from pathlib import Path

p = Path('competitions/rsna_knee/exp3_quad_coat_sota/notebook.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
print('Exp 3 Total cells:', len(nb['cells']))
for i in range(len(nb['cells'])-6, len(nb['cells'])):
    c = nb['cells'][i]
    src = "".join(c['source']).strip().replace("\n", " ")
    print(f"Cell {i} ({c['cell_type']}): {src[:120]}...")
