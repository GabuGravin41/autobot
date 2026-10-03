import json
from pathlib import Path

p = Path('competitions/rsna_knee/exp4_ryokucha_0946_sota/rsna-knee-d4-blend-0946-ours10.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
print('Total cells:', len(nb['cells']))
for i, c in enumerate(nb['cells'][:15]):
    src = ''.join(c['source']).strip().replace('\n', ' ')
    print(f"Cell {i} ({c['cell_type']}): {src[:100]}...")
