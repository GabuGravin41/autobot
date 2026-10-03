import json
import glob
import os

for p in glob.glob('competitions/arc_prize_2026/research_sota/*/*.ipynb'):
    print('==============================')
    print('Notebook:', p)
    with open(p, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print('Total cells:', len(nb.get('cells', [])))
    for i, cell in enumerate(nb.get('cells', [])):
        txt = ''.join(cell.get('source', []))
        matches = [l.strip() for l in txt.splitlines() if any(k in l.lower() for k in ['/kaggle/input', 'vllm', 'arcade', 'qwen', 'anim', 'score', 'weights', 'duck', 'python', 'sh '])]
        if matches:
            print(f"Cell {i} ({cell.get('cell_type')}):")
            for m in matches[:8]:
                print('  ', m[:140])
