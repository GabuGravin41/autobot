import json

with open('competitions/biohub_cell_tracking/sota_amanatar_optimized-biohub-max-score/optimized-biohub-max-score.ipynb', 'r', encoding='utf-8') as f:
    nb_old = json.load(f)
with open('competitions/biohub_cell_tracking/research_amanatar_opt/optimized-biohub-max-score.ipynb', 'r', encoding='utf-8') as f:
    nb_new = json.load(f)

print('Old cells:', len(nb_old['cells']))
print('New cells:', len(nb_new['cells']))

for i in range(min(len(nb_old['cells']), len(nb_new['cells']))):
    c_old = ''.join(nb_old['cells'][i]['source'])
    c_new = ''.join(nb_new['cells'][i]['source'])
    if c_old != c_new:
        print(f"Diff in cell {i}: old {len(c_old.splitlines())} lines vs new {len(c_new.splitlines())} lines")
