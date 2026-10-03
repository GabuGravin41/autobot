import json

with open('competitions/arc_prize_2026/research_sota/scottlegrand/taaf-flashnext-sheetu12b-0922.ipynb', 'r', encoding='utf-8') as f:
    nb_scott = json.load(f)
with open('competitions/arc_prize_2026/research_sota/woguoat/arc3-keith-repro-qwen38next.ipynb', 'r', encoding='utf-8') as f:
    nb_woguoat = json.load(f)

print('Scott cells:', len(nb_scott['cells']))
print('Woguoat cells:', len(nb_woguoat['cells']))

for i in range(min(len(nb_scott['cells']), len(nb_woguoat['cells']))):
    c1 = ''.join(nb_scott['cells'][i]['source'])
    c2 = ''.join(nb_woguoat['cells'][i]['source'])
    if c1 != c2:
        cell_type = nb_scott['cells'][i]['cell_type']
        print(f"Diff in cell {i} ({cell_type}):")
        print('  Scott lines:', len(c1.splitlines()), 'Woguoat lines:', len(c2.splitlines()))
        for l in c1.splitlines()[:3]:
            print('   [Scott]:', l[:100])
        for l in c2.splitlines()[:3]:
            print('   [Woguo]:', l[:100])
