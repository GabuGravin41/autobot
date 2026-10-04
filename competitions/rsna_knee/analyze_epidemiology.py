import pandas as pd
import numpy as np

train = pd.read_csv('competitions/rsna_knee/train.csv')
train_series = pd.read_csv('competitions/rsna_knee/train_series.csv')

targets = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 65)
print(f"RSNA KNEE GROUND-TRUTH EPIDEMIOLOGY AUDIT (N = {len(train)} studies)")
print("=" * 65)
for t in targets:
    pos = int(train[t].sum())
    rate = float(train[t].mean())
    print(f"  {t:<18}: {pos:4d} positive ({rate * 100:5.2f}%)")

print("\n" + "=" * 65)
print("TOP BIOMECHANICAL CO-OCCURRENCE PAIRS (Jaccard Index)")
print("=" * 65)
jaccards = []
for i in range(len(targets)):
    for j in range(i + 1, len(targets)):
        t1, t2 = targets[i], targets[j]
        both = int(((train[t1] == 1) & (train[t2] == 1)).sum())
        either = int(((train[t1] == 1) | (train[t2] == 1)).sum())
        jac = both / either if either > 0 else 0.0
        jaccards.append((jac, both, t1, t2))

jaccards.sort(reverse=True)
for jac, both, t1, t2 in jaccards[:12]:
    print(f"  {t1:<18} + {t2:<18}: Jaccard = {jac:.3f} (n = {both:4d})")

print("\n" + "=" * 65)
print("SERIES PLANE & CONTRAST PROTOCOL DISTRIBUTION")
print("=" * 65)
print(train_series.groupby(['Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression']).size())
