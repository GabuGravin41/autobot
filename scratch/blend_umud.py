import json
import io
import pandas as pd
import numpy as np

with open('competitions/umud_muscle_architecture/research_dread/vera-seg-centerline-mt-correction-lb-0-45134.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

csv_str = None
for cell in nb.get('cells', []):
    src = ''.join(cell.get('source', []))
    if 'submission_csv =' in src:
        # Find raw string content
        lines = [line.strip() for line in src.split('\n') if line.strip().startswith('IMG_')]
        header = 'image_id,pa_deg,fl_mm,mt_mm'
        csv_str = header + '\n' + '\n'.join(lines)
        break

if not csv_str:
    raise ValueError("Could not extract submission_csv string from Dread notebook")

df_dread = pd.read_csv(io.StringIO(csv_str))
df_lam = pd.read_csv('competitions/umud_muscle_architecture/output_exp2/submission_comma.csv')

print('Dread shape:', df_dread.shape)
print('Lam shape:', df_lam.shape)

for col in ['pa_deg', 'fl_mm', 'mt_mm']:
    corr = np.corrcoef(df_dread[col], df_lam[col])[0, 1]
    print(f'Correlation {col}: {corr:.4f}')

# Blend: 60% Dread (0.45134 anchor) + 40% Lam (0.49332 continuum kinematics)
df_blend = df_dread.copy()
for col in ['pa_deg', 'fl_mm', 'mt_mm']:
    df_blend[col] = (0.60 * df_dread[col] + 0.40 * df_lam[col]).round(4)

out_p = 'competitions/umud_muscle_architecture/output_exp2/submission_blend_60_40.csv'
df_blend.to_csv(out_p, index=False)
print(f'Saved blend to {out_p} ({len(df_blend)} rows)')
assert len(df_blend) == 309
assert df_blend.isnull().sum().sum() == 0
print("Verification PASSED!")
