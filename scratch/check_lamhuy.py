import io
import pandas as pd

with open('competitions/umud_muscle_architecture/research_lamhuy/variational-ultrasound-kinematics-ensemble.py', encoding='utf-8') as f:
    text = f.read()

start_s = 'CLINICAL_BENCHMARK_CSV = """'
start_idx = text.find(start_s)
end_idx = text.find('"""', start_idx + len(start_s))
csv_data = text[start_idx + len(start_s):end_idx]
df = pd.read_csv(io.StringIO(csv_data))
print('Clinical benchmark rows:', len(df))
print('Columns:', df.columns.tolist())
print(df.head(3))
