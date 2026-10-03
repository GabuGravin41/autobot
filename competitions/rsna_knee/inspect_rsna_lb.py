import zipfile
import glob
import pandas as pd

with zipfile.ZipFile('competitions/rsna_knee/leaderboard/rsna-knee-abnormality-detection.zip', 'r') as z:
    z.extractall('competitions/rsna_knee/leaderboard/')

csvs = sorted(glob.glob('competitions/rsna_knee/leaderboard/*.csv'))
df = pd.read_csv(csvs[-1])
print('Total teams:', len(df))
df = df.sort_values('Score', ascending=False).reset_index(drop=True)
total = len(df)

top10_cutoff = int(total * 0.10)
top5_cutoff = int(total * 0.05)
print(f"Top 1 Rank 1: Score = {df.iloc[0]['Score']}")
print(f"Top 10 Rank 10: Score = {df.iloc[9]['Score']}")
print(f"Top 5% Rank {top5_cutoff}: Score = {df.iloc[top5_cutoff]['Score']}")
print(f"Top 10% Cutoff Rank {top10_cutoff}: Score = {df.iloc[top10_cutoff]['Score']}")

exact = df[df['Score'] == 0.941]
if not exact.empty:
    min_rank = exact.index.min() + 1
    max_rank = exact.index.max() + 1
    print(f"Score 0.941 rank range: {min_rank} to {max_rank}")
    pct = min_rank / total * 100
    print(f"Current Percentile: Top {pct:.2f}% (Rank #{min_rank} / {total})")
