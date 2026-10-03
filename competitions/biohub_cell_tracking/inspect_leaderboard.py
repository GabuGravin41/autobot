import glob
import pandas as pd

files = sorted(glob.glob('competitions/biohub_cell_tracking/leaderboard/*.csv'))
latest_file = files[-1]
print('Using latest leaderboard file:', latest_file)

df = pd.read_csv(latest_file)
print('Total teams:', len(df))
df = df.sort_values('Score', ascending=False).reset_index(drop=True)
total = len(df)

top1_pct = int(total * 0.01)
top3_pct = int(total * 0.03)
top5_pct = int(total * 0.05)
top10_pct = int(total * 0.10)

print(f"Top 1% Rank {top1_pct}: Score = {df.iloc[top1_pct]['Score']}")
print(f"Top 3% Rank {top3_pct}: Score = {df.iloc[top3_pct]['Score']}")
print(f"Top 5% Rank {top5_pct}: Score = {df.iloc[top5_pct]['Score']}")
print(f"Top 10% Rank {top10_pct}: Score = {df.iloc[top10_pct]['Score']}")

our_rows = df[df['Score'] >= 0.954]
print(f"Teams with score >= 0.954: {len(our_rows)}")
our_exact = df[df['Score'] == 0.954]
if not our_exact.empty:
    print(f"Score 0.954 rank range: {our_exact.index.min() + 1} to {our_exact.index.max() + 1}")

print("\nScores around Rank 100-130:")
for r in range(100, 130, 5):
    print(f"Rank {r}: Score = {df.iloc[r]['Score']}")
