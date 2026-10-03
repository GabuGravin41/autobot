import pandas as pd
import numpy as np

sub_exp5 = pd.read_csv("scratch/exp5_output/submission.csv")
sub_exp1 = pd.read_csv("competitions/rsna_knee/output_exp1/submission_0939_parent_exact.csv")

targets = [c for c in sub_exp5.columns if c != "StudyInstanceUID"]

print("=" * 80)
print("RSNA KNEE SUBMISSION COMPARISON AUDIT: EXP 5 vs PREVIOUS SOTA (EXP 1/3)")
print("=" * 80)

print(f"{'Target':<18} | {'Exp1 (0.939) Ranks':<22} | {'Exp5 (0.501) Ranks':<22} | {'Rank Match?':<12}")
print("-" * 80)

matches = 0
correlations = []
for t in targets:
    r_exp1 = sub_exp1[t].rank().tolist()
    r_exp5 = sub_exp5[t].rank().tolist()
    match = (r_exp1 == r_exp5)
    if match:
        matches += 1
    corr = np.corrcoef(sub_exp1[t], sub_exp5[t])[0, 1]
    correlations.append(corr)
    print(f"{t:<18} | {str(r_exp1):<22} | {str(r_exp5):<22} | {str(match):<12}")

print("-" * 80)
print(f"Total Rank Alignment: {matches}/{len(targets)} ({matches/len(targets)*100:.1f}%)")
print(f"Mean Pearson Correlation across targets: {np.nanmean(correlations):.3f}")
print("=" * 80)

print("\n--- DETAILED VALUE DISTRIBUTION ANALYSIS ---")
print("Exp 1 (0.939 Parent Exact):")
print(sub_exp1[targets].describe().round(3).T[['mean', 'std', 'min', 'max']])

print("\nExp 5 (Knee-ACT Architecture):")
print(sub_exp5[targets].describe().round(3).T[['mean', 'std', 'min', 'max']])
