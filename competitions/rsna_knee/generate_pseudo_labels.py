import pandas as pd
from research_nlp_pseudo_labels import parse_report_for_pathologies, TARGETS

df = pd.read_csv('train.csv')
print(f"Loaded {len(df)} studies from train.csv")

parsed_records = []
for idx, row in df.iterrows():
    report = row['Report'] if pd.notna(row['Report']) else ""
    preds = parse_report_for_pathologies(report)
    preds['StudyInstanceUID'] = row['StudyInstanceUID']
    parsed_records.append(preds)
    
out_df = pd.DataFrame(parsed_records)
# Reorder columns
out_df = out_df[['StudyInstanceUID'] + TARGETS]
out_df.to_csv('rsna_knee_pseudo_labels.csv', index=False)
print(f"Saved {len(out_df)} pseudo-labels to rsna_knee_pseudo_labels.csv")
