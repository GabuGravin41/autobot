import re
from pathlib import Path

src_p = Path("competitions/umud_muscle_architecture/research_lamhuy/variational-ultrasound-kinematics-ensemble.py")
with open(src_p, "r", encoding="utf-8") as f:
    code = f.read()

# Replace the submission save cell to be robust
old_export_prefix = "# Export Official Comma-Separated Submission"
target_idx = code.find(old_export_prefix)

if target_idx != -1:
    end_of_block = code.find("print(\"+----------------------------------------------------------------+\")\nprint(\"| CLINICAL BIOMECHANICAL KINEMATICS SCORECARD", target_idx)
    if end_of_block != -1:
        new_export = """# Export Official Submission with Dynamic Delimiter Matching
def detect_delimiter():
    for p in [Path('/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/sample_submission.csv'),
              Path('/kaggle/input/umud-challenge-muscle-architecture-in-ultrasound-data/sample_submission.csv'),
              Path('sample_submission.csv')]:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                first = f.readline()
                return ';' if ';' in first else ','
    return ';'

delim = detect_delimiter()
print(f'Detected official sample_submission delimiter: {delim!r}')

submission_path = Path('/kaggle/working/submission.csv')
df_opt[['image_id', 'pa_deg', 'fl_mm', 'mt_mm']].to_csv(submission_path, sep=delim, index=False, float_format='%.4f')
df_opt[['image_id', 'pa_deg', 'fl_mm', 'mt_mm']].to_csv('submission.csv', sep=delim, index=False, float_format='%.4f')

# Verification checks
assert submission_path.exists(), 'Submission file was not written.'
df_sub = pd.read_csv(submission_path, sep=delim)
assert len(df_sub) == 309, f'Expected 309 rows, found {len(df_sub)}'
assert list(df_sub.columns) == ['image_id', 'pa_deg', 'fl_mm', 'mt_mm'], 'Invalid submission columns.'
assert not df_sub.isnull().values.any(), 'Submission contains null or NaN values.'

"""
        code = code[:target_idx] + new_export + code[end_of_block:]
        print("Replaced export block successfully!")
    else:
        print("Could not find end_of_block!")
else:
    print("Could not find target_idx!")

dst_p = Path("competitions/umud_muscle_architecture/exp2_variational_kinematics_unet/main.py")
with open(dst_p, "w", encoding="utf-8") as f:
    f.write(code)

print("Wrote UMUD Exp 2 main.py successfully!")
