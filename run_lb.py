import sys, subprocess, os, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

comps = {
    'traffic': '2026-ieee-big-data-traffic-flow-bench',
    'biohub': 'biohub-cell-tracking-during-development',
    'emulation': 'ieee-bigdata-cup-2026-ai-emulation-challenge',
    'arc': 'arc-prize-2026-arc-agi-3'
}

for key, c in comps.items():
    print('=' * 50)
    print('COMPETITION:', c)
    print('=' * 50)
    out_dir = 'competitions/lb_' + key
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run('python -X utf8 -m kaggle competitions leaderboard -c ' + c + ' --download -p ' + out_dir, shell=True, capture_output=True)
    zip_files = [f for f in os.listdir(out_dir) if f.endswith('.zip')]
    if zip_files:
        df = pd.read_csv(os.path.join(out_dir, zip_files[0]))
        total_teams = len(df)
        top10_n = max(1, int(total_teams * 0.10))
        top10_score = df.iloc[top10_n - 1]['Score']
        print('Total Teams:', total_teams, '| Top 10% cutoff rank: #', top10_n, 'Score:', top10_score)
        print('Top 5 Teams:')
        for i, r in df.head(5).iterrows():
            print('  #', i + 1, r['TeamName'], '->', r['Score'])
        found = False
        for i, r in df.iterrows():
            s = str(r['TeamName']).lower()
            if 'dalton' in s or 'gabriel' in s or 'autobot' in s:
                print('  >>> OUR TEAM:', r['TeamName'], '| Rank: #', i + 1, '/', total_teams, '| Score:', r['Score'])
                found = True
                break
        if not found:
            print('  >>> Our team not found on leaderboard.')
    print()
