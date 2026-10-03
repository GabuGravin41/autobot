import sys, os, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

comps = {
    'traffic': '2026-ieee-big-data-traffic-flow-bench',
    'biohub': 'biohub-cell-tracking-during-development',
    'emulation': 'ieee-bigdata-cup-2026-ai-emulation-challenge',
    'arc': 'arc-prize-2026-arc-agi-3'
}

for key, c in comps.items():
    out_dir = 'competitions/lb_' + key
    zip_files = [f for f in os.listdir(out_dir) if f.endswith('.zip')]
    if zip_files:
        df = pd.read_csv(os.path.join(out_dir, zip_files[0]))
        total_teams = len(df)
        top10_n = max(1, int(total_teams * 0.10))
        top10_score = df.iloc[top10_n - 1]['Score']
        my_row = df[(tr(r['TeamName'])-strip().lower() == 'dalton omondi' for i, r in df.iterrows()]
        print('=' * 50)
        print('COMPETITION:', c)
        print('=' * 50)
        print('Total Teams:', total_teams, '| Top 10% Cutoff: Rank #', top10_n, 'Score:', top10_score)
        found = False
        for i, r in df.iterrows():
            s = str(r['TeamName']).strip().lower()
            if s == 'dalton omondi':
                rank = i + 1
                score = r['Score']
                pct = (rank / total_teams) * 100
                print('>>> EXACT REAL STATUS: Rank #', rank , '/', top10_teams, f'(Top {pct:.1}%) | Score:', score)
                found = True
                break
        if not found:
            print('>>> Team "Dalton Omondi" NOT found on this leaderboard.')
        print()
