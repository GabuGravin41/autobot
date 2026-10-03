import sqlite3
import shutil
import tempfile
import os
from pathlib import Path

history_paths = list((Path(os.environ['LOCALAPPDATA']) / 'Google' / 'Chrome' / 'User Data').glob('**/History'))
print(f"Found {len(history_paths)} history files")

for hp in history_paths:
    if "System Profile" in str(hp):
        continue
    print(f"\nChecking: {hp}")
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
            tmp_name = tmp.name
        shutil.copy2(hp, tmp_name)
        conn = sqlite3.connect(tmp_name)
        cur = conn.cursor()
        cur.execute("""
            SELECT url, title, datetime(last_visit_time/1000000-11644473600, 'unixepoch') as visited
            FROM urls
            WHERE lower(url) LIKE '%eris%' OR lower(title) LIKE '%eris%'
               OR lower(url) LIKE '%shipd%' OR lower(title) LIKE '%shipd%'
               OR lower(url) LIKE '%benchmark%' OR lower(title) LIKE '%benchmark%'
            ORDER BY last_visit_time DESC LIMIT 25
        """)
        rows = cur.fetchall()
        print(f"Matches ({len(rows)}):")
        for r in rows:
            clean_title = (r[1] or "").encode("ascii", errors="replace").decode()
            print(f"  [{r[2]}] {clean_title} -> {r[0]}")
        conn.close()
    except Exception as e:
        print(f"Error checking {hp}: {e}")
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass

print("\n--- Checking Cookies ---")
cookie_paths = list((Path(os.environ['LOCALAPPDATA']) / 'Google' / 'Chrome' / 'User Data').glob('**/Network/Cookies'))
for cp in cookie_paths:
    if "System Profile" in str(cp):
        continue
    tmp_c = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
            tmp_c = tmp.name
        shutil.copy2(cp, tmp_c)
        conn = sqlite3.connect(tmp_c)
        cur = conn.cursor()
        cur.execute("SELECT host_key, name, path, is_secure, expires_utc FROM cookies WHERE host_key LIKE '%shipd%' OR host_key LIKE '%clerk%'")
        rows = cur.fetchall()
        if rows:
            print(f"Found {len(rows)} cookies in {cp}:")
            for r in rows:
                print(f"  {r[0]} | {r[1]} (secure={r[3]})")
        conn.close()
    except Exception as e:
        print(f"Error checking cookies in {cp}: {e}")
    finally:
        if tmp_c and os.path.exists(tmp_c):
            try:
                os.remove(tmp_c)
            except Exception:
                pass
