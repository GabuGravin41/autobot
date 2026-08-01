"""
Find which Profile contains Overleaf cookies
"""
import os
import sys
import sqlite3
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def find_overleaf():
    user_data = Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data"))
    profiles = ["Default", "Profile 1", "Profile 2", "Profile 4", "Profile 7", "Profile 10", "Profile 11"]
    
    print("🔍 Searching for 'overleaf' in all Chrome Profile Cookies...\n")
    for p in profiles:
        cookie_path = user_data / p / "Network" / "Cookies"
        if not cookie_path.exists():
            cookie_path = user_data / p / "Cookies"

        if not cookie_path.exists():
            continue

        tmp_db = Path("tmp") / f"cookies_overleaf_{p}.sqlite"
        tmp_db.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(cookie_path, tmp_db)
            conn = sqlite3.connect(tmp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT host_key FROM cookies WHERE host_key LIKE '%overleaf%'")
            rows = cursor.fetchall()
            conn.close()

            if rows:
                print(f"  ✅ Overleaf Cookies FOUND in Profile: '{p}' (Count: {len(rows)})")
            else:
                print(f"  - Profile '{p}': No Overleaf cookies.")
        except Exception as e:
            print(f"  - Profile '{p}': Error reading ({e})")

if __name__ == "__main__":
    find_overleaf()
