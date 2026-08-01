"""
Find which Chrome Profile Directory contains active cookies for grok.com, overleaf.com, chatgpt.com
"""
import os
import sys
import sqlite3
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def scan_profile_cookies():
    user_data = Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data"))
    
    profiles = ["Default", "Profile 1", "Profile 2", "Profile 4", "Profile 7", "Profile 10", "Profile 11"]
    
    print("🔍 Scanning Chrome Profiles for logged-in domains (grok, overleaf, openai, google)...\n")

    for p in profiles:
        cookie_path = user_data / p / "Network" / "Cookies"
        if not cookie_path.exists():
            cookie_path = user_data / p / "Cookies"

        if not cookie_path.exists():
            print(f"  Profile '{p}': No Cookies database found.")
            continue

        # Copy cookie db to temp file to read SQLite without lock
        tmp_db = Path("tmp") / f"cookies_{p}.sqlite"
        tmp_db.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(cookie_path, tmp_db)
            conn = sqlite3.connect(tmp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT host_key FROM cookies WHERE host_key LIKE '%grok%' OR host_key LIKE '%overleaf%' OR host_key LIKE '%openai%' OR host_key LIKE '%google%' OR host_key LIKE '%x.ai%'")
            rows = cursor.fetchall()
            hosts = set(r[0] for r in rows)
            conn.close()

            print(f"  Profile '{p}': Found {len(hosts)} target domain cookies!")
            if hosts:
                print(f"     Domains: {sorted(list(hosts))[:10]}")
        except Exception as e:
            print(f"  Profile '{p}': Could not read cookies ({e})")

if __name__ == "__main__":
    scan_profile_cookies()
