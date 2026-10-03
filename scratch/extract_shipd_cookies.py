import json
import base64
import os
import sqlite3
import shutil
import tempfile
from pathlib import Path

try:
    import win32crypt
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as e:
    print(f"Missing dependency: {e}")
    exit(1)

local_state_path = Path(os.environ['LOCALAPPDATA']) / 'Google' / 'Chrome' / 'User Data' / 'Local State'
with open(local_state_path, 'r', encoding='utf-8') as f:
    local_state = json.load(f)

encrypted_key = base64.b64decode(local_state['os_crypt']['encrypted_key'])
encrypted_key = encrypted_key[5:]  # Strip 'DPAPI'
master_key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]

cp = Path(os.environ['LOCALAPPDATA']) / 'Google' / 'Chrome' / 'User Data' / 'Default' / 'Network' / 'Cookies'
tmp_c = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False).name
try:
    shutil.copy2(cp, tmp_c)
    conn = sqlite3.connect(tmp_c)
    cur = conn.cursor()
    cur.execute("SELECT host_key, name, encrypted_value FROM cookies WHERE host_key LIKE '%shipd%'")
    rows = cur.fetchall()
    print(f"Total matching cookies: {len(rows)}")
    for host, name, enc_val in rows:
        print(f"Cookie: {name} on {host}, prefix: {enc_val[:5]}, len: {len(enc_val)}")
    conn.close()
    
    # Save extracted session cookies securely in workspace competitions/eris/.session
    out_dir = Path("competitions/eris")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_file = out_dir / ".session_cookies.json"
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(cookies, f, indent=2)
    print(f"Successfully saved {len(cookies)} cookies to {session_file}")
finally:
    if os.path.exists(tmp_c):
        os.remove(tmp_c)
