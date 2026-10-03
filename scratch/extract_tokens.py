import json
import base64
import os
import re
from pathlib import Path

def decode_jwt(raw):
    try:
        parts = raw.split(b'.')
        if len(parts) < 2:
            return None, None
        def pad(b):
            return b + b'=' * ((4 - len(b) % 4) % 4)
        header = json.loads(base64.urlsafe_b64decode(pad(parts[0])))
        payload = json.loads(base64.urlsafe_b64decode(pad(parts[1])))
        return header, payload
    except Exception:
        return None, None

ldb_dir = Path(os.environ['LOCALAPPDATA']) / 'Google' / 'Chrome' / 'User Data' / 'Default' / 'Local Storage' / 'leveldb'
matched = []
for p in ldb_dir.glob('*'):
    if not p.is_file():
        continue
    try:
        data = p.read_bytes()
    except Exception:
        continue
    tokens = re.findall(rb'ey[a-zA-Z0-9_\-]{20,}\.ey[a-zA-Z0-9_\-]{20,}\.[a-zA-Z0-9_\-]{20,}', data)
    for t in set(tokens):
        h, pl = decode_jwt(t)
        if pl and any(k in str(pl).lower() for k in ['shipd', 'clerk', 'eris', 'dalton']):
            matched.append((p.name, pl, t.decode('ascii')))
            print(f"Match in {p.name}: iss={pl.get('iss')}, sub={pl.get('sub')}, azp={pl.get('azp')}, exp={pl.get('exp')}")

print(f"Total matched tokens: {len(matched)}")
if matched:
    token_str = matched[0][2]
    out_file = Path("competitions/eris/.token.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(token_str)
    print(f"Wrote token to {out_file}")
