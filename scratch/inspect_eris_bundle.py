import urllib.request
import re
import json

url = 'https://shipd.ai/quests/eris/assets/_problemId-DrEYokec.js'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode('utf-8')
        print(f"Length of bundle: {len(content)}")
        
        # Look for convex queries or mutations
        fn_refs = re.findall(r'[a-zA-Z0-9_]+\s*:\s*\{\s*path\s*:\s*["\']([^"\']+)["\']', content)
        print("Explicit paths:", fn_refs)
        
        # Look for any function calls like useQuery or query
        queries = re.findall(r'useQuery\(([^,)]+)', content)
        print("useQuery calls:", list(set(queries))[:10])
        
        # Print strings in bundle that look like endpoints or actions
        strings = re.findall(r'["\']([a-zA-Z0-9_\-\.\/]{4,50})["\']', content)
        interesting = [s for s in set(strings) if any(x in s.lower() for x in ['benchmark', 'rubric', 'solution', 'convex', 'submit', 'gate'])]
        print("Interesting strings:", sorted(interesting)[:30])
except Exception as e:
    print("Error:", e)
