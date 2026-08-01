import sys
import requests
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

url = "https://openrouter.ai/api/v1/models"
resp = requests.get(url)
models = resp.json().get("data", [])

print(f"Total models available on OpenRouter: {len(models)}")
print("\nSample top model IDs:")
for m in models[:30]:
    print(f"  - {m['id']} (Context: {m.get('context_length', 'N/A')})")
