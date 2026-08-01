"""
Fetch exact OpenRouter Model IDs from OpenRouter API
"""
import sys
import requests
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def fetch_top_models():
    url = "https://openrouter.ai/api/v1/models"
    resp = requests.get(url)
    if resp.status_code != 200:
        print("Failed to fetch models")
        return

    data = resp.json()
    models = data.get("data", [])

    keywords = ["claude", "gpt-4o", "deepseek", "grok", "gemini-2", "gemini-1.5-pro"]
    
    matched = []
    for m in models:
        mid = m.get("id", "")
        name = m.get("name", "")
        ctx = m.get("context_length", 0)
        for kw in keywords:
            if kw in mid.lower() or kw in name.lower():
                matched.append((mid, name, ctx))
                break

    # Sort by context length descending
    matched.sort(key=lambda x: x[2], reverse=True)

    print("🔥 TOP FRONTIER MODELS ON OPENROUTER:\n")
    for mid, name, ctx in matched[:20]:
        print(f"  - Model ID: '{mid}' | Name: '{name}' | Context Window: {ctx:,} tokens")

if __name__ == "__main__":
    fetch_top_models()
