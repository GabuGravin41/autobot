"""
Test Claude 3.5 Sonnet Model IDs on OpenRouter
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_claude():
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    claude_variants = [
        "anthropic/claude-3.5-sonnet:beta",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-opus",
        "anthropic/claude-3.5-haiku",
    ]

    print("🧠 Testing Claude Model IDs on OpenRouter...\n")
    for model in claude_variants:
        print(f"👉 Testing '{model}'...")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Respond with CLAUDE_ONLINE"}],
                max_tokens=10,
                temperature=0.1,
            )
            content = resp.choices[0].message.content
            print(f"  ✅ SUCCESS for '{model}'! Response: '{content}'")
        except Exception as e:
            print(f"  ❌ FAILED for '{model}': {e}")

if __name__ == "__main__":
    test_claude()
