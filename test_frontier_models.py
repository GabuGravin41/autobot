"""
OpenRouter Frontier Model Verification Script
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def discover_frontier_models():
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    frontier_candidates = [
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat",
        "openai/gpt-4o",
        "x-ai/grok-2-1212",
        "google/gemini-2.0-flash-001",
    ]

    print("🚀 Discovering Available Frontier Models on OpenRouter...\n")

    working_frontier_models = []
    for model in frontier_candidates:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Respond with OK"}],
                max_tokens=10,
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            print(f"  ✅ AVAILABLE: {model} (Response: '{content}')")
            working_frontier_models.append(model)
        except Exception as e:
            print(f"  ❌ UNAVAILABLE: {model} -> {e}")

    print(f"\nFound {len(working_frontier_models)} working frontier models!")
    return working_frontier_models

if __name__ == "__main__":
    discover_frontier_models()
