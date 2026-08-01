"""
Live OpenRouter API Model Discovery Test
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_models():
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    candidates = [
        "google/gemini-flash-1.5",
        "google/gemini-pro-1.5",
        "anthropic/claude-3.5-haiku",
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-4o-mini",
    ]

    working_model = None
    for model in candidates:
        print(f"🔍 Testing OpenRouter model: {model}...")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say AUTOBOT_ONLINE"}],
                temperature=0.1,
            )
            content = resp.choices[0].message.content
            print(f"  ✅ SUCCESS with {model}! Response: '{content[:60]}'")
            working_model = model
            break
        except Exception as e:
            print(f"  ❌ {model} failed: {e}")

    if working_model:
        # Update .env with working model
        env_file = os.path.join(os.getcwd(), ".env")
        lines = []
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        new_lines = []
        model_updated = False
        for line in lines:
            if line.startswith("AUTOBOT_LLM_MODEL="):
                new_lines.append(f"AUTOBOT_LLM_MODEL={working_model}\n")
                model_updated = True
            else:
                new_lines.append(line)
        if not model_updated:
            new_lines.append(f"AUTOBOT_LLM_MODEL={working_model}\n")
            
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
            
        print(f"\n🎉 Updated .env with verified working model: {working_model}")

if __name__ == "__main__":
    test_models()
