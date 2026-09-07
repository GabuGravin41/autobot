"""
Test OpenAI GPT-4o, Gemini 2.5 Pro, and DeepSeek Chat on OpenRouter
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_top_frontier():
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    models_to_test = [
        ("openai/gpt-4o", "GPT-4o (128k Context Window)"),
        ("google/gemini-2.5-pro", "Gemini 2.5 Pro (1.05M Context Window)"),
        ("google/gemini-2.5-flash", "Gemini 2.5 Flash (1.05M Context Window)"),
        ("deepseek/deepseek-chat", "DeepSeek V3 (64k Context Window)"),
    ]

    print("👑 TESTING TOP-TIER FRONTIER MODELS ON YOUR OPENROUTER KEY:\n")

    working = []
    for model_id, label in models_to_test:
        print(f"👉 Testing {label} [{model_id}]...")
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are Autobot Frontier Agent."},
                    {"role": "user", "content": "Confirm operational status. Reply 'FRONTIER_ONLINE'."},
                ],
                max_tokens=20,
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            print(f"  ✅ SUCCESS! Response: '{content}'")
            working.append(model_id)
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

    if working:
        selected_model = working[0]
        env_file = os.path.join(os.getcwd(), ".env")
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.startswith("AUTOBOT_LLM_MODEL="):
                new_lines.append(f"AUTOBOT_LLM_MODEL={selected_model}\n")
            else:
                new_lines.append(line)
                
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        print(f"\n🎉 Set primary model in .env to: {selected_model}")

if __name__ == "__main__":
    test_top_frontier()
