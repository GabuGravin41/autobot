"""
Test script to verify Anthropic adapter functionality.
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from autobot.agent.anthropic_adapter import get_anthropic_llm_client

def main():
    print("Testing Anthropic Adapter initialization...")
    client = get_anthropic_llm_client(api_key="sk-ant-test-key-mock")
    if client and hasattr(client, "chat") and hasattr(client.chat, "completions"):
        print("[SUCCESS] AnthropicOpenAIAdapter successfully initialized and exposes chat.completions.create!")
    else:
        print("[FAIL] Adapter initialization failed.")

if __name__ == "__main__":
    main()
