"""
Phase 1 — Simple: Launch Chrome, verify with screenshot, navigate with keyboard.
No Playwright. No CDP. Just pyautogui + screenshots + LLM vision.
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pyautogui
import pyperclip
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# LLM client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def screenshot(name="screenshot"):
    """Take screenshot, return base64."""
    os.makedirs("tmp", exist_ok=True)
    path = f"tmp/{name}.png"
    img = pyautogui.screenshot()
    img.save(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    print(f"  📸 {path} ({img.size[0]}x{img.size[1]})")
    return b64


def ask_vision(b64, question):
    """Send screenshot to LLM, get text answer."""
    resp = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64}", "detail": "high"
            }},
        ]}],
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content


async def main():
    print("=" * 60)
    print("PHASE 1: Launch Chrome → Verify → Navigate to Grok")
    print("=" * 60)

    # 1. Just launch Chrome. That's it.
    print("\n1. Launching Chrome...")
    os.system(f'start "" "{CHROME_EXE}"')
    await asyncio.sleep(3.0)

    # 2. Screenshot — what's on screen?
    print("\n2. What's on screen?")
    b64 = screenshot("p1_after_launch")
    state = ask_vision(b64, 
        "What do you see? Is Chrome open? Is there a profile picker? "
        "Or is a browser window with a webpage visible? Answer in 1-2 sentences.")
    print(f"  → {state}")

    # 3. If profile picker, click the right one. If browser is open, skip to navigation.
    if "profile" in state.lower() or "who" in state.lower():
        print("\n3. Profile picker detected. Finding Person 1...")
        coords = ask_vision(b64,
            f"The screen is 1920x1080 pixels. This screenshot IS the full screen. "
            f"Find the profile tile labeled 'Person 1' (it has a notebook avatar). "
            f"Return ONLY the x,y pixel coordinates of its CENTER as JSON: "
            f'{{\"x\": <number>, \"y\": <number>}}')
        print(f"  → {coords}")
        try:
            c = json.loads(coords.strip().strip('`').replace('json','').strip())
            print(f"  Clicking ({c['x']}, {c['y']})...")
            pyautogui.click(c['x'], c['y'])
            await asyncio.sleep(3.0)
        except:
            # Fallback: Person 1 is row 2, col 1 based on observed screenshot
            print("  Parse failed, using fallback (422, 580)...")
            pyautogui.click(422, 580)
            await asyncio.sleep(3.0)
    else:
        print("\n3. Chrome already open — skipping profile picker.")

    # 4. Verify Chrome is now open with a browser window
    print("\n4. Verifying browser window...")
    b64 = screenshot("p1_browser_open")
    state2 = ask_vision(b64, "Is a Chrome browser window open with a webpage or new tab? Yes or no, and what do you see? 1 sentence.")
    print(f"  → {state2}")

    # 5. Navigate to grok.com
    print("\n5. Navigating to grok.com...")
    pyautogui.hotkey('ctrl', 'l')  # Focus address bar
    await asyncio.sleep(0.5)
    pyperclip.copy('https://grok.com')
    pyautogui.hotkey('ctrl', 'v')
    await asyncio.sleep(0.3)
    pyautogui.press('enter')
    await asyncio.sleep(5.0)

    # 6. Final check — are we on Grok? Are we logged in?
    print("\n6. Checking Grok status...")
    b64 = screenshot("p1_grok_check")
    final = ask_vision(b64,
        "Is this the Grok website? Am I logged in (I see a chat input box) "
        "or logged out (I see a Sign in button)? Answer: logged_in / logged_out / not_grok")
    print(f"  → {final}")

    if "logged_in" in final.lower() or "chat" in final.lower() or "input" in final.lower():
        print("\n✅ PHASE 1 PASSED — Chrome open, logged in to Grok!")
    elif "logged_out" in final.lower() or "sign in" in final.lower():
        print("\n⚠️ PHASE 1 PARTIAL — Grok loaded but not logged in.")
    else:
        print(f"\n⚠️ PHASE 1 UNCLEAR — check tmp/p1_grok_check.png")

    print("\nDone. All screenshots in tmp/")


if __name__ == "__main__":
    asyncio.run(main())
