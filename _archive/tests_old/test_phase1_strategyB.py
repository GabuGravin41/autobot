"""
Phase 1 — Strategy B v2: Fixed coordinate estimation.

The LLM correctly identifies profiles but gives wrong coordinates.
Fix: Tell the LLM the exact screen dimensions (1920x1080) and ask for
absolute pixel coordinates based on the full screenshot.
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

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def take_screenshot(filename="tmp/desktop_screenshot.png"):
    import pyautogui
    os.makedirs("tmp", exist_ok=True)
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    w, h = screenshot.size
    print(f"  📸 Screenshot saved: {filename} ({w}x{h})")
    return filename, w, h


def screenshot_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_llm_sync(screenshot_b64, question):
    """Synchronous LLM call with screenshot."""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "high",
                }},
            ],
        }],
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content


async def main():
    import pyautogui

    print("=" * 70)
    print("🧪 PHASE 1 — STRATEGY B v2: Fixed Coordinates")
    print("=" * 70)

    # Step 1: Kill Chrome
    print("\n🧹 Step 1: Killing existing Chrome processes...")
    subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
    await asyncio.sleep(2.0)

    # Step 2: Launch Chrome normally
    print("\n🚀 Step 2: Launching Chrome normally...")
    os.system(f'start "" "{CHROME_EXE}"')
    await asyncio.sleep(4.0)

    # Step 3: Screenshot
    print("\n📸 Step 3: Taking screenshot...")
    ss_path, screen_w, screen_h = take_screenshot("tmp/phase1b_v2_step3.png")
    ss_b64 = screenshot_to_base64(ss_path)

    # Step 4: Ask LLM with EXACT screen dimensions
    print("\n🧠 Step 4: Asking LLM to find 'Person 1' profile tile...")
    analysis = ask_llm_sync(ss_b64, f"""This is a screenshot of a {screen_w}x{screen_h} pixel screen.

The Chrome "Who's using Chrome?" profile picker is visible.

I need you to find the profile tile for "Person 1" (Dalton O). 
It has an avatar with a notebook/pencil and Euler's identity (e^iπ + 1 = 0).

CRITICAL: The screenshot is EXACTLY {screen_w}x{screen_h} pixels. 
The coordinates you return must be absolute pixel coordinates within this 
{screen_w}x{screen_h} image. The top-left corner is (0,0). The bottom-right is ({screen_w-1},{screen_h-1}).

Look at where the "Person 1" tile is positioned on screen. 
It should be in the lower row of the profile grid.
Return the CENTER of the "Person 1" tile as x,y coordinates.

Respond ONLY with JSON, no markdown:
{{"found": true, "x": <int>, "y": <int>, "confidence": "high/medium/low"}}""")

    print(f"  LLM response: {analysis}")

    # Parse coordinates
    try:
        json_text = analysis.strip()
        if "```" in json_text:
            json_text = json_text.split("```json")[-1].split("```")[0] if "```json" in json_text else json_text.split("```")[1].split("```")[0]
        result = json.loads(json_text.strip())
        x, y = result["x"], result["y"]
        print(f"  📍 LLM says Person 1 is at ({x}, {y}), confidence: {result.get('confidence', '?')}")
    except Exception as e:
        print(f"  ⚠️ Parse failed: {e}. Using hardcoded fallback coordinates...")
        # From visual inspection of the screenshot: Person 1 is row 2, col 1
        # The profile grid is centered around x=725, each tile ~200px wide, ~150px tall
        # Row 1 y≈410, Row 2 y≈580. Col 1 x≈422, Col 2 x≈622, Col 3 x≈825, Col 4 x≈1028
        x, y = 422, 580

    # Sanity check — coords should be within screen bounds and in the center area
    if x < 150 or x > 1770 or y < 100 or y > 980:
        print(f"  ⚠️ Coordinates ({x},{y}) look suspicious. Using hardcoded fallback...")
        x, y = 422, 580

    # Step 5: Click the profile
    print(f"\n🖱️ Step 5: Clicking Person 1 profile at ({x}, {y})...")
    pyautogui.click(x, y)
    await asyncio.sleep(4.0)

    # Step 6: Verify Chrome opened
    print("\n📸 Step 6: Screenshot after clicking profile...")
    ss_path2, _, _ = take_screenshot("tmp/phase1b_v2_step6_after_click.png")
    ss_b64_2 = screenshot_to_base64(ss_path2)

    verify = ask_llm_sync(ss_b64_2, """Is Chrome now open with a browser window? 
What page is loaded? Is this a regular Chrome window or still the profile picker?
Respond briefly in 1-2 sentences.""")
    print(f"  🧠 Verification: {verify}")

    # Step 7: Navigate to grok.com
    print("\n🧭 Step 7: Navigating to grok.com...")
    await asyncio.sleep(1.0)
    pyautogui.hotkey('ctrl', 'l')
    await asyncio.sleep(0.5)
    # Use pyperclip to paste URL (more reliable than typewrite for URLs)
    import pyperclip
    pyperclip.copy('https://grok.com')
    pyautogui.hotkey('ctrl', 'v')
    await asyncio.sleep(0.3)
    pyautogui.press('enter')
    await asyncio.sleep(5.0)

    # Step 8: Final verification
    print("\n📸 Step 8: Final screenshot — checking Grok login status...")
    ss_path3, _, _ = take_screenshot("tmp/phase1b_v2_step8_grok.png")
    ss_b64_3 = screenshot_to_base64(ss_path3)

    final = ask_llm_sync(ss_b64_3, """Look at this screenshot. 
1. Is the Grok website (grok.com) loaded?
2. Am I LOGGED IN? (I should see a chat input box, not a "Sign in" button)
3. What exactly do you see on screen?

Respond with JSON:
{"grok_loaded": true/false, "logged_in": true/false, "description": "what you see"}""")
    print(f"  🧠 Final check: {final}")

    print("\n" + "=" * 70)
    print("Phase 1 Strategy B v2 complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
