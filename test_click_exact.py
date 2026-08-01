"""
Refined Phase 1 Test — Click exactly on Person 1 card using estimated center (290, 720).
"""
import asyncio
import os
import subprocess
import sys
import pyautogui
import pyperclip

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def take_screenshot(name):
    os.makedirs("tmp", exist_ok=True)
    path = f"tmp/{name}.png"
    img = pyautogui.screenshot()
    img.save(path)
    print(f"  📸 Saved {path} ({img.size[0]}x{img.size[1]})")


async def main():
    print("=" * 60)
    print("🎯 REFINED CLICK TEST — TARGETING PERSON 1")
    print("=" * 60)

    # 1. Kill existing Chrome
    print("\n🧹 Step 1: Killing existing Chrome...")
    subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
    await asyncio.sleep(2.0)

    # 2. Launch Chrome
    print("\n🚀 Step 2: Launching Chrome normally...")
    os.system(f'start "" "{CHROME_EXE}"')
    await asyncio.sleep(4.0)

    take_screenshot("click_test_1_launched")

    # 3. Click exactly at (290, 720) — expected center of Person 1 card
    print("\n🖱️ Step 3: Clicking (290, 720)...")
    pyautogui.click(290, 720)
    await asyncio.sleep(4.0)

    take_screenshot("click_test_2_after_click")

    # 4. Navigate to grok.com
    print("\n🧭 Step 4: Navigating to grok.com...")
    pyautogui.hotkey('ctrl', 'l')
    await asyncio.sleep(0.5)
    pyperclip.copy('https://grok.com')
    pyautogui.hotkey('ctrl', 'v')
    await asyncio.sleep(0.3)
    pyautogui.press('enter')
    await asyncio.sleep(6.0)

    take_screenshot("click_test_3_grok")

    # 5. Check if we succeeded
    # We will let the LLM check the screenshot in the next step.
    print("\nDone. Please check the screenshots in tmp/ to see if Grok loaded successfully.")


if __name__ == "__main__":
    asyncio.run(main())
