"""
Phase 1 Test — Keyboard navigation on Chrome profile picker.
"""
import asyncio
import os
import subprocess
import sys
import pyautogui

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
    print("🎯 KEYBOARD NAVIGATION TEST — CHROME PROFILE PICKER")
    print("=" * 60)

    # 1. Kill existing Chrome
    print("\n🧹 Step 1: Killing existing Chrome...")
    subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
    await asyncio.sleep(2.0)

    # 2. Launch Chrome normally
    print("\n🚀 Step 2: Launching Chrome normally...")
    os.system(f'start "" "{CHROME_EXE}"')
    await asyncio.sleep(4.0)

    take_screenshot("kb_test_1_launched")

    # 3. Focus and navigate using keyboard
    # First, click in the center of the window to ensure it has focus
    print("\n🖱️ Step 3: Clicking center of screen to focus window...")
    pyautogui.click(960, 540)
    await asyncio.sleep(0.5)

    # Press Tab to focus the first profile card (Dalton)
    print("⌨️ Pressing Tab to focus first profile card...")
    pyautogui.press('tab')
    await asyncio.sleep(0.5)
    take_screenshot("kb_test_2_after_tab")

    # Press Down Arrow to move focus to the profile card below it (Person 1)
    print("⌨️ Pressing Down Arrow to select Person 1...")
    pyautogui.press('down')
    await asyncio.sleep(0.5)
    take_screenshot("kb_test_3_after_down")

    # Press Enter to open the profile
    print("⌨️ Pressing Enter to open the profile...")
    pyautogui.press('enter')
    await asyncio.sleep(5.0)  # Wait for Chrome to open with selected profile

    take_screenshot("kb_test_4_after_enter")

    print("\nDone. Please check the screenshots in tmp/ to see if the profile opened.")

if __name__ == "__main__":
    asyncio.run(main())
