"""
Launch Chrome normally, take a screenshot, and report the state.
"""
import asyncio
import os
import sys
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def take_screenshot():
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/chrome_launch_check.png"
    img = pyautogui.screenshot()
    img.save(path)
    print(f"  📸 Screenshot saved to {path} ({img.size[0]}x{img.size[1]})")

async def main():
    print("🚀 Launching Chrome normally...")
    os.system(f'start "" "{CHROME_EXE}"')
    await asyncio.sleep(4.0)  # Wait for Chrome to open
    take_screenshot()
    print("✅ Chrome launched. Screenshot taken. Please check tmp/chrome_launch_check.png")

if __name__ == "__main__":
    asyncio.run(main())
