import asyncio
import os
import sys
import time
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("🎯 Focusing Chrome window and clicking Person 1 profile at (290, 720)...")
    # Click once to focus window
    pyautogui.click(290, 720)
    time.sleep(0.3)
    # Click again to open profile
    pyautogui.click(290, 720)
    time.sleep(4.0)

    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/after_person1_click.png"
    img = pyautogui.screenshot()
    img.save(out_path)
    print(f"📸 Saved screenshot after click: {out_path}")

if __name__ == "__main__":
    main()
