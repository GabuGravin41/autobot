import os
import sys
import time
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("👇 Clicking Chrome icon on Windows Taskbar at (710, 965)...")
    pyautogui.click(710, 965)
    time.sleep(1.5)

    os.makedirs("tmp", exist_ok=True)
    path1 = "tmp/after_taskbar_click.png"
    pyautogui.screenshot().save(path1)
    print(f"📸 Saved screenshot 1: {path1}")

    print("🎯 Clicking Person 1 profile card at (290, 720)...")
    pyautogui.click(290, 720)
    time.sleep(4.0)

    path2 = "tmp/after_profile_click.png"
    pyautogui.screenshot().save(path2)
    print(f"📸 Saved screenshot 2: {path2}")

if __name__ == "__main__":
    main()
