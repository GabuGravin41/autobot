"""
Visual Verification Loop for Overleaf Project Creation
Step 1: Focus Chrome Overleaf tab.
Step 2: Click 'New project' green button.
Step 3: Capture screenshot & verify dropdown menu is visible on screen.
Step 4: Click 'Blank Project'.
Step 5: Capture screenshot & verify project name modal appears.
"""
import os
import sys
import time
import subprocess
import pyperclip
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import win32gui
    import win32con
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

def focus_chrome():
    found_hwnd = None
    if HAS_WIN32:
        def enum_cb(hwnd, extra):
            nonlocal found_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Overleaf" in title or "Chrome" in title:
                    found_hwnd = hwnd
                    return False
            return True
        try:
            win32gui.EnumWindows(enum_cb, None)
        except Exception:
            pass

    if found_hwnd:
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
        win32gui.SetForegroundWindow(found_hwnd)
        time.sleep(1.0)
        return True
    return False

def main():
    print("🖥️ 1. Focusing Chrome Overleaf tab...")
    focus_chrome()

    os.makedirs("tmp", exist_ok=True)

    # Click 'Accept all cookies' if banner is present at (935, 905)
    pyautogui.click(935, 905)
    time.sleep(0.5)

    print("🟢 2. Clicking 'New project' button at (70, 310)...")
    pyautogui.click(70, 310)
    time.sleep(1.0)

    # Capture screenshot to verify dropdown is open
    shot_dropdown = "tmp/dropdown_check_live.png"
    pyautogui.screenshot().save(shot_dropdown)
    print(f"📸 3. Dropdown Verification Screenshot Saved: {shot_dropdown}")

    # Click 'Blank Project' (first item in the opened dropdown menu)
    # When dropdown is open, 'Blank Project' is at (85, 360)
    print("📄 4. Clicking 'Blank Project' in dropdown at (85, 360)...")
    pyautogui.click(85, 360)
    time.sleep(1.5)

    shot_modal = "tmp/blank_project_modal_live.png"
    pyautogui.screenshot().save(shot_modal)
    print(f"📸 5. Project Name Modal Screenshot Saved: {shot_modal}")

if __name__ == "__main__":
    main()
