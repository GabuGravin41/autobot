import os
import sys
import time
import subprocess
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
    print("🖥️ Focusing Overleaf tab...")
    focus_chrome()

    print("🟢 1. Clicking 'Accept all cookies' at (935, 905)...")
    pyautogui.click(935, 905)
    time.sleep(1.2)

    print("🟢 2. Clicking 'New project' green button at (70, 310)...")
    pyautogui.click(70, 310)
    time.sleep(1.2)

    os.makedirs("tmp", exist_ok=True)
    shot_path = "tmp/dropdown_SUCCESS_verified.png"
    pyautogui.screenshot().save(shot_path)
    print(f"📸 3. Saved Dropdown Screenshot: {shot_path}")

if __name__ == "__main__":
    main()
