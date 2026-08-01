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
    focus_chrome()

    # Click on the green New project button area (x=70, y=310)
    print("🟢 Clicking New project button...")
    pyautogui.click(70, 310)
    time.sleep(0.5)

    os.makedirs("tmp", exist_ok=True)
    shot1 = "tmp/kb_new_proj_step1.png"
    pyautogui.screenshot().save(shot1)
    print(f"📸 Screenshot 1: {shot1}")

    # Press Down arrow to navigate menu
    pyautogui.press("down")
    time.sleep(0.5)
    pyautogui.press("enter")
    time.sleep(1.5)

    shot2 = "tmp/kb_new_proj_step2.png"
    pyautogui.screenshot().save(shot2)
    print(f"📸 Screenshot 2: {shot2}")

if __name__ == "__main__":
    main()
