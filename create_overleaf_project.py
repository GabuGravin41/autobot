"""
Overleaf Project Creation & Code Paste Test
Rules:
- Focus Chrome window (tab 5 with Overleaf).
- Click 'New project' -> 'Blank Project'.
- Type project name 'Perovskite_Polariton_Memory' and hit Enter.
- Capture screenshot of Overleaf editor.
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
                if "Chrome" in title or "Overleaf" in title or "Grok" in title:
                    found_hwnd = hwnd
                    return False
            return True
        win32gui.EnumWindows(enum_cb, None)

    if found_hwnd:
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
            win32gui.SetForegroundWindow(found_hwnd)
            time.sleep(1.0)
            return True
        except Exception:
            pass
    return False

def main():
    print("🖥️ Focusing Chrome Overleaf tab...")
    focus_chrome()
    time.sleep(1.0)

    # Click 'New project' green button on top left (x=70, y=310)
    print("🟢 Clicking 'New project' button...")
    pyautogui.click(70, 310)
    time.sleep(1.5)

    os.makedirs("tmp", exist_ok=True)
    shot_menu = "tmp/overleaf_menu_opened.png"
    pyautogui.screenshot().save(shot_menu)
    print(f"📸 Overleaf Menu Screenshot: {shot_menu}")

    # Click 'Blank Project' (first item in popup menu at x=100, y=365)
    print("📄 Selecting 'Blank Project'...")
    pyautogui.click(100, 365)
    time.sleep(1.5)

    # Type project name in modal popup and press Enter
    print("✍️ Typing project name 'Perovskite_Polariton_Memory'...")
    pyperclip.copy("Perovskite_Polariton_Memory")
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.press("enter")

    print("⏳ Waiting 6 seconds for project editor to load...")
    time.sleep(6.0)

    shot_editor = "tmp/overleaf_editor_ready.png"
    pyautogui.screenshot().save(shot_editor)
    print(f"📸 Overleaf Editor Ready Screenshot: {shot_editor}")

if __name__ == "__main__":
    main()
