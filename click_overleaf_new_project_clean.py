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
                if "Chrome" in title or "Overleaf" in title:
                    found_hwnd = hwnd
                    return False
            return True
        win32gui.EnumWindows(enum_cb, None)

    if found_hwnd:
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
        win32gui.SetForegroundWindow(found_hwnd)
        time.sleep(1.0)
        return True
    return False

def main():
    print("🖥️ Focusing Overleaf window...")
    focus_chrome()

    print("🍪 Clicking 'Accept all cookies' banner on bottom right...")
    pyautogui.click(930, 905)
    time.sleep(1.0)

    print("🟢 Clicking 'New project' green button...")
    pyautogui.click(70, 310)
    time.sleep(1.0)

    os.makedirs("tmp", exist_ok=True)
    shot_menu = "tmp/overleaf_dropdown_visible.png"
    pyautogui.screenshot().save(shot_menu)
    print(f"📸 Saved dropdown screenshot: {shot_menu}")

    print("📄 Clicking 'Blank Project'...")
    pyautogui.click(80, 360)
    time.sleep(1.5)

    print("✍️ Typing project name and hitting Enter...")
    pyperclip.copy("Perovskite_Polariton_Memory")
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.press("enter")

    print("⏳ Waiting 8 seconds for Overleaf LaTeX editor to load...")
    time.sleep(8.0)

    shot_editor = "tmp/overleaf_editor_opened_clean.png"
    pyautogui.screenshot().save(shot_editor)
    print(f"📸 Saved editor screenshot: {shot_editor}")

if __name__ == "__main__":
    main()
