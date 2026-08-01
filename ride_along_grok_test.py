"""
Ride-Along Chrome Automation for Grok & Overleaf
Rules:
- NEVER kill active Chrome processes.
- If Chrome is open, focus and ride along on the active session.
- If Chrome is closed, launch standard Chrome with Default profile.
- Visually verify window focus and element location before sending inputs.
"""
import os
import sys
import time
import subprocess
import pyperclip
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

try:
    import win32gui
    import win32con
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

def focus_or_launch_chrome(url="https://grok.com"):
    """
    Focuses an existing open Chrome window, or launches standard Chrome if closed.
    NO TASKKILL. ALWAYS RIDE ALONG ON EXISTING CHROME.
    """
    found_hwnd = None
    if HAS_WIN32:
        def enum_cb(hwnd, extra):
            nonlocal found_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Chrome" in title or "Grok" in title or "Overleaf" in title:
                    found_hwnd = hwnd
                    return False
            return True
        win32gui.EnumWindows(enum_cb, None)

    if found_hwnd:
        print(f"🎯 Chrome window found (hwnd={found_hwnd}). Bringing to front...")
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
            win32gui.SetForegroundWindow(found_hwnd)
            time.sleep(1.5)
            return True
        except Exception as e:
            print(f"⚠️ win32 focus warning: {e}")

    # If Chrome window not found, launch Chrome cleanly (WITHOUT killing existing instances)
    print(f"🚀 Chrome window not focused. Launching native Chrome to {url}...")
    cmd = f'start "" "{CHROME_EXE}" --profile-directory="Default" "{url}"'
    os.system(cmd)
    time.sleep(5.0)

    if HAS_WIN32:
        def enum_cb2(hwnd, extra):
            nonlocal found_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Chrome" in title or "Grok" in title or "Overleaf" in title:
                    found_hwnd = hwnd
                    return False
            return True
        win32gui.EnumWindows(enum_cb2, None)
        if found_hwnd:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
            win32gui.SetForegroundWindow(found_hwnd)
            time.sleep(1.0)
    return True

def main():
    print("=========================================")
    print("🌐 Ride-Along Chrome Execution Starting...")
    print("=========================================")

    focus_or_launch_chrome("https://grok.com")

    # Dismiss any leftover banners
    pyautogui.press("escape")
    time.sleep(0.5)

    os.makedirs("tmp", exist_ok=True)
    shot1 = "tmp/ride_along_step1.png"
    pyautogui.screenshot().save(shot1)
    print(f"📸 Step 1 Screenshot Saved: {shot1}")

    # Navigate to grok.com explicitly via Ctrl+L -> Ctrl+V if not already on grok.com
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.3)
    pyperclip.copy("https://grok.com")
    pyautogui.hotkey("ctrl", "v")
    pyautogui.press("enter")
    time.sleep(4.0)

    shot2 = "tmp/ride_along_grok_loaded.png"
    pyautogui.screenshot().save(shot2)
    print(f"📸 Grok Page Loaded Screenshot: {shot2}")

if __name__ == "__main__":
    main()
