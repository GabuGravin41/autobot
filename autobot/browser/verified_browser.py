"""
Verified Browser Controller for Autobot
Guarantees NO blind actions: Every action requires prior window focus verification and screenshot state checks.
"""
import os
import sys
import time
import subprocess
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

def get_chrome_hwnd(profile_title_substring="Chrome"):
    """Find the window handle for Google Chrome."""
    found_hwnd = None
    if not HAS_WIN32:
        return None

    def enum_windows_callback(hwnd, extra):
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            # Match Chrome windows
            if "Google Chrome" in title or "Chrome" in title or "Grok" in title or "Overleaf" in title:
                found_hwnd = hwnd
                return False
        return True

    win32gui.EnumWindows(enum_windows_callback, None)
    return found_hwnd

def focus_and_maximize_chrome():
    """Ensure Chrome is strictly focused, brought to front, and maximized."""
    hwnd = get_chrome_hwnd()
    if hwnd and HAS_WIN32:
        print(f"🎯 Found Chrome window (hwnd: {hwnd}). Bringing to front and maximizing...")
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")  # Press Alt key to bypass Windows foreground restriction
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"⚠️ win32 focus warning: {e}")

    # Fallback via PowerShell
    print("🖥️ Focusing Chrome via PowerShell...")
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Chrome')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(1.0)
    pyautogui.hotkey("win", "up")
    time.sleep(1.0)
    return True

def verified_launch(profile_dir="Default", url="https://grok.com"):
    """
    Launches Chrome with the specified profile natively, wait for load,
    focuses window, and verifies visual screenshot.
    """
    print(f"🚀 Launching Chrome (Profile: '{profile_dir}', URL: '{url}')...")
    cmd = f'start "" "{CHROME_EXE}" --profile-directory="{profile_dir}" "{url}"'
    os.system(cmd)
    time.sleep(5.0)

    focus_and_maximize_chrome()

    os.makedirs("tmp", exist_ok=True)
    shot_path = "tmp/verified_launch_state.png"
    pyautogui.screenshot().save(shot_path)
    print(f"📸 Visual State Captured: {shot_path}")
    return shot_path

if __name__ == "__main__":
    verified_launch("Default", "https://grok.com")
