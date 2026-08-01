import os
import sys
import time
import pyautogui
import win32gui
import win32con
import win32com.client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def focus_chrome():
    found_hwnd = None
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
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
            win32gui.SetForegroundWindow(found_hwnd)
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"Focus error: {e}")
    return False

def main():
    print("🚀 STEP 1: Launching/Focusing Chrome with Default profile (daltonomondi588@gmail.com)...")
    os.system(f'start "" "{CHROME_EXE}" --profile-directory="Default" "https://grok.com"')
    time.sleep(5.0)

    focus_chrome()
    pyautogui.press("escape")
    time.sleep(0.5)

    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/step1_chrome_grok_verified.png"
    pyautogui.screenshot().save(out_path)
    print(f"📸 Step 1 Screenshot Saved: {out_path}")

if __name__ == "__main__":
    main()
