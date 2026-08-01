import os
import sys
import time
import pyautogui
import win32gui
import win32con
import win32com.client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def force_chrome_foreground():
    print("🔎 Searching for Chrome window...")
    found_hwnd = None

    def enum_cb(hwnd, extra):
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Chrome" in title or "Grok" in title or "Overleaf" in title:
                print(f"  FOUND: hwnd={hwnd}, title='{title}'")
                found_hwnd = hwnd
                return False
        return True

    win32gui.EnumWindows(enum_cb, None)

    if found_hwnd:
        print(f"⚡ Bringing hwnd={found_hwnd} to foreground...")
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%") # Press Alt key
        win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
        win32gui.SetForegroundWindow(found_hwnd)
        time.sleep(1.5)
        return True
    else:
        print("❌ Chrome window not found.")
        return False

def main():
    if force_chrome_foreground():
        os.makedirs("tmp", exist_ok=True)
        out_path = "tmp/chrome_win32_focused.png"
        pyautogui.screenshot().save(out_path)
        print(f"📸 Saved screenshot: {out_path}")

if __name__ == "__main__":
    main()
