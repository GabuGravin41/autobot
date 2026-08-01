import os
import sys
import time
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def focus_chrome_win32():
    try:
        import win32gui, win32con, win32process
        def enum_handler(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Chrome" in title:
                    print(f"  Found window: '{title}' (hwnd: {hwnd})")
                    # Restore if minimized
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    # Bring to top
                    win32gui.SetForegroundWindow(hwnd)
                    return False
            return True
        win32gui.EnumWindows(enum_handler, None)
    except Exception as e:
        print(f"win32gui error: {e}")

def main():
    print("🖥️ Searching and bringing Chrome window to foreground...")
    focus_chrome_win32()
    time.sleep(2.0)
    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/chrome_foreground_win32.png"
    pyautogui.screenshot().save(out_path)
    print(f"📸 Saved screenshot: {out_path}")

if __name__ == "__main__":
    main()
