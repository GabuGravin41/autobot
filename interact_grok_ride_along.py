"""
Ride-Along Grok Interaction Test
Wait for grok.com rendering, send Turn 1 prompt, and verify response visually.
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
                if "Chrome" in title or "Grok" in title or "Overleaf" in title:
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
    print("🖥️ Focusing open Chrome window...")
    focus_chrome()

    print("⏳ Waiting 6 seconds for grok.com page rendering...")
    time.sleep(6.0)

    os.makedirs("tmp", exist_ok=True)
    shot_rendered = "tmp/grok_rendered_state.png"
    pyautogui.screenshot().save(shot_rendered)
    print(f"📸 Rendered State Screenshot: {shot_rendered}")

    prompt = "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory."
    print(f"💬 Pasting prompt: {prompt}")

    pyperclip.copy(prompt)
    time.sleep(0.3)

    # Click in middle center of page where Grok input box renders
    pyautogui.click(960, 550)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.press("enter")

    print("⏳ Waiting 15 seconds for Grok response...")
    time.sleep(15.0)

    shot_response = "tmp/grok_turn1_response.png"
    pyautogui.screenshot().save(shot_response)
    print(f"📸 Response Screenshot: {shot_response}")

if __name__ == "__main__":
    main()
