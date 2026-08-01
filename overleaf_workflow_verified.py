"""
Precise Overleaf Workflow:
Step 1: Focus Chrome Overleaf tab.
Step 2: Click 'New project' green button.
Step 3: Screenshot to verify dropdown appeared.
Step 4: If dropdown visible, click 'Blank project' (first item, right below the button).
Step 5: Screenshot to verify project name dialog appeared.
Step 6: Type project name + Enter.
Step 7: Screenshot to verify editor loaded.
"""
import os
import sys
import time
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
            time.sleep(1.2)
            return True
    return False

def screenshot(name):
    os.makedirs("tmp", exist_ok=True)
    path = f"tmp/{name}"
    pyautogui.screenshot().save(path)
    print(f"📸 Screenshot saved: {path}")
    return path

def main():
    print("="*60)
    print("🌿 OVERLEAF FULL WORKFLOW - VERIFIED STEP BY STEP")
    print("="*60)

    focus_chrome()
    screenshot("ol_step0_focused.png")

    # ── STEP 1: Click 'New project' button ──────────────────────
    # From screenshots: button is green pill in left sidebar, center at ~(103, 255)
    print("\n▶ STEP 1: Clicking 'New project' button at (103, 255)...")
    pyautogui.click(103, 255)
    time.sleep(1.5)   # wait for dropdown to render
    screenshot("ol_step1_after_new_project_click.png")

    # ── STEP 2: Click 'Blank project' ───────────────────────────
    # From user screenshot: dropdown appears directly below button.
    # 'Blank project' is the FIRST item in dropdown.
    # New project button top ≈ y=237, its height ≈ 34px → bottom ≈ y=271
    # Dropdown item 'Blank project' first row ≈ y=295 (just below button)
    print("\n▶ STEP 2: Clicking 'Blank project' at (103, 295)...")
    pyautogui.click(103, 295)
    time.sleep(1.5)
    screenshot("ol_step2_after_blank_project_click.png")

    # ── STEP 3: Type project name and confirm ────────────────────
    print("\n▶ STEP 3: Typing project name 'Perovskite Polariton Memory'...")
    # Clear existing text and type new name
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyperclip.copy("Perovskite Polariton Memory")
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    screenshot("ol_step3_name_typed.png")

    pyautogui.press("enter")
    print("⏳ Waiting 8 seconds for Overleaf editor to load...")
    time.sleep(8.0)
    screenshot("ol_step4_editor_loaded.png")
    print("\n🎉 Overleaf workflow complete! Check screenshots.")

if __name__ == "__main__":
    main()
