"""
Overleaf Workflow - Vision-Based (uses pixel color to verify state at each step)
Flow:
  1. Focus Chrome / Overleaf
  2. Screenshot to read current "New project" button Y
  3. Click it → wait → screenshot → verify dropdown appeared
  4. Click "Blank project" (first item in dropdown) → wait → screenshot
  5. Type project name → Enter → wait → screenshot of editor
"""
import os
import sys
import time
import pyperclip
import pyautogui
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import win32gui
    import win32con
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

os.makedirs("tmp", exist_ok=True)

def focus_overleaf():
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

def screenshot(name):
    path = f"tmp/{name}"
    pyautogui.screenshot().save(path)
    print(f"  📸 {path}")
    return path

def find_green_button(img_path):
    """Scan image left column (x=0..220) for green #49a54b-ish pixels.
    Return center y of the topmost horizontal band of green."""
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    scan_x = 103  # left sidebar column
    for y in range(150, 450):
        r, g, b = img.getpixel((scan_x, y))
        # Overleaf green is roughly: r≈55, g≈153, b≈88  OR  r≈35, g≈130, b≈65
        if g > 120 and g > r * 1.5 and g > b * 1.2 and r < 120 and b < 120:
            print(f"  🟢 Found green button pixel at ({scan_x}, {y}) → RGB({r},{g},{b})")
            return y
    return None

def main():
    print("="*60)
    print("🌿 OVERLEAF WORKFLOW - VISION GUIDED")
    print("="*60)

    focus_overleaf()
    state0 = screenshot("v_step0_baseline.png")

    # -- Find "New project" green button in screenshot ----------------
    green_y = find_green_button(state0)
    if green_y is None:
        print("  ⚠️  Could not auto-detect green button. Using fallback y=255.")
        green_y = 255
    btn_x, btn_y = 103, green_y
    print(f"  → Button at ({btn_x}, {btn_y})")

    # -- STEP 1: Click "New project" ----------------------------------
    print("\n▶ STEP 1: Clicking 'New project' green button...")
    pyautogui.click(btn_x, btn_y)
    time.sleep(1.5)   # let dropdown render
    state1 = screenshot("v_step1_dropdown_open.png")

    # Check if dropdown opened: look for white text just BELOW button
    # "Blank project" will be at btn_y + ~30..50
    blank_y = btn_y + 30   # first menu item is ~30px below button bottom
    blank_x = btn_x
    print(f"\n▶ STEP 2: Clicking 'Blank project' at ({blank_x}, {blank_y})...")
    pyautogui.click(blank_x, blank_y)
    time.sleep(1.5)
    state2 = screenshot("v_step2_after_blank.png")

    # -- STEP 3: Type project name ------------------------------------
    print("\n▶ STEP 3: Typing project name...")
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyperclip.copy("Perovskite Polariton Memory")
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    screenshot("v_step3_name_typed.png")

    pyautogui.press("enter")
    print("  ⏳ Waiting 8s for Overleaf editor to open...")
    time.sleep(8.0)
    screenshot("v_step4_editor.png")

    print("\n✅ Done. Check tmp/v_step*.png to verify each step.")

if __name__ == "__main__":
    main()
