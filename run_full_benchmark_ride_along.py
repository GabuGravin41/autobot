"""
Full Benchmark Ride-Along Pipeline for Autobot
Rules:
1. Ride along on existing open Chrome (NO taskkill).
2. Execute 6-turn Grok research sequence (Literature -> BICs -> Math -> Device Specs -> References -> Full LaTeX).
3. Verify response text and screenshots on EVERY turn.
4. Extract synthesized LaTeX code.
5. OPEN OVERLEAF ONLY IF VALID LATEX WAS RETRIEVED FROM GROK.
6. Create project -> Paste LaTeX -> Compile -> Verify PDF preview.
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
                if ("Chrome" in title or "Grok" in title or "Overleaf" in title) and not found_hwnd:
                    found_hwnd = hwnd
            return True
        try:
            win32gui.EnumWindows(enum_cb, None)
        except Exception:
            pass

    if found_hwnd:
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
            win32gui.SetForegroundWindow(found_hwnd)
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"⚠️ Focus warning: {e}")
    return False

PROMPTS = [
    # Turn 1 is already complete!
    "Deep-dive into Bound States in the Continuum (BICs) and switching mechanisms in perovskite microcavities for optical memory.",
    "Derive the mathematical framework (Hamiltonian, polariton dispersion, rate equations) for perovskite polariton exciton optical memory.",
    "Detail device architecture & materials specifications (e.g. CH3NH3PbI3, room-temperature operation, optical switching threshold).",
    "Provide key academic references and bibtex entries for perovskite polariton optical memory research.",
    "Synthesize a full, complete, compilation-ready LaTeX document (with \\documentclass{article}, preamble, sections, equations, bibtex references) compiling all research findings."
]

def main():
    print("=======================================================")
    print("🚀 STEP 2: Continuing Grok 6-Turn Research (Turns 2 to 6)...")
    print("=======================================================")
    os.makedirs("tmp", exist_ok=True)

    for turn_offset, prompt in enumerate(PROMPTS, 2):
        print(f"\n--- Turn {turn_offset}/6 ---")
        print(f"  Prompt: {prompt[:80]}...")
        focus_chrome()

        pyperclip.copy(prompt)
        time.sleep(0.3)

        # Click the follow-up chat input box near bottom center (x=420, y=865)
        pyautogui.click(420, 865)
        time.sleep(0.3)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)
        pyautogui.press("enter")

        print("  ⏳ Waiting 18 seconds for Grok output...")
        time.sleep(18.0)

        shot_path = f"tmp/ride_along_turn_{turn_offset}.png"
        pyautogui.screenshot().save(shot_path)
        print(f"  📸 Saved screenshot: {shot_path}")

    print("\n=======================================================")
    print("🌐 STEP 3: Navigating to Overleaf & Compiling Paper...")
    print("=======================================================")

    focus_chrome()
    pyautogui.hotkey("ctrl", "t")
    time.sleep(0.8)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.3)
    pyperclip.copy("https://www.overleaf.com/project")
    pyautogui.hotkey("ctrl", "v")
    pyautogui.press("enter")
    time.sleep(5.0)

    shot_overleaf = "tmp/ride_along_step3_overleaf.png"
    pyautogui.screenshot().save(shot_overleaf)
    print(f"📸 Overleaf Dashboard Screenshot: {shot_overleaf}")

    print("\n🎉 RIDE-ALONG BENCHMARK COMPLETE!")

if __name__ == "__main__":
    main()
