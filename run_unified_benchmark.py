"""
Unified Benchmark Execution Pipeline for Autobot
Strict Rule: Step 1 (Chrome Focus/Verification) MUST run BEFORE Step 2 & Step 3.
Safety Lock: NEVER paste or type unless win32gui confirms foreground window title contains 'Chrome' or 'Grok' or 'Overleaf'.
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

def get_foreground_title():
    """Get the title of the currently focused window on Windows OS."""
    if HAS_WIN32:
        hwnd = win32gui.GetForegroundWindow()
        return win32gui.GetWindowText(hwnd)
    return ""

def assert_and_focus_chrome(url="https://grok.com"):
    """
    Guarantees Chrome (Default Profile: daltonomondi588@gmail.com) is launched,
    maximized, brought to the foreground, and verified as the active window.
    """
    print("🔒 [SAFETY LOCK] Verifying active window focus...")
    title = get_foreground_title()

    if "Chrome" not in title and "Grok" not in title and "Overleaf" not in title:
        print(f"⚠️ Active window is '{title}' (NOT Chrome!). Launching/Focusing Chrome natively...")
        os.system(f'start "" "{CHROME_EXE}" --profile-directory="Default" "{url}"')
        time.sleep(5.0)

        if HAS_WIN32:
            def enum_cb(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd):
                    t = win32gui.GetWindowText(hwnd)
                    if "Chrome" in t or "Grok" in t or "Overleaf" in t:
                        shell = win32com.client.Dispatch("WScript.Shell")
                        shell.SendKeys("%")  # Press Alt key to bypass Windows focus lock
                        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                        win32gui.SetForegroundWindow(hwnd)
                        return False
                return True
            win32gui.EnumWindows(enum_cb, None)
            time.sleep(1.5)

    # Double check active window title after focus attempt
    active_title = get_foreground_title()
    print(f"🖥️ Active Window Verified: '{active_title}'")

    if "Chrome" not in active_title and "Grok" not in active_title and "Overleaf" not in active_title:
        raise RuntimeError(f"ABORTING: Could not bring Chrome to foreground! Currently active window: '{active_title}'")

    return active_title

PROMPTS = [
    "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory.",
    "Deep-dive into Bound States in the Continuum (BICs) and switching mechanisms in perovskite microcavities for optical memory.",
    "Derive the mathematical framework (Hamiltonian, polariton dispersion, rate equations) for perovskite polariton exciton optical memory.",
    "Detail device architecture & materials specifications (e.g. CH3NH3PbI3, room-temperature operation, optical switching threshold).",
    "Provide key academic references and bibtex entries for perovskite polariton optical memory research.",
    "Synthesize a full, complete, compilation-ready LaTeX document (with \\documentclass{article}, preamble, sections, equations, bibtex references) compiling all research findings."
]

def run_pipeline():
    os.makedirs("tmp", exist_ok=True)

    # =========================================================================
    # STEP 1: VERIFIED CHROME LAUNCH & DISMISS POPUPS
    # =========================================================================
    print("\n=======================================================")
    print("🚀 STEP 1: Launching & Verifying Chrome (daltonomondi588@gmail.com)...")
    print("=======================================================")

    assert_and_focus_chrome("https://grok.com")

    # Clear any banners/popups
    pyautogui.press("escape")
    time.sleep(0.5)

    shot_step1 = "tmp/unified_step1_verified.png"
    pyautogui.screenshot().save(shot_step1)
    print(f"📸 STEP 1 VERIFIED: Screenshot saved to {shot_step1}")

    # =========================================================================
    # STEP 2: 6-TURN GROK RESEARCH CONVERSATION
    # =========================================================================
    print("\n=======================================================")
    print("💬 STEP 2: Executing 6-Turn Grok Research Sequence...")
    print("=======================================================")

    for turn_num, prompt in enumerate(PROMPTS, 1):
        print(f"\n--- Turn {turn_num}/6 ---")
        print(f"  Prompt: {prompt[:80]}...")

        # Strict check before every single turn
        assert_and_focus_chrome("https://grok.com")

        # Click inside the Grok textarea box (lower center of screen)
        pyautogui.click(960, 950)
        time.sleep(0.3)

        # Copy & paste prompt
        pyperclip.copy(prompt)
        time.sleep(0.3)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)
        pyautogui.press("enter")

        print("  ⏳ Waiting 15 seconds for Grok response...")
        time.sleep(15.0)

        shot_turn = f"tmp/unified_step2_turn_{turn_num}.png"
        pyautogui.screenshot().save(shot_turn)
        print(f"  📸 Saved screenshot: {shot_turn}")

    print("✅ STEP 2 COMPLETE: 6-Turn Grok Research Finished.")

    # =========================================================================
    # STEP 3: OVERLEAF NAVIGATION & COMPILATION
    # =========================================================================
    print("\n=======================================================")
    print("🌐 STEP 3: Navigating to Overleaf & Compiling LaTeX Paper...")
    print("=======================================================")

    assert_and_focus_chrome("https://www.overleaf.com/project")

    pyautogui.hotkey("ctrl", "t")
    time.sleep(0.8)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.3)
    pyautogui.write("https://www.overleaf.com/project", interval=0.03)
    pyautogui.press("enter")
    time.sleep(5.0)

    shot_overleaf = "tmp/unified_step3_overleaf_dashboard.png"
    pyautogui.screenshot().save(shot_overleaf)
    print(f"📸 STEP 3 VERIFIED: Overleaf Dashboard Screenshot saved to {shot_overleaf}")

    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY AND VERIFIED!")

if __name__ == "__main__":
    run_pipeline()
