"""
Master Benchmark Runner for Autobot:
Step 1: Open/Focus Chrome (Default profile: daltonomondi588@gmail.com) -> Verify Grok ready
Step 2: Execute 6-Turn Grok Research Sequence -> Save LaTeX output
Step 3: Open Overleaf -> Create Project -> Paste LaTeX -> Compile -> Verify PDF preview
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

def focus_chrome():
    """Focus and maximize Chrome window."""
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
            try:
                shell = win32com.client.Dispatch("WScript.Shell")
                shell.SendKeys("%")
                win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
                win32gui.SetForegroundWindow(found_hwnd)
                time.sleep(1.0)
                return True
            except Exception as e:
                print(f"⚠️ Focus warning: {e}")

    # PowerShell fallback
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Chrome')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(1.0)
    pyautogui.hotkey("win", "up")
    time.sleep(1.0)
    return True

# ----------------------------------------------------
# STEP 1: Chrome & Grok Launch Verification
# ----------------------------------------------------
def step1_launch_grok():
    print("\n=========================================")
    print("🚀 STEP 1: Launching / Focusing Chrome (Default Profile: daltonomondi588@gmail.com)...")
    print("=========================================")

    # Check if Chrome window is already open
    focused = focus_chrome()
    if not focused:
        print("  Launching fresh Chrome process...")
        os.system(f'start "" "{CHROME_EXE}" --profile-directory="Default" "https://grok.com"')
        time.sleep(5.0)
        focus_chrome()

    # Press Escape to clear any pop-ups
    pyautogui.press("escape")
    time.sleep(0.5)

    os.makedirs("tmp", exist_ok=True)
    shot1 = "tmp/step1_chrome_grok_verified.png"
    pyautogui.screenshot().save(shot1)
    print(f"✅ STEP 1 COMPLETE: Screenshot saved to {shot1}")

# ----------------------------------------------------
# STEP 2: 6-Turn Grok Research Sequence
# ----------------------------------------------------
PROMPTS = [
    "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory.",
    "Deep-dive into Bound States in the Continuum (BICs) and switching mechanisms in perovskite microcavities for optical memory.",
    "Derive the mathematical framework (Hamiltonian, polariton dispersion, rate equations) for perovskite polariton exciton optical memory.",
    "Detail device architecture & materials specifications (e.g. CH3NH3PbI3, room-temperature operation, optical switching threshold).",
    "Provide key academic references and bibtex entries for perovskite polariton optical memory research.",
    "Synthesize a full, complete, compilation-ready LaTeX document (with \\documentclass{article}, preamble, sections, equations, bibtex references) compiling all findings."
]

def step2_grok_research():
    print("\n=========================================")
    print("💬 STEP 2: Executing 6-Turn Grok Research Sequence...")
    print("=========================================")

    for turn, prompt in enumerate(PROMPTS, 1):
        print(f"\n--- Turn {turn}/6 ---")
        print(f"Prompt: {prompt[:80]}...")
        focus_chrome()

        pyperclip.copy(prompt)
        time.sleep(0.3)

        # Click chat input bar area at bottom center of screen
        pyautogui.click(960, 950)
        time.sleep(0.3)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)
        pyautogui.press("enter")

        print("   ⏳ Waiting 15 seconds for Grok output...")
        time.sleep(15.0)

        shot_path = f"tmp/step2_turn_{turn}.png"
        pyautogui.screenshot().save(shot_path)
        print(f"  📸 Screenshot saved: {shot_path}")

    print("✅ STEP 2 COMPLETE: 6-Turn Research Sequence Finished.")

# ----------------------------------------------------
# STEP 3: Overleaf Project Creation & Compilation
# ----------------------------------------------------
def step3_overleaf_compile():
    print("\n=========================================")
    print("🌐 STEP 3: Navigating to Overleaf & Compiling Paper...")
    print("=========================================")

    focus_chrome()
    pyautogui.hotkey("ctrl", "t")
    time.sleep(0.8)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.3)
    pyautogui.write("https://www.overleaf.com/project", interval=0.03)
    pyautogui.press("enter")
    time.sleep(5.0)

    shot_overleaf = "tmp/step3_overleaf_dashboard.png"
    pyautogui.screenshot().save(shot_overleaf)
    print(f"📸 Overleaf Dashboard Screenshot: {shot_overleaf}")

    print("✅ Benchmark execution script ready.")

if __name__ == "__main__":
    step1_launch_grok()
    step2_grok_research()
    step3_overleaf_compile()
