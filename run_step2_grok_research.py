"""
Step 2: 6-Turn Grok Research Sequence Execution for Autobot
"""
import os
import sys
import time
import subprocess
import pyperclip
import pyautogui
import win32gui
import win32con
import win32com.client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
        except Exception:
            pass
    return False

PROMPTS = [
    "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory.",
    "Deep-dive into Bound States in the Continuum (BICs) and switching mechanisms in perovskite microcavities for optical memory.",
    "Derive the mathematical framework (Hamiltonian, polariton dispersion, rate equations) for perovskite polariton exciton optical memory.",
    "Detail device architecture & materials specifications (e.g. CH3NH3PbI3, room-temperature operation, optical switching threshold).",
    "Provide key academic references and bibtex entries for perovskite polariton optical memory research.",
    "Synthesize a full, complete, compilation-ready LaTeX document (with \\documentclass{article}, preamble, sections, equations, bibtex references) compiling all research findings."
]

def main():
    print("💬 STEP 2: Beginning 6-Turn Grok Research Sequence...")
    os.makedirs("tmp", exist_ok=True)

    for turn_num, prompt in enumerate(PROMPTS, 1):
        print(f"\n--- Turn {turn_num}/6 ---")
        print(f"  Prompt: {prompt[:80]}...")
        focus_chrome()

        pyperclip.copy(prompt)
        time.sleep(0.3)

        # Click chat input box in the middle/lower center of the page
        pyautogui.click(960, 600)
        time.sleep(0.3)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)
        pyautogui.press("enter")

        print("  ⏳ Waiting 15 seconds for Grok output...")
        time.sleep(15.0)

        shot_path = f"tmp/step2_turn_{turn_num}.png"
        pyautogui.screenshot().save(shot_path)
        print(f"  📸 Saved screenshot: {shot_path}")

    print("\n✅ STEP 2 COMPLETE: All 6 turns executed.")

if __name__ == "__main__":
    main()
