"""
Grok Research Benchmark Runner for Autobot
Executes 6-turn research query on Grok with visual screenshot verification at every step.
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
    if HAS_WIN32:
        def enum_windows_callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Chrome" in title or "Grok" in title:
                    try:
                        shell = win32com.client.Dispatch("WScript.Shell")
                        shell.SendKeys("%")
                        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                        win32gui.SetForegroundWindow(hwnd)
                        return False
                    except Exception:
                        pass
            return True
        win32gui.EnumWindows(enum_windows_callback, None)
        time.sleep(0.5)
    else:
        ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Chrome')"
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        time.sleep(0.5)

def dismiss_popup():
    print("🧹 Dismissing 'Restore pages' pop-up...")
    focus_chrome()
    # Press Escape twice
    pyautogui.press("escape")
    time.sleep(0.3)
    pyautogui.press("escape")
    time.sleep(0.5)
    os.makedirs("tmp", exist_ok=True)
    shot_path = "tmp/popup_dismissed.png"
    pyautogui.screenshot().save(shot_path)
    print(f"📸 Screenshot after popup dismissal: {shot_path}")

PROMPTS = [
    "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory.",
    "Deep-dive into Bound States in the Continuum (BICs) and switching mechanisms in perovskite microcavities for optical memory.",
    "Derive the mathematical framework (Hamiltonian, polariton dispersion, rate equations) for perovskite polariton exciton optical memory.",
    "Detail device architecture & materials specifications (e.g. CH3NH3PbI3, room-temperature operation, optical switching threshold).",
    "Provide key academic references and bibtex entries for perovskite polariton optical memory research.",
    "Synthesize a full, complete, compilation-ready LaTeX document (\documentclass{article}, preamble, equations, tables, sectioning, references) compiling all research findings from above."
]

def send_prompt(prompt_text, turn_num):
    print(f"\n💬 Executing Turn {turn_num}/6...")
    print(f"   Prompt: {prompt_text[:80]}...")
    focus_chrome()

    # Copy prompt text to clipboard and paste to avoid typing errors
    pyperclip.copy(prompt_text)
    time.sleep(0.3)

    # Click near lower middle of page to focus chat input
    pyautogui.click(960, 920)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.press("enter")

    print("   ⏳ Waiting 15 seconds for Grok to generate response...")
    time.sleep(15.0)

    shot_path = f"tmp/grok_turn_{turn_num}.png"
    pyautogui.screenshot().save(shot_path)
    print(f"📸 Screenshot Turn {turn_num}: {shot_path}")

def main():
    dismiss_popup()
    for i, p in enumerate(PROMPTS, 1):
        send_prompt(p, i)
    print("\n✅ Grok Research Benchmark 6-Turn Sequence Completed!")

if __name__ == "__main__":
    main()
